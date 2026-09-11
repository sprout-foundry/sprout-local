package main

// ---------------------------------------------------------------------------
// chatllm — interactive terminal chat over in-process sinter inference.
//
// Patterns follow ../gmitllm: single Go binary, sinter (in-process MLX)
// engine, shared ~/dev/llm-models model root, LOCAL_MODEL_DIR override.
//
// REPL: read → expand multiline → dispatch slash commands → stream the
// response token by token → append to history → log the exchange.
// ---------------------------------------------------------------------------

import (
	"bufio"
	"context"
	"encoding/json"
	"flag"
	"fmt"
	"log"
	"os"
	"strings"
	"sync"

	"github.com/sprout-foundry/seed/core"
	"github.com/sprout-foundry/sinter/llm"
)

// Session defaults.
const (
	historyLimit = 200             // hard cap on retained messages (100 turns)
	logDirName   = ".sprout_local_sessions" // session log dir (bash tool heritage)
)

// currentGen tracks the in-flight generation so Ctrl-C can cancel it.
type activeGen struct {
	cancel context.CancelFunc
}

// pipeRequest is the stdin payload for -transcript mode: a full
// conversation transcript (sinter llm.ChatMessage is {role, content}).
type pipeRequest struct {
	Messages []llm.ChatMessage `json:"messages"`
}

var (
	mu         sync.Mutex
	currentGen *activeGen
)

// registerGen stores the active generation's cancel fn; the returned func
// clears it when generation completes normally.
func registerGen(cancel context.CancelFunc) func() {
	mu.Lock()
	gen := &activeGen{cancel: cancel}
	currentGen = gen
	mu.Unlock()
	return func() {
		mu.Lock()
		if currentGen == gen { // pointer identity
			currentGen = nil
		}
		mu.Unlock()
	}
}

func main() {
	// ── CLI flags ────────────────────────────────────────────────────────
	//   -m / --model-dir : model directory override (same as LOCAL_MODEL_DIR)
	//   -s / --system    : system prompt for the session
	//   -p / --prompt    : one-shot prompt (print response and exit)
	//   -no-log          : disable session logging
	//   -pull [name]     : download a catalog model, then continue (chat
	//                      starts with the freshly pulled model; bare -pull
	//                      lists available models)
	// ─────────────────────────────────────────────────────────────────────
	fs := flag.NewFlagSet("chatllm", flag.ExitOnError)
	flagModelDir := fs.String("m", "", "Model directory (overrides LOCAL_MODEL_DIR)")
	flagSystem := fs.String("s", "", "System prompt for the session")
	flagPrompt := fs.String("p", "", "One-shot prompt: stream the response and exit")
	flagNoLog := fs.Bool("no-log", false, "Disable session logging")
	flagPull := fs.Bool("pull", false, "Download a catalog model (name as next arg; bare -pull lists)")
	flagTranscript := fs.Bool("transcript", false, "Read a conversation from stdin and stream a reply (pipe mode for scripting)")
	flagEOM := fs.Bool("eom", false, "With -transcript: append an end-of-message marker line")
	flagServe := fs.Bool("serve", false, "Host the embedded web chat UI (WebSocket + sinter in-process)")
	flagAddr := fs.String("addr", "127.0.0.1:8321", "Listen address for -serve")
	flagVerbose := fs.Bool("v", false, "Verbose: show engine load/debug output")
	fs.Parse(os.Args[1:])

	// Engine chatter (sinter load/warmup lines) is filtered out unless -v
	// or CHATLLM_DEBUG is set. Installed before any model loads.
	installLogFilter(*flagVerbose || os.Getenv("CHATLLM_DEBUG") != "")

	// Skills (~/.chatllm/skills/*.json) load once at startup; /tools
	// reloads them in the REPL. Load failures are non-fatal noise.
	if n, errs := reloadSkills(); len(errs) > 0 {
		for _, err := range errs {
			log.Printf("skill: %v", err)
		}
	} else if n > 0 {
		log.Printf("loaded %d skill%s from %s", n, plural(n), skillsDir())
	}

	// -serve: warm the default model's system+tools prefix in the
	// background so the first web turn delta-prefills instead of paying
	// the full prefill. Loading the weights is the cost; the server was
	// going to do it on first request anyway.
	if *flagServe {
		executor := core.ToolExecutor(core.NoopExecutor)
		if toolsRequested {
			executor = newToolExecutor(nil)
		}
		dir := resolveModelDir()
		if dir != "" {
			go warmModel(dir, *flagSystem, toolsRequested, executor)
		}
	}

	// -pull: fetch the model first, then drop into normal startup with the
	// new model selected. A bare -pull lists the catalog and exits.
	if *flagPull {
		name := strings.TrimSpace(strings.Join(fs.Args(), " "))
		if name == "" {
			printPullList()
			return
		}
		m, err := findCatalogModel(name)
		if err != nil {
			log.Fatalf("Fatal: %v", err)
		}
		dest, err := downloadModel(ctxBg(), m)
		if err != nil {
			log.Fatalf("Fatal: %v", err)
		}
		fmt.Printf("Pulled %s → %s\n", m.Name, dest)
		// Select the freshly pulled model for this run.
		os.Setenv("LOCAL_MODEL_DIR", dest)
	}

	// -m override: applied before the model loads. resolveModelDir reads the
	// env var, so we set it for this process.
	if *flagModelDir != "" {
		os.Setenv("LOCAL_MODEL_DIR", *flagModelDir)
	}
	// -serve: host the embedded web UI + WebSocket API. Session logging is
	// REPL-only, so this path never touches the session log.
	if *flagServe {
		serveHosts(*flagAddr)
		return
	}

	if !*flagNoLog {
		logPath = defaultLogPath()
		logEnabled = true
	}

	// Startup probe — mirrors gmitllm's modelBackend(): fail fast with a
	// clear message instead of a deep sinter load error on first message.
	engine, modelPath := modelBackend()

	startSignalWatch()

	// One-shot mode: single completion, no REPL.
	if *flagPrompt != "" {
		runOneShot(*flagSystem, *flagPrompt, engine, modelPath)
		return
	}

	// Transcript mode: read a JSON conversation from stdin, stream the
	// assistant reply, and exit. Used by chat/server.py to drive the same
	// engine over a pipe. With -eom, an end-of-message marker line is
	// appended for consumers that stream without EOF visibility.
	if *flagTranscript {
		if err := runOneShotPipe(engine, modelPath, *flagNoLog, *flagEOM); err != nil {
			log.Fatalf("Fatal: %v", err)
		}
		return
	}

	runREPL(engine, modelPath, *flagSystem)
}

// runOneShotPipe reads {"messages":[{role,content}...]} JSON from stdin,
// streams the assistant response to stdout, and (with eom) finishes with
// the endOfMessage marker line. Diagnostics stay on stderr. When the
// transcript has no messages, nothing is generated — with eom the marker
// is still emitted so pipe consumers see a well-formed (empty) exchange.
func runOneShotPipe(engine, modelPath string, noLog, eom bool) error {
	dec := json.NewDecoder(bufio.NewReader(os.Stdin))
	var req pipeRequest
	if err := dec.Decode(&req); err != nil {
		return fmt.Errorf("transcript: decode stdin JSON: %w", err)
	}

	if len(req.Messages) == 0 {
		if eom {
			fmt.Println(endOfMessage)
		}
		return nil
	}

	text, err := streamChat(ctxBg(), req.Messages, stdoutPrinter().writeDelta)
	fmt.Println()
	if err != nil {
		return err
	}
	if eom {
		fmt.Println(endOfMessage)
	}

	if !noLog {
		var system string
		if req.Messages[0].Role == "system" {
			system = req.Messages[0].Content
		}
		last := req.Messages[len(req.Messages)-1]
		appendLog("user", modelPath, system, last.Content, "")
		appendLog("assistant", modelPath, system, "", text)
	}
	return nil
}

// runOneShot streams a single system+user completion and exits.
func runOneShot(systemPrompt, prompt, engine, modelPath string) {
	messages := []llm.ChatMessage{}
	if systemPrompt != "" {
		messages = append(messages, llm.ChatMessage{Role: "system", Content: systemPrompt})
	}
	messages = append(messages, llm.ChatMessage{Role: "user", Content: prompt})

	text, err := streamChat(ctxBg(), messages, stdoutPrinter().writeDelta)
	fmt.Println() // newline after the streamed response
	if err != nil {
		log.Fatalf("Fatal: %v", err)
	}
	appendLog("user", modelPath, systemPrompt, prompt, "")
	appendLog("assistant", modelPath, systemPrompt, "", text)
}

// runREPL is the interactive loop: read → expand multiline → dispatch →
// run the seed agent turn → log. activeModel is the session's model
// directory; slash commands re-point it. The agent is rebuilt on model
// switches (state carries over via export/import).
func runREPL(engine, modelPath, systemPrompt string) {
	reader := bufio.NewReader(os.Stdin)
	activeModel := modelPath
	_ = activeModel // used by dispatchCommand via replState

	ui := &termUI{reader: reader}
	st := &replState{systemPrompt: systemPrompt, model: modelPath, ui: ui}
	st.newAgent()

	printWelcome(engine, modelPath)

	for {
		fmt.Print("\n" + ansiStyle(">>>", ansiMagenta) + " ")
		line, err := reader.ReadString('\n')
		if err != nil { // EOF (Ctrl-D) ends the session
			fmt.Println()
			break
		}
		line = strings.TrimSpace(line)
		if line == "" {
			continue
		}

		// Multiline input: a line that is exactly """ opens a block that
		// runs until a closing """ line.
		if line == `"""` {
			line = readMultiline(reader)
			if line == "" {
				continue
			}
		}

		// Slash commands do not touch the conversation history.
		if strings.HasPrefix(line, "/") {
			if cmd, args := splitCommand(line); dispatchCommand(cmd, args, st) {
				break // /exit, /quit, /q
			}
			continue
		}

		// ── Chat turn ────────────────────────────────────────────────
		turnErr := runChatTurn(st, line)
		if turnErr != nil {
			fmt.Printf("%s %v\n", ansiStyle("Error:", ansiRed), turnErr)
			appendLog("error", st.model, st.systemPrompt, line, errString(turnErr))
		}
	}
	fmt.Println("Bye!")
}

// replState carries everything slash commands and chat turns need: the
// seed agent, its provider (for the display sink), the session model,
// system prompt, and terminal UI.
type replState struct {
	agent        *core.Agent
	provider     *sinterProvider
	model        string
	systemPrompt string
	ui           *termUI
}

// termUI adapts the terminal to seed's UI interface: prompts and
// confirmations read from the REPL's stdin reader (shared with the main
// loop so input ordering stays sane), output goes to stdout.
type termUI struct {
	reader *bufio.Reader
}

func (u *termUI) Prompt(message string) (string, error) {
	fmt.Print(message + " ")
	line, err := u.reader.ReadString('\n')
	return strings.TrimSpace(line), err
}

// Confirm implements seed's y/N gate. Empty reply = no.
func (u *termUI) Confirm(message string) (bool, error) {
	fmt.Printf("%s [y/N] ", message)
	line, err := u.reader.ReadString('\n')
	if err != nil {
		return false, err
	}
	switch strings.ToLower(strings.TrimSpace(line)) {
	case "y", "yes":
		return true, nil
	}
	return false, nil
}

func (u *termUI) Print(message string)     { fmt.Print(message) }
func (u *termUI) PrintLine(message string) { fmt.Println(message) }

// newAgent builds a fresh seed agent (no conversation state). Used at
// startup and by /new.
func (s *replState) newAgent() {
	s.rebuildAgent(false)
}

// rebuildAgent builds the seed agent for the current model/prompt/tools
// state. Executor choice: NoopExecutor (zero tools) when tools are off —
// seed then sends no Tools to the provider and the model is never told
// the protocol exists. With carry, the previous agent's conversation
// state is exported and re-imported, so /model, /pull, /system and /tools
// keep the chat (seed's state survives model switches; the web UI already
// relies on that).
func (s *replState) rebuildAgent(carry bool) {
	var saved []byte
	if carry && s.agent != nil {
		saved, _ = s.agent.ExportState()
	}
	executor := core.ToolExecutor(core.NoopExecutor)
	if toolsRequested {
		executor = newToolExecutor(s.ui)
	}
	s.provider = newSinterProvider(s.model)
	agent, err := core.NewAgent(core.Options{
		Provider:       s.provider,
		Executor:       executor,
		UI:             s.ui,
		SystemPrompt:   s.systemPrompt,
		MaxIterations:  maxToolSteps,
		EventPublisher: &replEvents{},
		Debug:          os.Getenv("SPROUT_LOCAL_SEED_DEBUG") != "",
		// In-process sinter has no transient network errors; a failure is
		// real (OOM, context overflow). Retry just stalls the UI.
		RetryConfig: core.RetryConfig{MaxAttempts: 1},
	})
	if err != nil {
		fmt.Printf("%s seed agent: %v\n", ansiStyle("Error:", ansiRed), err)
		s.agent = nil
		return
	}
	if len(saved) > 0 {
		if err := agent.ImportState(saved); err != nil {
			fmt.Printf("%s restoring history: %v\n", ansiStyle("Error:", ansiRed), err)
		}
	}
	s.agent = agent
}

// runChatTurn runs one user→assistant exchange through the seed agent
// loop (query → LLM → tool execution → final answer). The agent owns
// history, compaction, and tool iteration; chatllm supplies streaming
// display, tool-call status lines, and session logging.
func runChatTurn(st *replState, userLine string) error {
	printer := stdoutPrinter()
	st.provider.SetDisplay(printer.writeDelta)
	defer st.provider.SetDisplay(nil)

	startTurnMetrics()
	fmt.Println() // blank line before the response
	streamCtx, cancel := context.WithCancel(context.Background())
	done := registerGen(cancel)
	defer done()
	defer cancel()

	text, err := st.agent.RunStream(streamCtx, userLine)
	printer.Close()
	fmt.Println()
	fmt.Printf("%s%s%s\n", ansiGray, st.metricsLine(), ansiReset)

	if err != nil {
		if strings.Contains(errString(err), "context canceled") ||
			strings.Contains(errString(err), "interrupted") {
			fmt.Println("(cancelled — history unchanged)")
			return nil
		}
		return err
	}

	logTurn(st, userLine, text)
	return nil
}

// metricsLine renders the whole turn's stats for REPL display (sums every
// generation in the turn — tool round-trips included).
func (s *replState) metricsLine() string {
	m := turnMetrics()
	if !turnMetricsActive || m.GenTokens == 0 {
		return ""
	}
	return "  " + m.String()
}

// logTurn records a tools-on exchange in the session log: the user line
// plus every tool call/result the agent ran this turn.
func logTurn(st *replState, userLine, reply string) {
	appendLog("user", st.model, st.systemPrompt, userLine, "")
	for _, m := range st.agent.State().Messages() {
		if m.Role == "tool" {
			appendLog("tool", st.model, st.systemPrompt,
				m.Meta["tool_name"], firstLine(m.Content))
		}
	}
	appendLog("assistant", st.model, st.systemPrompt, "", reply)
}

// toolStreamFilter suppresses <tool_call>…</tool_call> blocks from the
// streamed display while the raw text still accumulates for parsing.
// Without tools enabled it is a pass-through. Clean deltas go to both the
// terminal printer and (optionally) seed's stream handler, so the agent's
// content buffer matches what the user saw. Implementation: hold back a
// tag-sized tail of the stream, watch for "<tool_call>"; once seen, go
// silent until the matching "</tool_call>" has passed. Text after the
// block (a preamble or follow-up prose) flows again normally.
type toolStreamFilter struct {
	printer *streamPrinter // terminal display (may be nil)
	onDelta func(string)   // seed stream handler feed (may be nil)
	tools   bool
	tail    string // held-back characters not yet emitted
	inCall  bool   // inside a <tool_call> block
}

const toolCallTag = "<tool_call>"
const toolCallTagEnd = "</tool_call>"
const streamHoldback = len(toolCallTag) + 8 // tag + slack for split spans

// emit fans a clean delta out to the terminal and the agent's buffer.
func (f *toolStreamFilter) emit(s string) {
	if s == "" {
		return
	}
	if f.printer != nil {
		f.printer.Write(s)
	}
	if f.onDelta != nil {
		f.onDelta(s)
	}
}

func (f *toolStreamFilter) write(delta string) {
	if !f.tools {
		f.emit(delta)
		return
	}
	f.tail += delta
	for {
		if f.inCall {
			end := strings.Index(f.tail, toolCallTagEnd)
			if end < 0 {
				// Keep only a holdback in case the closing tag is split.
				if len(f.tail) > streamHoldback {
					f.tail = f.tail[len(f.tail)-streamHoldback:]
				}
				return
			}
			f.tail = f.tail[end+len(toolCallTagEnd):]
			f.inCall = false
			continue
		}
		start := strings.Index(f.tail, toolCallTag)
		if start < 0 {
			// No opening tag in the buffer: emit all but the holdback.
			if len(f.tail) > streamHoldback {
				clean := f.tail[:len(f.tail)-streamHoldback]
				f.tail = f.tail[len(f.tail)-streamHoldback:]
				f.emit(clean)
			}
			return
		}
		if start > 0 {
			f.emit(f.tail[:start])
		}
		f.tail = f.tail[start+len(toolCallTag):]
		f.inCall = true
	}
}

// close flushes the held-back tail. If a call block is still open, only
// raw call markup is dropped: any prose in the tail (the model's words
// before it spiraled) still reaches the display, minus a partial "<…"
// tag fragment.
func (f *toolStreamFilter) close() {
	if !f.tools {
		return
	}
	if f.inCall {
		// Inside <tool_call>…: drop markup, but the model may have written
		// prose before the block opened — that already streamed. The tail
		// here is call markup (possibly unterminated); show nothing.
		return
	}
	if f.tail != "" {
		f.emit(f.tail)
		f.tail = ""
	}
}

// trimHistory caps the web UI's per-connection history at historyLimit
// (the REPL migrated to seed's compaction; the web loop keeps the simple
// cap until it moves onto the agent too).
func trimHistory(messages *[]llm.ChatMessage) {
	if len(*messages) <= historyLimit {
		return
	}
	var system *llm.ChatMessage
	rest := *messages
	if rest[0].Role == "system" {
		system = &rest[0]
		rest = rest[1:]
	}
	if len(rest) > historyLimit-1 { // system message counts toward the cap
		rest = rest[len(rest)-(historyLimit-1):]
	}
	out := make([]llm.ChatMessage, 0, historyLimit+1)
	if system != nil {
		out = append(out, *system)
	}
	out = append(out, rest...)
	*messages = out
}

// firstLine returns the first line of s, for one-line status display.
func firstLine(s string) string {
	if i := strings.IndexByte(s, '\n'); i >= 0 {
		return s[:i]
	}
	return s
}

// readMultiline reads lines until a closing """ line. Returns the joined
// text, or "" if EOF hit before any content.
func readMultiline(reader *bufio.Reader) string {
	var lines []string
	for {
		line, err := reader.ReadString('\n')
		if err != nil {
			break
		}
		if strings.TrimSpace(line) == `"""` {
			break
		}
		lines = append(lines, strings.TrimRight(line, "\n"))
	}
	return strings.Join(lines, "\n")
}

// ─── Commands ────────────────────────────────────────────────────────────

// splitCommand separates "/cmd args…" into (cmd, args).
func splitCommand(line string) (string, string) {
	line = strings.TrimSpace(strings.TrimPrefix(line, "/"))
	if idx := strings.IndexAny(line, " \t"); idx != -1 {
		return line[:idx], strings.TrimSpace(line[idx+1:])
	}
	return line, ""
}

// dispatchCommand runs a slash command. Returns true when the REPL should
// exit. Commands operate on the replState: the seed agent (history),
// session model, and system prompt.
func dispatchCommand(cmd, args string, st *replState) bool {
	switch cmd {
	case "exit", "quit", "q":
		return true
	case "new":
		st.newAgent() // fresh agent = fresh conversation state
		fmt.Println("New conversation started.")
	case "system":
		handleSystemCommand(args, st)
	case "model":
		handleModelCommand(args, &st.model)
		if st.model != "" {
			st.rebuildAgent(true) // new model, history kept
		}
	case "models":
		printModelList(st.model)
	case "pull":
		handlePullCommand(args, &st.model)
		if st.model != "" {
			st.rebuildAgent(true)
		}
	case "tools":
		handleToolsCommand(args)
		st.rebuildAgent(true) // executor swap: NoopExecutor ↔ toolExecutor
	case "history":
		printHistory(st.agent)
	case "help":
		printHelp()
	default:
		fmt.Printf("Unknown command /%s — try /help\n", cmd)
	}
	return false
}

// handleSystemCommand shows (no args) or sets (args) the system prompt,
// rebuilding the agent so the new prompt takes effect from the next turn.
func handleSystemCommand(args string, st *replState) {
	if args == "" {
		if st.systemPrompt == "" {
			fmt.Println("No system prompt set.")
		} else {
			fmt.Printf("System prompt: %s\n", st.systemPrompt)
		}
		return
	}
	st.systemPrompt = args
	st.rebuildAgent(true)
	fmt.Println("System prompt set.")
}

// printHistory dumps the seed agent's conversation state.
func printHistory(agent *core.Agent) {
	if agent == nil {
		fmt.Println("(no conversation)")
		return
	}
	msgs := agent.State().Messages()
	if len(msgs) == 0 {
		fmt.Println("(no messages yet)")
		return
	}
	for _, m := range msgs {
		switch m.Role {
		case "system":
			fmt.Printf("[system]    %s\n", m.Content)
		case "user":
			fmt.Printf("[user]      %s\n", m.Content)
		case "assistant":
			if len(m.ToolCalls) > 0 {
				names := make([]string, 0, len(m.ToolCalls))
				for _, tc := range m.ToolCalls {
					names = append(names, tc.Function.Name)
				}
				fmt.Printf("[assistant] %s %s\n", m.Content, strings.Join(names, ","))
				continue
			}
			fmt.Printf("[assistant] %s\n", m.Content)
		case "tool":
			fmt.Printf("[tool:%s]   %s\n", m.Meta["tool_name"], firstLine(m.Content))
		default:
			fmt.Printf("[%s] %s\n", m.Role, m.Content)
		}
	}
}

// ─── Display helpers ─────────────────────────────────────────────────────

func printWelcome(engine, modelPath string) {
	fmt.Printf("chatllm — %s engine, model: %s\n", engine, modelPath)
	fmt.Println("Type /help for commands. Ctrl-C cancels the current response; /exit quits.")
}

func printHelp() {
	fmt.Println("Commands:")
	fmt.Println("  /new             Start a new conversation (drops history, keeps system prompt)")
	fmt.Println("  /system [text]   Show or set the system prompt")
	fmt.Println("  /model [ref]     Show or switch the model (history is kept)")
	fmt.Println("  /models          List installed models (* marks the session model)")
	fmt.Println("  /pull [name]     Download a catalog model, switch to it (bare /pull lists)")
	fmt.Println("  /history         Show the conversation so far")
	fmt.Println("  /tools [on|off|yolo]  Toggle tool calling (read_file, write_file, run_command, web_fetch)")
	fmt.Println("  /exit            Quit (also /quit, /q)")
	fmt.Println()
	fmt.Println("Model refs: name under " + modelsRoot() + ", catalog name, or a path.")
	fmt.Println(`Multiline: type """ alone on a line, type your text, end with """ on its own line.`)
	fmt.Println(`One-shot:   chatllm -p "your question"`)
	fmt.Println(`Web UI:     chatllm -serve [-addr 127.0.0.1:8321]`)
}

// errString safely extracts the message from an error.
func errString(err error) string {
	if err == nil {
		return ""
	}
	return err.Error()
}

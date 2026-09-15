package main

// ---------------------------------------------------------------------------
// webui.go — embedded single-binary web chat.
//
// `chatllm -serve` hosts the UI (static assets embedded at build time via
// go:embed) plus a WebSocket API (/ws), model listing (/models), model
// downloads (/pull), and the OpenAI-compatible /v1 endpoint (apiserver.go)
// over in-process sinter inference. UI_SPROUT_LOCAL_DEV=1 serves assets
// from disk (chatllm/ui/) instead, so the UI can be edited without a
// rebuild.
//
// Each WebSocket connection owns a seed agent (tools + skills enabled via
// the per-turn "tools" flag; run_command auto-declines — see ensureAgent).
// Conversations persist to ~/.sprout-local/conversations/ after every turn.
//
// Protocol (JSON text frames; deltas are raw text):
//   client → server: {"model":"…","prompt":"…","tools":bool}
//                    {"list":true} | {"load":"id"} | {"delete":"id"}
//                    {"new_chat":true} | {"resume":"id"}
//                    {"pull":"name"} (download a catalog model into the
//                                     models root, then re-list)
//   server → client: raw text deltas; then {"done":true}
//                    {"status":"…"} (working / loading model … / generating)
//                    {"tool":"name"} / {"tool_done":"name","ok":bool}
//                    {"metrics":{prompt_tokens,gen_tokens,tps,pp,ctx…}}
//                    {"note":"…"} | {"error":"…"}
//                    {"pulled":"name","dir":"…"} (download complete)
//                    {"conv":"id"} | {"conversations":[…]} | {"transcript":[…]}
//                    {"deleted":"id"}
// ---------------------------------------------------------------------------

import (
	"context"
	"embed"
	"encoding/json"
	"errors"
	"fmt"
	"io/fs"
	"log"
	"net/http"
	"os"
	"path/filepath"
	"sort"
	"strings"
	"sync"
	"time"

	"github.com/gorilla/websocket"
	"github.com/sprout-foundry/seed/core"
	"github.com/sprout-foundry/sinter/llm"
	"github.com/sprout-foundry/sinter/llm/catalog"

	"github.com/sprout-foundry/sprout-local/internal/config"
	"github.com/sprout-foundry/sprout-local/internal/conversations"
	"github.com/sprout-foundry/sprout-local/internal/download"
	"github.com/sprout-foundry/sprout-local/internal/mdterm"
	"github.com/sprout-foundry/sprout-local/internal/paths"
	"github.com/sprout-foundry/sprout-local/internal/sysinfo"
	"github.com/sprout-foundry/sprout-local/internal/urlfetch"
)

//go:embed all:ui
var uiFS embed.FS

var upgrader = websocket.Upgrader{
	// The UI is served by this same binary; allow any origin so the page
	// also works when opened from a different local port during dev.
	CheckOrigin: func(r *http.Request) bool { return true },
}

type wsClient struct {
	conn *websocket.Conn
	send chan []byte

	mu       sync.Mutex // serializes frame writes from the pump
	agent    *core.Agent
	provider *sinterProvider
	agentKey string // model+tools config the agent was built for ("|")
	model    string // selected model name ("" = process default)
	dir      string // resolved model directory ("" = not yet resolved)
	conv     *conversations.Conversation
}

// ensureAgent returns the connection's seed agent, rebuilding it when the
// model or tools flag changed (conversation carried via state export).
// The web executor is the full tool set with one difference from the
// REPL: run_command's y/N confirm cannot be answered over the socket, so
// its UI.Confirm always declines — file tools and skills run agentically,
// shell commands stay REPL-only.
func (s *webServer) ensureAgent(c *wsClient, modelDir string, tools bool) error {
	key := modelDir + "|" + fmt.Sprintf("%v", tools)
	if c.agent != nil && c.agentKey == key {
		return nil
	}
	var saved []byte
	if c.agent != nil {
		saved, _ = c.agent.ExportState() // carry the conversation across
	}
	executor := core.ToolExecutor(core.NoopExecutor)
	if tools {
		executor = newToolExecutor(&decliningUI{})
	}
	c.provider = newSinterProvider(modelDir)
	c.provider.SetDisplayMode(mdterm.Raw) // web socket: plain text, no ANSI
	agent, err := core.NewAgent(core.Options{
		Provider:       c.provider,
		Executor:       executor,
		SystemPrompt:   webSystemPrompt(tools),
		MaxIterations:  config.MaxToolSteps,
		EventPublisher: &webEvents{c: c},
		RetryConfig:    core.RetryConfig{MaxAttempts: 1},
	})
	if err != nil {
		return err
	}
	if len(saved) > 0 {
		_ = agent.ImportState(saved)
	}
	c.agent = agent
	c.agentKey = key
	c.dir = modelDir

	// The model just loaded (or is about to be needed): warm the exact
	// system+tools prefix this agent will send, so the first turn on this
	// configuration delta-prefills. No-op if the slot already matches.
	go warmModel(modelDir, webSystemPrompt(tools), tools, executor)
	return nil
}

// warmPrefixesFor renders the prompt prefixes worth warming for a model
// with the given system prompt and tool set: exactly the static head
// (system message with the tools block) that every first turn begins
// with. FormatChatPrefix output is a true prefix of any later
// FormatChat on the same conversation head, which is what sinter's slot
// matcher requires. The tools block is rendered in the target model's
// protocol (from its directory), not the session's startup protocol —
// a mismatched block would never match the slot the provider actually
// renders on the first turn.
func warmPrefixesFor(m *llm.Model, modelDir, systemPrompt string, tools []core.Tool) []string {
	msgs := []llm.ChatMessage{{Role: "system", Content: systemPrompt}}
	if len(tools) > 0 {
		block := toolPromptBlockFromSeedFor(tools, toolProtocolForModelDir(modelDir))
		msgs[0].Content = strings.TrimRight(msgs[0].Content, "\n") + "\n\n" + block
	}
	return []string{m.FormatChatPrefix(msgs)}
}

// warmModel loads dir (if needed) and pre-fills the system-prefix KV slot
// so the first user turn delta-prefills instead of paying the full
// prefill. Fire-and-forget: errors are logged only.
func warmModel(modelDir, systemPrompt string, tools bool, executor core.ToolExecutor) {
	m, err := loadModelDir(modelDir)
	if err != nil {
		log.Printf("warm: %v", err)
		return
	}
	var toolsList []core.Tool
	if executor != nil {
		toolsList = executor.GetTools()
	}
	for _, prefix := range warmPrefixesFor(m, modelDir, systemPrompt, toolsList) {
		if err := m.WarmSystemPrefix(prefix); err != nil {
			log.Printf("warm: %v", err)
		}
	}
	log.Printf("chatllm: warmed system prefix for %s", filepath.Base(modelDir))
}

// webSystemPrompt returns the system prompt for web connections. With
// tools on, it must tell the model that file operations are REAL here —
// small models otherwise fall back to their pretrained "here's some code
// to copy-paste" behavior for create/write asks, declining to use tools
// that are actually available.
func webSystemPrompt(tools bool) string {
	env := config.EnvironmentContext()
	if !tools {
		return env
	}
	return env + "\n\nYou are a helpful assistant working on the user's local machine. " +
		"When the user asks you to create, write, save, or generate files or code, " +
		"actually perform the action with your tools (write_file for file creation, " +
		"read_file to inspect existing files) instead of printing code for the user to copy. " +
		"Choose sensible file names and directories when unspecified."
}

// decliningUI is seed's UI for web connections: nothing to confirm on,
// so run_command declines itself (reported to the model as a declined
// result it can narrate).
type decliningUI struct{}

func (u *decliningUI) Prompt(string) (string, error) { return "", nil }
func (u *decliningUI) Confirm(string) (bool, error)  { return false, nil }
func (u *decliningUI) Print(string)                  {}
func (u *decliningUI) PrintLine(string)              {}

// webEvents forwards seed tool events to the browser as chips.
type webEvents struct{ c *wsClient }

func (e *webEvents) Publish(eventType string, data interface{}) {
	switch eventType {
	case core.EventTypeToolStart:
		e.c.enqueue(map[string]string{"tool": eventString(data, "tool_name")})
	case core.EventTypeToolEnd:
		ok := eventString(data, "status") == core.ToolStatusCompleted
		e.c.enqueue(map[string]any{
			"tool_done": eventString(data, "tool_name"),
			"ok":        ok,
			"args":      eventJSON(data, "arguments"),
			"result":    truncateResultForDisplay(eventString(data, "result")),
		})
	}
}

// agentTranscript extracts the visible conversation (no system messages)
// from the agent's state.
func agentTranscript(a *core.Agent) []conversations.StoredMsg {
	if a == nil {
		return nil
	}
	out := []conversations.StoredMsg{}
	for _, m := range a.State().Messages() {
		if m.Role == "system" {
			continue
		}
		out = append(out, conversations.StoredMsg{Role: m.Role, Content: m.Content})
	}
	return out
}

// importTranscript builds agent state from persisted messages (used by
// resume/load so a rebuilt agent continues with full context).
func importTranscript(a *core.Agent, msgs []conversations.StoredMsg) error {
	state := core.AgentState{}
	for _, m := range msgs {
		state.Messages = append(state.Messages, core.Message{Role: m.Role, Content: m.Content})
	}
	b, err := json.Marshal(state)
	if err != nil {
		return err
	}
	return a.ImportState(b)
}

// ─── Model discovery ─────────────────────────────────────────────────────

// availableModels lists the MLX model directories under the shared models
// root (the same root -pull downloads into). Empty when an explicit model
// dir env (SPROUT_LOCAL_MODEL_DIR or the legacy LOCAL_MODEL_DIR) pins the
// process to one model.
func availableModels() []string {
	if _, pinned := modelDirEnv(); pinned != "" {
		return nil // pinned: no listing
	}
	root := paths.ModelsRoot()
	entries, err := os.ReadDir(root)
	if err != nil {
		return nil
	}
	var names []string
	for _, e := range entries {
		if e.IsDir() && paths.IsModelDir(filepath.Join(root, e.Name())) {
			names = append(names, e.Name())
		}
	}
	sort.Strings(names)
	return names
}

// resolveWebModelDir maps a UI model name to a model directory. Bare names
// resolve against the shared models root; absolute paths pass through.
// Empty name → the process default (resolveModelDir).
func resolveWebModelDir(name string) (string, error) {
	name = strings.TrimSpace(name)
	if name == "" {
		dir := resolveModelDir()
		if dir == "" {
			return "", fmt.Errorf("no model configured (set SPROUT_LOCAL_MODEL_DIR or -m)")
		}
		return dir, nil
	}
	if filepath.IsAbs(name) {
		if paths.IsModelDir(name) {
			return name, nil
		}
		return "", fmt.Errorf("%s is not an MLX-format model directory", name)
	}
	cand := filepath.Join(paths.ModelsRoot(), name)
	if paths.IsModelDir(cand) {
		return cand, nil
	}
	return "", fmt.Errorf("unknown model %q", name)
}

// ─── WebSocket handling ──────────────────────────────────────────────────

func (c *wsClient) enqueue(v any) {
	b, err := json.Marshal(v)
	if err != nil {
		return
	}
	select {
	case c.send <- b:
	default: // queue full: client too slow, drop the frame
	}
}

// serveWS upgrades and runs one connection: a writer pump drains c.send,
// the read loop turns incoming prompts into generations. Each connection
// gets its own context so a disconnect cancels an in-flight generation.
func (s *webServer) serveWS(w http.ResponseWriter, r *http.Request) {
	conn, err := upgrader.Upgrade(w, r, nil)
	if err != nil {
		return
	}
	defer conn.Close()

	c := &wsClient{conn: conn, send: make(chan []byte, 64)}
	ctx, cancel := context.WithCancel(context.Background())
	defer cancel()

	done := make(chan struct{})
	defer close(done)
	go func() { // writer pump: frame writes must not interleave
		for {
			select {
			case b := <-c.send:
				c.mu.Lock()
				err := conn.WriteMessage(websocket.TextMessage, b)
				c.mu.Unlock()
				if err != nil {
					return
				}
			case <-done:
				// drain so a generating turn never blocks on a full queue
				for {
					select {
					case <-c.send:
					default:
						return
					}
				}
			}
		}
	}()

	for {
		_, raw, err := conn.ReadMessage()
		if err != nil {
			return // disconnect — deferred cancel stops any in-flight turn
		}
		var msg struct {
			Model   string `json:"model"`
			Prompt  string `json:"prompt"`
			Tools   bool   `json:"tools"`
			List    bool   `json:"list"`
			Load    string `json:"load"`
			Delete  string `json:"delete"`
			NewChat bool   `json:"new_chat"`
			Resume  string `json:"resume"` // client reconnecting: restore server history if this conv exists
			Pull    string `json:"pull"`   // download a catalog model into the models root
		}
		if err := json.Unmarshal(raw, &msg); err != nil {
			c.enqueue(map[string]string{"error": "bad request: " + err.Error()})
			continue
		}

		// Model download frame (no generation).
		if msg.Pull != "" {
			s.handlePull(c, msg.Pull)
			continue
		}

		// Conversation-management frames (no generation).
		if msg.List {
			c.enqueue(map[string]any{"conversations": conversations.ListConversations()})
			continue
		}
		if msg.Resume != "" {
			// Reconnect after Stop/refresh: restore the saved conversation
			// into a fresh agent and re-send the transcript (after a page
			// refresh the client's thread is empty; after a mid-chat Stop
			// the rebuild is idempotent — same completed turns).
			if conv, err := conversations.LoadConversation(msg.Resume); err == nil {
				if dir, err := resolveWebModelDir(convModelName(conv)); err == nil {
					if aerr := s.ensureAgentFor(c, dir, false, conv); aerr == nil {
						c.enqueue(map[string]any{"conv": conv.ID, "transcript": agentTranscript(c.agent)})
					}
				}
			}
			c.enqueue(map[string]any{"conversations": conversations.ListConversations()})
			continue
		}
		if msg.Load != "" {
			s.loadConversation(c, msg.Load)
			continue
		}
		if msg.Delete != "" {
			if err := conversations.DeleteConversation(msg.Delete); err != nil {
				c.enqueue(map[string]string{"error": "delete failed: " + err.Error()})
			} else {
				c.enqueue(map[string]any{"deleted": msg.Delete})
			}
			c.enqueue(map[string]any{"conversations": conversations.ListConversations()})
			continue
		}
		if msg.NewChat {
			s.startConversation(c)
			continue
		}

		if strings.TrimSpace(msg.Prompt) == "" {
			continue
		}
		s.handleTurn(ctx, c, msg.Model, msg.Tools, msg.Prompt)
	}
}

// convModelName picks the model a saved conversation ran on, falling back
// to the process default when unknown.
func convModelName(conv *conversations.Conversation) string {
	if conv.Model != "" {
		return conv.Model
	}
	return ""
}

// ensureAgentFor is ensureAgent with an initial conversation: a fresh
// agent whose state is imported from the saved transcript.
func (s *webServer) ensureAgentFor(c *wsClient, modelDir string, tools bool, conv *conversations.Conversation) error {
	c.conv = conv
	if err := s.ensureAgent(c, modelDir, tools); err != nil {
		return err
	}
	if c.agent != nil {
		_ = importTranscript(c.agent, conv.Messages)
	}
	return nil
}

// startConversation begins a fresh conversation: new id, empty history.
// Replies with the sidebar refresh and the new conversation id.
func (s *webServer) startConversation(c *wsClient) {
	c.mu.Lock()
	c.conv = &conversations.Conversation{
		ID:        conversations.NewConversationID(),
		CreatedAt: time.Now(),
		UpdatedAt: time.Now(),
	}
	c.agent = nil // fresh agent on the next turn; state starts empty
	c.mu.Unlock()
	c.enqueue(map[string]any{"conv": c.conv.ID, "conversations": conversations.ListConversations()})
}

// pullInFlight serializes handlePull: hf downloads to a shared models root,
// and two concurrent pulls of the same (or any) model race on the same
// destination. A download can take minutes; the UI disables its own buttons
// but a second tab (or a refresh mid-download) would happily start another.
var pullMu struct {
	sync.Mutex
	active bool
}

// handlePull downloads a catalog model (name or unique prefix) into the
// shared models root, then refreshes the model listing. Runs in the
// connection's read loop, so a slow download blocks further frames —
// acceptable: the UI is busy-waiting on the download anyway. On success it
// reports the new model dir so the client can select it; a RAM-gate
// refusal or missing hf CLI surfaces as an error frame.
func (s *webServer) handlePull(c *wsClient, name string) {
	m, err := download.FindCatalogModel(name)
	if err != nil {
		c.enqueue(map[string]string{"error": err.Error()})
		return
	}
	pullMu.Lock()
	if pullMu.active {
		pullMu.Unlock()
		c.enqueue(map[string]string{"error": "a model download is already running — wait for it to finish"})
		return
	}
	pullMu.active = true
	pullMu.Unlock()
	defer func() {
		pullMu.Lock()
		pullMu.active = false
		pullMu.Unlock()
	}()
	dest, err := download.DownloadModel(ctxBg(), m)
	if err != nil {
		c.enqueue(map[string]string{"error": err.Error()})
		return
	}
	c.enqueue(map[string]any{
		"pulled":        m.Name,
		"dir":           dest,
		"model":         filepath.Base(dest),
		"conversations": conversations.ListConversations(),
	})
}

// loadConversation restores a saved conversation into a fresh agent and
// streams the transcript back for rendering.
func (s *webServer) loadConversation(c *wsClient, id string) {
	conv, err := conversations.LoadConversation(id)
	if err != nil {
		c.enqueue(map[string]string{"error": "load failed: " + err.Error()})
		return
	}
	dir, err := resolveWebModelDir(convModelName(conv))
	if err != nil {
		c.enqueue(map[string]string{"error": err.Error()})
		return
	}
	if err := s.ensureAgentFor(c, dir, false, conv); err != nil {
		c.enqueue(map[string]string{"error": "agent: " + err.Error()})
		return
	}
	// The transcript is the only frame that carries a message list; the
	// client rebuilds its thread from it wholesale.
	c.enqueue(map[string]any{"conv": conv.ID, "model": conv.Model, "transcript": agentTranscript(c.agent)})
	c.enqueue(map[string]any{"conversations": conversations.ListConversations()})
}

// handleTurn runs one user→assistant exchange through the connection's
// seed agent: tools execute, tool events stream as chips, the answer
// streams as raw deltas. The conversation persists to disk after the turn
// (partial answers included on cancel).
func (s *webServer) handleTurn(ctx context.Context, c *wsClient, modelName string, tools bool, prompt string) {
	// Tell the client immediately that the turn started — model loading
	// can take many seconds on first use, and silence reads as broken.
	startTurnMetrics()
	c.enqueue(map[string]string{"status": "working"})

	if modelName != "" && modelName != c.model {
		c.model = modelName
	}
	if _, err := resolveWebModelDir(c.model); err != nil {
		c.enqueue(map[string]string{"error": err.Error()})
		return
	}
	dir, _ := resolveWebModelDir(c.model)

	// (Re)build the agent when the model or tools flag changed.
	if err := s.ensureAgent(c, dir, tools); err != nil {
		c.enqueue(map[string]string{"error": "agent: " + err.Error()})
		return
	}

	// Loading a cold model can take 10-20s; say so before it happens.
	if !isModelLoaded(dir) {
		c.enqueue(map[string]string{"status": "loading model " + filepath.Base(dir) + "…"})
	}

	// #url enrichment: readable text of referenced pages is appended to
	// the user turn (same protocol as the old Python server). Fetch
	// failures become notes; the chat continues without the page.
	content := prompt
	for _, u := range urlfetch.ExtractPromptURLs(prompt) {
		page, err := urlfetch.FetchReadable(ctx, u)
		if err != nil {
			c.enqueue(map[string]string{"note": fmt.Sprintf("couldn't fetch %s: %v", u, err)})
			continue
		}
		content += fmt.Sprintf("\n\nContent of %s:\n%s", u, page)
	}

	// First prompt without an explicit new-chat: start a conversation now
	// so the turn is captured from the beginning.
	if c.conv == nil {
		c.conv = &conversations.Conversation{
			ID:        conversations.NewConversationID(),
			CreatedAt: time.Now(),
		}
		c.enqueue(map[string]any{"conv": c.conv.ID})
	}

	// Stream: the provider suppresses <tool_call> markup; deltas go
	// straight to the socket (blocking send, cancel-aware).
	c.provider.ResetTurnCount()
	streamCtx, cancel := context.WithCancel(ctx)
	display := func(delta string) {
		select {
		case c.send <- []byte(delta):
		case <-streamCtx.Done():
		}
	}
	c.provider.SetDisplay(display)
	c.provider.SetIterationBoundary(func() {
		// A new assistant message begins: close the current bubble with a
		// visible marker; the client opens the next one on the first
		// delta/chip that follows.
		c.enqueue(map[string]string{"iter": "next"})
		c.enqueue(map[string]string{"status": "continuing…"})
	})

	text, err := c.agent.RunStream(streamCtx, content)
	c.provider.SetDisplay(nil)
	cancel()

	// Surface guard stops and spirals so "the text just stopped" is
	// explainable.
	if LastMetrics.GuardReason != "" {
		c.enqueue(map[string]string{"note": "Stopped a runaway generation (" + LastMetrics.GuardReason + ")."})
	} else if text != "" && isSpiral(text) {
		c.enqueue(map[string]string{"note": "The reply was looping — stopped it. Try rephrasing or narrowing the ask."})
	}

	if err != nil {
		// Interrupted (Stop/refresh): keep the partial the client already
		// saw, persist what completed, and let the client reconnect.
		if ctx.Err() != nil || strings.Contains(errString(err), "interrupt") {
			if partial := lastAssistantText(c.agent); partial != "" {
				c.syncConversation(partial)
			}
			return
		}
		// Out of tool steps: the tool results are already in the
		// conversation — one more forced generation (on a fresh context —
		// the first one is canceled above) turns them into an answer
		// instead of an error frame.
		if errors.Is(err, core.ErrMaxIterations) {
			c.enqueue(map[string]string{"status": "tool budget reached — forcing a final answer…"})
			graceCtx, graceCancel := context.WithCancel(ctx)
			defer graceCancel()
			c.provider.ResetTurnCount()
			c.provider.SetDisplay(func(delta string) {
				select {
				case c.send <- []byte(delta):
				case <-graceCtx.Done():
				}
			})
			text, err = c.agent.RunStream(graceCtx, graceFinalQuery)
			c.provider.SetDisplay(nil)
			graceCancel()
			if err != nil {
				c.enqueue(map[string]string{"error": err.Error()})
				return
			}
			c.syncConversation(text)
			c.enqueue(map[string]bool{"done": true})
			return
		}
		c.enqueue(map[string]string{"error": err.Error()})
		return
	}

	c.syncConversation(text)

	// Post-answer self-check against the enriched turn (sinter, not an
	// external service). A confident "No" surfaces as a note frame before
	// done, so the UI renders it as a system warning on the answer.
	if ok, err := factCheck(dir, content, text); err == nil && !ok {
		c.enqueue(map[string]string{"note": "Looks like my last answer might be off — you might want to double check that."})
	}

	// Metrics frame: llama.cpp-style stats for the whole turn (every
	// generation, tool round-trips included).
	m := turnMetrics()
	c.enqueue(map[string]any{"metrics": map[string]any{
		"prompt_tokens": m.PromptTokens,
		"gen_tokens":    m.GenTokens,
		"tps":           m.TPS(),
		"pp":            m.PP(),
		"context_used":  m.ContextUsed,
		"context_size":  m.ContextSize,
	}})

	c.enqueue(map[string]bool{"done": true})
}

// syncConversation commits the agent's current transcript to the
// conversation record (title from the first user turn, capped at
// historyLimit).
func (c *wsClient) syncConversation(lastReply string) {
	c.mu.Lock()
	defer c.mu.Unlock()
	if c.conv == nil || c.agent == nil {
		return
	}
	msgs := agentTranscript(c.agent)
	if len(msgs) > historyLimit {
		msgs = msgs[len(msgs)-historyLimit:]
	}
	c.conv.Model = c.model
	c.conv.Messages = msgs
	if c.conv.Title == "" {
		c.conv.Title = conversations.ConversationTitleFromStored(msgs)
	}
	c.conv.UpdatedAt = time.Now()
	conversations.SaveConversation(c.conv)
}

// lastAssistantText returns the newest assistant message in the agent
// state ("" when none) — the partial answer of an interrupted turn.
func lastAssistantText(a *core.Agent) string {
	if a == nil {
		return ""
	}
	msgs := a.State().Messages()
	for i := len(msgs) - 1; i >= 0; i-- {
		if msgs[i].Role == "assistant" && msgs[i].Content != "" {
			return msgs[i].Content
		}
	}
	return ""
}

// handlePreview serves a sandbox file (cwd or /tmp) for browser viewing —
// the preview links on write_file tool chips. Path validation matches
// resolveToolPath; HTML/SVG are served as their real content types so the
// browser renders them, everything else downloads as plain text.
func handlePreview(w http.ResponseWriter, r *http.Request) {
	rel := r.PathValue("path")
	path, err := resolveToolPath(rel)
	if err != nil {
		http.Error(w, "outside sandbox", http.StatusForbidden)
		return
	}
	data, err := os.ReadFile(path)
	if err != nil {
		http.Error(w, "not found", http.StatusNotFound)
		return
	}
	switch strings.ToLower(filepath.Ext(path)) {
	case ".html", ".htm":
		w.Header().Set("Content-Type", "text/html; charset=utf-8")
	case ".svg":
		w.Header().Set("Content-Type", "image/svg+xml")
	case ".css":
		w.Header().Set("Content-Type", "text/css; charset=utf-8")
	case ".js":
		w.Header().Set("Content-Type", "text/javascript; charset=utf-8")
	case ".png":
		w.Header().Set("Content-Type", "image/png")
	case ".jpg", ".jpeg":
		w.Header().Set("Content-Type", "image/jpeg")
	case ".gif":
		w.Header().Set("Content-Type", "image/gif")
	default:
		w.Header().Set("Content-Type", "text/plain; charset=utf-8")
	}
	w.Header().Set("X-Content-Type-Options", "nosniff")
	_, _ = w.Write(data)
}

// ─── HTTP handlers ───────────────────────────────────────────────────────

type webServer struct{}

func (s *webServer) handleModels(w http.ResponseWriter, r *http.Request) {
	w.Header().Set("Content-Type", "application/json")
	// No model installed → default stays "" so the UI's first-run
	// download panel triggers (models:[] + default:"" is the panel's
	// exact precondition). filepath.Base("") would be "." — a bogus
	// entry that suppresses the panel.
	def := ""
	if dir := resolveModelDir(); dir != "" {
		def = filepath.Base(dir)
	}
	names := availableModels()
	if names == nil && def != "" { // pinned to a single model
		names = []string{def}
	}
	// Default model first so the UI preselects it.
	sort.Slice(names, func(i, j int) bool {
		if (names[i] == def) != (names[j] == def) {
			return names[i] == def
		}
		return names[i] < names[j]
	})
	json.NewEncoder(w).Encode(map[string]any{
		"models":  names,
		"default": def,
	})
}

// handlePullCatalog lists the downloadable catalog models with this
// machine's RAM tier for each and the RAM-recommended default. The UI uses
// this to offer downloads when no models are installed (a model-less
// machine otherwise has an empty picker and no way in).
func (s *webServer) handlePullCatalog(w http.ResponseWriter, r *http.Request) {
	ram := sysinfo.TotalSystemRAM()
	suggested := catalog.SuggestedForRAM(ram)
	type entry struct {
		Name string `json:"name"`
		Repo string `json:"repo"`
		Tag  string `json:"tag,omitempty"`
	}
	entries := make([]entry, 0, len(catalog.ModelCatalog))
	for _, m := range catalog.ModelCatalog {
		tag := ""
		switch {
		case ram != 0 && m.MinRAMSelect != 0 && ram < m.MinRAMSelect:
			tag = "needs more RAM"
		case ram != 0 && m.MinRAMSuggested != 0 && ram < m.MinRAMSuggested && m.MinRAMSuggested != ^uint64(0):
			tag = "tight fit"
		}
		if m.Name == suggested.Name {
			tag = "recommended for this machine"
		}
		entries = append(entries, entry{Name: m.Name, Repo: m.HFRepo, Tag: tag})
	}
	w.Header().Set("Content-Type", "application/json")
	json.NewEncoder(w).Encode(map[string]any{
		"catalog":     entries,
		"suggested":   suggested.Name,
		"installed":   availableModels(),
		"models_root": paths.ModelsRoot(),
	})
}

// uiRoot returns the embedded UI tree, or the on-disk tree when
// UI_SPROUT_LOCAL_DEV=1 (edit the UI without rebuilding).
func uiRoot() fs.FS {
	if os.Getenv("UI_SPROUT_LOCAL_DEV") == "1" {
		if st, err := os.Stat("ui"); err == nil && st.IsDir() {
			if sub, err := fs.Sub(os.DirFS("."), "ui"); err == nil {
				return sub
			}
		}
		log.Println("UI_SPROUT_LOCAL_DEV=1 but ./ui not found; using embedded UI")
	}
	sub, err := fs.Sub(uiFS, "ui")
	if err != nil {
		log.Fatalf("embedded ui broken: %v", err)
	}
	return sub
}

// serveHosts runs chatllm -serve: embedded UI plus /ws, /models, and the
// OpenAI-compatible /v1 endpoint. Startup is quiet: one banner with the
// address, engine, and default model (use -v for engine detail).
//
// A model-less machine is not fatal here: the UI's model picker offers
// -pull downloads, so we log a warning and serve anyway (the REPL path
// would instead run the first-run walkthrough).
func serveHosts(addr string) {
	mux := http.NewServeMux()
	s := &webServer{}
	mux.HandleFunc("/ws", s.serveWS)
	mux.HandleFunc("/models", s.handleModels)
	mux.HandleFunc("/pull", s.handlePullCatalog)
	mux.HandleFunc("/preview/{path...}", handlePreview)
	mux.Handle("/", http.FileServer(http.FS(uiRoot())))
	serveAPI(mux)

	// A model-less machine is not fatal under -serve: the UI's model
	// picker offers -pull downloads, so warn and serve anyway. (The REPL
	// path would instead run the first-run walkthrough; only -serve takes
	// this route.)
	if resolveModelDir() == "" {
		log.Printf("no models installed — download one via the web UI model picker")
		fmt.Printf("chatllm serving on http://%s — chat UI, model picker, OpenAI-style /v1 API (no models installed yet)\n", addr)
	} else {
		engine, modelPath := modelBackend()
		fmt.Printf("chatllm serving on http://%s — chat UI, model picker, OpenAI-style /v1 API\n", addr)
		fmt.Printf("engine %s, default model %s (Ctrl-C to stop)\n", engine, filepath.Base(modelPath))
	}
	if err := http.ListenAndServe(addr, mux); err != nil {
		log.Fatalf("serve: %v", err)
	}
}

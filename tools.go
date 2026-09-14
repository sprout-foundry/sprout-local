package main

// ---------------------------------------------------------------------------
// tools.go — minimal tool calling for chatllm.
//
// Sinter's llm.ChatMessage is {Role, Content} and the qwen3.5 chat template
// renders roles verbatim, so tool calling is done entirely at the text
// layer, using the exact protocol in the model's own chat_template.jinja:
//
//   • tool declarations:  "# Tools" block in the system message
//   • tool call:          <tool_call><function=name><parameter=k>v</parameter></function></tool_call>
//   • tool result:        user-role message containing <tool_response>…</tool_response>
//
// The visible REPL loop never changes: the user types, the answer streams,
// tool round-trips happen invisibly in between.
// ---------------------------------------------------------------------------

import (
	"context"
	"encoding/json"
	"fmt"
	"os"
	"os/exec"
	"path/filepath"
	"regexp"
	"sort"
	"strconv"
	"strings"

	"github.com/sprout-foundry/sinter/llm"
)

// Session-tunable defaults. initTunables (runtime.go) applies
// SPROUT_LOCAL_MAX_STEPS, SPROUT_LOCAL_TOOL_RESULT_CAP,
// SPROUT_LOCAL_COMMAND_TIMEOUT and SPROUT_LOCAL_MAX_TOKENS; the -max-steps
// and -max-tokens flags override the environment.
const (
	// defaultMaxToolSteps caps tool round-trips per user turn, so a model
	// that keeps calling tools can't loop forever (or burn the token
	// budget). Default raised from 4: with self-correcting errors, a
	// lookup usually needs 1–2 steps and a broken first guess shouldn't
	// kill the turn. 100 = effectively "until the model stops on its
	// own"; the graceful-exhaustion path still bounds a runaway.
	defaultMaxToolSteps = 100
	// defaultToolResultCap bounds each tool result fed back to the model.
	defaultToolResultCap = 6000 // characters
	// defaultCommandTimeout bounds run_command executions, in seconds.
	defaultCommandTimeout = 30
)

// toolSpec describes one tool: declaration fields (name, description,
// parameters) plus the native Go implementation.
type toolSpec struct {
	name        string
	description string
	parameters  []toolParam
	run         func(ctx context.Context, args map[string]string) (string, error)
}

type toolParam struct {
	name        string
	typeName    string
	description string
	required    bool
}

// toolRegistry is the session's tool set. Deterministic order (read →
// write → run → fetch) keeps the prompt stable for sinter's prefix cache.
var toolRegistry = []toolSpec{
	{
		name:        "read_file",
		description: "Read a text file. Use this to inspect existing files.",
		parameters: []toolParam{
			{name: "path", typeName: "string", description: "File path", required: true},
		},
		run: toolRunReadFile,
	},
	{
		name:        "write_file",
		description: "Create or overwrite a file with the given content.",
		parameters: []toolParam{
			{name: "path", typeName: "string", description: "File path", required: true},
			{name: "content", typeName: "string", description: "Full file content", required: true},
		},
		run: toolRunWriteFile,
	},
	{
		name: "run_command",
		description: "Run one shell command via /bin/sh and return stdout+stderr combined. " +
			"Pipes and quoted arguments work, so `ifconfig | grep inet` is fine. " +
			"For quick lookups (ls, cat, grep, git status, ifconfig, …). " +
			"Chaining with ; & && || and file redirects (>, >>, <, <<) are not available — " +
			"stderr is captured automatically, so no 2>/dev/null is needed. " +
			"Requires user confirmation at runtime.",
		parameters: []toolParam{
			{name: "command", typeName: "string", description: "The shell command line", required: true},
		},
		run: toolRunCommand,
	},
	{
		name:        "web_fetch",
		description: "Fetch a web page and return its readable text. Use for current information or URLs the user shares.",
		parameters: []toolParam{
			{name: "url", typeName: "string", description: "URL to fetch", required: true},
		},
		run: toolRunWebFetch,
	},
}

// toolDeclJSON renders a toolSpec as the JSON declaration object the chat
// template expects inside <tools>…</tools> (name/description/parameters
// with JSON-schema types).
func toolDeclJSON(t toolSpec) string {
	type param struct {
		Type        string `json:"type"`
		Description string `json:"description"`
	}
	type props map[string]param
	type decl struct {
		Name        string `json:"name"`
		Description string `json:"description"`
		Parameters  struct {
			Type       string   `json:"type"`
			Properties props    `json:"properties"`
			Required   []string `json:"required,omitempty"`
		} `json:"parameters"`
	}
	var d decl
	d.Name = t.name
	d.Description = t.description
	d.Parameters.Type = "object"
	d.Parameters.Properties = props{}
	for _, p := range t.parameters {
		d.Parameters.Properties[p.name] = param{Type: p.typeName, Description: p.description}
		if p.required {
			d.Parameters.Required = append(d.Parameters.Required, p.name)
		}
	}
	b, err := json.Marshal(d)
	if err != nil {
		return fmt.Sprintf(`{"name":%q,"description":%q}`, t.name, t.description)
	}
	return string(b)
}

// renderToolCallText renders one tool call in the model's native markup.
// indentFirst is true when content precedes the call (template: two
// newlines before the first block).
func renderToolCallText(name string, args map[string]string, afterContent bool) string {
	var sb strings.Builder
	if afterContent {
		sb.WriteString("\n\n")
	}
	sb.WriteString("<tool_call>\n<function=" + name + ">\n")
	for k, v := range args {
		sb.WriteString("<parameter=" + k + ">\n" + v + "\n</parameter>\n")
	}
	sb.WriteString("</function>\n</tool_call>")
	return sb.String()
}

// appendToolResult appends a tool result as a user-role <tool_response>
// message, matching chat_template.jinja's rendering of role=tool turns.
func appendToolResult(out []llm.ChatMessage, content string) []llm.ChatMessage {
	return append(out, llm.ChatMessage{
		Role:    "user",
		Content: "<tool_response>\n" + content + "\n</tool_response>",
	})
}

// renderToolPromptBlock renders the "# Tools" block for arbitrary specs —
// the shared body behind toolPromptBlock (static registry) and
// toolPromptBlockFromSeed (dynamic, from the executor).
func renderToolPromptBlock(specs []toolSpec) string {
	return renderToolPromptBlockFor(specs, "qwen")
}

// renderToolPromptBlockFor renders the "# Tools" block in either supported
// protocol. "qwen" is the qwen3.5 <tool_call>/<function=name> markup;
// "minicpm5" is MiniCPM5's native format — <function name="…"><param
// name="…">v</param></function> inside <tools>…</tools>, per its official
// chat_template.jinja (tool usage guidelines quoted from the template).
func renderToolPromptBlockFor(specs []toolSpec, protocol string) string {
	if protocol == "minicpm5" {
		var sb strings.Builder
		sb.WriteString("# Tools\n\nYou are provided with function signatures within <tools></tools> XML tags:\n<tools>")
		for _, t := range specs {
			sb.WriteString("\n" + miniCPM5ToolDeclJSON(t))
		}
		sb.WriteString("\n</tools>\n\nTool usage guidelines:\n- You may call zero or more functions. If no function calls are needed, just answer normally and do not include any <function ... </function>.\n- When calling a function, return an XML object within <function ... </function> using:\n<function name=\"function-name\"><param name=\"param-name\">param-value</param></function>\n- param-value may be multi-line. If it contains <, & or newline characters, wrap it in a CDATA block: <param name=\"param-name\"><![CDATA[...multi-line value...]]></param>")
		return sb.String()
	}
	var sb strings.Builder
	sb.WriteString("# Tools\n\nYou have access to the following functions:\n\n<tools>")
	for _, t := range specs {
		sb.WriteString("\n" + toolDeclJSON(t))
	}
	sb.WriteString("\n</tools>\n\nIf you choose to call a function ONLY reply in the following format with NO suffix:\n\n")
	sb.WriteString("<tool_call>\n<function=example_function_name>\n<parameter=example_parameter_1>\nvalue_1\n</parameter>\n<parameter=example_parameter_2>\nThis is the value for the second parameter\nthat can span\nmultiple lines\n</parameter>\n</function>\n</tool_call>\n\n")
	sb.WriteString(`<IMPORTANT>
Reminder:
- Function calls MUST follow the specified format: an inner <function=...></function> block must be nested within <tool_call></tool_call> XML tags
- Required parameters MUST be specified
- You may provide optional reasoning for your function call in natural language BEFORE the function call, but NOT after
- If there is no function call available, answer the question like normal with your current knowledge and do not tell the user about function calls
</IMPORTANT>`)
	return sb.String()
}

// miniCPM5ToolDeclJSON renders a toolSpec as the JSON declaration MiniCPM5's
// template expects (its template serializes each tool with tojson — the
// OpenAI-style function envelope).
func miniCPM5ToolDeclJSON(t toolSpec) string {
	type param struct {
		Type        string `json:"type"`
		Description string `json:"description"`
	}
	type props map[string]param
	type fn struct {
		Name        string `json:"name"`
		Description string `json:"description"`
		Parameters  struct {
			Type       string   `json:"type"`
			Properties props    `json:"properties"`
			Required   []string `json:"required,omitempty"`
		} `json:"parameters"`
	}
	type decl struct {
		Type     string `json:"type"`
		Function fn     `json:"function"`
	}
	var d decl
	d.Type = "function"
	d.Function.Name = t.name
	d.Function.Description = t.description
	d.Function.Parameters.Type = "object"
	d.Function.Parameters.Properties = props{}
	for _, p := range t.parameters {
		d.Function.Parameters.Properties[p.name] = param{Type: p.typeName, Description: p.description}
		if p.required {
			d.Function.Parameters.Required = append(d.Function.Parameters.Required, p.name)
		}
	}
	b, err := json.Marshal(d)
	if err != nil {
		return fmt.Sprintf(`{"type":"function","function":{"name":%q,"description":%q}}`, t.name, t.description)
	}
	return string(b)
}

// toolPromptBlock renders the "# Tools" system block for the static
// registry in the session's tool protocol.
func toolPromptBlock() string {
	return renderToolPromptBlockFor(toolRegistry, toolProtocol())
}

// ─── Protocol selection ──────────────────────────────────────────────────

// toolProtocolForModelDir reports the tool-call protocol a model directory
// speaks, from config.json model_type + tokenizer.json control tokens:
// "minicpm5" for MiniCPM5 (native <function name="…"><param …/></function>
// XML), "qwen" for everything else (the qwen3.5 <tool_call>/<function=name>
// markup).
func toolProtocolForModelDir(dir string) string {
	data, err := os.ReadFile(filepath.Join(dir, "config.json"))
	if err == nil {
		var cfg struct {
			ModelType string `json:"model_type"`
		}
		if json.Unmarshal(data, &cfg) == nil && strings.EqualFold(cfg.ModelType, "llama") &&
			isMiniCPM5Tokenizer(filepath.Join(dir, "tokenizer.json")) {
			return "minicpm5"
		}
	}
	return "qwen"
}

// isMiniCPM5Tokenizer reads a tokenizer.json and reports whether it carries
// MiniCPM5's /think and /no_think control tokens.
func isMiniCPM5Tokenizer(path string) bool {
	data, err := os.ReadFile(path)
	if err != nil {
		return false
	}
	var tok struct {
		AddedTokens []struct {
			Content string `json:"content"`
		} `json:"added_tokens"`
	}
	if json.Unmarshal(data, &tok) != nil {
		return false
	}
	hasThink, hasNoThink := false, false
	for _, t := range tok.AddedTokens {
		switch t.Content {
		case "/think":
			hasThink = true
		case "/no_think":
			hasNoThink = true
		}
	}
	return hasThink && hasNoThink
}

// sessionProtocol is the active session's tool protocol, set whenever a
// model directory resolves (startup, /model, /pull). Empty = qwen default.
var sessionProtocol string

// toolProtocol returns the active session's tool protocol.
func toolProtocol() string {
	if sessionProtocol != "" {
		return sessionProtocol
	}
	return "qwen"
}

// setSessionModelProtocol records the protocol for the given model directory.
func setSessionModelProtocol(dir string) {
	if dir != "" {
		sessionProtocol = toolProtocolForModelDir(dir)
	}
}

// ─── Parsing ─────────────────────────────────────────────────────────────

// parsedToolCall is one extracted call: tool name + arguments.
type parsedToolCall struct {
	name      string
	args      map[string]string
	truncated bool // ran out of tokens mid-call; recovered best-effort
}

// extractToolCalls finds tool-call blocks in a model response using the
// session's tool protocol (both parsers also run when the primary finds
// nothing, so a model that drifts between markups still gets caught).
// Text outside the blocks is returned too, so the caller can show it as
// the assistant's preamble.
func extractToolCalls(text string) (remaining string, calls []parsedToolCall) {
	return extractToolCallsFor(text, toolProtocol())
}

// extractToolCallsFor is extractToolCalls with an explicit protocol.
func extractToolCallsFor(text, protocol string) (remaining string, calls []parsedToolCall) {
	if protocol == "minicpm5" {
		remaining, calls = extractMiniCPM5Calls(text)
		if len(calls) > 0 {
			return remaining, calls
		}
		// Fall through to the qwen parser: the model may fall back to the
		// more common markup it saw in pretraining.
		return extractQwenToolCalls(text)
	}
	return extractQwenToolCalls(text)
}

// extractMiniCPM5Calls finds every <function name="…">…</function> block
// (MiniCPM5 native format) in a model response.
func extractMiniCPM5Calls(text string) (remaining string, calls []parsedToolCall) {
	var plain strings.Builder
	rest := text
	for {
		start, end := findMiniCPM5Call(rest)
		if start < 0 {
			plain.WriteString(rest)
			break
		}
		plain.WriteString(rest[:start])
		body := rest[start:end]
		if c := parseMiniCPM5CallBody(body); c.name != "" {
			calls = append(calls, c)
		}
		if end >= len(rest) {
			break
		}
		rest = rest[end:]
	}
	return plain.String(), calls
}

// findMiniCPM5Call locates the next <function name="…"> opening and its
// matching </function> close in s. Returns start=-1 when none remains.
// end points just past </function> (or len(s) for an unterminated call).
func findMiniCPM5Call(s string) (start, end int) {
	open := regexp.MustCompile(`<function\s+name="[A-Za-z0-9_.-]+"\s*>`).FindStringIndex(s)
	if open == nil {
		return -1, -1
	}
	start = open[0]
	close := strings.Index(s[open[1]:], "</function>")
	if close < 0 {
		// Unterminated — the out-of-tokens case; the parser recovers
		// name+params from the truncated body.
		return start, len(s)
	}
	return start, open[1] + close + len("</function>")
}

// extractQwenToolCalls is the original qwen-protocol block scanner.
func extractQwenToolCalls(text string) (remaining string, calls []parsedToolCall) {
	var plain strings.Builder
	rest := text
	for {
		start := strings.Index(rest, "<tool_call>")
		if start < 0 {
			plain.WriteString(rest)
			break
		}
		plain.WriteString(rest[:start])
		rest = rest[start+len("<tool_call>"):]

		end := strings.Index(rest, "</tool_call>")
		if end < 0 {
			// Unterminated block: the model ran out of tokens mid-call.
			// Treat the remainder as a call attempt anyway.
			if c := parseToolCallBody(rest); c.name != "" {
				calls = append(calls, c)
			}
			rest = ""
			break
		}
		if c := parseToolCallBody(rest[:end]); c.name != "" {
			calls = append(calls, c)
		}
		rest = rest[end+len("</tool_call>"):]
	}
	return plain.String(), calls
}

// parseToolCallBody parses "<function=name>…</function>" into a call.
func parseToolCallBody(body string) parsedToolCall {
	body = strings.TrimSpace(body)
	fm := regexp.MustCompile(`(?s)<function=([A-Za-z0-9_.-]+)>(.*)</function>`).FindStringSubmatch(body)
	if fm == nil {
		// MiniCPM5 protocol: <function name="tool">…</param></function>.
		// Try it before declaring the call malformed.
		if c := parseMiniCPM5CallBody(body); c.name != "" {
			return c
		}
		// Unterminated call (ran out of tokens mid-parameter — common when
		// a model writes a large file). Recover the tool name and whatever
		// parameters completed: a truncated write beats no write.
		fm2 := regexp.MustCompile(`(?s)^\s*<function=([A-Za-z0-9_.-]+)>(.*)$`).FindStringSubmatch(body)
		if fm2 == nil {
			return parsedToolCall{}
		}
		call := parsedToolCall{name: fm2[1], args: map[string]string{}, truncated: true}
		pm := regexp.MustCompile(`(?s)<parameter=([A-Za-z0-9_.-]+)>\n?(.*?)(?:</parameter>|$)`).FindAllStringSubmatch(fm2[2], -1)
		for _, p := range pm {
			call.args[p[1]] = strings.TrimSpace(p[2])
		}
		return call
	}
	call := parsedToolCall{name: fm[1], args: map[string]string{}}
	pm := regexp.MustCompile(`(?s)<parameter=([A-Za-z0-9_.-]+)>\n?(.*?)</parameter>`).FindAllStringSubmatch(fm[2], -1)
	for _, p := range pm {
		call.args[p[1]] = strings.TrimSpace(p[2])
	}
	return call
}

// miniCPM5CallRe matches MiniCPM5's native tool-call markup:
// <function name="name"><param name="k">v</param>…</function>. The param
// value may be wrapped in a CDATA block for multi-line / special-char values.
var miniCPM5CallRe = regexp.MustCompile(`(?s)<function\s+name="([A-Za-z0-9_.-]+)"\s*>(.*)</function>`)

// miniCPM5ParamRe matches one <param name="k">v</param>, CDATA optional.
var miniCPM5ParamRe = regexp.MustCompile(`(?s)<param\s+name="([A-Za-z0-9_.-]+)"\s*>\s*(?:<!\[CDATA\[(.*?)\]\]>|(.*?))</param>`)

// parseMiniCPM5CallBody parses MiniCPM5's <function name="…">…</function>
// into a call. Returns the zero call when the body doesn't use that format.
func parseMiniCPM5CallBody(body string) parsedToolCall {
	fm := miniCPM5CallRe.FindStringSubmatch(strings.TrimSpace(body))
	if fm == nil {
		// Unterminated (out of tokens mid-call): recover the name and any
		// complete params.
		fm2 := regexp.MustCompile(`(?s)^\s*<function\s+name="([A-Za-z0-9_.-]+)"\s*>(.*)$`).FindStringSubmatch(body)
		if fm2 == nil {
			return parsedToolCall{}
		}
		call := parsedToolCall{name: fm2[1], args: map[string]string{}, truncated: true}
		for _, p := range miniCPM5ParamRe.FindAllStringSubmatch(fm2[2], -1) {
			call.args[p[1]] = strings.TrimSpace(firstNonEmpty(p[2], p[3]))
		}
		return call
	}
	call := parsedToolCall{name: fm[1], args: map[string]string{}}
	for _, p := range miniCPM5ParamRe.FindAllStringSubmatch(fm[2], -1) {
		call.args[p[1]] = strings.TrimSpace(firstNonEmpty(p[2], p[3]))
	}
	return call
}

// firstNonEmpty returns the first non-empty string.
func firstNonEmpty(vals ...string) string {
	for _, v := range vals {
		if v != "" {
			return v
		}
	}
	return ""
}

// renderMiniCPM5ToolCallText renders one tool call in MiniCPM5's native
// markup (the format its own template teaches and its parser expects).
// Values containing <, & or newlines are wrapped in CDATA, matching the
// template's rule.
func renderMiniCPM5ToolCallText(name string, args map[string]string, afterContent bool) string {
	var sb strings.Builder
	if afterContent {
		sb.WriteString("\n\n")
	}
	sb.WriteString(`<function name="` + name + `">`)
	for _, k := range sortedArgKeys(args) {
		v := args[k]
		if strings.ContainsAny(v, "<&\n") {
			sb.WriteString(`<param name="` + k + `"><![CDATA[` + v + `]]></param>`)
		} else {
			sb.WriteString(`<param name="` + k + `">` + v + `</param>`)
		}
	}
	sb.WriteString(`</function>`)
	return sb.String()
}

// miniCPM5ToolCallBlock wraps a rendered call in the <tool_call> envelope
// MiniCPM5 may emit around calls (harmless to parse either way).
func sortedArgKeys(args map[string]string) []string {
	keys := make([]string, 0, len(args))
	for k := range args {
		keys = append(keys, k)
	}
	sort.Strings(keys)
	return keys
}

// ─── Execution ───────────────────────────────────────────────────────────

// execToolCall runs one parsed call against the registry. The error return
// covers pure Go failures (unknown tool, sandbox violation); tool-level
// errors (missing file, failed command) are reported *as the result text*
// so the model can see them and react.
func execToolCall(ctx context.Context, call parsedToolCall) (string, error) {
	var spec *toolSpec
	for i := range toolRegistry {
		if toolRegistry[i].name == call.name {
			spec = &toolRegistry[i]
			break
		}
	}
	if spec == nil {
		return "", fmt.Errorf("unknown tool %q", call.name)
	}
	// Required-argument check (all current tools take only required args).
	for _, p := range spec.parameters {
		if p.required && strings.TrimSpace(call.args[p.name]) == "" {
			return "", fmt.Errorf("tool %q: missing required parameter %q", call.name, p.name)
		}
	}
	res, err := spec.run(ctx, call.args)
	if err != nil {
		return fmt.Sprintf("error: %v", err), nil
	}
	return res, nil
}

// resolveToolPath resolves a tool path argument and enforces the sandbox:
// relative paths stay under the working directory, absolute paths must be
// the workspace, /tmp, /private/tmp (darwin symlink), or $TMPDIR.
func resolveToolPath(p string) (string, error) {
	p = strings.TrimSpace(p)
	if p == "" {
		return "", fmt.Errorf("empty path")
	}
	if !filepath.IsAbs(p) {
		p = filepath.Join(cwd(), p)
	}
	p = filepath.Clean(p)

	for _, root := range sandboxRoots() {
		if hasPathPrefix(p, root) {
			return p, nil
		}
	}
	return "", fmt.Errorf("path %q is outside the sandbox (allowed: %s, /tmp, $TMPDIR) — use a relative path from the working directory or a /tmp path", p, cwd())
}

// sandboxRoots lists the allowed absolute roots. os.TempDir() is $TMPDIR on
// macOS (/var/folders/…); /tmp is added explicitly so plain /tmp paths work
// too (symlink resolution maps it to /private/tmp).
func sandboxRoots() []string {
	roots := []string{cwd(), "/tmp"}
	if tmp := os.TempDir(); tmp != "" && tmp != "/tmp" {
		roots = append(roots, tmp)
	}
	return roots
}

// hasPathPrefix reports whether path is root itself or lies under root.
// Both sides go through evalPathExisting: symlinks are resolved from the
// deepest existing ancestor (macOS /tmp → /private/tmp, $TMPDIR spindumps).
// This is a gate against accidents, not a security boundary.
func hasPathPrefix(path, root string) bool {
	path = filepath.Clean(evalPathExisting(path))
	root = filepath.Clean(evalPathExisting(root))
	if path == root {
		return true
	}
	return strings.HasPrefix(path, root+string(filepath.Separator))
}

// evalPathExisting resolves symlinks starting at the deepest ancestor of p
// that exists, then re-joins the leaf. Paths that exist resolve exactly.
func evalPathExisting(p string) string {
	if rp, err := filepath.EvalSymlinks(p); err == nil {
		return rp
	}
	base := filepath.Base(p)
	for dir := filepath.Dir(p); dir != p; dir = filepath.Dir(dir) {
		if rp, err := filepath.EvalSymlinks(dir); err == nil {
			return filepath.Join(rp, base)
		}
	}
	return p
}

// cwd returns the process working directory, or "." if it cannot be read.
func cwd() string {
	wd, err := os.Getwd()
	if err != nil {
		return "."
	}
	return wd
}

// truncateToolResult caps a tool result at toolResultCap characters.
func truncateToolResult(s string) string {
	if len(s) <= toolResultCap {
		return s
	}
	return s[:toolResultCap] + "\n…[truncated]"
}

// truncateResultForDisplay shortens a tool result for the one-line REPL
// status line (full text still goes to the model). The raw first line is
// often a comment banner ("##" in /etc/hosts), which tells the approving
// human nothing — skip lines that carry no content before falling back.
func truncateResultForDisplay(s string) string {
	const max = 120
	for _, line := range strings.Split(s, "\n") {
		trimmed := strings.TrimSpace(line)
		if trimmed == "" || strings.HasPrefix(trimmed, "#") {
			continue
		}
		r := []rune(trimmed)
		if len(r) <= max {
			return string(r)
		}
		return string(r[:max]) + "…"
	}
	r := []rune(firstLine(s))
	if len(r) <= max {
		return string(r)
	}
	return string(r[:max]) + "…"
}

// toolResultMessage wraps a tool result in the <tool_response> envelope the
// chat template expects, exactly as chat_template.jinja renders role=tool
// messages (inside a user turn).
func toolResultMessage(name, result string) llm.ChatMessage {
	content := fmt.Sprintf("<tool_response>\n%s\n</tool_response>", result)
	return llm.ChatMessage{Role: "user", Content: content}
}

// ─── /tools command ──────────────────────────────────────────────────────

// handleToolsCommand shows/toggles tool mode. "yolo" also skips the
// run_command confirmation prompt. The choice persists in tools.json and
// is restored on the next launch.
func handleToolsCommand(args string) {
	n, errs := reloadSkills()
	for _, err := range errs {
		fmt.Printf("%s skill: %v\n", ansiStyle("Error:", ansiRed), err)
	}
	switch strings.ToLower(strings.TrimSpace(args)) {
	case "", "show":
		if !toolsRequested {
			fmt.Printf("Tools are off. /tools on enables: %s\n", toolNames())
			if n > 0 {
				fmt.Printf("Installed skill%s: %s\n", plural(n), skillNames())
			}
			return
		}
		status := "run_command asks before running"
		if toolSafetyBypass {
			status = "run_command does not ask (yolo)"
		}
		fmt.Printf("Tools are on; %s.\nTools: %s\n", status, toolNames())
		if n > 0 {
			fmt.Printf("Skills: %s\n", skillNames())
		}
	case "on", "off":
		toolsRequested = args == "on"
		toolSafetyBypass = false
		if toolsRequested {
			fmt.Printf("Tools are on; run_command asks before running.\nTools: %s\n", toolNames())
		} else {
			fmt.Println("Tools are off.")
		}
		if toolsRequested && n > 0 {
			fmt.Printf("Skills: %s\n", skillNames())
		}
	case "yolo":
		toolsRequested = true
		toolSafetyBypass = true
		fmt.Printf("Tools are on; run_command does not ask (yolo).\nTools: %s\n", toolNames())
		if n > 0 {
			fmt.Printf("Skills: %s\n", skillNames())
		}
	default:
		fmt.Println("usage: /tools [on|off|yolo]")
		return
	}
	saveToolsPreference()
}

// toolNames lists registry names for status lines.
func toolNames() string {
	names := make([]string, 0, len(toolRegistry))
	for _, t := range toolRegistry {
		names = append(names, t.name)
	}
	return strings.Join(names, ", ")
}

// skillNames lists loaded skill names.
func skillNames() string {
	names := make([]string, 0, len(skillRegistry))
	for _, s := range skillRegistry {
		names = append(names, s.Name)
	}
	return strings.Join(names, ", ")
}

// plural returns "" or "s".
func plural(n int) string {
	if n == 1 {
		return ""
	}
	return "s"
}

// ─── Tool implementations ────────────────────────────────────────────────

func toolRunReadFile(ctx context.Context, args map[string]string) (string, error) {
	path, err := resolveToolPath(args["path"])
	if err != nil {
		return "", err
	}
	b, err := os.ReadFile(path)
	if err != nil {
		return "", err
	}
	s := string(b)
	if len(s) > toolResultCap {
		s = s[:toolResultCap] + "\n…[truncated]"
	}
	if s == "" {
		return "(empty file)", nil
	}
	return s, nil
}

func toolRunWriteFile(ctx context.Context, args map[string]string) (string, error) {
	path, err := resolveToolPath(args["path"])
	if err != nil {
		return "", err
	}
	content := args["content"]
	// Templates emit raw newlines for multi-line values; accept JSON-style
	// "\n" escapes too (some models stringify the whole value).
	if !strings.Contains(content, "\n") && strings.Contains(content, `\n`) {
		if unquoted, err := strconv.Unquote(`"` + content + `"`); err == nil {
			content = unquoted
		}
	}
	// Models say "create folder X with files" and write into X/ directly —
	// create missing parent directories instead of failing.
	if dir := filepath.Dir(path); dir != "." {
		if err := os.MkdirAll(dir, 0o755); err != nil {
			return "", err
		}
	}
	if err := os.WriteFile(path, []byte(content), 0o644); err != nil {
		return "", err
	}
	return fmt.Sprintf("wrote %d bytes to %s", len(content), path), nil
}

// commandDenyChars are the characters refused in yolo mode (no per-command
// user approval there, so command chaining and substitution stay off the
// table). The interactive mode runs the command under /bin/sh after an
// explicit y/N confirm — the confirmed literal text is the consent, and
// pipes/quotes are exactly what make lookups useful.
const commandDenyChars = ";`$\n\r"

// commandDenyNames maps each denied rune to its plain name for the error.
var commandDenyNames = map[rune]string{
	';':  "command chaining (;)",
	'`':  "command substitution (backticks)",
	'$':  "variable expansion ($)",
	'\n': "newline",
	'\r': "newline",
}

// toolRunCommand runs the command. Confirmation is owned by the seed
// executor (UI.Confirm or the yolo bypass) — this function just executes.
// Interactive (confirmed) runs go through /bin/sh so pipes and quoting
// work; yolo runs skip the shell and stay single-command (exec argv split,
// the characters above refused up front).
func toolRunCommand(ctx context.Context, args map[string]string) (string, error) {
	cmdline := strings.TrimSpace(args["command"])
	if cmdline == "" {
		return "", fmt.Errorf("empty command")
	}
	ctx, cancel := context.WithTimeout(ctx, commandTimeout)
	defer cancel()

	var cmd *exec.Cmd
	if toolSafetyBypass {
		for _, r := range cmdline {
			if name, bad := commandDenyNames[r]; bad {
				return "", fmt.Errorf("refusing %s in yolo mode — run one simple command (e.g. %q)",
					name, strings.Fields(cmdline)[0])
			}
		}
		fields := strings.Fields(cmdline)
		cmd = exec.CommandContext(ctx, fields[0], fields[1:]...)
	} else {
		cmd = exec.CommandContext(ctx, "/bin/sh", "-c", cmdline)
	}

	cmd.Dir = cwd()
	out, err := cmd.CombinedOutput()
	s := strings.TrimRight(string(out), "\n")
	if len(s) > toolResultCap {
		s = s[:toolResultCap] + "\n…[truncated]"
	}
	if err != nil {
		if s == "" {
			return "", err
		}
		return s + "\n(exit status: " + err.Error() + ")", nil
	}
	if s == "" {
		return "(no output)", nil
	}
	return s, nil
}

func toolRunWebFetch(ctx context.Context, args map[string]string) (string, error) {
	rawURL := strings.TrimSpace(args["url"])
	text, err := fetchReadable(ctx, rawURL)
	if err != nil {
		return "", err
	}
	if len(text) > toolResultCap {
		text = text[:toolResultCap] + "\n…[truncated]"
	}
	return text, nil
}

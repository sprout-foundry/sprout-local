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
	"strconv"
	"strings"
	"time"

	"github.com/sprout-foundry/sinter/llm"
)

const (
	// maxToolSteps caps tool round-trips per user turn, so a model that
	// keeps calling tools can't loop forever (or burn the token budget).
	maxToolSteps = 4
	// toolResultCap bounds each tool result fed back to the model.
	toolResultCap = 4000 // characters
	// commandTimeout bounds run_command executions.
	commandTimeout = 30 * time.Second
)

// toolsRequested is the REPL session switch, toggled by /tools. Tools add
// prompt tokens and free-form calls to a small model, so they start off.
var toolsRequested bool

// toolSafetyBypass skips the run_command confirmation prompt (set by
// `/tools yolo`). Every other tool runs without prompting.
var toolSafetyBypass bool

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
		name:        "run_command",
		description: "Run a short shell command in the current working directory and return stdout+stderr. For quick lookups only (ls, cat, grep, git status, …). Requires user confirmation at runtime.",
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

// toolPromptBlock renders the "# Tools" system block for the static
// registry, mirroring the exact wording of qwen3.5's chat_template.jinja.
func toolPromptBlock() string {
	return renderToolPromptBlock(toolRegistry)
}

// ─── Parsing ─────────────────────────────────────────────────────────────

// parsedToolCall is one extracted call: tool name + arguments.
type parsedToolCall struct {
	name      string
	args      map[string]string
	truncated bool // ran out of tokens mid-call; recovered best-effort
}

// extractToolCalls finds every <tool_call>…</tool_call> block in a model
// response and parses name + parameters. Malformed blocks are skipped.
// Text outside the blocks is returned too, so the caller can show it as
// the assistant's preamble.
func extractToolCalls(text string) (remaining string, calls []parsedToolCall) {
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
	return "", fmt.Errorf("path %q is outside the sandbox (allowed: cwd, /tmp, $TMPDIR)", p)
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
// status line (full text still goes to the model).
func truncateResultForDisplay(s string) string {
	const max = 120
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
// run_command confirmation prompt.
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
	}
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

// commandAllowBits are the bits of any argument word that immediately mark
// a command as unsafe (shell metacharacters used for chaining, redirection
// or substitution). run_command is for quick lookups and awk-style
// one-liners: $ is allowed (awk '{sum+=$1}'), pipes/chaining/redirects are
// not.
const commandAllowBits = "|;&`><\n\r"

// toolRunCommand runs the command. Confirmation is owned by the seed
// executor (UI.Confirm or the yolo bypass) — this function just executes.
// Shell-free (no sh -c): arguments split on whitespace, so
// chaining/redirection metacharacters are rejected up front.
func toolRunCommand(ctx context.Context, args map[string]string) (string, error) {
	cmdline := strings.TrimSpace(args["command"])
	if cmdline == "" {
		return "", fmt.Errorf("empty command")
	}
	for _, r := range cmdline {
		if strings.ContainsRune(commandAllowBits, r) {
			return "", fmt.Errorf("refusing command with %q — run_command takes a single simple command", string(r))
		}
	}
	fields := strings.Fields(cmdline)

	ctx, cancel := context.WithTimeout(ctx, commandTimeout)
	defer cancel()
	cmd := exec.CommandContext(ctx, fields[0], fields[1:]...)
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

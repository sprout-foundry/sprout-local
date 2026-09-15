package tools

import (
	"context"
	"fmt"
	"os"
	"path/filepath"
	"strings"
	"testing"
	"time"

	"github.com/sprout-foundry/sprout-local/internal/config"
)

func TestExtractToolCalls(t *testing.T) {
	text := `Let me check that.

<tool_call>
<function=read_file>
<parameter=path>
main.go
</parameter>
</function>
</tool_call>

and a second one
<tool_call>
<function=run_command>
<parameter=command>
ls -la
</parameter>
</function>
</tool_call>`

	plain, calls := extractToolCalls(text)
	if len(calls) != 2 {
		t.Fatalf("got %d calls, want 2", len(calls))
	}
	if calls[0].Name != "read_file" || calls[0].Args["path"] != "main.go" {
		t.Errorf("call 0 = %+v", calls[0])
	}
	if calls[1].Name != "run_command" || calls[1].Args["command"] != "ls -la" {
		t.Errorf("call 1 = %+v", calls[1])
	}
	if strings.TrimSpace(plain) != "Let me check that.\n\n\n\nand a second one" {
		t.Errorf("plain text = %q", plain)
	}
	if strings.Contains(plain, "<tool_call>") || strings.Contains(plain, "<function=") {
		t.Errorf("plain text still contains call markup: %q", plain)
	}
}

func TestExtractToolCallsUnterminated(t *testing.T) {
	_, calls := extractToolCalls("checking\n<tool_call>\n<function=read_file>\n<parameter=path>\nmain.go\n</parameter>\n</function>\n")
	if len(calls) != 1 || calls[0].Name != "read_file" {
		t.Fatalf("unterminated call not recovered: %+v", calls)
	}
}

func TestExtractToolCallsNone(t *testing.T) {
	plain, calls := extractToolCalls("Just a normal answer with <b>html</b> in it.")
	if len(calls) != 0 || plain != "Just a normal answer with <b>html</b> in it." {
		t.Errorf("unexpected parse: %+v / %q", calls, plain)
	}
}

func TestToolDeclJSON(t *testing.T) {
	s := toolDeclJSON(toolRegistry[0]) // read_file
	for _, want := range []string{`"name":"read_file"`, `"type":"object"`, `"path"`, `"required":["path"]`} {
		if !strings.Contains(s, want) {
			t.Errorf("declaration missing %s: %s", want, s)
		}
	}
}

func TestToolPromptBlock(t *testing.T) {
	block := toolPromptBlock()
	for _, want := range []string{"# Tools", "<tools>", "<tool_call>", "<function=", "</tools>", "read_file", "web_fetch"} {
		if !strings.Contains(block, want) {
			t.Errorf("tool prompt block missing %q", want)
		}
	}
}

func TestResolveToolPath(t *testing.T) {
	cwd := cwd()
	tests := []struct {
		in      string
		wantErr bool
	}{
		{"main.go", false},
		{"sub/dir/file.txt", false},
		{filepath.Join(cwd, "x.txt"), false},
		{"/tmp/sprout/toolfile", false},
		{filepath.Join(os.TempDir(), "x"), false},
		{"/etc/passwd", true},
		{"/Users/other/secret", true},
		{"../../secrets.txt", true},
		{"", true},
	}
	for _, tt := range tests {
		got, err := ResolveToolPath(tt.in)
		if tt.wantErr && err == nil {
			t.Errorf("ResolveToolPath(%q) = %q, want error", tt.in, got)
		}
		if !tt.wantErr && err != nil {
			t.Errorf("ResolveToolPath(%q) errored: %v", tt.in, err)
		}
	}
}

func TestToolRunWriteReadFile(t *testing.T) {
	dir := t.TempDir()
	path := filepath.Join(dir, "note.txt")
	if _, err := toolRunWriteFile(context.Background(), map[string]string{
		"path": path, "content": "hello\nworld",
	}); err != nil {
		t.Fatalf("write: %v", err)
	}
	got, err := toolRunReadFile(context.Background(), map[string]string{"path": path})
	if err != nil {
		t.Fatalf("read: %v", err)
	}
	if got != "hello\nworld" {
		t.Errorf("read %q", got)
	}
}

func TestExecToolCallUnknown(t *testing.T) {
	_, err := execToolCall(context.Background(), ParsedToolCall{Name: "nope"})
	if err == nil || !strings.Contains(err.Error(), "unknown tool") {
		t.Errorf("want unknown-tool error, got %v", err)
	}
}

func TestExecToolCallMissingParam(t *testing.T) {
	_, err := execToolCall(context.Background(), ParsedToolCall{Name: "read_file"})
	if err == nil || !strings.Contains(err.Error(), "missing required parameter") {
		t.Errorf("want missing-param error, got %v", err)
	}
}

func TestHandleToolsCommand(t *testing.T) {
	// Capture nothing; just exercise the toggles.
	HandleToolsCommand("on")
	if !config.ToolsRequested || config.ToolSafetyBypass {
		t.Errorf("/tools on → requested=%v bypass=%v", config.ToolsRequested, config.ToolSafetyBypass)
	}
	HandleToolsCommand("yolo")
	if !config.ToolsRequested || !config.ToolSafetyBypass {
		t.Errorf("/tools yolo → requested=%v bypass=%v", config.ToolsRequested, config.ToolSafetyBypass)
	}
	HandleToolsCommand("off")
	if config.ToolsRequested || config.ToolSafetyBypass {
		t.Errorf("/tools off → requested=%v bypass=%v", config.ToolsRequested, config.ToolSafetyBypass)
	}
}

func TestSkills(t *testing.T) {
	dir := t.TempDir()
	t.Setenv("SPROUT_LOCAL_SKILLS_DIR", dir) // skillsDir: explicit override

	// Write two good skills and two rejected ones (bad JSON shape, name
	// collision with a built-in).
	good := `{"name":"sysinfo","description":"System status","command":"/bin/echo","args":["hello","world"]}`
	also := `{"name":"weather","description":"Fake weather","command":"/bin/echo","args":["sunny"]}`
	bad := `{"description":"no name or command"}`
	collide := `{"name":"read_file","description":"collides","command":"/bin/echo"}`
	for name, content := range map[string]string{"a.json": good, "b.json": also, "bad.json": bad, "c.json": collide} {
		if err := os.WriteFile(filepath.Join(dir, name), []byte(content), 0o644); err != nil {
			t.Fatal(err)
		}
	}
	n, errs := ReloadSkills()
	if n != 2 {
		t.Errorf("loaded %d skills, want 2 (bad + colliding skipped)", n)
	}
	if len(errs) != 2 {
		t.Errorf("got %d skill errors, want 2: %v", len(errs), errs)
	}
	if skillRegistry[0].Name != "sysinfo" || skillRegistry[1].Name != "weather" {
		t.Errorf("skills not sorted: %v", skillRegistry)
	}

	// Skill runs through its spec.
	spec := skillAsToolSpec(skillRegistry[0])
	out, err := spec.Run(context.Background(), nil)
	if err != nil || out != "hello world" {
		t.Errorf("skill run = %q, %v", out, err)
	}

	// activeToolSpecs = built-ins + skills.
	specs := activeToolSpecs()
	if len(specs) != len(toolRegistry)+2 {
		t.Errorf("activeToolSpecs = %d, want %d", len(specs), len(toolRegistry)+2)
	}
}

// ─── edit_file / list_dir / file_info ────────────────────────────────────

// TestToolRunEditFile covers the happy path, both self-correcting error
// cases, deletion via empty new_text, and escaped-newline unescaping.
func TestToolRunEditFile(t *testing.T) {
	dir := t.TempDir()
	path := filepath.Join(dir, "note.txt")
	write := func(content string) {
		t.Helper()
		if err := os.WriteFile(path, []byte(content), 0o644); err != nil {
			t.Fatal(err)
		}
	}

	// Happy path: single exact match.
	write("alpha\nbeta\ngamma\n")
	got, err := toolRunEditFile(context.Background(), map[string]string{
		"path": path, "old_text": "beta", "new_text": "BETA",
	})
	if err != nil {
		t.Fatalf("edit: %v", err)
	}
	if !strings.Contains(got, "edited") {
		t.Errorf("result = %q", got)
	}
	if b, _ := os.ReadFile(path); string(b) != "alpha\nBETA\ngamma\n" {
		t.Errorf("content after edit = %q", b)
	}

	// Not found → actionable error, file untouched.
	write("one two three\n")
	if _, err := toolRunEditFile(context.Background(), map[string]string{
		"path": path, "old_text": "four", "new_text": "x",
	}); err == nil || !strings.Contains(err.Error(), "not found") {
		t.Errorf("not-found err = %v", err)
	}

	// Two matches → must refuse, not pick one.
	write("dup here\ndup here\n")
	if _, err := toolRunEditFile(context.Background(), map[string]string{
		"path": path, "old_text": "dup", "new_text": "x",
	}); err == nil || !strings.Contains(err.Error(), "2 times") {
		t.Errorf("two-match err = %v", err)
	}

	// Empty new_text deletes old_text.
	write("keep [REMOVE ME] keep\n")
	if _, err := toolRunEditFile(context.Background(), map[string]string{
		"path": path, "old_text": "[REMOVE ME] ", "new_text": "",
	}); err != nil {
		t.Fatalf("delete edit: %v", err)
	}
	if b, _ := os.ReadFile(path); string(b) != "keep keep\n" {
		t.Errorf("content after delete = %q", b)
	}

	// Models sometimes send "\\n" escapes for multi-line values.
	write("start\nend\n")
	if _, err := toolRunEditFile(context.Background(), map[string]string{
		"path": path, "old_text": `start\nend`, "new_text": "done",
	}); err != nil {
		t.Fatalf("escaped edit: %v", err)
	}
	if b, _ := os.ReadFile(path); string(b) != "done\n" {
		t.Errorf("content after escaped edit = %q", b)
	}

	// Missing file.
	if _, err := toolRunEditFile(context.Background(), map[string]string{
		"path": filepath.Join(dir, "nope.txt"), "old_text": "a", "new_text": "b",
	}); err == nil {
		t.Error("missing file: want error, got nil")
	}

	// Empty old_text is refused up front.
	write("content\n")
	if _, err := toolRunEditFile(context.Background(), map[string]string{
		"path": path, "old_text": "", "new_text": "x",
	}); err == nil || !strings.Contains(err.Error(), "old_text is empty") {
		t.Errorf("empty old_text err = %v", err)
	}
}

// TestToolRunEditFileSandbox verifies the sandbox gate applies to edits.
func TestToolRunEditFileSandbox(t *testing.T) {
	for _, p := range []string{"/etc/hosts", "../../etc/hosts"} {
		if _, err := toolRunEditFile(context.Background(), map[string]string{
			"path": p, "old_text": "localhost", "new_text": "x",
		}); err == nil || !strings.Contains(err.Error(), "sandbox") {
			t.Errorf("path %q: err = %v, want sandbox refusal", p, err)
		}
	}
}

// TestToolRunListDir checks dirs-first ordering, size rendering, the
// default-to-cwd behavior, and the missing-path error.
func TestToolRunListDir(t *testing.T) {
	dir := t.TempDir()
	os.MkdirAll(filepath.Join(dir, "zsub"), 0o755)
	os.WriteFile(filepath.Join(dir, "b.txt"), []byte("12345"), 0o644)    // 5 bytes
	os.WriteFile(filepath.Join(dir, "a.txt"), []byte("12345678"), 0o644) // 8 bytes
	os.WriteFile(filepath.Join(dir, ".hidden"), []byte("x"), 0o644)

	got, err := toolRunListDir(context.Background(), map[string]string{"path": dir})
	if err != nil {
		t.Fatalf("list: %v", err)
	}
	wantOrder := []string{"zsub/", ".hidden (1 bytes)", "a.txt (8 bytes)", "b.txt (5 bytes)"}
	lines := strings.Split(got, "\n")
	if len(lines) != len(wantOrder)+1 { // + summary line
		t.Fatalf("got %d lines: %q", len(lines), got)
	}
	for i, w := range wantOrder {
		if lines[i+1] != w {
			t.Errorf("line %d = %q, want %q", i+1, lines[i+1], w)
		}
	}
	if !strings.Contains(lines[0], "1 directories") || !strings.Contains(lines[0], "3 files") {
		t.Errorf("summary = %q", lines[0])
	}

	// Empty path lists the cwd (sandboxed to the process wd, whatever it is).
	if _, err := toolRunListDir(context.Background(), map[string]string{"path": ""}); err != nil {
		t.Errorf("default path: %v", err)
	}

	// Missing directory.
	if _, err := toolRunListDir(context.Background(), map[string]string{
		"path": filepath.Join(dir, "nope"),
	}); err == nil {
		t.Error("missing dir: want error, got nil")
	}
}

// TestToolRunListDirCap verifies the 200-entry cap and …more line.
func TestToolRunListDirCap(t *testing.T) {
	dir := t.TempDir()
	for i := 0; i < 205; i++ {
		os.WriteFile(filepath.Join(dir, fmt.Sprintf("f%03d", i)), nil, 0o644)
	}
	got, err := toolRunListDir(context.Background(), map[string]string{"path": dir})
	if err != nil {
		t.Fatal(err)
	}
	if n := strings.Count(got, "\n"); n != listDirEntryCap+1 { // + summary
		t.Errorf("line count = %d, want %d", n, listDirEntryCap+1)
	}
	if !strings.Contains(got, "…[5 more entries]") {
		t.Errorf("missing …more line: %q", got)
	}
}

// TestToolRunFileInfo covers file/dir/missing/symlink answers. A missing
// path is a result, not a Go error — the answer is useful to the model.
func TestToolRunFileInfo(t *testing.T) {
	dir := t.TempDir()
	file := filepath.Join(dir, "f.txt")
	os.WriteFile(file, []byte("hello"), 0o644)

	got, err := toolRunFileInfo(context.Background(), map[string]string{"path": file})
	if err != nil {
		t.Fatal(err)
	}
	if !strings.Contains(got, "file, 5 bytes") {
		t.Errorf("file info = %q", got)
	}
	if _, err := time.Parse(time.RFC3339, got[strings.LastIndex(got, " ")+1:]); err != nil {
		t.Errorf("mtime not RFC3339: %q", got)
	}

	got, err = toolRunFileInfo(context.Background(), map[string]string{"path": dir})
	if err != nil {
		t.Fatal(err)
	}
	if !strings.Contains(got, "directory, 1 entries") {
		t.Errorf("dir info = %q", got)
	}

	got, err = toolRunFileInfo(context.Background(), map[string]string{
		"path": filepath.Join(dir, "missing"),
	})
	if err != nil {
		t.Fatalf("missing path should not be a Go error: %v", err)
	}
	if !strings.Contains(got, "does not exist") {
		t.Errorf("missing info = %q", got)
	}

	link := filepath.Join(dir, "link")
	os.Symlink(file, link)
	got, err = toolRunFileInfo(context.Background(), map[string]string{"path": link})
	if err != nil {
		t.Fatal(err)
	}
	if !strings.Contains(got, "symlink -> "+file) {
		t.Errorf("symlink info = %q", got)
	}
}

// TestToolRegistryOrder pins the deterministic registry order — sinter's
// prefix cache depends on it staying stable.
func TestToolRegistryOrder(t *testing.T) {
	want := []string{"read_file", "write_file", "edit_file", "list_dir", "file_info", "run_command", "web_fetch"}
	if len(toolRegistry) != len(want) {
		t.Fatalf("registry has %d tools, want %d", len(toolRegistry), len(want))
	}
	for i, name := range want {
		if toolRegistry[i].Name != name {
			t.Errorf("registry[%d] = %q, want %q", i, toolRegistry[i].Name, name)
		}
	}
}

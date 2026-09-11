package main

import (
	"context"
	"os"
	"path/filepath"
	"strings"
	"testing"
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
	if calls[0].name != "read_file" || calls[0].args["path"] != "main.go" {
		t.Errorf("call 0 = %+v", calls[0])
	}
	if calls[1].name != "run_command" || calls[1].args["command"] != "ls -la" {
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
	if len(calls) != 1 || calls[0].name != "read_file" {
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
		got, err := resolveToolPath(tt.in)
		if tt.wantErr && err == nil {
			t.Errorf("resolveToolPath(%q) = %q, want error", tt.in, got)
		}
		if !tt.wantErr && err != nil {
			t.Errorf("resolveToolPath(%q) errored: %v", tt.in, err)
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
	_, err := execToolCall(context.Background(), parsedToolCall{name: "nope"})
	if err == nil || !strings.Contains(err.Error(), "unknown tool") {
		t.Errorf("want unknown-tool error, got %v", err)
	}
}

func TestExecToolCallMissingParam(t *testing.T) {
	_, err := execToolCall(context.Background(), parsedToolCall{name: "read_file"})
	if err == nil || !strings.Contains(err.Error(), "missing required parameter") {
		t.Errorf("want missing-param error, got %v", err)
	}
}

func TestHandleToolsCommand(t *testing.T) {
	// Capture nothing; just exercise the toggles.
	handleToolsCommand("on")
	if !toolsRequested || toolSafetyBypass {
		t.Errorf("/tools on → requested=%v bypass=%v", toolsRequested, toolSafetyBypass)
	}
	handleToolsCommand("yolo")
	if !toolsRequested || !toolSafetyBypass {
		t.Errorf("/tools yolo → requested=%v bypass=%v", toolsRequested, toolSafetyBypass)
	}
	handleToolsCommand("off")
	if toolsRequested || toolSafetyBypass {
		t.Errorf("/tools off → requested=%v bypass=%v", toolsRequested, toolSafetyBypass)
	}
}

func TestSkills(t *testing.T) {
	dir := t.TempDir()
	t.Setenv("HOME", dir) // skillsDir derives from home
	skills := filepath.Join(dir, ".chatllm", "skills")
	if err := os.MkdirAll(skills, 0o755); err != nil {
		t.Fatal(err)
	}

	// Write two good skills and two rejected ones (bad JSON shape, name
	// collision with a built-in).
	good := `{"name":"sysinfo","description":"System status","command":"/bin/echo","args":["hello","world"]}`
	also := `{"name":"weather","description":"Fake weather","command":"/bin/echo","args":["sunny"]}`
	bad := `{"description":"no name or command"}`
	collide := `{"name":"read_file","description":"collides","command":"/bin/echo"}`
	for name, content := range map[string]string{"a.json": good, "b.json": also, "bad.json": bad, "c.json": collide} {
		if err := os.WriteFile(filepath.Join(skills, name), []byte(content), 0o644); err != nil {
			t.Fatal(err)
		}
	}
	n, errs := reloadSkills()
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
	out, err := spec.run(context.Background(), nil)
	if err != nil || out != "hello world" {
		t.Errorf("skill run = %q, %v", out, err)
	}

	// activeToolSpecs = built-ins + skills.
	specs := activeToolSpecs()
	if len(specs) != len(toolRegistry)+2 {
		t.Errorf("activeToolSpecs = %d, want %d", len(specs), len(toolRegistry)+2)
	}
}

func TestToolStreamFilter(t *testing.T) {
	var out strings.Builder
	printer := newStreamPrinter(&out, mdRaw)
	f := &toolStreamFilter{printer: printer, tools: true}
	f.write("Let me check. <tool_call>\n<function=read_fi")
	f.write("le>\n<parameter=path>\nmain.go\n</parameter>\n</function>\n</tool_call> done.")
	f.close()
	printer.Close()
	got := out.String()
	if strings.Contains(got, "<tool_call>") || strings.Contains(got, "function=") {
		t.Errorf("markup leaked to display: %q", got)
	}
	if !strings.Contains(got, "Let me check.") || !strings.Contains(got, "done.") {
		t.Errorf("prose lost: %q", got)
	}
}

func TestToolStreamFilterPassThrough(t *testing.T) {
	var out strings.Builder
	printer := newStreamPrinter(&out, mdRaw)
	f := &toolStreamFilter{printer: printer, tools: false}
	f.write("plain answer, no filtering")
	f.close()
	printer.Close()
	if got := out.String(); got != "plain answer, no filtering" {
		t.Errorf("pass-through broken: %q", got)
	}
}

// TestToolStreamFilterSeedBuffer verifies the seed feed matches the
// terminal: markup suppressed, prose delivered to both sinks.
func TestToolStreamFilterSeedBuffer(t *testing.T) {
	var out strings.Builder
	var seed strings.Builder
	printer := newStreamPrinter(&out, mdRaw)
	f := &toolStreamFilter{printer: printer, tools: true, onDelta: func(s string) { seed.WriteString(s) }}
	f.write("Check. <tool_call>\n<function=read_file>\n<parameter=path>\nx\n</parameter>\n</function>\n</tool_call> Done.")
	f.close()
	printer.Close()
	for _, sink := range map[string]string{"term": out.String(), "seed": seed.String()} {
		if strings.Contains(sink, "<tool_call>") {
			t.Errorf("%s got markup: %q", sink, sink)
		}
		if !strings.Contains(sink, "Check.") || !strings.Contains(sink, "Done.") {
			t.Errorf("%s lost prose: %q", sink, sink)
		}
	}
}

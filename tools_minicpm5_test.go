package main

import (
	"fmt"
	"os"
	"path/filepath"
	"strings"
	"testing"
)

// TestMiniCPM5ProtocolDetection pins model-dir → protocol resolution.
func TestMiniCPM5ProtocolDetection(t *testing.T) {
	dir := t.TempDir()
	writeModel := func(name, modelType string, withMiniTokens bool) string {
		d := filepath.Join(dir, name)
		if err := os.MkdirAll(d, 0o755); err != nil {
			t.Fatal(err)
		}
		cfg := fmt.Sprintf(`{"model_type": %q}`, modelType)
		if err := os.WriteFile(filepath.Join(d, "config.json"), []byte(cfg), 0o644); err != nil {
			t.Fatal(err)
		}
		tok := `{"added_tokens": []}`
		if withMiniTokens {
			tok = `{"added_tokens": [{"id": 130080, "content": "/think"}, {"id": 130081, "content": "/no_think"}]}`
		}
		if err := os.WriteFile(filepath.Join(d, "tokenizer.json"), []byte(tok), 0o644); err != nil {
			t.Fatal(err)
		}
		return d
	}
	mini := writeModel("mini", "llama", true)
	if got := toolProtocolForModelDir(mini); got != "minicpm5" {
		t.Fatalf("MiniCPM5 dir should resolve to minicpm5 protocol, got %q", got)
	}
	qwen := writeModel("qwen", "qwen3_5_text", false)
	if got := toolProtocolForModelDir(qwen); got != "qwen" {
		t.Fatalf("qwen dir should resolve to qwen protocol, got %q", got)
	}
	// A llama model_type WITHOUT MiniCPM5's control tokens stays qwen
	// (plain Llama chat would need its own template — out of scope).
	plainLlama := writeModel("llama", "llama", false)
	if got := toolProtocolForModelDir(plainLlama); got != "qwen" {
		t.Fatalf("plain llama dir should stay qwen protocol, got %q", got)
	}
}

// TestExtractMiniCPM5Calls pins the MiniCPM5 tool-call parser: plain
// params, CDATA params, prose before/after, and unterminated recovery.
func TestExtractMiniCPM5Calls(t *testing.T) {
	text := `Let me read that file.
<function name="read_file"><param name="path">/tmp/notes.txt</param></function>`
	plain, calls := extractToolCallsFor(text, "minicpm5")
	if len(calls) != 1 || calls[0].name != "read_file" || calls[0].args["path"] != "/tmp/notes.txt" {
		t.Fatalf("simple call parse failed: %+v", calls)
	}
	if want := "Let me read that file.\n"; plain != want {
		t.Fatalf("plain text mismatch: %q", plain)
	}

	cdata := `<function name="write_file"><param name="path">/tmp/out.txt</param><param name="content"><![CDATA[line one
line two <with> markup & stuff]]></param></function>`
	_, calls = extractToolCallsFor(cdata, "minicpm5")
	if len(calls) != 1 || calls[0].name != "write_file" {
		t.Fatalf("CDATA call parse failed: %+v", calls)
	}
	if calls[0].args["content"] != "line one\nline two <with> markup & stuff" {
		t.Fatalf("CDATA content mismatch: %q", calls[0].args["content"])
	}

	// Unterminated: recover name + completed params.
	trunc := `<function name="write_file"><param name="path">/tmp/x</param><param name="content">partial dat`
	_, calls = extractToolCallsFor(trunc, "minicpm5")
	if len(calls) != 1 || calls[0].name != "write_file" || !calls[0].truncated || calls[0].args["path"] != "/tmp/x" {
		t.Fatalf("truncated call recovery failed: %+v", calls)
	}
}

// TestRenderMiniCPM5ToolCallText pins history rendering: CDATA wrapping for
// values with special chars, plain otherwise.
func TestRenderMiniCPM5ToolCallText(t *testing.T) {
	got := renderMiniCPM5ToolCallText("write_file", map[string]string{
		"path":    "/tmp/a.txt",
		"content": "two\nlines",
	}, false)
	// Args render in sorted key order (content before path) — deterministic
	// for the prefix cache.
	want := `<function name="write_file"><param name="content"><![CDATA[two` +
		"\nlines]]></param><param name=\"path\">/tmp/a.txt</param></function>"
	if got != want {
		t.Fatalf("render mismatch:\n got %q\nwant %q", got, want)
	}
}

// TestQwenParseUnaffectedByMiniCPM5 guards the qwen parser against the new
// fallback: a qwen-protocol body must still parse, and must not be
// misinterpreted as MiniCPM5 markup.
func TestQwenParseUnaffectedByMiniCPM5(t *testing.T) {
	body := "<tool_call>\n<function=read_file>\n<parameter=path>\n/tmp/x\n</parameter>\n</function>\n</tool_call>"
	_, calls := extractToolCallsFor(body, "qwen")
	if len(calls) != 1 || calls[0].name != "read_file" || calls[0].args["path"] != "/tmp/x" {
		t.Fatalf("qwen parse broken: %+v", calls)
	}
}

// TestToolStreamFilterMiniCPM5 checks the display filter suppresses the
// MiniCPM5 markup span.
func TestToolStreamFilterMiniCPM5(t *testing.T) {
	f := &toolStreamFilter{tools: true, protocol: "minicpm5"}
	var out strings.Builder
	f.printer = newStreamPrinter(&out, mdRaw)
	f.write("prose ")
	f.write(`<function name="read_file"><param name="path">/tmp/x</param>`)
	f.write(`</function>`)
	f.write(" after")
	f.close()
	got := out.String()
	if strings.Contains(got, "function") || strings.Contains(got, "param") {
		t.Fatalf("markup leaked to display: %q", got)
	}
	if !strings.Contains(got, "prose") || !strings.Contains(got, "after") {
		t.Fatalf("prose lost: %q", got)
	}
}

// homeModelDir resolves a directory under the shared models root for tests.
func homeModelDir(name string) string {
	return filepath.Join(modelsRoot(), name)
}
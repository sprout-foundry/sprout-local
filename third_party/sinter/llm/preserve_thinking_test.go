package llm

import (
	"encoding/json"
	"os"
	"path/filepath"
	"strings"
	"testing"
)

// writeQwen38Tokenizer writes files shaped like the real Qwen3.8-27B
// artifacts: tokenizer.json with an NFC normalizer, chat_template.jinja
// with the preserve_thinking kwarg, and tokenizer_config.json with the
// (misleadingly 3.5-shaped) pretokenize_regex the mlx-community conversion
// ships — the regex must NOT drive family detection.
func writeQwen38Tokenizer(t *testing.T, dir string) {
	t.Helper()
	tj := `{
	  "model": {"type": "BPE", "vocab": {"a": 10, "b": 11}, "merges": []},
	  "normalizer": {"type": "NFC"},
	  "added_tokens": [
	    {"id": 1, "content": "<|im_start|>"},
	    {"id": 2, "content": "<|im_end|>"},
	    {"id": 3, "content": "<think>"},
	    {"id": 4, "content": "</think>"}
	  ]
	}`
	if err := os.WriteFile(filepath.Join(dir, "tokenizer.json"), []byte(tj), 0o644); err != nil {
		t.Fatal(err)
	}
	// Reference-template excerpt: the two markers detection keys on.
	jinja := `{%- if enable_thinking is undefined or enable_thinking is true %}
{%- set resolved_reasoning_effort = reasoning_effort|default('xhigh') %}
{%- endif %}
{%- if preserve_thinking is undefined or preserve_thinking is true %}
{{- '<|im_start|>' + message.role + '\n<think>\n' + reasoning_content + '\n</think>\n\n' + content }}
{%- endif %}
{%- if add_generation_prompt %}{{- '<|im_start|>assistant\n' }}{%- if enable_thinking is defined and enable_thinking is false %}{{- '<think>\n\n</think>\n\n' }}{%- else %}{{- '<think>\n' }}{%- endif %}{%- endif %}`
	if err := os.WriteFile(filepath.Join(dir, "chat_template.jinja"), []byte(jinja), 0o644); err != nil {
		t.Fatal(err)
	}
	// The 3.5-shaped regex the real 27b conversion ships — presence must
	// not flip the family back.
	tc := map[string]string{
		"pretokenize_regex": "(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\\r\\n\\p{L}\\p{N}]?[\\p{L}\\p{M}]+|\\p{N}| ?[^\\s\\p{L}\\p{M}\\p{N}]+[\\r\\n]*|\\s*[\\r\\n]+|\\s+(?!\\S)|\\s+",
	}
	data, _ := json.Marshal(tc)
	if err := os.WriteFile(filepath.Join(dir, "tokenizer_config.json"), data, 0o644); err != nil {
		t.Fatal(err)
	}
}

func writeQwen35Tokenizer(t *testing.T, dir string) {
	t.Helper()
	tj := `{
	  "model": {"type": "BPE", "vocab": {"a": 10, "b": 11}, "merges": []},
	  "added_tokens": [
	    {"id": 1, "content": "<|im_start|>"},
	    {"id": 2, "content": "<|im_end|>"},
	    {"id": 3, "content": "<think>"},
	    {"id": 4, "content": "</think>"}
	  ]
	}`
	if err := os.WriteFile(filepath.Join(dir, "tokenizer.json"), []byte(tj), 0o644); err != nil {
		t.Fatal(err)
	}
	tc := map[string]string{
		"pretokenize_regex": "(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\\r\\n\\p{L}\\p{N}]?[\\p{L}\\p{M}]+|\\p{N}| ?[^\\s\\p{L}\\p{M}\\p{N}]+[\\r\\n]*|\\s*[\\r\\n]+|\\s+(?!\\S)|\\s+",
	}
	data, _ := json.Marshal(tc)
	if err := os.WriteFile(filepath.Join(dir, "tokenizer_config.json"), data, 0o644); err != nil {
		t.Fatal(err)
	}
}

// TestQwen38TokenizerDetection pins family detection: preserve-thinking is
// detected from chat_template.jinja's preserve_thinking marker, NFC from
// tokenizer.json's normalizer section. The pretokenize_regex is not a
// marker — the real 27b conversion ships a 3.5-shaped regex.
func TestQwen38TokenizerDetection(t *testing.T) {
	dir38 := t.TempDir()
	writeQwen38Tokenizer(t, dir38)
	tok38, err := LoadTokenizer(filepath.Join(dir38, "tokenizer.json"))
	if err != nil {
		t.Fatal(err)
	}
	if !tok38.marksPreserveThinking {
		t.Error("Qwen3.8 chat_template.jinja not detected as preserve-thinking")
	}
	if !tok38.nfcNormalize {
		t.Error("Qwen3.8 tokenizer.json normalizer not detected as NFC")
	}

	dir35 := t.TempDir()
	writeQwen35Tokenizer(t, dir35)
	tok35, err := LoadTokenizer(filepath.Join(dir35, "tokenizer.json"))
	if err != nil {
		t.Fatal(err)
	}
	if tok35.marksPreserveThinking {
		t.Error("Qwen3.5-style config wrongly detected as preserve-thinking")
	}
	if tok35.nfcNormalize {
		t.Error("Qwen3.5-style config wrongly detected as NFC-normalizing")
	}
}

// TestPreserveThinkingTemplate pins the Qwen3.8 reference template shapes:
// history keeps the reasoning trace (<think>\n trace \n</think>\n\n), the
// on cue is an open <think>\n, the off cue is the closed empty block, and
// the history replay prefix property holds for KV-cache reuse.
func TestPreserveThinkingTemplate(t *testing.T) {
	dir := t.TempDir()
	writeQwen38Tokenizer(t, dir)
	tok, err := LoadTokenizer(filepath.Join(dir, "tokenizer.json"))
	if err != nil {
		t.Fatal(err)
	}

	turn1 := []ChatMessage{
		{Role: "system", Content: "Be concise."},
		{Role: "user", Content: "hi"},
	}
	p1 := tok.formatPreserveThinkingChat(turn1, true)
	wantCue := "<|im_start|>assistant\n<think>\n"
	if !strings.HasSuffix(p1, wantCue) {
		t.Fatalf("thinking-on cue mismatch:\n got %q\nwant suffix %q", p1, wantCue)
	}

	p1off := tok.formatPreserveThinkingChat(turn1, false)
	if !strings.HasSuffix(p1off, "<|im_start|>assistant\n<think>\n\n</think>\n\n") {
		t.Fatalf("thinking-off cue mismatch: %q", p1off)
	}

	turn2 := append(turn1,
		ChatMessage{Role: "assistant", ReasoningContent: "  Let me think.  ", Content: "Hello."},
		ChatMessage{Role: "user", Content: "more"},
	)
	p2 := tok.formatPreserveThinkingChat(turn2, true)
	if !strings.Contains(p2, "<|im_start|>assistant\n<think>\nLet me think.\n</think>\n\nHello.") {
		t.Fatalf("history assistant turn must carry trimmed reasoning trace:\n%s", p2)
	}
	// History-replay prefix property: turn2's prompt extends turn1's prompt
	// minus the cue (KV-cache reuse across turns).
	if !strings.HasPrefix(p2, strings.TrimSuffix(p1, wantCue)) {
		t.Fatal("turn-2 prompt does not extend turn-1 prompt (breaks KV prefix-cache reuse)")
	}

	// An assistant turn with no reasoning content renders an empty (but
	// present) trace — matches the reference template's '' default.
	turn3 := []ChatMessage{
		{Role: "assistant", Content: "direct"},
	}
	p3 := tok.formatPreserveThinkingBody(turn3)
	if !strings.HasPrefix(p3, "<|im_start|>assistant\n<think>\n\n</think>\n\ndirect") {
		t.Fatalf("empty-reasoning history shape mismatch: %q", p3)
	}
}

// TestQwen35TemplateUnchanged guards the 3.5 template against regression
// from the preserve-thinking branch: ChatMessage.ReasoningContent must be
// ignored there (history stays the closed empty block).
func TestQwen35TemplateUnchanged(t *testing.T) {
	tok := &Tokenizer{
		vocab:         map[string]int{"<|im_start|>": 1},
		specialTokens: map[string]int{"<|im_start|>": 1},
	}
	msgs := []ChatMessage{
		{Role: "assistant", ReasoningContent: "should not appear", Content: "answer"},
	}
	got := tok.formatQwenChat(msgs)
	want := "<|im_start|>assistant\n<think>\n\n</think>\n\nanswer<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n"
	if got != want {
		t.Fatalf("3.5 template changed:\n got %q\nwant %q", got, want)
	}
}

// TestNFCNormalizationEncoding pins that a preserve-thinking tokenizer
// NFC-normalizes input before pre-tokenization: the decomposed "café"
// (e + U+0301) must encode identically to the precomposed form.
func TestNFCNormalizationEncoding(t *testing.T) {
	dir := t.TempDir()
	writeQwen38Tokenizer(t, dir)
	tok, err := LoadTokenizer(filepath.Join(dir, "tokenizer.json"))
	if err != nil {
		t.Fatal(err)
	}
	decomposed := "caf\u0065\u0301" // e + combining acute
	precomposed := "caf\u00e9"
	if got := nfcNormalize(decomposed); got != precomposed {
		t.Fatalf("NFC failed: %q != %q", got, precomposed)
	}
	a := tok.Encode(decomposed)
	b := tok.Encode(precomposed)
	if len(a) == 0 || len(b) == 0 {
		t.Fatal("empty encoding")
	}
	if len(a) != len(b) {
		t.Fatalf("decomposed input encoded to %d tokens, precomposed to %d", len(a), len(b))
	}
	for i := range a {
		if a[i] != b[i] {
			t.Fatalf("token %d differs: %d vs %d", i, a[i], b[i])
		}
	}
}

// TestReasoningFnStreamsTrace pins the ReasoningFn callback contract: while
// inThinkBlock, trace tokens reach ReasoningFn and (by default) not the
// main output; the closing marker appends the template newline.
func TestReasoningFnStreamsTrace(t *testing.T) {
	tok := &Tokenizer{
		specialTokens: map[string]int{
			"<|im_start|>": 1,
			"<|im_end|>":   2,
			"<think>":      3,
			"</think>":     4,
		},
		idToTok: map[int]string{7: "hi", 8: "there"},
		vocab:   map[string]int{"<|im_start|>": 1, "<|im_end|>": 2, "<think>": 3, "</think>": 4},
	}
	m := &Model{tokenizer: tok}
	m.initThinking()

	var trace strings.Builder
	var output strings.Builder

	cfg := GenerateConfig{ReasoningFn: func(s string) { trace.WriteString(s) }}
	m.inThinkBlock = false

	// <think> opens the block (dropped).
	m.shouldFilterToken(3, cfg)
	// Trace tokens reach ReasoningFn.
	m.shouldFilterToken(7, cfg)
	m.shouldFilterToken(8, cfg)
	// </think> closes (drops the marker, appends newline to the trace).
	m.shouldFilterToken(4, cfg)
	// Answer tokens now go to the main path, not ReasoningFn.
	if m.shouldFilterToken(7, cfg) {
		t.Error("answer token wrongly filtered after </think>")
	}

	if got, want := trace.String(), "hithere\n"; got != want {
		t.Fatalf("trace = %q, want %q", got, want)
	}
	_ = output
}

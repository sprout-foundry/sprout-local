//go:build arm64 && cgo && (darwin || (linux && ggml))

package llm

import (
	"reflect"
	"strings"
	"testing"
)

// TestQwenPreTokenizeNewlineSemantics pins the pre-tokenizer's newline
// handling to HuggingFace's Qwen Split-regex behavior. The previous
// whitespace-attaching splitter merged newline runs into the following word
// ("system\nYou" → ["system", "ĠYou"]) and dropped trailing newlines, so
// every multi-line (chat) prompt encoded to different token IDs than the
// reference tokenizer. Expected segments below are the byte-level-encoded
// forms verified token-for-token against tokenizers' Tokenizer for
// Qwen3.5/Qwen3.6 vocabularies.
func TestQwenPreTokenizeNewlineSemantics(t *testing.T) {
	cases := []struct {
		in   string
		want []string
	}{
		// \s*[\r\n]+ keeps newline runs standalone: Ċ = \n
		{"system\nYou", []string{"system", "Ċ", "You"}},
		{"a\n\nb", []string{"a", "ĊĊ", "b"}},
		{"end\n", []string{"end", "Ċ"}},
		{"x \n", []string{"x", "ĠĊ"}},
		{"  \n c", []string{"ĠĠĊ", "Ġc"}},
		// punct runs absorb trailing newlines (BPE re-splits: ?! + ... + Ċ —
		// verified token-for-token against HF: [14556 25153 1076 198 3480])
		{"hello?!...\nnext", []string{"hello", "?!...Ċ", "next"}},
		// \t is a valid optional-prefix char for a letter run; BPE splits
		// ĉhere into ĉ + here (HF: [5999 197 6527])
		{"tab\there", []string{"tab", "ĉhere"}},
		// \s+(?!\S) leaves one space for the following word
		{"1 2 3", []string{"1", "Ġ", "2", "Ġ", "3"}},
		// contractions and single digits
		{"it's 42", []string{"it", "'s", "Ġ", "4", "2"}},
	}
	for _, tc := range cases {
		got := qwenPreTokenize(tc.in)
		if !reflect.DeepEqual(got, tc.want) {
			t.Errorf("qwenPreTokenize(%q)\n  got  %v\n  want %v", tc.in, got, tc.want)
		}
	}
}

// TestQwenSplitTrailingNewlineNotDropped guards the exact regression that
// collapsed 2-bit MoE generation: a trailing "\n" after the last special
// token (chat templates end "<think>\n") used to vanish entirely. <x is one
// pre-token (punct prefix + letters); the punct token absorbs its newline.
func TestQwenSplitTrailingNewlineNotDropped(t *testing.T) {
	got := qwenSplit("assistant\n<x>\n")
	want := []string{"assistant", "\n", "<x", ">\n"}
	if !reflect.DeepEqual(got, want) {
		t.Fatalf("qwenSplit dropped or merged newline runs\n  got  %q\n  want %q", got, want)
	}
}

// TestDecodeByteLevelRoundTrip checks decode inverts the byte-level mapping.
func TestDecodeByteLevelRoundTrip(t *testing.T) {
	for _, s := range []string{"hello world", "a\nb", "tab\there", "  spaces  ", "punct!?..."} {
		enc := qwenByteEncode(s)
		dec := decodeByteLevel(enc)
		if dec != s {
			t.Errorf("round trip %q → %q → %q", s, enc, dec)
		}
	}
}

// TestFormatChat_HistoryReplayMatchesGeneration guards the KV prefix cache's
// core invariant: the prompt for turn N must be an exact prefix of the
// prompt for turn N+1 once the assistant's turn-N reply is appended as
// history. If FormatChat renders a completed assistant turn differently
// than it rendered the same position as a live "generate now" cue (e.g.
// dropping the <think></think> marker), every multi-turn conversation
// silently loses KV-cache reuse and re-prefills from scratch on every call.
func TestFormatChat_HistoryReplayMatchesGeneration(t *testing.T) {
	tok := &Tokenizer{}

	turn1 := []ChatMessage{
		{Role: "system", Content: "You are a helpful assistant."},
		{Role: "user", Content: "list files"},
	}
	prompt1 := tok.formatQwenChat(turn1)

	turn2 := []ChatMessage{
		{Role: "system", Content: "You are a helpful assistant."},
		{Role: "user", Content: "list files"},
		{Role: "assistant", Content: "<tool_call>\n<function=ls>\n</function>\n</tool_call>\n"},
		{Role: "user", Content: "<tool_response>\na.go\n</tool_response>"},
	}
	prompt2 := tok.formatQwenChat(turn2)

	if !strings.HasPrefix(prompt2, prompt1) {
		t.Fatalf("turn-2 prompt is not an exact prefix extension of turn-1 prompt (breaks KV prefix-cache reuse)\nturn1=%q\nturn2=%q", prompt1, prompt2)
	}
}

func TestFormatLFM2Chat_HistoryReplayMatchesGeneration(t *testing.T) {
	tok := &Tokenizer{}

	turn1 := []ChatMessage{
		{Role: "system", Content: "List of tools: []"},
		{Role: "user", Content: "list files"},
	}
	prompt1 := tok.formatLFM2Chat(turn1)

	turn2 := []ChatMessage{
		{Role: "system", Content: "List of tools: []"},
		{Role: "user", Content: "list files"},
		{Role: "assistant", Content: "<|tool_call_start|>[ls()]<|tool_call_end|>"},
		{Role: "user", Content: "a.go"},
	}
	prompt2 := tok.formatLFM2Chat(turn2)

	if !strings.HasPrefix(prompt2, prompt1) {
		t.Fatalf("turn-2 prompt is not an exact prefix extension of turn-1 prompt (breaks KV prefix-cache reuse)\nturn1=%q\nturn2=%q", prompt1, prompt2)
	}
}

// TestVocabHasMultiDigitToken pins the digit-split detection: a vocab with
// merged 2-3 digit tokens ("12", "123") uses the \p{N}{1,3} pre-split
// (MiniCPM5 / Llama-3 style); a Qwen vocab with only single-digit entries
// does not.
func TestVocabHasMultiDigitToken(t *testing.T) {
	with := map[string]int{"hello": 0, "12": 1, "123": 2}
	if !vocabHasMultiDigitToken(with) {
		t.Fatal("merged digit tokens present but not detected")
	}
	qwen := map[string]int{"hello": 0, "1": 1, "2": 2, "Ġ12": 3}
	if vocabHasMultiDigitToken(qwen) {
		t.Fatal("single-digit vocab flagged as digitGroup")
	}
}

// TestQwenPreTokenizeDigitGroup pins the \p{N}{1,3} digit split used by
// MiniCPM5: digit runs chunk into groups of up to 3 before BPE, matching
// HF's Split("\p{N}{1,3}") pretokenizer. Byte-level-encoded segments shown.
func TestQwenPreTokenizeDigitGroup(t *testing.T) {
	cases := []struct {
		in   string
		want []string
	}{
		{"2026", []string{"202", "6"}},
		{"12345", []string{"123", "45"}},
		{"42", []string{"42"}},
		{"v1.2.3", []string{"v", "1", ".", "2", ".", "3"}},
		{"a 1234 b", []string{"a", "Ġ", "123", "4", "Ġb"}},
	}
	for _, tc := range cases {
		got := qwenPreTokenizeMode(tc.in, true)
		if !reflect.DeepEqual(got, tc.want) {
			t.Errorf("qwenPreTokenizeMode(%q, digitGroup)\n  got  %v\n  want %v", tc.in, got, tc.want)
		}
	}
	// single-digit mode unchanged
	if got := qwenPreTokenizeMode("2026", false); !reflect.DeepEqual(got, []string{"2", "0", "2", "6"}) {
		t.Errorf("digitGroup=false should split single digits, got %v", got)
	}
}

// TestMiniCPM5ChatTemplate pins the MiniCPM5 chat rendering: the generation
// cue is <|im_start|>assistant\n + a bare newline (NOT a closed think
// block — an empty think prefix makes the model answer with <|im_end|>
// immediately), while history assistant turns carry the empty think block.
// History-replay prefix property must hold for KV-cache reuse.
func TestMiniCPM5ChatTemplate(t *testing.T) {
	tok := &Tokenizer{
		vocab:         map[string]int{"/think": 1, "/no_think": 2, "<|im_start|>": 3},
		specialTokens: map[string]int{"/think": 1, "/no_think": 2, "<|im_start|>": 3},
	}
	if !tok.isMiniCPM5() {
		t.Fatal("MiniCPM5 vocab not detected")
	}
	turn1 := []ChatMessage{
		{Role: "system", Content: "Be concise."},
		{Role: "user", Content: "hi"},
	}
	p1 := tok.formatMiniCPM5Chat(turn1)
	want := "<|im_start|>system\nBe concise.<|im_end|>\n<|im_start|>user\nhi<|im_end|>\n<|im_start|>assistant\n\n\n"
	if p1 != want {
		t.Fatalf("generation cue mismatch:\n got %q\nwant %q", p1, want)
	}

	turn2 := append(turn1,
		ChatMessage{Role: "assistant", Content: "Hello."},
		ChatMessage{Role: "user", Content: "more"},
	)
	p2 := tok.formatMiniCPM5Chat(turn2)
	// The trailing "ĊĊ-run" cue is replaced in history by the empty think
	// block + content; the shared prefix ends right after the assistant cue.
	if !strings.HasPrefix(p2, p1[:len(p1)-2]) {
		t.Fatalf("turn-2 prompt does not extend turn-1 prompt (breaks KV prefix-cache reuse)\np1=%q\np2=%q", p1, p2)
	}
	if !strings.Contains(p2, "<|im_start|>assistant\n<think>\n\n</think>\n\nHello.") {
		t.Fatal("history assistant turn missing the empty think block")
	}
}

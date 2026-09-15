package chatmodel

import "testing"

func TestGemmaStripThinking(t *testing.T) {
	in := "before <|channel>thought secret reasoning <channel|> after"
	want := "before  after"
	if got := GemmaStripThinking(in); got != want {
		t.Errorf("GemmaStripThinking = %q, want %q", got, want)
	}
	// Unterminated thought: drop the rest
	in = "answer <|channel>thought never closed"
	if got := GemmaStripThinking(in); got != "answer " {
		t.Errorf("unterminated thought: got %q", got)
	}
	// No marker: passthrough
	if got := GemmaStripThinking("plain"); got != "plain" {
		t.Errorf("passthrough: got %q", got)
	}
}

func TestStripThinkingTag(t *testing.T) {
	in := "answer <thinking>hidden</thinking> tail"
	want := "answer  tail"
	if got := StripThinkingTag(in); got != want {
		t.Errorf("StripThinkingTag = %q, want %q", got, want)
	}
	multiline := "a <thinking>\nline1\nline2\n</thinking>\nb"
	if got := StripThinkingTag(multiline); got != "a \nb" {
		t.Errorf("multiline strip: got %q", got)
	}
}

func TestStripWrapperTag(t *testing.T) {
	tests := []struct{ in, want string }{
		{"<answer>The answer</answer>", "The answer"},
		{"<commit>msg</commit>", "msg"},
		{"<answer>unclosed", "unclosed"}, // stray open tag stripped, content kept
		{"plaintext\nsome text", "some text"},
		{"  <answer>x</answer>  ", "x"},
		{"plain text", "plain text"},
	}
	for _, tt := range tests {
		if got := StripWrapperTag(tt.in); got != tt.want {
			t.Errorf("StripWrapperTag(%q) = %q, want %q", tt.in, got, tt.want)
		}
	}
}

func TestStripOutputNoise(t *testing.T) {
	tests := []struct{ in, want string }{
		{`"quoted"`, "quoted"},
		{"'single'", "single"},
		{"`code`", "code"},
		{"  spaced  ", "spaced"},
		{"no noise", "no noise"},
	}
	for _, tt := range tests {
		if got := StripOutputNoise(tt.in); got != tt.want {
			t.Errorf("StripOutputNoise(%q) = %q, want %q", tt.in, got, tt.want)
		}
	}
}
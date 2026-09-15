package main

import (
	"bufio"
	"strings"
	"testing"

	"github.com/sprout-foundry/sprout-local/internal/config"
)

func TestSplitCommand(t *testing.T) {
	tests := []struct {
		line, cmd, args string
	}{
		{"/help", "help", ""},
		{"/new", "new", ""},
		{"/system be terse", "system", "be terse"},
		{"/system   spaced\targs ", "system", "spaced\targs"},
		{"exit", "exit", ""}, // /-prefix is optional in dispatch, not here
		{"/q", "q", ""},
		{"/", "", ""},
	}
	for _, tt := range tests {
		cmd, args := splitCommand(tt.line)
		if cmd != tt.cmd || args != tt.args {
			t.Errorf("splitCommand(%q) = (%q, %q), want (%q, %q)", tt.line, cmd, args, tt.cmd, tt.args)
		}
	}
}

func TestDispatchCommand(t *testing.T) {
	st := &replState{ui: &termUI{reader: bufio.NewReader(strings.NewReader(""))}}
	st.newAgent()

	// /exit returns true (REPL should exit)
	if !dispatchCommand("exit", "", st) {
		t.Error("/exit should signal REPL exit")
	}
	if !dispatchCommand("quit", "", st) {
		t.Error("/quit should signal REPL exit")
	}

	// /system set + show (rebuilds the agent with the new prompt)
	if dispatchCommand("system", "you are a pirate", st) {
		t.Error("/system must not signal exit")
	}
	if st.systemPrompt != "you are a pirate" {
		t.Errorf("/system set %q, want %q", st.systemPrompt, "you are a pirate")
	}
	if st.agent == nil {
		t.Error("/system left no agent behind")
	}

	// /new rebuilds the agent (fresh conversation)
	if dispatchCommand("new", "", st) {
		t.Error("/new must not signal exit")
	}
	if st.agent == nil {
		t.Error("/new left no agent behind")
	}

	// /tools toggles and rebuilds the executor
	config.ToolsRequested = false
	if dispatchCommand("tools", "on", st) {
		t.Error("/tools must not signal exit")
	}
	if !config.ToolsRequested {
		t.Error("/tools on did not set the flag")
	}
	dispatchCommand("tools", "off", st)
	if config.ToolsRequested {
		t.Error("/tools off did not clear the flag")
	}
}

func TestTrimHistory(t *testing.T) {
	t.Skip("trimHistory retired — seed owns compaction now")
}

func TestGemmaStripThinking(t *testing.T) {
	in := "before <|channel>thought secret reasoning <channel|> after"
	want := "before  after"
	if got := gemmaStripThinking(in); got != want {
		t.Errorf("gemmaStripThinking = %q, want %q", got, want)
	}
	// Unterminated thought: drop the rest
	in = "answer <|channel>thought never closed"
	if got := gemmaStripThinking(in); got != "answer " {
		t.Errorf("unterminated thought: got %q", got)
	}
	// No marker: passthrough
	if got := gemmaStripThinking("plain"); got != "plain" {
		t.Errorf("passthrough: got %q", got)
	}
}

func TestStripThinkingTag(t *testing.T) {
	in := "answer <thinking>hidden</thinking> tail"
	want := "answer  tail"
	if got := stripThinkingTag(in); got != want {
		t.Errorf("stripThinkingTag = %q, want %q", got, want)
	}
	multiline := "a <thinking>\nline1\nline2\n</thinking>\nb"
	if got := stripThinkingTag(multiline); got != "a \nb" {
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
		if got := stripWrapperTag(tt.in); got != tt.want {
			t.Errorf("stripWrapperTag(%q) = %q, want %q", tt.in, got, tt.want)
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
		if got := stripOutputNoise(tt.in); got != tt.want {
			t.Errorf("stripOutputNoise(%q) = %q, want %q", tt.in, got, tt.want)
		}
	}
}

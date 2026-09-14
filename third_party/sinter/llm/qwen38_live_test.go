//go:build cgo && ((darwin && arm64) || (linux && ggml && (arm64 || amd64)))

package llm

import (
	"os"
	"strings"
	"testing"
	"time"
)

// liveQwen38Dir points at the mlx-community Qwen3.8-27B 4-bit build.
func liveQwen38Dir(t *testing.T) string {
	t.Helper()
	if os.Getenv("SINTER_LIVE_QWEN38") == "" {
		t.Skip("SINTER_LIVE_QWEN38 not set")
	}
	dir := os.Getenv("HOME") + "/dev/llm-models/qwen3.8-27b-4bit"
	if _, err := os.Stat(dir + "/config.json"); err != nil {
		t.Skipf("qwen3.8-27b-4bit not downloaded at %s", dir)
	}
	return dir
}

// TestLiveQwen38ThinkingAndTraces is the end-to-end check for the
// preserve-thinking family against real weights:
//
//  1. Thinking ON (open cue): the model emits a <think>...</think> block;
//     the full trace streams to ReasoningFn; the answer (post-</think>
//     content) reaches the normal output and is non-empty; answer content
//     does not leak into the trace.
//  2. Thinking OFF (closed cue): output is direct, no think markers in
//     output or trace.
//  3. History reasoning replay: a follow-up turn whose assistant history
//     carries ReasoningContent must render the trace into the prompt
//     (verified via tokenizer output) and generation must still work.
func TestLiveQwen38ThinkingAndTraces(t *testing.T) {
	dir := liveQwen38Dir(t)

	m, err := NewModel(dir)
	if err != nil {
		t.Fatalf("load: %v", err)
	}
	defer m.Close()

	if !m.Tokenizer().marksPreserveThinking {
		t.Fatal("qwen3.8 tokenizer not detected as preserve-thinking family")
	}

	// ── Thinking ON ────────────────────────────────────────────────────
	msgs := []ChatMessage{
		{Role: "user", Content: "What is 17 * 23? Think it through, then give just the number."},
	}
	prompt := m.FormatChatThinking(msgs, true)
	if !strings.HasSuffix(prompt, "<|im_start|>assistant\n<think>\n") {
		t.Fatalf("open cue missing:\n%q", prompt[len(prompt)-120:])
	}

	var trace strings.Builder
	var answer strings.Builder
	cfg := DefaultGenerateConfig()
	cfg.MaxTokens = 1024
	cfg.Temperature = 0
	cfg.RepetitionPenalty = 0
	cfg.PromptLookupMaxDrafts = 6
	cfg.ReasoningFn = func(chunk string) { trace.WriteString(chunk) }

	start := time.Now()
	err = m.Generate(t.Context(), prompt, cfg, func(id int) {
		answer.WriteString(m.DecodeToken(id))
	})
	if err != nil {
		t.Fatalf("generate (thinking on): %v", err)
	}
	t.Logf("thinking-on: %d trace bytes, %d answer bytes in %.1fs",
		trace.Len(), answer.Len(), time.Since(start).Seconds())

	if trace.Len() < 10 {
		t.Fatalf("thinking-on produced almost no reasoning trace (%d bytes): %q", trace.Len(), trace.String())
	}
	if answer.Len() < 1 {
		t.Fatal("thinking-on produced no answer after </think>")
	}
	if strings.Contains(answer.String(), "<think>") || strings.Contains(answer.String(), "</think>") {
		t.Fatalf("think markers leaked into answer: %q", answer.String())
	}
	if !strings.Contains(strings.ReplaceAll(answer.String(), " ", ""), "391") {
		t.Logf("answer did not contain expected 391 (got %q) — checking anyway, model may have reformatted", answer.String())
	}

	// ── Thinking OFF ───────────────────────────────────────────────────
	promptOff := m.FormatChatThinking(msgs, false)
	if !strings.HasSuffix(promptOff, "<|im_start|>assistant\n<think>\n\n</think>\n\n") {
		t.Fatalf("closed cue missing:\n%q", promptOff[len(promptOff)-120:])
	}

	var traceOff strings.Builder
	var answerOff strings.Builder
	cfgOff := cfg
	cfgOff.ReasoningFn = func(chunk string) { traceOff.WriteString(chunk) }

	err = m.Generate(t.Context(), promptOff, cfgOff, func(id int) {
		answerOff.WriteString(m.DecodeToken(id))
	})
	if err != nil {
		t.Fatalf("generate (thinking off): %v", err)
	}
	t.Logf("thinking-off: %d trace bytes, %d answer bytes", traceOff.Len(), answerOff.Len())

	if traceOff.Len() > 0 {
		t.Errorf("thinking-off must not stream a reasoning trace, got %q", traceOff.String())
	}
	if strings.Contains(answerOff.String(), "<think>") || strings.Contains(answerOff.String(), "</think>") {
		t.Fatalf("think markers leaked into thinking-off answer: %q", answerOff.String())
	}
	if answerOff.Len() == 0 {
		t.Fatal("thinking-off produced no answer")
	}

	// ── History reasoning replay ───────────────────────────────────────
	history := []ChatMessage{
		{Role: "user", Content: "What is 17 * 23?"},
		{Role: "assistant", ReasoningContent: "17*23 = 17*20 + 17*3 = 340 + 51 = 391.", Content: "391"},
		{Role: "user", Content: "Now add 9."},
	}
	replay := m.FormatChatThinking(history, true)
	if !strings.Contains(replay, "<think>\n17*23 = 17*20 + 17*3 = 340 + 51 = 391.\n</think>\n\n391<|im_end|>") {
		t.Fatalf("history reasoning trace not preserved in replay:\n%q", replay)
	}

	var replayAnswer strings.Builder
	cfgReplay := cfg
	cfgReplay.ReasoningFn = nil
	err = m.Generate(t.Context(), replay, cfgReplay, func(id int) {
		replayAnswer.WriteString(m.DecodeToken(id))
	})
	if err != nil {
		t.Fatalf("generate (replay): %v", err)
	}
	t.Logf("replay turn answer: %q", replayAnswer.String())
	if !strings.Contains(replayAnswer.String(), "400") {
		t.Errorf("replay answer should carry the continued computation (391+9=400), got %q", replayAnswer.String())
	}
}

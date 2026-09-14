package llm

import (
	"testing"
)

// TestFilterTokenZeroCollidesWithPad pins the id-0 collision: when a
// tokenizer has no <think>/</think> added tokens (gemma4), initThinking
// leaves thinkID/endThinkID at 0, and shouldFilterToken must not treat
// token id 0 (gemma's <pad>) as a think-boundary marker. Today it does:
// one emitted id-0 token would flip inThinkBlock on and silently filter
// every following token until another id-0 closes it.
func TestFilterTokenZeroCollidesWithPad(t *testing.T) {
	tok := &Tokenizer{
		specialTokens: map[string]int{
			"<pad>":    0,
			"<eos>":    1,
			"<bos>":    2,
			"<turn|>":  3,
			"<|think|>": 98,
		},
	}
	m := &Model{tokenizer: tok}
	m.initThinking()

	if m.thinkID != 0 || m.endThinkID != 0 {
		t.Fatalf("expected think/end ids 0 for a no-think-marker tokenizer, got %d/%d", m.thinkID, m.endThinkID)
	}

	// Id 0 must not toggle the think filter state.
	if m.shouldFilterToken(0, GenerateConfig{}) {
		t.Error("token id 0 (<pad>) should not be dropped as a think marker when the model has no think tokens")
	}
	if m.inThinkBlock {
		t.Error("token id 0 must not open a phantom think block")
	}

	// A model that DOES have think tokens (qwen3.5/minicpm5 shape): the
	// markers must still open/close the block.
	tok2 := &Tokenizer{
		specialTokens: map[string]int{
			"<|im_start|>": 1,
			"<|im_end|>":   2,
			"<think>":      248068,
			"</think>":     248069,
			"<pad>":        0,
		},
	}
	m2 := &Model{tokenizer: tok2}
	m2.initThinking()
	if !m2.shouldFilterToken(248068, GenerateConfig{}) {
		t.Error("<think> marker should be dropped and open the block")
	}
	if !m2.shouldFilterToken(7, GenerateConfig{}) {
		t.Error("content token inside think block should be filtered")
	}
	if m2.shouldFilterToken(7, GenerateConfig{ThinkingTokens: true}) {
		t.Error("content token inside think block should pass with ThinkingTokens=true")
	}
	if !m2.shouldFilterToken(248069, GenerateConfig{}) {
		t.Error("</think> marker should be dropped and close the block")
	}
	if m2.shouldFilterToken(7, GenerateConfig{}) {
		t.Error("content token after </think> should pass")
	}
}

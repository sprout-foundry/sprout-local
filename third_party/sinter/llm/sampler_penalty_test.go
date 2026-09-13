//go:build cgo && ((darwin && arm64) || (linux && ggml && (arm64 || amd64)))

package llm

import "testing"

// Regression guard for the repetition-penalty window. The window must be
// the TAIL of prompt-plus-generated: generated tokens are appended after
// the prompt so they survive the window crop. The original code built
// the window as append(generated, tokenIDs...) and then cropped the tail,
// which meant any prompt of 64+ tokens — every real chat prompt — evicted
// all generated tokens from the window. With the penalty effectively
// blind to what the model was producing, sampled chat fell into verbatim
// repeat loops (observed live on MiniCPM5-2B: a full answer block
// re-emitted, then "a leaf from an ordinary tree," forever).
func TestRepetitionPenaltyWindow(t *testing.T) {
	// 300-token prompt, 10 generated: every generated token must be in
	// the window, and the window must be the last N prompt+gen tokens.
	prompt := make([]int, 300)
	for i := range prompt {
		prompt[i] = i + 1
	}
	gen := []int{9001, 9002, 9003, 9004, 9005, 9006, 9007, 9008, 9009, 9010}

	w := repetitionPenaltyWindow(prompt, gen)
	if len(w) != repetitionPenaltyWindowTokens {
		t.Fatalf("window len = %d, want %d", len(w), repetitionPenaltyWindowTokens)
	}
	// The tail of prompt(300)+gen(10) = last N-10 prompt tokens + all 10 gen.
	first := len(prompt) + len(gen) - repetitionPenaltyWindowTokens
	for i, id := range gen {
		if w[len(w)-10+i] != id {
			t.Fatalf("window[%d] = %d, want generated token %d", len(w)-10+i, w[len(w)-10+i], id)
		}
	}
	if w[0] != prompt[first] { // first surviving prompt token
		t.Fatalf("window[0] = %d, want %d (prompt token)", w[0], prompt[first])
	}

	// Short prompt: window is the whole prompt+gen, order preserved.
	w = repetitionPenaltyWindow([]int{5, 6}, []int{7})
	if len(w) != 3 || w[0] != 5 || w[1] != 6 || w[2] != 7 {
		t.Fatalf("short window = %v, want [5 6 7]", w)
	}

	// Empty generated (before the first decode step's append): window is
	// the prompt tail.
	w = repetitionPenaltyWindow(prompt, nil)
	if len(w) != repetitionPenaltyWindowTokens || w[len(w)-1] != 300 {
		t.Fatalf("prompt-only window last = %d (len %d), want 300 (len %d)",
			w[len(w)-1], len(w), repetitionPenaltyWindowTokens)
	}

	// Boundary: exactly at the window size → no crop; one over → drops
	// exactly one token.
	exact := make([]int, 255)
	for i := range exact {
		exact[i] = i + 1
	}
	w = repetitionPenaltyWindow(exact, []int{42})
	if len(w) != repetitionPenaltyWindowTokens || w[0] != 1 || w[len(w)-1] != 42 {
		t.Fatalf("exact-boundary window: len %d, first %d, last %d; want %d, 1, 42",
			len(w), w[0], w[len(w)-1], repetitionPenaltyWindowTokens)
	}
	w = repetitionPenaltyWindow(append(exact, 43), []int{44})
	if len(w) != repetitionPenaltyWindowTokens || w[0] != 2 || w[len(w)-1] != 44 {
		t.Fatalf("one-over window: len %d, first %d, last %d; want %d, 2, 44",
			len(w), w[0], w[len(w)-1], repetitionPenaltyWindowTokens)
	}

	// Inputs must not be aliased: cropping must not share backing arrays
	// with the caller's slices (the decode loop keeps mutating them).
	w = repetitionPenaltyWindow(prompt, gen)
	w[0] = -1
	if prompt[first] == -1 {
		t.Fatal("window aliases the input slices")
	}
}

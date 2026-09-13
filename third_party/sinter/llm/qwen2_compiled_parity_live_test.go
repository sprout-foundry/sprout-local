//go:build darwin && arm64 && cgo

package llm_test

import (
	"context"
	"fmt"
	"os"
	"testing"
	"time"

	"github.com/sprout-foundry/sinter/llm"
	_ "github.com/sprout-foundry/sinter/llm/qwen2" // registers qwen2 + llama
)

// qwen2ParityModelDir returns the model dir for the qwen2/llama compiled-decode
// live tests: SINTER_QWEN2_PARITY_MODEL wins, else SINTER_MTP_PARITY_MODEL.
// Empty when neither is set → the test skips.
func qwen2ParityModelDir() string {
	if d := os.Getenv("SINTER_QWEN2_PARITY_MODEL"); d != "" {
		return d
	}
	return os.Getenv("SINTER_MTP_PARITY_MODEL")
}

// TestQwen2CompiledDecodeParityLiveModel verifies the qwen2/llama compiled
// decode path (CompiledGreedyArchitecture; default ON for greedy decoding
// below the context cutoff, SINTER_COMPILED_DECODE=0 opts out) against the
// eager per-token path. Same assertions as TestCompiledDecodeParityLiveModel:
// the compiled path runs, both paths are self-deterministic, and output is
// non-empty. Byte-identical eager-vs-compiled streams are NOT asserted for
// the same shape-specialized accumulation-order reason documented there.
func TestQwen2CompiledDecodeParityLiveModel(t *testing.T) {
	dir := qwen2ParityModelDir()
	if dir == "" {
		t.Skip("SINTER_QWEN2_PARITY_MODEL / SINTER_MTP_PARITY_MODEL not set")
	}

	prompts := []string{
		"Hello",
		"The capital of France is",
		"Write a short poem about the ocean.",
		"func main() { fmt.Println(\nReturn the next token of this Go snippet.",
	}

	for _, prompt := range prompts {
		prompt := prompt
		t.Run(prompt, func(t *testing.T) {
			model, err := llm.NewModel(dir)
			if err != nil {
				t.Fatalf("NewModel(%q): %v", dir, err)
			}
			defer model.Close()

			cfg := llm.DefaultGenerateConfig()
			cfg.MaxTokens = 24
			cfg.Temperature = 0
			cfg.RepetitionPenalty = 0
			// Disable prompt-lookup so the compiled branch is the one under
			// test (lookup takes priority in generateLocked).
			cfg.PromptLookupMaxDrafts = 0

			gen := func() []int {
				var toks []int
				if err := model.Generate(context.Background(), prompt, cfg, func(id int) {
					toks = append(toks, id)
				}); err != nil {
					t.Fatalf("Generate(%q): %v", prompt, err)
				}
				return toks
			}

			gen() // cold: full prefill, populates this instance's slot

			os.Setenv("SINTER_COMPILED_DECODE", "0")
			plain1 := gen() // warm eager (opted out)
			plain2 := gen() // eager determinism control
			os.Unsetenv("SINTER_COMPILED_DECODE")

			compiled1 := gen() // warm compiled (default on)
			compiled2 := gen() // compiled determinism control

			if len(compiled1) == 0 {
				t.Fatalf("compiled decode produced no tokens for %q", prompt)
			}
			if os.Getenv("SINTER_LOCAL_DEBUG") == "1" {
				t.Logf("compiled tokens: %v", compiled1)
			}

			equal := func(a, b []int) bool {
				if len(a) != len(b) {
					return false
				}
				for i := range a {
					if a[i] != b[i] {
						return false
					}
				}
				return true
			}

			if !equal(plain1, plain2) {
				t.Errorf("eager decode is not deterministic across calls for %q:\n  %v\n  %v", prompt, plain1, plain2)
			}
			if !equal(compiled1, compiled2) {
				t.Errorf("compiled decode is not deterministic across calls for %q:\n  %v\n  %v", prompt, compiled1, compiled2)
			}
		})
	}
}

// TestQwen2CompiledDecodeBenchLiveModel measures eager vs compiled decode
// throughput at a ~500-token prompt and 100 generated tokens — the
// short-context regime where compiled decode wins on qwen35 (~+14%) and
// where the MiniCPM5-2B tool-calling workload lives. Decode tok/s is measured
// from onToken timestamps (first token marks prefill's end); both paths are
// run warm (a short throwaway generation primes the prefix slot and, for the
// compiled path, the closure compile) so the numbers reflect steady-state
// decode, not one-time staging.
func TestQwen2CompiledDecodeBenchLiveModel(t *testing.T) {
	dir := qwen2ParityModelDir()
	if dir == "" {
		t.Skip("SINTER_QWEN2_PARITY_MODEL / SINTER_MTP_PARITY_MODEL not set")
	}

	// ~500-token prompt: enough to clear the short-context floor but stay well
	// below compiledCtxLimit (4096).
	const promptWords = 500
	words := []string{"func", "return", "error", "nil", "string", "int", "struct",
		"interface", "package", "import", "context", "time", "sync", "mutex",
		"append", "len", "make", "the", "and", "of", "to", "in", "a", "is"}
	var prompt string
	prompt += "Summarize the following text in one sentence.\n\n"
	for i := 0; i < promptWords; i++ {
		prompt += words[i%len(words)]
		prompt += " "
	}
	prompt += "\n\nReturn ONLY the summary."

	const genTokens = 100

	model, err := llm.NewModel(dir)
	if err != nil {
		t.Fatalf("NewModel(%q): %v", dir, err)
	}
	defer model.Close()

	setEnv := func(val string) {
		if val == "" {
			os.Unsetenv("SINTER_COMPILED_DECODE")
		} else {
			os.Setenv("SINTER_COMPILED_DECODE", val)
		}
	}

	// oneGenerate runs a single generation with the compiled path enabled
	// (SINTER_COMPILED_DECODE unset, i.e. default on) or opted out (set to
	// "0"), returning prefill/decode wall time.
	oneGenerate := func(maxTok int) (prefillMs, decodeMs, n int) {
		cfg := llm.DefaultGenerateConfig()
		cfg.MaxTokens = maxTok
		cfg.Temperature = 0
		cfg.RepetitionPenalty = 0
		cfg.PromptLookupMaxDrafts = 0
		var firstAt, lastAt time.Time
		start := time.Now()
		if err := model.Generate(context.Background(), prompt, cfg, func(id int) {
			now := time.Now()
			if firstAt.IsZero() {
				firstAt = now
			}
			lastAt = now
			n++
		}); err != nil {
			return 0, 0, n
		}
		if firstAt.IsZero() {
			return 0, 0, n
		}
		return int(firstAt.Sub(start).Milliseconds()), int(lastAt.Sub(firstAt).Milliseconds()), n
	}

	// warm primes the prefix slot and (for the compiled path) the one-time
	// closure compile so the timed run below is steady-state.
	warm := func() {
		cfg := llm.DefaultGenerateConfig()
		cfg.MaxTokens = 4
		cfg.Temperature = 0
		cfg.RepetitionPenalty = 0
		cfg.PromptLookupMaxDrafts = 0
		_ = model.Generate(context.Background(), prompt, cfg, nil)
	}

	// run warms the path then times a full run under the given env value.
	run := func(envVal string) (prefillMs, decodeMs, n int) {
		setEnv(envVal)
		warm()
		return oneGenerate(genTokens)
	}

	// Eager (compiled path opted out).
	eagerPrefill, eagerDecode, eagerN := run("0")
	// Compiled (default on).
	compPrefill, compDecode, compN := run("")

	eagerTps, compTps := 0.0, 0.0
	if eagerDecode > 0 {
		eagerTps = float64(eagerN-1) / (float64(eagerDecode) / 1000.0)
	}
	if compDecode > 0 {
		compTps = float64(compN-1) / (float64(compDecode) / 1000.0)
	}
	deltaPct := 0.0
	if eagerTps > 0 {
		deltaPct = (compTps - eagerTps) / eagerTps * 100
	}
	t.Logf("eager:   prefill=%dms decode=%dms (%d tok) %.1f tok/s", eagerPrefill, eagerDecode, eagerN, eagerTps)
	t.Logf("compiled: prefill=%dms decode=%dms (%d tok) %.1f tok/s", compPrefill, compDecode, compN, compTps)
	t.Logf("compiled vs eager decode: %+.1f%%", deltaPct)
	fmt.Printf("qwen2 compiled-vs-eager decode @~%d-token prompt, %d gen: eager=%.1f tok/s compiled=%.1f tok/s (%+.1f%%)\n",
		promptWords, genTokens, eagerTps, compTps, deltaPct)
}

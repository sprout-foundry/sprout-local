// Command sinter-chat is a minimal end-to-end example of the gomlx module:
// it loads a local model, streams tokens to the terminal, and exercises
// the full engine path (safetensors load -> dequant -> prefill -> decode
// -> detokenize) that real applications use.
//
// It doubles as the module's e2e smoke test:
//
//	go run ./examples/chat -model ~/.cache/sprout/models/qwen3-0.6b
//
// The llm package compiles on every platform via its stub; the stub's
// Generate returns an error, which this example surfaces. So the same
// source documents the API on any platform and runs for real on Apple
// Silicon (MLX/Metal) and GGML Linux.
package main

import (
	"context"
	"flag"
	"fmt"
	"os"
	"path/filepath"
	"strings"
	"time"

	"github.com/sprout-foundry/sinter/llm"
	_ "github.com/sprout-foundry/sinter/llm/all"
)

func main() {
	modelDir := flag.String("model", "", "path to a model directory (safetensors + config.json + tokenizer.json)")
	prompt := flag.String("prompt", "Say exactly: hello world", "user prompt")
	sys := flag.String("system", "You are a helpful assistant. Be concise.", "system prompt")
	maxTokens := flag.Int("max-tokens", 120, "generation cap")
	timeout := flag.Duration("timeout", 5*time.Minute, "overall generation timeout")
	thinking := flag.Bool("thinking", false, "prefix an empty <think></think> block (Qwen3-family chat template)")
	flag.Parse()

	if *modelDir == "" {
		fmt.Fprintln(os.Stderr, `usage: sinter-chat -model <dir> [-prompt text] [-system text] [-max-tokens n]

  -model      model directory containing config.json, tokenizer.json and
              *.safetensors (e.g. a mlx-community Qwen export)
  -prompt     user message rendered through the chat template
  -thinking   emit an empty <think></think> block first (Qwen3 thinking models)

exit codes: 0 ok, 1 generation/load error, 2 usage error`)
		os.Exit(2)
	}

	dir := expandHome(*modelDir)
	if _, err := os.Stat(filepath.Join(dir, "config.json")); err != nil {
		fmt.Fprintf(os.Stderr, "sinter-chat: %s does not look like a model dir (no config.json): %v\n", dir, err)
		os.Exit(2)
	}

	start := time.Now()
	model, err := llm.NewModel(dir)
	if err != nil {
		fmt.Fprintf(os.Stderr, "sinter-chat: load: %v\n", err)
		os.Exit(1)
	}
	defer model.Close()
	loadSecs := time.Since(start).Seconds()

	ctx, cancel := context.WithTimeout(context.Background(), *timeout)
	defer cancel()

	// The engine takes a raw prompt string; rendering a chat template is
	// the caller's job. This is the Qwen-family format - check your
	// model's tokenizer_config.json (chat_template) for other families.
	rendered := fmt.Sprintf(
		"<|im_start|>system\n%s<|im_end|>\n<|im_start|>user\n%s<|im_end|>\n<|im_start|>assistant\n",
		*sys, *prompt)
	if *thinking {
		rendered += "<think>\n\n</think>\n\n"
	}

	start = time.Now()
	var tokens int
	err = model.Generate(ctx, rendered, llm.GenerateConfig{
		MaxTokens:         *maxTokens,
		Temperature:       0.0, // greedy: deterministic, best for smoke tests
		TopP:              1.0,
		TopK:              1,
		RepetitionPenalty: 0,
	}, func(tokenID int) {
		tokens++
		fmt.Print(model.DecodeToken(tokenID))
	})
	fmt.Println()
	secs := time.Since(start).Seconds()
	if err != nil {
		fmt.Fprintf(os.Stderr, "sinter-chat: generate: %v\n", err)
		os.Exit(1)
	}
	fmt.Fprintf(os.Stderr, "[load %.1fs | %d tokens | %.1f tok/s]\n",
		loadSecs, tokens, float64(tokens)/secs)
}

func expandHome(p string) string {
	if p == "~" || strings.HasPrefix(p, "~/") {
		if home, err := os.UserHomeDir(); err == nil {
			return filepath.Join(home, strings.TrimPrefix(p, "~"))
		}
	}
	return p
}

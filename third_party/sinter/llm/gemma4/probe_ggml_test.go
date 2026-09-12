//go:build darwin && arm64 && cgo

package gemma4

import (
	"context"
	"os"
	"testing"

	"github.com/sprout-foundry/sinter/llm"
	"github.com/sprout-foundry/sinter/tensor"
)

// TestProbeGGMLLogits runs DebugDecodeComparison under the ggml backend to
// see whether the forward pass produces a real distribution.
func TestProbeGGMLLogits(t *testing.T) {
	b := tensor.DetectBackend()
	t.Logf("backend=%q", b.Name())

	modelDir := os.ExpandEnv("$HOME/dev/llm-models/gemma-4-e2b-it-5bit")
	if _, err := os.Stat(modelDir + "/model.safetensors"); err != nil {
		t.Skip("model not found")
	}
	m, err := llm.NewModel(modelDir)
	if err != nil {
		t.Fatalf("NewModel: %v", err)
	}
	defer m.Close()

	prompt := m.FormatChat([]llm.ChatMessage{{Role: "user", Content: "What is 2+2? Answer with just the number."}})
	var toks []int
	err = m.Generate(context.Background(), prompt, llm.GenerateConfig{
		MaxTokens: 16, Temperature: 0, ThinkingTokens: false,
	}, func(id int) { toks = append(toks, id) })
	if err != nil {
		t.Fatalf("Generate: %v", err)
	}
	text := ""
	for _, id := range toks {
		text += m.DecodeToken(id)
	}
	t.Logf("tokens=%v text=%q", toks, text)


}

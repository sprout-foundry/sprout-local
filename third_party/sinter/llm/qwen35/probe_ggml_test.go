//go:build darwin && arm64 && cgo && ggml

package qwen35

import (
	"context"
	"os"
	"testing"

	"github.com/sprout-foundry/sinter/llm"
	_ "github.com/sprout-foundry/sinter/tensor/ggml"
	"github.com/sprout-foundry/sinter/tensor"
)

func TestProbeQwen35GGML(t *testing.T) {
	b := tensor.DetectBackend()
	t.Logf("backend=%q", b.Name())
	if b.Name() != "MTL0" && b.Name() != "CPU" {
		t.Skip("not ggml")
	}

	modelDir := os.ExpandEnv("$HOME/dev/llm-models/qwen3.5-4b-4bit")
	if _, err := os.Stat(modelDir + "/model.safetensors"); err != nil {
		t.Skip("model not found")
	}
	m, err := llm.NewModel(modelDir)
	if err != nil {
		t.Fatalf("NewModel: %v", err)
	}
	defer m.Close()

	prompt := m.FormatChat([]llm.ChatMessage{{Role: "user", Content: "Count from 1 to 10, one number per line, then stop."}})
	var tokens []int
	err = m.Generate(context.Background(), prompt, llm.GenerateConfig{
		MaxTokens: 64, Temperature: 0, ThinkingTokens: false,
	}, func(id int) { tokens = append(tokens, id) })
	if err != nil {
		t.Fatalf("Generate: %v", err)
	}
	text := ""
	for _, id := range tokens {
		text += m.DecodeToken(id)
	}
	t.Logf("tokens=%v text=%q", tokens, text)
	if len(tokens) == 0 {
		t.Fatal("no tokens — logits broken under ggml")
	}
}

//go:build darwin && arm64 && cgo

package chatmodel

// Register all sinter model architectures on Apple Silicon (MLX backend).
import (
	_ "github.com/sprout-foundry/sinter/llm/gemma4"
	_ "github.com/sprout-foundry/sinter/llm/qwen2"
	_ "github.com/sprout-foundry/sinter/llm/qwen35"
	_ "github.com/sprout-foundry/sinter/mlx"
)

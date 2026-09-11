// Package all registers every model architecture shipped with sinter
// (qwen2, qwen3, qwen3.5 dense + MoE, gemma4, lfm2) plus the MLX (Metal)
// and GGML tensor backends.
//
// Import it blank when you just want everything to work:
//
//	import _ "github.com/sprout-foundry/sinter/llm/all"
//
// To keep binaries lean, blank-import only what you need instead:
//
//	import _ "github.com/sprout-foundry/sinter/llm/qwen2"
//	import _ "github.com/sprout-foundry/sinter/mlx" // registers the Metal backend
package all

import (
	_ "github.com/sprout-foundry/sinter/llm/gemma4"
	_ "github.com/sprout-foundry/sinter/llm/lfm2"
	_ "github.com/sprout-foundry/sinter/llm/qwen2"
	_ "github.com/sprout-foundry/sinter/llm/qwen3"
	_ "github.com/sprout-foundry/sinter/llm/qwen35"
	_ "github.com/sprout-foundry/sinter/mlx"
)

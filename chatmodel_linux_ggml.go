//go:build linux && (arm64 || amd64) && cgo && ggml

package main

// Register the model architectures sinter supports on Linux via the GGML
// backend (Termux builds with -tags ggml). gemma4 is excluded: every file
// in that package is darwin/arm64-only, so it cannot compile here.
import (
	_ "github.com/sprout-foundry/sinter/llm/qwen2"
	_ "github.com/sprout-foundry/sinter/llm/qwen35"
)

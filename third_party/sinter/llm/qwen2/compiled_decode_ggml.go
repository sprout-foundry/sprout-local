//go:build linux && (arm64 || amd64) && cgo && ggml

// Package qwen2 stub implementations of the compiled decode path for the
// non-MLX (GGML) backend. The MLX-compiled closure is unavailable, so
// PrepareCompiledDecode always fails and the Model layer falls back to the
// eager decode path (the interface assertion still succeeds, but the error
// return declines the compiled path).
package qwen2

import (
	"fmt"

	"github.com/sprout-foundry/sinter/llm"
	"github.com/sprout-foundry/sinter/tensor"
)

// compiledDecode is the non-MLX placeholder: PrepareCompiledDecode always
// fails, so nothing dereferences a populated one.
type compiledDecode struct{}

// errCompiledUnavailable is returned by every compiled-decode entry point on
// the GGML backend.
var errCompiledUnavailable = fmt.Errorf("qwen2: compiled decode unavailable on non-MLX backend")

// PrepareCompiledDecode always fails on non-MLX backends; the Model layer
// falls back to the eager decode path (interface assertion governs).
func (q *Qwen2) PrepareCompiledDecode(promptLen, maxTokens int, cache *llm.KVCache) error {
	return errCompiledUnavailable
}

// ForwardDecodeCompiled always fails on non-MLX backends.
func (q *Qwen2) ForwardDecodeCompiled(tokenArr tensor.Array, pos int) (tensor.Array, error) {
	return nil, errCompiledUnavailable
}

// ReleaseCompiledDecode is a no-op on non-MLX backends.
func (q *Qwen2) ReleaseCompiledDecode() {}

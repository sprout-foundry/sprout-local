//go:build cgo && ((darwin && arm64 && !mlx) || (linux && ggml && (arm64 || amd64)))

package qwen2

import "github.com/sprout-foundry/sinter/tensor"

// swigluClosure is a stub for non-MLX backends. It always returns nil,
// so the eager path in swiglu() is used instead.
func (q *Qwen2) swigluClosure(_ int) any { return nil }

// applySwigluClosure is unreachable on non-MLX backends (swigluClosure always returns nil).
func (q *Qwen2) applySwigluClosure(_ any, _ tensor.Array) (tensor.Array, error) {
	return nil, nil
}

// freeSwigluClosures is a no-op on non-MLX backends.
func freeSwigluClosures(_ *Qwen2) {}

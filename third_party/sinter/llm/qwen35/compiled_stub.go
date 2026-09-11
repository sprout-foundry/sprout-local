//go:build linux && (arm64 || amd64) && cgo && ggml

package qwen35

// compiledDecode is the MLX-only compiled decode closure state (defined in
// compiled_decode_mlx.go, which is darwin/arm64-only). forward.go keeps a
// `cd *compiledDecode` field but it is only exercised on the MLX backend —
// on the GGML backend it stays nil (PrepareCompiledDecode/ForwardDecodeCompiled
// always fail and the eager path is used), so an empty type is sufficient to
// let the linux+ggml build compile.
//
// Local patch for upstream sinter (github.com/sprout-foundry/sinter) v0.1.1:
// forward.go references this type but no non-darwin file defines it, so
// `go build -tags ggml` fails with "undefined: compiledDecode" on Linux.
// Re-delete this file once the upstream fix is merged and bump the replace.
type compiledDecode struct{}

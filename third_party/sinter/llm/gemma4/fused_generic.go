//go:build linux && ggml && (arm64 || amd64) && cgo

package gemma4

import (
	"github.com/sprout-foundry/sinter/tensor"
)

// On the GGML backend the fused MLX shims don't exist; these helpers are
// eager compositions of interface ops.

// geluFused degrades to the eager tanh approximation.
func geluFused(x tensor.Array, b tensor.Backend, s tensor.Stream) (tensor.Array, error) {
	return geluApprox(x, b, s)
}

// gegluFused degrades to eager gelu + multiply.
func gegluFused(gate, up tensor.Array, b tensor.Backend, s tensor.Stream) (tensor.Array, error) {
	act, err := geluApprox(gate, b, s)
	if err != nil {
		return nil, err
	}
	defer act.Free()
	return b.Multiply(act, up, s)
}

// lessEqual is the saturating-sigmoid comparison on GGML.
func lessEqual(a, b tensor.Array, backend tensor.Backend, s tensor.Stream) (tensor.Array, error) {
	return lessEqualSigmoid(a, b, backend, s)
}

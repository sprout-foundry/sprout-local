//go:build darwin && arm64 && cgo

package gemma4

import (
	"github.com/sprout-foundry/sinter/mlx"
	"github.com/sprout-foundry/sinter/tensor"
)

// geluFused uses the MLX C shim (1 CGO call instead of ~15) when the arrays
// are MLX-backed; otherwise falls back to the eager tanh approximation.
func geluFused(x tensor.Array, b tensor.Backend, s tensor.Stream) (tensor.Array, error) {
	if mx, ok := x.(*mlx.Array); ok {
		if out, err := mlx.Gemma4GELU(mx, s.(*mlx.Stream)); err == nil {
			return out, nil
		}
	}
	return geluApprox(x, b, s)
}

// gegluFused uses the MLX C shim for gelu(gate)*up in one call when the
// arrays are MLX-backed; otherwise falls back to eager ops.
func gegluFused(gate, up tensor.Array, b tensor.Backend, s tensor.Stream) (tensor.Array, error) {
	if mg, ok := gate.(*mlx.Array); ok {
		if mu, ok2 := up.(*mlx.Array); ok2 {
			if out, err := mlx.Gemma4GeGLU(mg, mu, s.(*mlx.Stream)); err == nil {
				return out, nil
			}
		}
	}
	act, err := geluApprox(gate, b, s)
	if err != nil {
		return nil, err
	}
	defer act.Free()
	return b.Multiply(act, up, s)
}

// lessEqual uses MLX's native comparison when MLX-backed.
func lessEqual(a, b tensor.Array, backend tensor.Backend, s tensor.Stream) (tensor.Array, error) {
	if ma, ok := a.(*mlx.Array); ok {
		if mb, ok2 := b.(*mlx.Array); ok2 {
			return mlx.LessEqual(ma, mb, s.(*mlx.Stream))
		}
	}
	// Integer positions: saturating-sigmoid comparison is exact.
	return lessEqualSigmoid(a, b, backend, s)
}

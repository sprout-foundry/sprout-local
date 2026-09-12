//go:build cgo && ((darwin && arm64) || (linux && ggml && (arm64 || amd64)))

package gemma4

import (
	"github.com/sprout-foundry/sinter/tensor"
)

// lessEqualSigmoid builds a<=b from backend-generic ops:
//
//	le(a,b) = 1 - sigmoid((a-b-0.5) * K)   with K a large constant
//
// Inputs are integer positions, so a-b-0.5 never lands on the sigmoid's
// 0.5-crossing: a<=b gives a negative argument (sigmoid→0, le→1) and a>b a
// positive one (sigmoid→1, le→0). The -0.5 shift matters: without it a==b
// would produce exactly 0.5 instead of a clean boolean.
func lessEqualSigmoid(a, b tensor.Array, backend tensor.Backend, s tensor.Stream) (tensor.Array, error) {
	diff, err := backend.Subtract(a, b, s)
	if err != nil {
		return nil, err
	}
	defer diff.Free()
	halfK, err := backend.NewArrayFromFloat32([]float32{50}, []int{1})
	if err != nil {
		return nil, err
	}
	defer halfK.Free()
	diff, err = backend.Subtract(diff, halfK, s)
	if err != nil {
		return nil, err
	}
	defer diff.Free()
	k, err := backend.NewArrayFromFloat32([]float32{100}, []int{1})
	if err != nil {
		return nil, err
	}
	defer k.Free()
	scaled, err := backend.Multiply(diff, k, s)
	if err != nil {
		return nil, err
	}
	defer scaled.Free()
	sig, err := backend.Sigmoid(scaled, s)
	if err != nil {
		return nil, err
	}
	defer sig.Free()
	one, err := backend.NewArrayFromFloat32([]float32{1}, []int{1})
	if err != nil {
		return nil, err
	}
	defer one.Free()
	return backend.Subtract(one, sig, s)
}

//go:build darwin && arm64 && cgo

package mlx

import (
	"math"
	"testing"
)

// TestProbeMLXRopeFreqs compares the freqs-table path against GGML's probe
// input to pin down MLX's pairing convention and inf handling.
func TestProbeMLXRopeFreqs(t *testing.T) {
	s, err := DefaultGPUStream()
	if err != nil {
		t.Fatal(err)
	}
	vals := []float32{1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16}
	x, err := NewArrayFromFloat32(vals, []int{1, 1, 2, 8})
	if err != nil {
		t.Fatal(err)
	}
	freqs := []float32{1, 2, float32(math.Inf(1)), float32(math.Inf(1))}
	f, err := NewArrayFromFloat32(freqs, []int{4})
	if err != nil {
		t.Fatal(err)
	}
	out, err := FastRoPE(x, 8, false, 0, 1.0, 0, f, s)
	if err != nil {
		t.Fatal(err)
	}
	d, err := out.Float32Data()
	if err != nil {
		t.Fatal(err)
	}
	t.Logf("mlx freqs-path out=%v", d)
}
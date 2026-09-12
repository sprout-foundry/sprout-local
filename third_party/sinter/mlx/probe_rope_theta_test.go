//go:build darwin && arm64 && cgo

package mlx

import (
	"math"
	"testing"
)

// TestProbeMLXRopeThetaExtract drives FastRoPE with unit vectors so the
// output directly yields (cos, sin) of the angle MLX applies to each pair
// at each position: set x[p][i]=1 (pair i's first dim), then out[p][i]=cos,
// out[p][i+D/2]=sin.
func TestProbeMLXRopeThetaExtract(t *testing.T) {
	s, err := DefaultGPUStream()
	if err != nil {
		t.Fatal(err)
	}
	const D, S = 8, 4
	for _, freqs := range [][]float32{
		{1, 2, float32(math.Inf(1)), float32(math.Inf(1))},
		{1, 3, 5, 7},
	} {
		f, err := NewArrayFromFloat32(freqs, []int{len(freqs)})
		if err != nil {
			t.Fatal(err)
		}
		// One unit vector per position, cycling pairs.
		vals := make([]float32, S*D)
		pairAt := make([]int, S)
		for p := 0; p < S; p++ {
			pairAt[p] = (p*2) % (D / 2)
			vals[p*D+pairAt[p]] = 1
		}
		x, err := NewArrayFromFloat32(vals, []int{1, 1, S, D})
		if err != nil {
			t.Fatal(err)
		}
		out, err := FastRoPE(x, D, false, 0, 1.0, 0, f, s)
		if err != nil {
			t.Fatal(err)
		}
		d, err := out.Float32Data()
		if err != nil {
			t.Fatal(err)
		}
		t.Logf("freqs=%v", freqs)
		for p := 0; p < S; p++ {
			i := pairAt[p]
			c := d[p*D+i]
			sn := d[p*D+D/2+i]
			theta := math.Atan2(float64(sn), float64(c))
			t.Logf("pos=%d pair=%d cos=%.6f sin=%.6f theta=%.6f", p+0, i, c, sn, theta)
		}
	}
}
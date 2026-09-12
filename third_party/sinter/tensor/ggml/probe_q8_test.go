//go:build (darwin || linux) && (arm64 || amd64) && cgo && ggml

package ggml

import (
	"encoding/binary"
	"math"
	"os"
	"testing"
)

func TestProbeQ8RoundTrip(t *testing.T) {
	raw, err := os.ReadFile("/tmp/sprout/lp_ref/qproj_w.f32")
	if err != nil {
		t.Skip("no ref weights")
	}
	n := len(raw) / 4
	data := make([]float32, n)
	for i := range data {
		data[i] = math.Float32frombits(binary.LittleEndian.Uint32(raw[i*4:]))
	}
	const inDim = 1536
	outDim := n / inDim

	b := &GGMLBackend{}
	s, _ := b.DefaultStream()
	arr, err := b.NewArrayQ8_0(data, []int{outDim, inDim})
	if err != nil {
		t.Fatal(err)
	}
	e0 := make([]float32, inDim)
	e0[0] = 1
	x, _ := b.NewArrayFromFloat32(e0, []int{1, 1, inDim})
	out, err := b.MatMul(x, arr, s)
	if err != nil {
		t.Fatal(err)
	}
	d, _ := out.Float32Data()
	// Q8_0: max relative error ~0.4%
	err1 := math.Abs(float64(d[0]-data[0])) / math.Max(1e-9, math.Abs(float64(data[0])))
	t.Logf("roundtrip w[0][0]=%v want %v relerr=%v", d[0], data[0], err1)
	if err1 > 0.01 {
		t.Errorf("Q8_0 error too large")
	}
}

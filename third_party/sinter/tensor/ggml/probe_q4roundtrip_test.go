//go:build (darwin || linux) && (arm64 || amd64) && cgo && ggml

package ggml

import (
	"encoding/binary"
	"math"
	"os"
	"testing"

	"github.com/sprout-foundry/sinter/tensor"
)

// TestProbeQ4RoundTrip pushes the mlx-dequantized q_proj weights through
// NewArrayQ4_0 and reads back per-row samples to find where requantization
// breaks.
func TestProbeQ4RoundTrip(t *testing.T) {
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
	t.Logf("weights [out=%d, in=%d]", outDim, inDim)

	b := &GGMLBackend{}
	s, _ := b.DefaultStream()
	arr, err := b.NewArrayQ4_0(data, []int{outDim, inDim})
	if err != nil {
		t.Fatal(err)
	}

	// Read back via x=identity? Simpler: gather row0 by matmul with unit vector
	// e0: [1,1,inDim]
	e0 := make([]float32, inDim)
	e0[0] = 1
	x, _ := b.NewArrayFromFloat32(e0, []int{1, 1, inDim})
	out, err := b.MatMul(x, arr, s)
	if err != nil {
		t.Fatal(err)
	}
	d, err := out.Float32Data()
	if err != nil {
		t.Fatal(err)
	}
	// d[0] should equal data[0] (w[0,0])
	t.Logf("roundtrip w[0][0]=%v want %v", d[0], data[0])
	_ = tensor.Float32
}

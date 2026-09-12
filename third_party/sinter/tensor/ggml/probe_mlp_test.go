//go:build (darwin || linux) && (arm64 || amd64) && cgo && ggml

package ggml

import (
	"math"
	"testing"

	"github.com/sprout-foundry/sinter/tensor"
)

func TestProbeGELUChain(t *testing.T) {
	b := &GGMLBackend{}
	s, _ := b.DefaultStream()

	// replicate geluApprox on [-1.48, 0.296, 1.76, -4.6, 0.72]
	vals := []float32{-1.48, 0.296, 1.76, -4.6, 0.72}
	x, err := b.NewArrayFromFloat32(vals, []int{len(vals)})
	if err != nil {
		t.Fatal(err)
	}
	defer x.Free()

	// tanh path
	th, err := b.Tanh(x, s)
	if err != nil {
		t.Fatal(err)
	}
	defer th.Free()
	td, _ := th.Float32Data()
	t.Logf("tanh=%v", td)

	// power 3
	p3, err := b.Power(x, 3, s)
	if err != nil {
		t.Fatal(err)
	}
	p3d, _ := p3.Float32Data()
	p3.Free()
	t.Logf("x^3=%v", p3d)
	for i, v := range vals {
		want := v * v * v
		if math.Abs(float64(p3d[i]-want)) > 0.01 {
			t.Errorf("x^3[%d]=%v want %v", i, p3d[i], want)
		}
	}

	// exp
	e, err := b.Exp(x, s)
	if err != nil {
		t.Fatal(err)
	}
	ed, _ := e.Float32Data()
	e.Free()
	t.Logf("exp=%v", ed)
	for i, v := range vals {
		want := float32(math.Exp(float64(v)))
		if math.Abs(float64(ed[i]-want)) > 0.01 {
			t.Errorf("exp[%d]=%v want %v", i, ed[i], want)
		}
	}

	// sigmoid
	sg, err := b.Sigmoid(x, s)
	if err != nil {
		t.Fatal(err)
	}
	sgd, _ := sg.Float32Data()
	sg.Free()
	t.Logf("sigmoid=%v", sgd)
	_ = tensor.Float32
}

func TestProbeFastRoPE(t *testing.T) {
	b := &GGMLBackend{}
	s, _ := b.DefaultStream()

	// x: [B=1, H=1, S=2, D=8] row-major (MLX attention convention)
	vals := []float32{1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16}
	x, err := b.NewArrayFromFloat32(vals, []int{1, 1, 2, 8})
	if err != nil {
		t.Fatal(err)
	}
	defer x.Free()
	x4, err := b.Reshape(x, []int{1, 2, 1, 8}, s)
	if err != nil {
		t.Fatal(err)
	}
	_ = x4

	// freqs for dims=8 (numFreqs=4), rotated=4 (rotatedFreqs=2): [1, base^0.5, inf, inf]
	freqs := []float32{1, 2, float32(math.Inf(1)), float32(math.Inf(1))}
	f, err := b.NewArrayFromFloat32(freqs, []int{4})
	if err != nil {
		t.Fatal(err)
	}
	defer f.Free()

	out, err := b.FastRoPE(x, 8, false, 0, 1.0, 0, f, s)
	if err != nil {
		t.Fatal(err)
	}
	defer out.Free()
	d, err := out.Float32Data()
	if err != nil {
		t.Fatal(err)
	}
	t.Logf("rope out=%v", d)
	// MLX fast_rope contract (pinned by mlx/probe_rope_theta_test.go):
	// theta_i = pos/freqs[i] (table holds base^(2i/d)), +inf → identity,
	// half-split pairs (i, i+D/2) rotated in place.
	want := make([]float32, 16)
	for p := 0; p < 2; p++ {
		for i := 0; i < 4; i++ {
			fi := float64(freqs[i])
			th := 0.0
			if !math.IsInf(fi, 0) && fi != 0 {
				th = float64(p) / fi
			}
			e := float64(vals[p*8+i])
			o := float64(vals[p*8+4+i])
			want[p*8+i] = float32(e*math.Cos(th) - o*math.Sin(th))
			want[p*8+4+i] = float32(o*math.Cos(th) + e*math.Sin(th))
		}
	}
	t.Logf("want     =%v", want)
	for i := range want {
		if math.Abs(float64(d[i]-want[i])) > 1e-3 {
			t.Errorf("out[%d]=%v want %v", i, d[i], want[i])
		}
	}
}

func TestProbeFastRoPEBase(t *testing.T) {
	b := &GGMLBackend{}
	s, _ := b.DefaultStream()

	// x: [B=1, H=1, S=2, D=8], base=10000, no freqs, offset=0
	vals := []float32{1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16}
	x, err := b.NewArrayFromFloat32(vals, []int{1, 1, 2, 8})
	if err != nil {
		t.Fatal(err)
	}
	defer x.Free()

	out, err := b.FastRoPE(x, 8, false, 10000.0, 1.0, 0, nil, s)
	if err != nil {
		t.Fatal(err)
	}
	defer out.Free()
	d, err := out.Float32Data()
	if err != nil {
		t.Fatal(err)
	}
	t.Logf("base-path out=%v", d)

	// Reference: MLX fast_rope base contract — theta_i = pos·base^(-2i/d),
	// half-split pairs (i, i+D/2) rotated in place. Output-identical to
	// MLX (verified by mlx/probe_rope_test.go).
	want := make([]float32, 16)
	for p := 0; p < 2; p++ {
		for i := 0; i < 4; i++ {
			th := float64(p) * math.Pow(10000, -float64(2*i)/8)
			e := float64(vals[p*8+i])
			o := float64(vals[p*8+4+i])
			want[p*8+i] = float32(e*math.Cos(th) - o*math.Sin(th))
			want[p*8+4+i] = float32(o*math.Cos(th) + e*math.Sin(th))
		}
	}
	t.Logf("want          =%v", want)
	bad := 0
	for i := range want {
		if math.Abs(float64(d[i]-want[i])) > 1e-3 {
			bad++
		}
	}
	if bad > 0 {
		t.Errorf("%d/%d mismatch", bad, len(want))
	}
}

func TestProbeQ4MatMul(t *testing.T) {
	b := &GGMLBackend{}
	s, _ := b.DefaultStream()

	// W: [out=4, in=32], x: [1, 2, 32] (ne0 must be %32)
	wData := make([]float32, 4*32)
	for i := 0; i < 32; i++ {
		wData[0*32+i] = 1 // row0 = ones
		wData[1*32+i] = 0
		wData[2*32+i] = float32(i) // row2 = i
		wData[3*32+i] = 0
	}
	wData[1*32+5] = 2 // row2 single spike
	w, err := b.NewArrayQ4_0(wData, []int{4, 32})
	if err != nil {
		t.Fatal(err)
	}
	xv := make([]float32, 2*32)
	for i := range xv {
		xv[i] = 1
	}
	x, err := b.NewArrayFromFloat32(xv, []int{1, 2, 32})
	if err != nil {
		t.Fatal(err)
	}
	out, err := b.MatMul(x, w, s)
	if err != nil {
		t.Fatal(err)
	}
	d, err := out.Float32Data()
	if err != nil {
		t.Fatal(err)
	}
	t.Logf("qmatmul out shape=%v d=%v", out.Shape(), d)
	// want per token: [sum=32, spike=2, sum(i)=496, 0]
	want := []float32{32, 2, 496, 0, 32, 2, 496, 0}
	for i, v := range want {
		if math.Abs(float64(d[i]-v)) > 0.05 {
			t.Errorf("d[%d]=%v want %v", i, d[i], v)
		}
	}
}

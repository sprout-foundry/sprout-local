//go:build darwin && arm64 && cgo

package gemma4

import (
	"encoding/binary"
	"math"
	"os"
	"testing"

	"github.com/sprout-foundry/sinter/llm"
	"github.com/sprout-foundry/sinter/tensor"
)

// TestProbeLinearParity loads the real q_proj weights under the active
// backend and compares a Forward pass against a pure-Go reference matmul.
func TestProbeLinearParity(t *testing.T) {
	b := tensor.DetectBackend()
	t.Logf("backend=%q", b.Name())

	modelDir := os.ExpandEnv("$HOME/dev/llm-models/gemma-4-e2b-it-5bit")
	if _, err := os.Stat(modelDir + "/model.safetensors"); err != nil {
		t.Skip("model not found")
	}

	sf, err := llm.OpenSafetensors(modelDir + "/model.safetensors")
	if err != nil {
		t.Fatalf("open: %v", err)
	}
	defer sf.Close()

	s, _ := b.DefaultStream()
	quant := &llm.QuantConfig{GroupSize: 64, Bits: 5, Mode: "affine"}
	lin, err := llm.LoadLinear(sf, "language_model.model.layers.0.self_attn.q_proj", b, s, quant)
	if err != nil {
		t.Fatalf("load linear: %v", err)
	}
	defer lin.Free()

	// input [1, 2, 1536]
	in := make([]float32, 2*1536)
	for i := range in {
		in[i] = float32(i%17) * 0.01
	}
	x, err := b.NewArrayFromFloat32(in, []int{1, 2, 1536})
	if err != nil {
		t.Fatal(err)
	}
	out, err := lin.Forward(x, b, s)
	if err != nil {
		t.Fatalf("forward: %v", err)
	}
	d, err := out.Float32Data()
	if err != nil {
		t.Fatal(err)
	}

	// Reference: dequantize the raw triplet in Go and matmul.
	w, err := sf.Get("language_model.model.layers.0.self_attn.q_proj.weight", b, s)
	if err != nil {
		t.Fatalf("get w: %v", err)
	}
	scales, err := sf.Get("language_model.model.layers.0.self_attn.q_proj.scales", b, s)
	if err != nil {
		t.Fatalf("get scales: %v", err)
	}
	bias, err := sf.Get("language_model.model.layers.0.self_attn.q_proj.biases", b, s)
	if err != nil {
		t.Logf("no biases: %v", err)
		bias = nil
	}
	f32w, wshape, err := llm.DequantizeForTest(w, scales, bias, 5, 64)
	if err != nil {
		t.Fatalf("dequant: %v", err)
	}
	_ = wshape
	// y = x @ W^T ; q_proj: [2048, 1536]
	outDim := len(f32w) / 1536
	ref := make([]float32, 2*outDim)
	for t_ := 0; t_ < 2; t_++ {
		for o := 0; o < outDim; o++ {
			var sum float32
			for i := 0; i < 1536; i++ {
				sum += in[t_*1536+i] * f32w[o*1536+i]
			}
			ref[t_*outDim+o] = sum
		}
	}
	if dir := os.Getenv("GEMMA4_LAYERS_DUMP"); dir != "" {
		buf := make([]byte, 0, len(d)*4)
		for _, v := range d {
			buf = binary.LittleEndian.AppendUint32(buf, math.Float32bits(v))
		}
		_ = os.WriteFile(dir+"/qproj_out.f32", buf, 0o644)
	}
	// compare against mlx dequantized weights
	refW, err := os.ReadFile(os.Getenv("GEMMA4_REF_W"))
	if err == nil {
		wv := make([]float32, len(refW)/4)
		for i := range wv {
			wv[i] = math.Float32frombits(binary.LittleEndian.Uint32(refW[i*4:]))
		}
		// Diag 1: does the Go affine dequant match mlx's dequantized W?
		if len(wv) == len(f32w) {
			wmd := float32(0)
			wbad := 0
			for i := range wv {
				dd := f32w[i] - wv[i]
				if dd < 0 {
					dd = -dd
				}
				if dd > wmd {
					wmd = dd
				}
				if dd > 0.01 {
					wbad++
				}
			}
			t.Logf("diag1 go-dequant-W vs mlx-W: maxdiff=%v bad(>0.01)=%d/%d", wmd, wbad, len(wv))
		} else {
			t.Logf("diag1 skipped: len mismatch go=%d mlx=%d", len(f32w), len(wv))
		}
		// Diag 2: ggml out vs pure-Go matmul with the SAME Go-dequantized W.
		md2 := float32(0)
		for i := range d {
			if i >= len(ref) {
				break
			}
			dd := d[i] - ref[i]
			if dd < 0 {
				dd = -dd
			}
			if dd > md2 {
				md2 = dd
			}
		}
		t.Logf("diag2 ggml-out vs go-W matmul: maxdiff=%v", md2)
		outDim := len(wv) / 1536
		ref2 := make([]float32, 2*outDim)
		for t_ := 0; t_ < 2; t_++ {
			for o := 0; o < outDim; o++ {
				var sum float32
				for i := 0; i < 1536; i++ {
					sum += in[t_*1536+i] * wv[o*1536+i]
				}
				ref2[t_*outDim+o] = sum
			}
		}
		md := float32(0)
		for i := range ref2 {
			dd := d[i] - ref2[i]
			if dd < 0 {
				dd = -dd
			}
			if dd > md {
				md = dd
			}
		}
		t.Logf("vs-mlx-dequant: maxdiff=%v", md)
	}
	_ = ref
	_ = f32w
	_ = wshape
}

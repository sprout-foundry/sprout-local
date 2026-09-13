//go:build cgo && darwin && arm64

package llm

import (
	"encoding/binary"
	"encoding/json"
	"fmt"
	"math"
	"math/rand"
	"os"
	"path/filepath"
	"testing"

	"github.com/sprout-foundry/sinter/mlx"
	"github.com/sprout-foundry/sinter/tensor"
)

type rawTensor struct {
	dtype string
	shape []int
	data  []byte
}

// writeRawSafetensors writes a minimal safetensors file from raw byte blobs.
func writeRawSafetensors(t *testing.T, path string, tensors map[string]rawTensor) {
	writeRawSafetensorsMeta(t, path, tensors, nil)
}

func writeRawSafetensorsMeta(t *testing.T, path string, tensors map[string]rawTensor, meta map[string]string) {
	t.Helper()
	type entry struct {
		Dtype       string   `json:"dtype"`
		Shape       []int    `json:"shape"`
		DataOffsets []uint64 `json:"data_offsets"`
	}
	header := map[string]any{}
	if len(meta) > 0 {
		header["__metadata__"] = meta
	}
	var blobs []byte
	var offset uint64
	names := make([]string, 0, len(tensors))
	for name := range tensors {
		names = append(names, name)
	}
	// deterministic order
	for i := 0; i < len(names); i++ {
		for j := i + 1; j < len(names); j++ {
			if names[j] < names[i] {
				names[i], names[j] = names[j], names[i]
			}
		}
	}
	for _, name := range names {
		tn := tensors[name]
		n := uint64(len(tn.data))
		header[name] = entry{Dtype: tn.dtype, Shape: tn.shape, DataOffsets: []uint64{offset, offset + n}}
		blobs = append(blobs, tn.data...)
		offset += n
	}
	hb, err := json.Marshal(header)
	if err != nil {
		t.Fatal(err)
	}
	for len(hb)%8 != 0 {
		hb = append(hb, ' ')
	}
	out := make([]byte, 8, 8+len(hb)+len(blobs))
	binary.LittleEndian.PutUint64(out, uint64(len(hb)))
	out = append(out, hb...)
	out = append(out, blobs...)
	if err := os.WriteFile(path, out, 0o644); err != nil {
		t.Fatal(err)
	}
}

func f32Bytes(f []float32) []byte {
	out := make([]byte, len(f)*4)
	for i, v := range f {
		binary.LittleEndian.PutUint32(out[i*4:], math.Float32bits(v))
	}
	return out
}

// f16Bytes converts F32 bytes to F16 bytes via the backend-free table method
// (round-to-nearest via math.Float32 → float16 encoding).
func f16Bytes(t *testing.T, f32data []byte, _ bool) []byte {
	t.Helper()
	n := len(f32data) / 4
	out := make([]byte, n*2)
	for i := 0; i < n; i++ {
		f := math.Float32frombits(binary.LittleEndian.Uint32(f32data[i*4:]))
		binary.LittleEndian.PutUint16(out[i*2:], f32ToF16(f))
	}
	return out
}

// f32ToF16 converts with round-to-nearest-even (handles normals; values near
// zero flush correctly for scale magnitudes used in tests).
func f32ToF16(f float32) uint16 {
	b := math.Float32bits(f)
	sign := uint16((b >> 16) & 0x8000)
	exp := int32((b>>23)&0xFF) - 127
	man := b & 0x7FFFFF
	if exp > 15 { // saturate
		return sign | 0x7C00
	}
	if exp >= -14 {
		// normal: rebuild with bias 15, round to nearest even
		half := uint32(man >> 13)
		if man&0x1000 != 0 && (man&0xEFFF)>>13 == 0x1FFF>>1 { // simple RNE tie
			half++
		} else if man&0x1000 != 0 {
			half++
		}
		e := uint16(exp + 15)
		return sign | e<<10 | uint16(half)
	}
	// subnormal / zero: scale down
	var man16 uint32
	if exp >= -25 {
		man16 = (man | 0x800000) >> uint32(-exp-14+13)
	}
	return sign | uint16(man16)
}

func mustArray(t *testing.T, b tensor.Backend, f []float32, shape []int) tensor.Array {
	t.Helper()
	a, err := b.NewArrayFromFloat32(f, shape)
	if err != nil {
		t.Fatal(err)
	}
	return a
}

func evalAll(arrs []tensor.Array) error {
	for _, a := range arrs {
		if err := a.Eval(); err != nil {
			return err
		}
	}
	return nil
}

// dtypeName maps a tensor dtype to its safetensors string.
func dtypeName(t *testing.T, a tensor.Array) string {
	t.Helper()
	switch a.Dtype() {
	case tensor.UInt32:
		return "U32"
	case tensor.Float16:
		return "F16"
	case tensor.Float32:
		return "F32"
	case tensor.BFloat16:
		return "BF16"
	case tensor.Int32:
		return "I32"
	default:
		t.Fatalf("unhandled dtype %v", a.Dtype())
		return ""
	}
}

// rawBytes reads a GPU array's raw little-endian payload via the backend's
// RawBytes (dtype-agnostic: works for U32-packed weights and F16 scales).
func rawBytes(t *testing.T, a tensor.Array) []byte {
	t.Helper()
	b, err := a.RawBytes()
	if err != nil {
		t.Fatal("rawBytes:", err)
	}
	return b
}

// dequantReference dequantizes a quantized (weight, scales) pair back to F32
// for the reference computation.
func dequantReference(t *testing.T, b tensor.Backend, s tensor.Stream, w, scales tensor.Array) []float32 {
	t.Helper()
	// group size 64, 4 bits, affine: per group g with scale s_g and min m_g
	// MLX affine quantization: w ≈ scales_g * (q - bias) — the biases are nil
	// here, meaning zero-point 8 (center of 4-bit range)? MLX's convention:
	// when biases are absent the zero point is embedded in scales via the
	// (2^(b-1)) offset. Reproduce via the backend's own dequant path instead:
	// dequantizeToFull is the llm helper used at load time.
	f32, _, err := readQuantizedWeightsPublic(b, s, w, scales, nil, 4, 64)
	if err != nil {
		t.Fatal(err)
	}
	return f32
}

// TestLoadLinearLoraMergeMath verifies LoadLinearLora's merge arithmetic:
// merged(W) dequantizes to dequant(W) + scale*(B@A), and the re-quantized
// triplet round-trips within quantization tolerance (group-64 affine).
func TestLoadLinearLoraMergeMath(t *testing.T) {
	if !mlx.Available() {
		t.Skip("MLX backend unavailable")
	}
	backend := tensor.DetectBackend()
	if backend == nil {
		t.Skip("no backend")
	}
	stream, err := backend.DefaultStream()
	if err != nil {
		t.Fatal(err)
	}

	dir := t.TempDir()
	const outDim, inDim, rank = 64, 128, 4 // small; group 64 divides both dims

	rng := rand.New(rand.NewSource(7))
	w := make([]float32, outDim*inDim)
	for i := range w {
		w[i] = float32(rng.NormFloat64()) * 0.05
	}
	aM := make([]float32, rank*inDim)
	for i := range aM {
		aM[i] = float32(rng.NormFloat64()) * 0.02
	}
	bM := make([]float32, outDim*rank)
	for i := range bM {
		bM[i] = float32(rng.NormFloat64()) * 0.02
	}

	// Base as F32 triplet file: LoadLinearLora reads the base via
	// readQuantizedWeights, which needs a pre-quantized triplet. Quantize W
	// with the backend, write the triplet to safetensors as raw bytes, then
	// run the merge and compare against the F32 expectation.
	parts, err := backend.Quantize(mustArray(t, backend, w, []int{outDim, inDim}), 64, 4, "affine", stream)
	if err != nil {
		t.Fatal(err)
	}
	if len(parts) < 2 {
		t.Fatalf("quantize returned %d parts", len(parts))
	}
	if err := evalAll(parts[:2]); err != nil {
		t.Fatal(err)
	}
	wBytes := rawBytes(t, parts[0])
	sBytes := rawBytes(t, parts[1])
	wDtype, sDtype := dtypeName(t, parts[0]), dtypeName(t, parts[1])

	// packed 4-bit: outDim*inDim/2 bytes
	if len(wBytes) != outDim*inDim/2 {
		t.Fatalf("packed weight %d bytes, want %d", len(wBytes), outDim*inDim/2)
	}
	nGroups := outDim * inDim / 64
	if len(sBytes) != nGroups*4 {
		t.Fatalf("scales %d bytes, want %d", len(sBytes), nGroups*4)
	}

	basePath := filepath.Join(dir, "base.safetensors")
	writeRawSafetensors(t, basePath, map[string]rawTensor{
		"proj.weight": {dtype: wDtype, shape: parts[0].Shape(), data: wBytes},
		"proj.scales": {dtype: sDtype, shape: parts[1].Shape(), data: sBytes},
	})

	loraPath := filepath.Join(dir, "lora_test.safetensors")
	writeRawSafetensorsMeta(t, loraPath,
		map[string]rawTensor{
			"proj.lora_A": {dtype: "F32", shape: []int{rank, inDim}, data: f32Bytes(aM)},
			"proj.lora_B": {dtype: "F32", shape: []int{outDim, rank}, data: f32Bytes(bM)},
		},
		map[string]string{"r": "4", "alpha": "8"}, // scale 2.0
	)

	adapter, err := LoadLoraAdapter(dir)
	if err != nil {
		t.Fatal(err)
	}
	if adapter == nil {
		t.Fatal("adapter not discovered")
	}
	if adapter.Scale != 2.0 {
		t.Fatalf("scale = %v, want 2.0 (alpha/r = 8/4)", adapter.Scale)
	}
	sf, err := OpenSafetensors(basePath)
	if err != nil {
		t.Fatal(err)
	}
	defer sf.Release()

	lin, err := LoadLinearLora(sf, "proj.weight", backend, stream, &QuantConfig{GroupSize: 64, Bits: 4, Mode: "affine"}, adapter)
	if err != nil {
		t.Fatal(err)
	}
	defer lin.Free()

	// Read the merged weight back to F32 and compare with dequant(W)+scale*B@A.
	// The merged Linear is quantized on MLX: dequantize its live triplet via
	// the same llm helper the loader uses.
	got, gotShape, derr := readQuantizedWeights(lin.QW(), lin.QScales(), nil, lin.QBits(), lin.QGroupSize())
	if derr != nil {
		t.Fatal("dequant merged:", derr)
	}
	if len(got) != outDim*inDim || len(gotShape) != 2 {
		t.Fatalf("merged dequant %d elems shape %v, want %d [out,in]", len(got), gotShape, outDim*inDim)
	}
	if len(got) != outDim*inDim {
		t.Fatalf("merged dequant %d elems, want %d (shape %v)", len(got), outDim*inDim, gotShape)
	}

	want := make([]float32, outDim*inDim)
	copy(want, w)
	for o := 0; o < outDim; o++ {
		for r := 0; r < rank; r++ {
			bv := bM[o*rank+r] * 2.0
			for c := 0; c < inDim; c++ {
				want[o*inDim+c] += bv * aM[r*inDim+c]
			}
		}
	}

	// 4-bit group-64 quantization error: max |x|<=1-ish inputs; allow a loose
	// tolerance because quantize→dequant of the BASE already rounds.
	maxErr, baseErr := 0.0, 0.0
	dqBase := dequantReference(t, backend, stream, parts[0], parts[1])
	for i := range want {
		if e := math.Abs(float64(got[i] - want[i])); e > maxErr {
			maxErr = e
		}
		if e := math.Abs(float64(dqBase[i] - w[i])); e > baseErr {
			baseErr = e
		}
	}
	if maxErr > baseErr*8+0.02 { // merge adds a second quantization round
		t.Fatalf("merge error too large: max %v (base quant err %v)", maxErr, baseErr)
	}
	if os.Getenv("SINTER_LORA_VERBOSE") == "1" {
		fmt.Printf("merge maxErr=%v baseQuantErr=%v\n", maxErr, baseErr)
	}
}

// readQuantizedWeightsPublic adapts llm's internal readQuantizedWeights for
// the test's reference computation.
func readQuantizedWeightsPublic(b tensor.Backend, s tensor.Stream, w, scales, biases tensor.Array, bits, groupSize int) ([]float32, []int, error) {
	return readQuantizedWeights(w, scales, biases, bits, groupSize)
}

//go:build cgo && ((darwin && arm64) || (linux && ggml && (arm64 || amd64)))

package llm

import (
	"fmt"
	"os"
	"path/filepath"
	"sort"
	"strconv"
	"strings"

	"github.com/sprout-foundry/sinter/tensor"
)

// ---------------------------------------------------------------------------
// lora.go — LoRA merge-at-load for pre-quantized models.
//
// Some checkpoints (e.g. Edge0-35B-A3B-preview, arch qwen3_5_moe) ship the
// language model as bare quantized triplets plus a *separate* LoRA adapter
// (lora_*.safetensors) of rank-`r` Recover-LoRA pairs:
//
//	<base>.lora_A  [r, in]    (F16)
//	<base>.lora_B  [out, r]   (F16)
//
// where <base> is the full base weight key (e.g.
// "language_model.model.layers.0.self_attn.q_proj"). Without the adapter the
// model runs as bare int4 (a few points worse); with it, every target weight
// is dequantized, adjusted by `scale * (B @ A)` (scale = alpha / r), and
// re-quantized at the SAME group size / bits / mode as the original triplet —
// so the merged weight drops straight into the existing quantized load path
// (MLX native-quant, or dequant-to-Q4_0/Q8_0 on GGML) unchanged.
//
// The adapter's scale and rank are read from its safetensors __metadata__
// ("alpha" and "r", both stored as strings); when either is absent the scale
// falls back to 1.0.
// ---------------------------------------------------------------------------

// loraPair holds the two adapter tensor keys for one target base weight.
type loraPair struct {
	aName string // "<base>.lora_A"
	bName string // "<base>.lora_B"
}

// LoraAdapter is a parsed LoRA adapter file. Targets maps each target base
// weight name (the safetensors key the LoRA modifies, without ".weight") to
// its A/B pair. Scale is alpha/r (PEFT convention).
type LoraAdapter struct {
	// File owns the adapter's mmap; callers must Release it. Single-file only
	// (a LoRA adapter is never sharded).
	File  *SafetensorsFile
	Scale float32
	Rank  int // `r`; 0 when the metadata omits it (shape-derived at merge time)
	Alpha int // raw `alpha`; 0 when the metadata omits it
	// Targets maps target base weight name → A/B pair. Names already carry the
	// model's full prefix (e.g. "language_model.model.layers.0.self_attn.q_proj").
	Targets map[string]loraPair
}

// loraAdapterFile returns the first lora_*.safetensors in modelDir
// (alphabetical), or "" when none. The bare name lora.safetensors is also
// accepted. prerouter_*.safetensors (the edge0 framework's expert-prefetch
// sidecar) is deliberately excluded — it is not a LoRA adapter.
func loraAdapterFile(modelDir string) string {
	var matches []string
	if fs, _ := filepath.Glob(filepath.Join(modelDir, "lora_*.safetensors")); len(fs) > 0 {
		matches = append(matches, fs...)
	} else if p := filepath.Join(modelDir, "lora.safetensors"); fileExists(p) {
		matches = append(matches, p)
	}
	sort.Strings(matches)
	for _, p := range matches {
		if strings.Contains(strings.ToLower(filepath.Base(p)), "prerouter") {
			continue
		}
		return p
	}
	return ""
}

// LoadLoraAdapter discovers and parses a LoRA adapter in modelDir. It returns
// (nil, nil) when the directory has no lora_*.safetensors file — callers
// treat nil as "no merge" and load the base weights untouched. A present
// adapter that carries no lora_A/lora_B pairs is likewise returned as nil
// (nothing to merge), so callers can simply proceed.
func LoadLoraAdapter(modelDir string) (*LoraAdapter, error) {
	path := loraAdapterFile(modelDir)
	if path == "" {
		return nil, nil
	}
	sf, err := OpenSafetensors(path)
	if err != nil {
		return nil, fmt.Errorf("open lora adapter %s: %w", filepath.Base(path), err)
	}

	a := &LoraAdapter{File: sf, Targets: map[string]loraPair{}}
	a.Scale, a.Alpha, a.Rank = loraScaleAndRank(sf.HeaderMetadata())

	// Pair up lora_A / lora_B tensors by their shared base weight name.
	var aNames []string
	for _, name := range sf.Keys() {
		if strings.HasSuffix(name, ".lora_A") {
			aNames = append(aNames, name)
		}
	}
	sort.Strings(aNames)
	for _, aName := range aNames {
		base := strings.TrimSuffix(aName, ".lora_A")
		bName := base + ".lora_B"
		if !sf.Has(bName) {
			continue // orphan A with no matching B
		}
		a.Targets[base] = loraPair{aName: aName, bName: bName}
	}
	if len(a.Targets) == 0 {
		sf.Release()
		return nil, nil // nothing to merge
	}
	return a, nil
}

// loraScaleAndRank parses the PEFT scale (alpha/r) and its parts from the
// adapter's __metadata__ ("alpha" and "r", stored as decimal strings). When
// either is missing or unparseable the scale is 1.0 (a safe default) and the
// corresponding part is 0.
func loraScaleAndRank(meta map[string]string) (scale float32, alpha, rank int) {
	var alphaF, rankF float64
	alphaSet, rankSet := false, false
	if v, ok := meta["alpha"]; ok {
		if f, err := strconv.ParseFloat(v, 64); err == nil {
			alphaF, alphaSet = f, true
		}
	}
	if v, ok := meta["r"]; ok {
		if f, err := strconv.ParseFloat(v, 64); err == nil {
			rankF, rankSet = f, true
		}
	}
	if alphaSet {
		alpha = int(alphaF)
	}
	if rankSet {
		rank = int(rankF)
	}
	if alphaSet && rankSet && rank > 0 {
		return float32(alphaF / rankF), alpha, rank
	}
	return 1.0, alpha, rank
}

// loraTensorF32 reads an adapter matrix (A/B, stored as F16) as row-major
// float32, casting through the backend when needed. The input array is NOT
// freed (the caller owns it); the temporary cast is freed internally.
func loraTensorF32(a tensor.Array, b tensor.Backend, s tensor.Stream) ([]float32, error) {
	if a.Dtype() == tensor.Float32 {
		return a.Float32Data()
	}
	casted, err := b.AsType(a, tensor.Float32, s)
	if err != nil {
		return nil, err
	}
	defer casted.Free()
	if err := casted.Eval(); err != nil {
		return nil, err
	}
	return casted.Float32Data()
}

// LoadLinearLora loads a LoRA-merged projection: the base weight is dequantized
// from its pre-quantized triplet, adjusted by `scale * (B @ A)`, and
// re-quantized at the same group size / bits / mode. It returns a *Linear built
// exactly as loadQuantizedTriplet would (MLX native-quant triplet, or a
// dequant-to-Q4_0/Q8_0 / F32-transposed weight on non-native backends), so the
// merged projection flows through the usual decode kernels untouched.
//
// `name` may be the base key or the "...weight" key; the base weight must be a
// pre-quantized triplet present in sf (weight + scales, biases optional) and
// an A/B pair present in a. Callers gate this on both conditions.
func LoadLinearLora(sf *SafetensorsFile, name string, b tensor.Backend, s tensor.Stream, quant *QuantConfig, a *LoraAdapter) (*Linear, error) {
	base := strings.TrimSuffix(name, ".weight")
	pair, ok := a.Targets[base]
	if !ok {
		return nil, fmt.Errorf("lora %s: no adapter pair", base)
	}

	groupSize := quant.GroupSize
	if groupSize == 0 {
		groupSize = 64
	}

	// 1. Base quantized triplet.
	w, err := sf.Get(base+".weight", b, s)
	if err != nil {
		return nil, fmt.Errorf("lora %s.weight: %w", base, err)
	}
	scales, err := sf.Get(base+".scales", b, s)
	if err != nil {
		w.Free()
		return nil, fmt.Errorf("lora %s.scales: %w", base, err)
	}
	var biases tensor.Array
	hasBiases := sf.Has(base + ".biases")
	if hasBiases {
		biases, err = sf.Get(base+".biases", b, s)
		if err != nil {
			w.Free()
			scales.Free()
			return nil, fmt.Errorf("lora %s.biases: %w", base, err)
		}
	}

	// Infer the real packed bits from the triplet shapes (models mix widths,
	// e.g. a 5-bit model with 6-bit embeddings) so the re-quantize matches.
	bits := inferQuantBits(w.Shape(), scales.Shape(), groupSize, quant.Bits)

	// 2. Dequantize the base to F32 ([out, in]); the dequant frees the triplet.
	f32, shape, err := readQuantizedWeights(w, scales, biases, bits, groupSize)
	if err != nil {
		w.Free()
		scales.Free()
		if biases != nil {
			biases.Free()
		}
		return nil, fmt.Errorf("lora dequantize %s: %w", base, err)
	}
	w.Free()
	scales.Free()
	if biases != nil {
		biases.Free()
	}
	if len(shape) != 2 {
		return nil, fmt.Errorf("lora %s: weight shape %v not 2-D", base, shape)
	}
	outDim := shape[0]
	inDim := shape[1]
	if len(f32) != outDim*inDim {
		return nil, fmt.Errorf("lora %s: %d elems != out*in %d", base, len(f32), outDim*inDim)
	}

	// 3. delta = scale * (B @ A): A [r, in], B [out, r] → [out, in].
	aArr, err := a.File.Get(pair.aName, b, s)
	if err != nil {
		return nil, fmt.Errorf("lora %s: %w", pair.aName, err)
	}
	aData, err := loraTensorF32(aArr, b, s)
	aArr.Free()
	if err != nil {
		return nil, fmt.Errorf("lora read %s: %w", pair.aName, err)
	}
	bArr, err := a.File.Get(pair.bName, b, s)
	if err != nil {
		return nil, fmt.Errorf("lora %s: %w", pair.bName, err)
	}
	bData, err := loraTensorF32(bArr, b, s)
	bArr.Free()
	if err != nil {
		return nil, fmt.Errorf("lora read %s: %w", pair.bName, err)
	}

	// Derive the rank from A's shape; cross-check against the base dims and B.
	// The BASE shape here is the packed U32 view ([out, in/8] for 4-bit), so
	// the logical in-dim is recovered via the scales shape (numGroups ×
	// groupSize) — read from sf (the BASE file), not the adapter file.
	logicalIn := inDim
	if sShape := sf.TensorShape(base + ".scales"); len(sShape) == 2 && sShape[1] > 0 {
		logicalIn = sShape[1] * groupSize
	}
	rank := a.Rank
	if s := a.File.TensorShape(pair.aName); len(s) == 2 {
		rank = s[0]
		if s[1] != logicalIn {
			return nil, fmt.Errorf("lora %s: A in-dim %d != base in-dim %d", base, s[1], logicalIn)
		}
	}
	if rank <= 0 || rank*logicalIn != len(aData) {
		return nil, fmt.Errorf("lora %s: A shape mismatch (rank %d, in %d, %d elems)", base, rank, logicalIn, len(aData))
	}
	if bs := a.File.TensorShape(pair.bName); len(bs) == 2 {
		if bs[0] != outDim || bs[1] != rank {
			return nil, fmt.Errorf("lora %s: B shape %v != [out %d, rank %d]", base, bs, outDim, rank)
		}
	}
	if len(bData) != outDim*rank {
		return nil, fmt.Errorf("lora %s: B %d elems != out*rank %d", base, len(bData), outDim*rank)
	}

	merged := make([]float32, len(f32))
	copy(merged, f32)
	for o := 0; o < outDim; o++ {
		bRow := bData[o*rank : (o+1)*rank]
		dst := merged[o*inDim : (o+1)*inDim]
		for r := 0; r < rank; r++ {
			bv := bRow[r] * a.Scale
			if bv == 0 {
				continue
			}
			aRow := aData[r*inDim : (r+1)*inDim]
			for c := 0; c < inDim; c++ {
				dst[c] += bv * aRow[c]
			}
		}
	}

	// 4. Materialize the merged weight and re-quantize at the base's layout.
	if b.NativeQuantization() {
		arr, err := b.NewArrayFromFloat32(merged, shape)
		if err != nil {
			return nil, fmt.Errorf("lora merged array %s: %w", base, err)
		}
		parts, err := b.Quantize(arr, groupSize, bits, quant.Mode, s)
		arr.Free()
		if err != nil {
			return nil, fmt.Errorf("lora requantize %s: %w", base, err)
		}
		if len(parts) < 2 {
			for _, p := range parts {
				p.Free()
			}
			return nil, fmt.Errorf("lora requantize %s: expected [weight, scales], got %d", base, len(parts))
		}
		for _, p := range parts {
			if err := p.Eval(); err != nil {
				for _, q := range parts {
					q.Free()
				}
				return nil, fmt.Errorf("lora eval %s: %w", base, err)
			}
		}
		// The base stored no biases: drop any the quantizer emitted so the
		// merged Linear matches the unmerged layout (qBiases == nil).
		l := &Linear{qW: parts[0], qScales: parts[1], qGroupSize: groupSize, qBits: bits, qMode: quant.Mode}
		if len(parts) > 2 {
			if hasBiases {
				l.qBiases = parts[2]
			} else {
				parts[2].Free()
			}
		}
		return l, nil
	}

	// Non-native backend: re-quantize the F32 merged weight exactly as the
	// dequant-to-full path does (Q4_0/Q8_0 on GGML, else F32 transposed).
	if qz, ok := b.(GGMLQuantizer); ok {
		if os.Getenv("SINTER_QUANT") == "q4_0" {
			arr, err := qz.NewArrayQ4_0(merged, shape)
			if err != nil {
				return nil, fmt.Errorf("lora q4_0 %s: %w", base, err)
			}
			return &Linear{wT: arr}, nil
		}
		arr, err := qz.NewArrayQ8_0(merged, shape)
		if err != nil {
			return nil, fmt.Errorf("lora q8_0 %s: %w", base, err)
		}
		return &Linear{wT: arr}, nil
	}
	wT, err := b.NewArrayFromFloat32(merged, shape)
	if err != nil {
		return nil, fmt.Errorf("lora merged array %s: %w", base, err)
	}
	wTT, err := b.Transpose(wT, s)
	wT.Free()
	if err != nil {
		return nil, fmt.Errorf("lora transpose %s: %w", base, err)
	}
	if err := wTT.Eval(); err != nil {
		wTT.Free()
		return nil, fmt.Errorf("lora eval transpose %s: %w", base, err)
	}
	return &Linear{wT: wTT}, nil
}

// fileExists reports whether path exists (used by loraAdapterFile).
func fileExists(path string) bool {
	_, err := os.Stat(path)
	return err == nil
}

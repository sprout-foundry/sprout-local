//go:build darwin && arm64 && cgo

// Compiled decode for the qwen2/llama architecture: the full single-token
// decode step (embedding gather, all decoder layers, final norm, logits head,
// argmax) traced once as an MLX closure and compiled via mlx_compile, so
// every subsequent token replays the cached Metal execution plan instead of
// re-walking the graph from Go.
//
// Qwen2/llama is pure full-attention (no Qwen3-style per-head QK RMSNorm, no
// hybrid linear-attention layers), so the closure I/O is simpler than
// qwen35's: every layer has fixed-capacity K/V buffers and no recurrent
// state. Each layer's new token K/V row is written in-graph via a
// Where(Equal(arange(C), pos)) scatter, attention runs over the padded buffer
// with an in-graph additive mask, and the updated whole buffers are the
// layer's outputs — one materialization per token, no host-side write. The
// closure body uses the same eager ops as the decode path (llm.RMSNorm, the
// linear projections, mlx fast-SDPA, the dynamic-offset RoPE) so numerics
// match eager, with NO_SIMPLIFY compile mode preventing reassociation that
// would flip near-tie bf16 argmaxes.
package qwen2

import (
	"fmt"
	"log"
	"os"
	"time"

	"github.com/sprout-foundry/sinter/llm"
	"github.com/sprout-foundry/sinter/mlx"
	"github.com/sprout-foundry/sinter/tensor"
)

// debugCompiledDecode gates verbose per-step timing logs.
func debugCompiledDecode() bool { return os.Getenv("SINTER_LOCAL_DEBUG") == "1" }

// compiledGrowStep rounds the fixed K/V capacity up so a single capacity
// serves most generation budgets. Mirrors llm.kvGrowStep's granularity.
const compiledGrowStep = 256

// roundUpTo rounds n up to a multiple of m (m >= 1).
func roundUpTo(n, m int) int {
	if m < 1 {
		m = 1
	}
	return ((n + m - 1) / m) * m
}

// compiledDecode holds one model's compiled decode closure plus its
// per-layer K/V buffers. Every layer is full-attention, so there is no
// recurrent state — just K and V per layer.
type compiledDecode struct {
	closure *mlx.Closure
	plain   *mlx.Closure // traced source; freed with the compiled closure
	// capacity is the per-layer sequence capacity. Shapes are constant for
	// the whole generation — a change would recompile.
	capacity int
	// kBufs/vBufs are the owned K/V buffers (zero-padded to capacity, with
	// the prefilled window in the front). They are rewritten in place
	// (value-semantics) via the closure outputs each step.
	kBufs, vBufs []tensor.Array

	backend tensor.Backend
	stream  tensor.Stream
	streamM *mlx.Stream

	traceCount int // diagnostics: how many times the Go body ran (should be 1)
}

// PrepareCompiledDecode sizes the fixed-capacity K/V buffers from the
// just-prefilled cache, traces the decode-step closure, and compiles it.
// Runs once per generation, before the decode loop.
func (q *Qwen2) PrepareCompiledDecode(promptLen, maxTokens int, cache *llm.KVCache) error {
	if q.cd != nil {
		q.ReleaseCompiledDecode()
	}
	if q.backend.Name() != "metal" {
		return fmt.Errorf("qwen2: compiled decode requires the MLX backend (got %q)", q.backend.Name())
	}

	cd := &compiledDecode{
		// Fixed-capacity K/V buffers (capacity rounded to compiledGrowStep):
		// shapes are constant for the whole generation, so one compilation
		// serves every step and every op traces with concrete shapes.
		capacity: roundUpTo(promptLen+maxTokens, compiledGrowStep),
		kBufs:    make([]tensor.Array, q.cfg.NumLayers),
		vBufs:    make([]tensor.Array, q.cfg.NumLayers),
		backend:  q.backend,
		stream:   q.stream,
		streamM:  q.stream.(*mlx.Stream),
	}
	fail := func(err error) error {
		q.freeCompiledState(cd)
		return err
	}

	n := cache.CachedLen()
	for i := 0; i < q.cfg.NumLayers; i++ {
		layer, err := cache.Get(i)
		if err != nil {
			return fail(err)
		}
		if layer == nil || layer.K == nil || layer.V == nil {
			return fail(fmt.Errorf("qwen2: compiled decode: layer %d has no K/V", i))
		}
		shape := layer.K.Shape()
		padShape := []int{shape[0], shape[1], cd.capacity, shape[3]}
		kb, err := q.backend.Zeros(padShape, layer.K.Dtype(), q.stream)
		if err != nil {
			return fail(fmt.Errorf("qwen2: compiled decode: alloc k buf: %w", err))
		}
		vb, err := q.backend.Zeros(padShape, layer.V.Dtype(), q.stream)
		if err != nil {
			kb.Free()
			return fail(fmt.Errorf("qwen2: compiled decode: alloc v buf: %w", err))
		}
		if err := q.copyPrefillWindow(cd, i, kb, vb, layer.K, layer.V, n); err != nil {
			return fail(err)
		}
		// copyPrefillWindow adopts the zero-padded buffers (see its
		// comment): on success it owns and assigns cd.kBufs[i]/vBufs[i].
	}

	// Trace inputs: dummy ids/pos plus the real (already materialized) K/V
	// buffers. The trace only consumes shapes; values are irrelevant.
	inputs, err := q.compiledDecodePlaceholders(cd)
	if err != nil {
		return fail(err)
	}
	plain, err := mlx.NewClosure(func(in []*mlx.Array) ([]*mlx.Array, error) {
		return q.compiledDecodeBody(cd, in)
	})
	if err != nil {
		for _, a := range inputs {
			a.Free()
		}
		return fail(fmt.Errorf("qwen2: compiled decode: new closure: %w", err))
	}
	// NO_SIMPLIFY+NO_FUSE: cache the scheduled execution plan (the CPU-side
	// win) but skip BOTH algebraic simplification (which can reorder
	// associativity and change bf16 rounding vs the eager op sequence) and
	// kernel fusion (which keeps fp32 intermediates where eager rounds to
	// bf16 between kernels). Either transform is enough to flip near-tie
	// argmax tokens and break parity with the eager decode path.
	if err := mlx.SetCompileMode(mlx.CompileModeNoSimplify); err != nil {
		return fail(fmt.Errorf("qwen2: compiled decode: set compile mode: %w", err))
	}
	compiled, err := plain.Compile(false) // fixed shapes by design
	if err != nil {
		plain.Free()
		for _, a := range inputs {
			a.Free()
		}
		return fail(fmt.Errorf("qwen2: compiled decode: compile: %w", err))
	}
	cd.closure = compiled
	cd.plain = plain
	q.cd = cd

	// First apply runs the trace on the placeholder inputs; its outputs are
	// discarded (values may be placeholder garbage) but this must happen
	// before the real loop so the first real token is a pure replay. Use
	// dummy state arrays so the real K/V buffers are never touched at trace
	// time. inputs[2:] hold the real arrays — swap them for zero clones of
	// the same shapes.
	dummies := make([]*mlx.Array, 0, len(inputs)-2)
	for _, a := range inputs[2:] {
		z, err := mlx.Zeros(a.Shape(), a.Dtype(), cd.streamM)
		if err != nil {
			inputs[0].Free()
			inputs[1].Free()
			q.ReleaseCompiledDecode()
			return fmt.Errorf("qwen2: compiled decode: trace dummy: %w", err)
		}
		dummies = append(dummies, z)
	}
	traceInputs := append([]*mlx.Array{inputs[0], inputs[1]}, dummies...)
	outs, err := compiled.Apply(traceInputs)
	for _, z := range dummies {
		z.Free()
	}
	// inputs[0:2] are the dummy ids/pos; inputs[2:] are the real K/V
	// buffers (owned by cd) and must NOT be freed here.
	inputs[0].Free()
	inputs[1].Free()
	if err != nil {
		q.ReleaseCompiledDecode()
		return fmt.Errorf("qwen2: compiled decode: trace apply: %w", err)
	}
	for _, o := range outs {
		o.Free()
	}
	return nil
}

// copyPrefillWindow copies a layer's populated window into the fresh
// fixed-capacity buffers. Takes full ownership of kb/vb (freed on every
// path) and, on success, adopts the SliceUpdate results into
// cd.kBufs[layerIdx]/cd.vBufs[layerIdx]: MLX arrays are value-semantics,
// so SliceUpdate returns NEW arrays carrying the prefilled window — the
// original zero-padded buffers are never written in place.
func (q *Qwen2) copyPrefillWindow(cd *compiledDecode, layerIdx int, kb, vb, srcK, srcV tensor.Array, n int) error {
	defer kb.Free()
	defer vb.Free()
	shape := srcK.Shape()
	if n <= 0 || n > shape[2] {
		n = shape[2]
	}
	start := []int{0, 0, 0, 0}
	stop := []int{shape[0], shape[1], n, shape[3]}
	wk, err := q.backend.SliceUpdate(kb, srcK, start, stop, q.stream)
	if err != nil {
		return fmt.Errorf("qwen2: prefill copy k: %w", err)
	}
	wv, err := q.backend.SliceUpdate(vb, srcV, start, stop, q.stream)
	if err != nil {
		wk.Free()
		return fmt.Errorf("qwen2: prefill copy v: %w", err)
	}
	if err := wk.Eval(); err != nil {
		wk.Free()
		wv.Free()
		return fmt.Errorf("qwen2: prefill copy k eval: %w", err)
	}
	if err := wv.Eval(); err != nil {
		wk.Free()
		wv.Free()
		return fmt.Errorf("qwen2: prefill copy v eval: %w", err)
	}
	cd.kBufs[layerIdx] = wk
	cd.vBufs[layerIdx] = wv
	return nil
}

// compiledDecodePlaceholders allocates the trace inputs: dummy ids/pos plus
// the real K/V buffers (already evaluated by prefill).
func (q *Qwen2) compiledDecodePlaceholders(cd *compiledDecode) ([]*mlx.Array, error) {
	inputs := make([]*mlx.Array, 0, 2+2*q.cfg.NumLayers)
	ids, err := q.backend.NewArrayFromInt64([]int64{0}, []int{1, 1})
	if err != nil {
		return nil, err
	}
	inputs = append(inputs, ids.(*mlx.Array))
	pos, err := q.backend.NewArrayFromInt32([]int32{0}, []int{1})
	if err != nil {
		ids.Free()
		return nil, err
	}
	inputs = append(inputs, pos.(*mlx.Array))
	for i := 0; i < q.cfg.NumLayers; i++ {
		if cd.kBufs[i] == nil || cd.vBufs[i] == nil {
			for _, a := range inputs {
				a.Free()
			}
			return nil, fmt.Errorf("qwen2: compiled decode: layer %d has no state", i)
		}
		inputs = append(inputs, cd.kBufs[i].(*mlx.Array), cd.vBufs[i].(*mlx.Array))
	}
	return inputs, nil
}

// ForwardDecodeCompiled runs one compiled decode step: applies the closure
// on the current state, writes the updated state back, and returns the
// argmax next token as a [1,1] int64 array with AsyncEval already called
// (same contract as ForwardDecodeArgmaxArray). tokenArr is borrowed — the
// caller owns it.
func (q *Qwen2) ForwardDecodeCompiled(tokenArr tensor.Array, pos int) (tensor.Array, error) {
	t0 := time.Now()
	cd := q.cd
	if cd == nil || cd.closure == nil {
		return nil, fmt.Errorf("qwen2: compiled decode not prepared")
	}
	posArr, err := q.backend.NewArrayFromInt32([]int32{int32(pos)}, []int{1})
	if err != nil {
		return nil, fmt.Errorf("qwen2: compiled decode: pos array: %w", err)
	}
	inputs := make([]*mlx.Array, 0, 2+2*q.cfg.NumLayers)
	inputs = append(inputs, tokenArr.(*mlx.Array), posArr.(*mlx.Array))
	for i := 0; i < q.cfg.NumLayers; i++ {
		inputs = append(inputs, cd.kBufs[i].(*mlx.Array), cd.vBufs[i].(*mlx.Array))
	}

	outs, err := cd.closure.Apply(inputs)
	posArr.Free()
	if err != nil {
		return nil, fmt.Errorf("qwen2: compiled decode apply: %w", err)
	}
	applyDone := time.Now()
	if debugCompiledDecode() {
		log.Printf("qwen2: compiled apply done (%d outputs) in %.3fs", len(outs), applyDone.Sub(t0).Seconds())
	}
	if len(outs) != 2*q.cfg.NumLayers+1 {
		for _, o := range outs {
			o.Free()
		}
		return nil, fmt.Errorf("qwen2: compiled decode: got %d outputs, want %d", len(outs), 2*q.cfg.NumLayers+1)
	}

	// Write the updated state back. The closure outputs the updated K/V
	// buffers (Where-scatter result); swap them in, freeing the previous
	// arrays. The lazy outputs hold their own refs, so dropping our wrapper
	// is safe.
	idx := 0
	for i := 0; i < q.cfg.NumLayers; i++ {
		cd.kBufs[i].Free()
		cd.vBufs[i].Free()
		cd.kBufs[i] = outs[idx]
		cd.vBufs[i] = outs[idx+1]
		idx += 2
	}
	logits := outs[idx]

	// The compiled replay evaluates its outputs before returning, so no
	// separate state dispatch is needed.

	// [1,1,vocab] argmax → [1,1] int64, dispatched without readback — same
	// tail as ForwardDecodeArgmaxArray.
	idxArr, err := mlx.ArgMax(logits, false, cd.streamM)
	logits.Free()
	if err != nil {
		return nil, fmt.Errorf("qwen2: compiled decode argmax: %w", err)
	}
	idx64, err := mlx.AsType(idxArr, mlx.Int64, cd.streamM)
	idxArr.Free()
	if err != nil {
		return nil, fmt.Errorf("qwen2: compiled decode argmax cast: %w", err)
	}
	next, err := mlx.Reshape(idx64, []int{1, 1}, cd.streamM)
	idx64.Free()
	if err != nil {
		return nil, fmt.Errorf("qwen2: compiled decode argmax reshape: %w", err)
	}
	if err := next.AsyncEval(); err != nil {
		next.Free()
		return nil, fmt.Errorf("qwen2: compiled decode async eval: %w", err)
	}
	if debugCompiledDecode() {
		log.Printf("qwen2: compiled step pos=%d total=%.3fs (apply=%.3f tail=%.3f)",
			pos, time.Since(t0).Seconds(), applyDone.Sub(t0).Seconds(), time.Since(applyDone).Seconds())
	}
	return next, nil
}

// ReleaseCompiledDecode frees the compiled closure and all owned state.
func (q *Qwen2) ReleaseCompiledDecode() {
	if q.cd == nil {
		return
	}
	q.freeCompiledState(q.cd)
	q.cd = nil
}

func (q *Qwen2) freeCompiledState(cd *compiledDecode) {
	if cd.closure != nil {
		cd.closure.Free()
		cd.closure = nil
	}
	if cd.plain != nil {
		cd.plain.Free()
		cd.plain = nil
	}
	for i := range cd.kBufs {
		if cd.kBufs[i] != nil {
			cd.kBufs[i].Free()
			cd.kBufs[i] = nil
		}
		if cd.vBufs[i] != nil {
			cd.vBufs[i].Free()
			cd.vBufs[i] = nil
		}
	}
}

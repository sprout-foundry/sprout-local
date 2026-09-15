package chatmodel

import (
	"context"
	"fmt"
	"log"
	"os"
	"path/filepath"
	"regexp"
	"sort"
	"strconv"
	"strings"
	"sync"
	"time"

	"github.com/sprout-foundry/sinter/llm"

	"github.com/sprout-foundry/sprout-local/internal/config"
	"github.com/sprout-foundry/sprout-local/internal/paths"
)

// Per-platform model registrations (sinter blank-imports) live in
// chatmodel_darwin.go and chatmodel_linux_ggml.go. gemma4 and the MLX
// backend are darwin-only, so they must not be imported on Linux.

// ---------------------------------------------------------------------------
// Sinter-native inference
//
// chatllm follows the gmitllm pattern: sinter
// (github.com/sprout-foundry/sinter) runs inference in-process against
// MLX-format model directories. No llama-server, no GGUF files, no HTTP.
//
// Unlike gmitllm (single completion), chatllm streams: the Generate call
// receives an onToken callback and each token ID is decoded with
// DecodeToken for incremental display.
// ---------------------------------------------------------------------------

var (
	modelMu    sync.Mutex
	modelCache = map[string]*llm.Model{}

	// genMu serializes Generate calls across surfaces: sinter runs on
	// one GPU. Non-streaming callers go through SerializeGeneration.
	genMu sync.Mutex

	// modelRecent is the most recently loaded/used model directory; the
	// eviction pass never evicts it.
	modelRecent string
)

// ResidentLimit is how many models may stay resident in the load cache.
// Two keeps instant swaps between a pair; each additional resident model
// pins its weights for the model's lifetime. Override with
// SPROUT_LOCAL_RESIDENT_MODELS (1 = always reload on switch).
func ResidentLimit() int {
	if v := os.Getenv("SPROUT_LOCAL_RESIDENT_MODELS"); v != "" {
		if n, err := strconv.Atoi(v); err == nil && n >= 1 {
			return n
		}
	}
	return 2
}

// LoadModelDir loads and caches a sinter model by directory, so switching
// models in the web UI only pays the load cost once per model. The cache
// is capped (evictModels) right here — every surface (REPL, web, OpenAI
// API, warming) funnels through this function, so no caller can leak
// resident models by switching.
func LoadModelDir(dir string) (*llm.Model, error) {
	modelMu.Lock()
	defer modelMu.Unlock()
	if m, ok := modelCache[dir]; ok {
		modelRecent = dir
		return m, nil
	}
	m, err := llm.NewModel(dir)
	if err != nil {
		return nil, fmt.Errorf("sinter failed to load %s: %w", dir, err)
	}
	modelCache[dir] = m
	modelRecent = dir
	log.Printf("sinter: loaded %s", dir)
	evictModelsLocked(ResidentLimit(), dir)
	return m, nil
}

// loadSinterModel lazily loads the default MLX model once per process.
// Returns (nil, nil) when no MLX-format model directory is
// configured/found — the caller surfaces the error to the user.
func loadSinterModel() (*llm.Model, error) {
	dir := paths.ResolveModelDir()
	if dir == "" {
		return nil, nil // no error: just not available
	}
	return LoadModelDir(dir)
}

// StreamChatDir is StreamChat against an explicit model directory (the
// REPL's session model, set by /model or /pull). An empty dir falls back
// to the process default. REPL-only: single-threaded, so it skips the
// GenMu serialization the web UI's StreamChatModel needs.
func StreamChatDir(ctx context.Context, modelDir string, messages []llm.ChatMessage, onDelta func(string)) (string, error) {
	if modelDir == "" {
		return StreamChat(ctx, messages, onDelta)
	}
	m, err := LoadModelDir(modelDir)
	if err != nil {
		return "", err
	}
	return generateChat(ctx, m, messages, onDelta)
}

// StreamChat renders the conversation via the model's chat template and
// streams generated tokens through onDelta as decoded text. The full
// response is also returned (already hygiene-stripped). Streaming is
// raw-text; stripStreamingNoise handles the visible layer only, while the
// returned value is fully cleaned for history/logging.
func StreamChat(ctx context.Context, messages []llm.ChatMessage, onDelta func(string)) (string, error) {
	m, err := loadSinterModel()
	if err != nil {
		return "", err
	}
	if m == nil {
		return "", fmt.Errorf("sinter: no MLX-format model directory configured")
	}
	return generateChat(ctx, m, messages, onDelta)
}

// generateChat runs one streaming completion on a loaded model and applies
// the full hygiene pass to the returned text. Shared by StreamChat
// (process default model) and StreamChatDir (session model).
// TurnMetrics is one generation's llama.cpp-style stats: prompt tokens,
// generation tokens, wall-clock timing, and context usage.
type TurnMetrics struct {
	PromptTokens int    // tokens in the rendered prompt (pp)
	PromptMs     int64  // prefill wall time
	GenTokens    int    // tokens generated
	GenMs        int64  // generation wall time (includes prefill)
	ContextUsed  int    // prompt + generated
	ContextSize  int    // model's window
	GuardReason  string // non-empty when the generation guard stopped it
}

// TPS returns generation tokens per second (0 when no time elapsed).
func (t TurnMetrics) TPS() float64 {
	if t.GenMs <= 0 {
		return 0
	}
	return float64(t.GenTokens) / (float64(t.GenMs) / 1000)
}

// PP returns prompt tokens per second from sinter's reported prefill wall
// time (LastPromptMs). 0 when unknown.
func (t TurnMetrics) PP() float64 {
	if t.PromptMs <= 0 || t.PromptTokens == 0 {
		return 0
	}
	return float64(t.PromptTokens) / (float64(t.PromptMs) / 1000)
}

// String renders a llama.cpp-style one-liner:
// "pp 3125.0 t/s · 49 tok · 29.3 t/s · ctx 563/128000" — prompt-processing
// rate first (from sinter's prefill split), then decode.
func (t TurnMetrics) String() string {
	var b strings.Builder
	if t.PP() > 0 {
		fmt.Fprintf(&b, "pp %.1f t/s · ", t.PP())
	}
	fmt.Fprintf(&b, "%d tok · %.1f t/s · ctx %d/%d",
		t.GenTokens, t.TPS(), t.ContextUsed, t.ContextSize)
	return b.String()
}

// LastMetrics holds the most recent generation's stats. For multi-step
// tool turns this is the final generation only; turn callers that want
// whole-turn totals accumulate via trackMetrics (see replState.metricsLine
// and the web metrics frame).
var LastMetrics TurnMetrics

// turnMetricsAcc accumulates every generation in the current turn. Reset
// by startTurnMetrics at turn start.
var turnMetricsAcc TurnMetrics
var turnMetricsActive bool

// startTurnMetrics clears the per-turn accumulator.
func StartTurnMetrics() {
	turnMetricsAcc = TurnMetrics{}
	turnMetricsActive = true
}

// trackMetrics folds a finished generation into the per-turn accumulator.
// The web/REPL surfaces read turnMetrics() after the turn.
func trackMetrics(m TurnMetrics) {
	turnMetricsAcc = addMetrics(turnMetricsAcc, m)
	LastMetrics = m
}

// CurrentTurnMetrics returns the accumulated per-turn stats.
func CurrentTurnMetrics() TurnMetrics { return turnMetricsAcc }

// TurnMetricsActive reports whether metric accumulation is on for this turn.
func TurnMetricsActive() bool { return turnMetricsActive }

// addMetrics sums two metric snapshots (context fields come from b, the
// latest generation's window position).
func addMetrics(a, b TurnMetrics) TurnMetrics {
	return TurnMetrics{
		PromptTokens: a.PromptTokens + b.PromptTokens,
		PromptMs:     a.PromptMs + b.PromptMs,
		GenTokens:    a.GenTokens + b.GenTokens,
		GenMs:        a.GenMs + b.GenMs,
		ContextUsed:  b.ContextUsed,
		ContextSize:  b.ContextSize,
	}
}

// ---------------------------------------------------------------------------
// Generation guard: small models occasionally emit chat-boundary tokens as
// literal text ("<|endoftext|><|im_start|>user…"), hallucinate a fake user
// turn, and then loop until the token budget burns (observed live: a 2000+
// token "Show me where they…" spiral). Sinter's loop only stops on the EOS
// token ID — spelled-out control tokens and repetition don't trip it — so
// chatllm watches the decoded stream and cancels the context itself.
// ---------------------------------------------------------------------------

// leakMarkers are chat-boundary tokens that must never appear in output
// text. Seeing one means the model has left its turn.
var leakMarkers = []string{
	"<|endoftext|>", "<|im_start|>", "<|im_end|>", "<|startoftext|>",
}

// genGuard watches a generation's decoded stream and cancels its context
// when the model leaks control tokens or falls into a repeat loop.
type genGuard struct {
	cancel    context.CancelFunc
	buf       strings.Builder
	triggered bool
	reason    string
}

// newGenGuard wraps parent so calling g.trigger cancels the generation.
func newGenGuard(parent context.Context) (context.Context, *genGuard) {
	ctx, cancel := context.WithCancel(parent)
	return ctx, &genGuard{cancel: cancel}
}

// feed ingests one decoded delta and reports whether the guard has
// triggered (cancellation requested). When true, stop emitting deltas.
//
// Cancellation is reserved for unambiguous failures: leaked chat-boundary
// control tokens. Repetition is NOT a cancel — legitimate output (CSS
// blocks, table rows, code idioms, poetry) repeats short spans, and the
// token budget already bounds any loop. Repetitive spirals are instead
// detected post-hoc by IsSpiral and (a) excluded from seed's
// truncated-continue logic via an explicit finish note, (b) reported to
// the user.
func (g *genGuard) feed(delta string) bool {
	if g.triggered {
		return true
	}
	g.buf.WriteString(delta)
	s := g.buf.String()

	for _, m := range leakMarkers {
		if strings.Contains(s, m) {
			g.trigger("control token " + m)
			return true
		}
	}
	return false
}

// IsSpiral reports whether the assistant PROSE (text before any
// <tool_call> block) is dominated by verbatim repetition of a single
// long span — the "Show me where they…" failure mode. Repetition inside
// tool parameters is excluded: generated files legitimately repeat
// idioms (CSS blocks, table rows, error-handling patterns). Used
// post-generation: spiral text is still delivered (the user saw it
// stream) but the reply is annotated so seed treats the turn as final.
func IsSpiral(text string) bool {
	prose := text
	if i := strings.Index(prose, "<tool_call>"); i >= 0 {
		prose = prose[:i]
	}
	const probeLen, minCount = 80, 4
	norm := normalizeForLoopCheck(prose)
	if len(norm) < probeLen*minCount {
		return false
	}
	probe := norm[len(norm)-probeLen:]
	count := strings.Count(norm, probe)
	if count < minCount {
		return false
	}
	// Require the repeats to dominate the prose (not merely appear): a
	// refrain at the end of an otherwise varied answer passes.
	covered := count * probeLen
	return covered*2 > len(norm)
}

// normalizeForLoopCheck collapses runs of whitespace to single spaces so
// line breaks and indentation don't create false matches.
func normalizeForLoopCheck(s string) string {
	var b strings.Builder
	b.Grow(len(s))
	prevSpace := true
	for _, r := range s {
		if r == ' ' || r == '\t' || r == '\n' || r == '\r' {
			if !prevSpace {
				b.WriteByte(' ')
			}
			prevSpace = true
			continue
		}
		b.WriteRune(r)
		prevSpace = false
	}
	return b.String()
}

func (g *genGuard) trigger(reason string) {
	if g.triggered {
		return
	}
	g.triggered = true
	g.reason = reason
	if g.cancel != nil {
		g.cancel()
	}
	log.Printf("chatllm: stopped generation early (%s)", reason)
}

// text returns the accumulated output with any leaked control-token text
// (and a trailing partial marker) removed.
func (g *genGuard) text() string {
	s := g.buf.String()
	cut := len(s)
	for _, m := range leakMarkers {
		if i := strings.Index(s, m); i >= 0 && i < cut {
			cut = i
		}
	}
	s = s[:cut]
	// Strip a trailing partial marker the model was in the middle of
	// spelling when the guard fired.
	for _, m := range leakMarkers {
		for l := len(m) - 1; l > 0; l-- {
			if strings.HasSuffix(s, m[:l]) {
				s = s[:len(s)-l]
				return s
			}
		}
	}
	return s
}

// IsModelLoaded reports whether dir is resident in the load cache.
func IsModelLoaded(dir string) bool {
	modelMu.Lock()
	defer modelMu.Unlock()
	_, ok := modelCache[dir]
	return ok
}

// RunGeneration is the single generation core every surface goes through
// (REPL via the seed provider, web UI, one-shot). One loop owns the
// cross-cutting concerns that used to be copy-pasted three times: config,
// metrics, the leak/repetition guard, and output hygiene.
//
//	onDelta receives raw decoded deltas as they stream (may be nil).
//	Returns the hygiened text; guard-triggered stops return the clean
//	prefix plus a "…[stopped: <reason>]" note appended by the caller.
func RunGeneration(
	ctx context.Context,
	m *llm.Model,
	rawModelDir string,
	messages []llm.ChatMessage,
	onDelta func(string),
) (text string, metrics TurnMetrics, err error) {
	rendered := m.FormatChat(messages)

	cfg := llm.DefaultGenerateConfig()
	cfg.MaxTokens = config.MaxTokens
	cfg.Temperature = temperature
	cfg.ThinkingTokens = false // filter <think>…</think> at the token layer

	gctx, guard := newGenGuard(ctx)
	var genTokens int
	start := time.Now()
	genErr := m.Generate(gctx, rendered, cfg, func(id int) {
		genTokens++
		delta := m.DecodeToken(id)
		if guard.feed(delta) {
			return // guard fired: stop emitting; ctx cancel ends the loop
		}
		if onDelta != nil {
			onDelta(delta)
		}
	})
	elapsed := time.Since(start)
	promptTokens := len(m.TokenizerEncode(rendered))

	metrics = TurnMetrics{
		PromptTokens: promptTokens,
		PromptMs:     m.LastPromptMs(),
		GenTokens:    genTokens,
		GenMs:        elapsed.Milliseconds(),
		ContextUsed:  promptTokens + genTokens,
		ContextSize:  m.ContextLength(),
		GuardReason:  guard.reason,
	}
	trackMetrics(metrics)
	if genErr != nil && !guard.triggered {
		// Real failure (not guard, not cancel): report it.
		if ctx.Err() == nil && !strings.Contains(errString(genErr), "cancel") {
			return "", metrics, fmt.Errorf("sinter generation failed: %w", genErr)
		}
		// Canceled mid-stream: return whatever streamed so callers can
		// keep the partial (REPL prints "(cancelled)" for context).
		return HygieneAll(guard.text()), metrics, genErr
	}

	guardText := guard.text()
	text = HygieneAll(guardText)
	if os.Getenv("SPROUT_LOCAL_RAW") == "1" {
		if text == "" && genTokens > 0 {
			// Everything was eaten post-generation: dump the unhygiened
			// stream so it's visible what the model actually produced
			// (thinking-token spirals show up here as empty output).
			dumpRawFull(rawModelDir, guardText, "all-text-removed-by-hygiene")
		} else if text != "" {
			dumpRaw(rawModelDir, text)
		}
	}
	return text, metrics, nil
}

// HygieneAll is the full post-generation cleanup pass (mirrors gmitllm):
// Gemma thought-channel spans, Qwen <thinking> blocks, wrapper tags,
// surrounding quote noise.
func HygieneAll(text string) string {
	text = GemmaStripThinking(text)
	text = StripThinkingTag(text)
	text = StripWrapperTag(text)
	return StripOutputNoise(text)
}

// errString safely extracts the message from an error.
func errString(err error) string {
	if err == nil {
		return ""
	}
	return err.Error()
}

// evictModelsLocked frees and drops cache entries so at most keep models
// stay resident. Called with modelMu held (from LoadModelDir); the active
// model is never evicted, and beyond that alphabetical order stands in
// for recency — with keep=2 and alternating use of two models this is a
// no-op, which is the common case.
func evictModelsLocked(keep int, recent string) {
	if len(modelCache) <= keep {
		return
	}
	names := make([]string, 0, len(modelCache))
	for n := range modelCache {
		names = append(names, n)
	}
	sort.Strings(names)
	for _, n := range names {
		if len(modelCache) <= keep {
			break
		}
		if n == recent {
			continue
		}
		if m := modelCache[n]; m != nil {
			m.Close() // free weights + prefix slots
		}
		delete(modelCache, n)
		log.Printf("chatllm: evicted %s (resident limit %d)", filepath.Base(n), keep)
	}
}

// Evict frees and drops cache entries so at most keep models stay
// resident (public entry; evictModelsLocked is the shared core that
// runs inside LoadModelDir as well, so every load path is capped).
func Evict(keep int, recent string) {
	modelMu.Lock()
	defer modelMu.Unlock()
	evictModelsLocked(keep, recent)
}

// dumpRawFull writes the unhygiened stream with a reason header — the
// forensic view when hygiene eats an entire generation.
func dumpRawFull(modelDir, text, note string) {
	dir := filepath.Join(paths.SessionsDir(), "raw")
	if err := os.MkdirAll(dir, 0o755); err != nil {
		return
	}
	name := fmt.Sprintf("%s_%s_FULL.txt", time.Now().Format("20060102_150405"),
		strings.ReplaceAll(filepath.Base(modelDir), "/", "_"))
	f, err := os.Create(filepath.Join(dir, name))
	if err != nil {
		return
	}
	defer f.Close()
	fmt.Fprintf(f, "model: %s\nnote: %s\ngen_tokens: %d\n---\n%s\n",
		modelDir, note, LastMetrics.GenTokens, text)
}

// dumpRaw appends one generation's raw (hygiene-passed) text to a
// timestamped file under <stateRoot>/sessions/raw/. Best effort: debugging
// output must never break the chat.
func dumpRaw(modelDir, text string) {
	dir := filepath.Join(paths.SessionsDir(), "raw")
	if err := os.MkdirAll(dir, 0o755); err != nil {
		return
	}
	name := fmt.Sprintf("%s_%s.txt", time.Now().Format("20060102_150405"),
		strings.ReplaceAll(filepath.Base(modelDir), "/", "_"))
	f, err := os.Create(filepath.Join(dir, name))
	if err != nil {
		return
	}
	defer f.Close()
	fmt.Fprintf(f, "model: %s\nguard: %q\nprompt_tokens: %d\ngen_tokens: %d\n---\n%s\n",
		modelDir, LastMetrics.GuardReason, LastMetrics.PromptTokens, LastMetrics.GenTokens, text)
}

// generateChat runs one streaming completion on a loaded model through the
// shared generation core.
func generateChat(ctx context.Context, m *llm.Model, messages []llm.ChatMessage, onDelta func(string)) (string, error) {
	text, _, err := RunGeneration(ctx, m, "", messages, onDelta)
	return text, err
}

// StreamChatModel is StreamChat against an explicit model directory — the
// web UI lets each connection pick a model from the shared models root.
func StreamChatModel(ctx context.Context, modelDir string, messages []llm.ChatMessage, onDelta func(string)) (string, error) {
	m, err := LoadModelDir(modelDir)
	if err != nil {
		return "", err
	}

	genMu.Lock() // one GPU: serialize Generate across connections
	defer genMu.Unlock()

	return generateChat(ctx, m, messages, onDelta)
}

// GemmaStripThinking removes thought-channel spans the model may emit
// (<|channel>thought …<channel|>), mirroring the template's strip_thinking
// macro. Ported from gmitllm sintermodel.go (origin sprout
// pkg/localmodel/gemma_tools.go).
func GemmaStripThinking(text string) string {
	if !strings.Contains(text, "<|channel>") {
		return text
	}
	var sb strings.Builder
	rest := text
	for {
		i := strings.Index(rest, "<|channel>thought")
		if i < 0 {
			sb.WriteString(rest)
			return sb.String()
		}
		sb.WriteString(rest[:i])
		rest = rest[i:]
		end := strings.Index(rest, "<channel|>")
		if end < 0 {
			return sb.String() // unterminated thought: drop the rest
		}
		rest = rest[end+len("<channel|>"):]
	}
}

func StripThinkingTag(s string) string {
	return regexp.MustCompile(`(?s)<thinking>.*?</thinking>`).ReplaceAllString(s, "")
}

// StripWrapperTag unwraps whole-response wrappers some models emit around
// their answer (e.g. <answer>…</answer>) and strips leading code-fence
// language markers ("plaintext", "text") some models prepend.
func StripWrapperTag(s string) string {
	s = strings.TrimSpace(s)
	if m := regexp.MustCompile(`^<(commit|answer|message|title)>`).FindString(s); m != "" {
		end := regexp.MustCompile(`</(commit|answer|message|title)>`).FindStringIndex(s)
		if end != nil {
			s = s[len(m):end[0]]
		} else {
			s = s[len(m):]
		}
		s = strings.TrimSpace(s)
	}
	s = regexp.MustCompile(`^(plaintext|text|markdown)\n?`).ReplaceAllString(s, "")
	return s
}

// StripOutputNoise trims surrounding quotes/backticks the model sometimes
// wraps responses in.
func StripOutputNoise(s string) string {
	return strings.TrimSpace(strings.Trim(s, "'\"`"))
}

func CtxBg() context.Context { return context.Background() }

// SerializeGeneration runs fn while holding the single-GPU generation
// lock, so non-streaming callers (the web fact-check) cannot interleave
// with streaming turns.
func SerializeGeneration(fn func()) {
	genMu.Lock()
	defer genMu.Unlock()
	fn()
}

// EndOfMessage is the sentinel line appended after one-shot output so
// stream consumers (chat/server.py) know the response is complete even
// when the final line looks like ordinary text.
const EndOfMessage = "[END_OF_MESSAGE]"

// Generation parameters for chat. Higher temperature than gmitllm's 0.2 —
// conversation benefits from a little more variety. The maxTokens cap
// lives in the config package (tunable via SPROUT_LOCAL_MAX_TOKENS or
// -max-tokens).
const temperature = 0.3

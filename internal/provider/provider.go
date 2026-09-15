package provider

// ---------------------------------------------------------------------------
// seedprovider.go — sinter as a seed core.Provider.
//
// seed (github.com/sprout-foundry/seed) owns the agent loop: query → LLM →
// tool calls → results → final answer, with compaction, retries, state
// export and interrupt handling. chatllm supplies two adapters:
//
//   • Provider — inference. seed's ChatRequest carries structured
//     Tools + Messages; this adapter renders them into the qwen3.5 text
//     protocol (# Tools system block, <tool_call> markup, <tool_response>
//     results) and runs sinter's FormatChat + Generate. Tool calls the
//     model emits as text are parsed back into structured seed.ToolCalls
//     at OnDone, so seed's loop drives everything.
//
//   • toolExecutor — the local tool set (tools.go registry) as a
//     seed core.ToolExecutor.
// ---------------------------------------------------------------------------

import (
	"context"
	"encoding/json"
	"fmt"
	"sort"
	"strings"

	"github.com/sprout-foundry/seed/core"
	"github.com/sprout-foundry/sinter/llm"

	"github.com/sprout-foundry/sprout-local/internal/chatmodel"
	"github.com/sprout-foundry/sprout-local/internal/config"
	"github.com/sprout-foundry/sprout-local/internal/mdterm"
	"github.com/sprout-foundry/sprout-local/internal/tools"
)

// contextWindow is qwen3.5-4b's max_position_embeddings. The template has
// no true document mask, so this is also the practical ceiling; seed uses
// it to schedule compaction.
const contextWindow = 262144

// Provider implements core.Provider against a sinter model directory.
// display (optional) receives clean, markup-free deltas for terminal
// display; seed's own stream handler still receives every delta via
// onDelta so its content buffer matches what the user saw.
type Provider struct {
	modelDir    string
	display     func(string)
	displayMode mdterm.Mode // styled (terminal) or raw (web socket)
	chatCount   int         // generations this turn (iteration boundary detection)
	// protocolName caches the model's tool protocol ("qwen" or "minicpm5").
	protocolName string
	// onIterationBoundary, when set, is called before each generation
	// after the first in a multi-step turn — the web UI starts a new
	// bubble per assistant message so tool chips interleave in order.
	onIterationBoundary func()
}

// NewProvider builds a provider for the given model directory.
func NewProvider(modelDir string) *Provider {
	return &Provider{modelDir: modelDir}
}

// protocol returns the tool protocol for this provider's model directory,
// resolved lazily (first call) and cached. Defaults to "qwen".
func (p *Provider) protocol() string {
	if p.protocolName == "" {
		p.protocolName = tools.ToolProtocolForModelDir(p.modelDir)
	}
	return p.protocolName
}

// SetDisplay wires (or clears, nil) the terminal delta sink.
func (p *Provider) SetDisplay(fn func(string)) { p.display = fn }

// SetDisplayMode picks the rendering for the display sink: mdterm.Styled
// for terminals (markdown → ANSI), mdterm.Raw for consumers that take
// plain text (the web socket — ANSI codes are meaningless there).
func (p *Provider) SetDisplayMode(mode mdterm.Mode) { p.displayMode = mode }

// ResetTurnCount clears the per-turn generation counter; surfaces call it
// at user-turn start so the iteration boundary only fires for generations
// 2..N *within* the same turn.
func (p *Provider) ResetTurnCount() { p.chatCount = 0 }

// SetIterationBoundary wires a callback fired at the start of each
// generation after the first (multi-step tool turns run several).
func (p *Provider) SetIterationBoundary(fn func()) { p.onIterationBoundary = fn }

// Info reports the session model and its context window.
func (p *Provider) Info() core.ProviderInfo {
	return core.ProviderInfo{
		Model:           p.modelDir,
		ContextSize:     contextWindow,
		MaxOutputTokens: config.MaxTokens,
	}
}

// EstimateTokens approximates the rendered prompt size. The chat template
// adds a per-message wrapper (~8 tokens); actual token counting would need
// the tokenizer, which is not exposed until the model loads. The 15%
// inflation over chars/4 keeps compaction slightly eager rather than late.
func (p *Provider) EstimateTokens(req *core.ChatRequest) int {
	n := 0
	for _, m := range req.Messages {
		n += len(m.Content)/4 + 8
	}
	return n * 115 / 100
}

// Chat runs a non-streaming completion.
func (p *Provider) Chat(ctx context.Context, req *core.ChatRequest) (*core.ChatResponse, error) {
	resp, err := p.chat(ctx, req, nil)
	if err != nil {
		return nil, err
	}
	return resp, nil
}

// ChatStream runs a streaming completion. Deltas already filtered of
// <tool_call> markup go through onContent; the full raw text is parsed and
// attached as structured ToolCalls in the returned response.
func (p *Provider) ChatStream(ctx context.Context, req *core.ChatRequest, handler core.StreamHandler) error {
	_, err := p.chat(ctx, req, handler)
	return err
}

// chat is the shared implementation: render → generate → parse → respond.
// When handler is non-nil, display-ready deltas stream to it (with
// <tool_call> markup suppressed by toolStreamFilter) and OnDone receives
// the final response.
func (p *Provider) chat(ctx context.Context, req *core.ChatRequest, handler core.StreamHandler) (*core.ChatResponse, error) {
	// Multi-step tool turns call Chat once per iteration; every call after
	// the first begins a new assistant message. Surfaces that render
	// per-message (the web thread) split their bubbles here so tool chips
	// interleave in order instead of piling up after one merged blob.
	if p.onIterationBoundary != nil && p.chatCount > 0 {
		p.onIterationBoundary()
	}
	p.chatCount++

	toolsOn := len(req.Tools) > 0
	msgs := renderSeedMessagesFor(req.Messages, p.protocol())
	if toolsOn {
		msgs = appendToolPromptFor(msgs, req.Tools, p.protocol())
	}

	var display *mdterm.StreamPrinter
	if p.display != nil {
		display = newStreamPrinterFunc(p.display, p.displayMode)
	}
	filter := &toolStreamFilter{tools: toolsOn, printer: display, protocol: p.protocol()}
	if handler != nil {
		filter.onDelta = handler.OnContent
	}

	m, err := chatmodel.LoadModelDir(p.modelDir)
	if err != nil {
		return nil, err
	}

	raw, _, err := chatmodel.RunGeneration(ctx, m, p.modelDir, msgs, func(delta string) {
		filter.write(delta)
	})
	if err != nil {
		return nil, err
	}

	text := chatmodel.HygieneAll(raw)
	plain, calls := tools.ExtractToolCallsFor(text, p.protocol())
	plain = strings.TrimSpace(plain)

	// A guard stop or a plain-prose spiral is final: mid-word/looping text
	// makes seed's truncation validator re-prompt the same spiral. Mark
	// the reply explicitly finished so the turn ends; the caller surfaces
	// a note to the user.
	if chatmodel.LastMetrics.GuardReason != "" || chatmodel.IsSpiral(plain) {
		if chatmodel.IsSpiral(plain) {
			plain = strings.TrimRight(plain, " \n") + "\n\n*(repetition detected — stopping here)*"
		} else {
			plain = strings.TrimRight(plain, " \n") +
				"\n\n*(stopped: " + chatmodel.LastMetrics.GuardReason + ")*"
		}
		if calls == nil {
			calls = []tools.ParsedToolCall{} // non-nil, empty: finish_reason stop
		}
	}

	resp := p.responseFor(plain, calls)
	filter.close() // emit any held-back tail before flushing the display
	if display != nil {
		display.Close() // flush a trailing partial line
	}
	if handler != nil {
		handler.OnDone(resp)
	}
	return resp, nil
}

// newStreamPrinterFunc builds a display-only StreamPrinter that forwards
// rendered output to fn (the terminal sink without owning stdout).
func newStreamPrinterFunc(fn func(string), mode mdterm.Mode) *mdterm.StreamPrinter {
	return mdterm.NewStreamPrinter(&funcWriter{fn}, mode)
}

// funcWriter adapts a func(string) to io.Writer.
type funcWriter struct{ fn func(string) }

func (w *funcWriter) Write(p []byte) (int, error) {
	w.fn(string(p))
	return len(p), nil
}

// responseFor builds a seed ChatResponse with structured tool calls.
func (p *Provider) responseFor(raw string, calls []tools.ParsedToolCall) *core.ChatResponse {
	tc := make([]core.ToolCall, 0, len(calls))
	for i, c := range calls {
		if c.Truncated {
			if c.Args == nil {
				c.Args = map[string]string{}
			}
			c.Args["_truncated"] = "1" // executor strips + annotates the result
		}
		args, err := json.Marshal(c.Args)
		if err != nil {
			args = []byte("{}")
		}
		tc = append(tc, core.ToolCall{
			ID:   fmt.Sprintf("call_%d", i),
			Type: "function",
			Function: core.ToolCallFunction{
				Name:      c.Name,
				Arguments: string(args),
			},
		})
	}
	return &core.ChatResponse{
		Model: p.modelDir,
		Choices: []core.ChatChoice{{
			Index:        0,
			Message:      core.Message{Role: "assistant", Content: raw, ToolCalls: tc},
			FinishReason: finishReason(calls),
		}},
	}
}

// finishReason mirrors OpenAI semantics: "tool_calls" when calls are
// present, "stop" otherwise (seed's loop keys off ToolCalls, but explicit
// is better than implicit).
func finishReason(calls []tools.ParsedToolCall) string {
	if len(calls) > 0 {
		return "tool_calls"
	}
	return "stop"
}

// renderSeedMessages converts seed messages into sinter ChatMessages.
// Assistant tool_calls become the model's native tool-call markup (so
// the history looks exactly like what the model itself emits), and tool
// results are wrapped in <tool_response> inside a user turn — mirroring
// chat_template.jinja.
func renderSeedMessages(msgs []core.Message) []llm.ChatMessage {
	return renderSeedMessagesFor(msgs, tools.ToolProtocol())
}

// renderSeedMessagesFor is renderSeedMessages with an explicit protocol.
func renderSeedMessagesFor(msgs []core.Message, protocol string) []llm.ChatMessage {
	out := make([]llm.ChatMessage, 0, len(msgs)+2)
	for _, m := range msgs {
		switch m.Role {
		case "system":
			out = append(out, llm.ChatMessage{Role: "system", Content: m.Content})
		case "user":
			out = append(out, llm.ChatMessage{Role: "user", Content: m.Content})
		case "assistant":
			content := m.Content
			for i, tc := range m.ToolCalls {
				var args map[string]string
				json.Unmarshal([]byte(tc.Function.Arguments), &args)
				content = renderToolCallTextFor(tc.Function.Name, args, i == 0 && strings.TrimSpace(content) != "", protocol)
			}
			out = append(out, llm.ChatMessage{Role: "assistant", Content: content})
		case "tool":
			out = tools.AppendToolResult(out, m.Content)
		}
	}
	// A turn ending in tool results leaves the model's turn open; the
	// template's generation cue (appended by sinter's FormatChat) closes it.
	return out
}

// renderToolCallTextFor renders one tool call in the protocol's native markup.
func renderToolCallTextFor(name string, args map[string]string, afterContent bool, protocol string) string {
	if protocol == "minicpm5" {
		return tools.RenderMiniCPM5ToolCallText(name, args, afterContent)
	}
	return tools.RenderToolCallText(name, args, afterContent)
}

// appendToolPrompt injects the "# Tools" block into the system message
// (creating one if needed) from seed's structured tool list.
func appendToolPrompt(msgs []llm.ChatMessage, seedTools []core.Tool) []llm.ChatMessage {
	return appendToolPromptFor(msgs, seedTools, tools.ToolProtocol())
}

// appendToolPromptFor is appendToolPrompt with an explicit protocol.
func appendToolPromptFor(msgs []llm.ChatMessage, seedTools []core.Tool, protocol string) []llm.ChatMessage {
	block := ToolPromptBlockFromSeedFor(seedTools, protocol)
	if len(msgs) > 0 && msgs[0].Role == "system" {
		msgs[0].Content = strings.TrimRight(msgs[0].Content, "\n") + "\n\n" + block
		return msgs
	}
	return append([]llm.ChatMessage{{Role: "system", Content: block}}, msgs...)
}

// toolPromptBlockFromSeed renders the # Tools block from seed Tool values.
func toolPromptBlockFromSeed(seedTools []core.Tool) string {
	return ToolPromptBlockFromSeedFor(seedTools, tools.ToolProtocol())
}

// toolPromptBlockFromSeedFor is toolPromptBlockFromSeed with an explicit protocol.
func ToolPromptBlockFromSeedFor(seedTools []core.Tool, protocol string) string {
	specs := make([]tools.ToolSpec, 0, len(seedTools))
	for _, t := range seedTools {
		params, _ := toolParamsFromSeed(t.Function.Parameters)
		specs = append(specs, tools.ToolSpec{
			Name:        t.Function.Name,
			Description: t.Function.Description,
			Parameters:  params,
		})
	}
	return tools.RenderToolPromptBlockFor(specs, protocol)
}

// toolParamsFromSeed converts a seed parameters schema (interface{} carrying
// our JSON) back into tools.ToolParam values.
func toolParamsFromSeed(schema interface{}) ([]tools.ToolParam, bool) {
	b, err := json.Marshal(schema)
	if err != nil {
		return nil, false
	}
	var parsed struct {
		Properties map[string]struct {
			Type        string `json:"type"`
			Description string `json:"description"`
		} `json:"properties"`
		Required []string `json:"required"`
	}
	if err := json.Unmarshal(b, &parsed); err != nil {
		return nil, false
	}
	// Deterministic order: sort property names.
	names := make([]string, 0, len(parsed.Properties))
	for n := range parsed.Properties {
		names = append(names, n)
	}
	sort.Strings(names)

	out := make([]tools.ToolParam, 0, len(names))
	for _, n := range names {
		p := parsed.Properties[n]
		req := false
		for _, r := range parsed.Required {
			if r == n {
				req = true
				break
			}
		}
		out = append(out, tools.ToolParam{Name: n, TypeName: p.Type, Description: p.Description, Required: req})
	}
	return out, true
}

// ─── Shared helpers ──────────────────────────────────────────────────────

// printerFromHandler was folded into chat(): the type-assert on
// *toolDisplayHandler happens there directly.

// toolDisplayHandler streams clean content to the terminal.
type toolDisplayHandler struct {
	printer *mdterm.StreamPrinter
}

func (h *toolDisplayHandler) OnContent(content string) {
	if h.printer != nil && content != "" {
		h.printer.Write(content)
	}
}

func (h *toolDisplayHandler) OnReasoning(string)        {}
func (h *toolDisplayHandler) OnDone(*core.ChatResponse) {}
func (h *toolDisplayHandler) OnError(err error) {
	if err != nil {
		fmt.Println()
		fmt.Printf("%s %v\n", mdterm.AnsiStyle("Error:", mdterm.AnsiRed), err)
	}
}

// compile-time interface checks
var (
	_ core.Provider      = (*Provider)(nil)
	_ core.StreamHandler = (*toolDisplayHandler)(nil)
)

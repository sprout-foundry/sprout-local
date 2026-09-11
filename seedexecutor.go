package main

// ---------------------------------------------------------------------------
// seedexecutor.go — the chatllm tool set as a seed core.ToolExecutor.
//
// seed hands us structured ToolCalls (arguments as JSON); we execute
// against the static registry in tools.go. The run_command y/N gate goes
// through seed's UI.Confirm so the confirmation flows through the same
// interface as every other prompt (REPL today, web UI later).
// ---------------------------------------------------------------------------

import (
	"context"
	"encoding/json"
	"fmt"
	"strings"

	"github.com/sprout-foundry/seed/core"
)

// toolExecutor executes the local registry via seed's ToolExecutor iface.
type toolExecutor struct {
	ui core.UI
}

func newToolExecutor(ui core.UI) *toolExecutor {
	if ui == nil {
		ui = core.NoopUI
	}
	return &toolExecutor{ui: ui}
}

// GetTools declares the registry to seed (which passes the list to the
// provider on every request): built-ins plus any user skills.
func (e *toolExecutor) GetTools() []core.Tool {
	specs := activeToolSpecs()
	out := make([]core.Tool, 0, len(specs))
	for _, t := range specs {
		params := map[string]interface{}{
			"type":       "object",
			"properties": map[string]interface{}{},
			"required":   []string{},
		}
		props := map[string]interface{}{}
		required := []string{}
		for _, p := range t.parameters {
			props[p.name] = map[string]string{"type": p.typeName, "description": p.description}
			if p.required {
				required = append(required, p.name)
			}
		}
		params["properties"] = props
		params["required"] = required
		out = append(out, core.Tool{
			Type: "function",
			Function: core.ToolFunction{
				Name:        t.name,
				Description: t.description,
				Parameters:  params,
			},
		})
	}
	return out
}

// Execute runs tool calls sequentially and returns one result message per
// call, in order. Tool-level failures become error-status tool messages
// (visible to the model), never Go errors — seed's retry machinery is for
// transport failures, not "file not found".
func (e *toolExecutor) Execute(ctx context.Context, calls []core.ToolCall) []core.Message {
	out := make([]core.Message, 0, len(calls))
	for _, call := range calls {
		var args map[string]string
		if err := json.Unmarshal([]byte(call.Function.Arguments), &args); err != nil {
			out = append(out, e.result(call, "error: malformed arguments: "+err.Error(), core.ToolStatusError))
			continue
		}
		result, err := e.run(ctx, call.Function.Name, args)
		if err != nil {
			out = append(out, e.result(call, "error: "+err.Error(), core.ToolStatusError))
			continue
		}
		out = append(out, e.result(call, result, core.ToolStatusCompleted))
	}
	return out
}

// run dispatches one call by name: built-ins first, then skills.
// run_command consults UI.Confirm unless the yolo bypass is set; skills
// are pinned command lines the user installed, so they run without an
// extra prompt (installing the skill was the consent).
func (e *toolExecutor) run(ctx context.Context, name string, args map[string]string) (string, error) {
	// Reserved provider marker: the call was recovered from an
	// unterminated <tool_call> (token budget ran out mid-parameter).
	truncated := args["_truncated"] == "1"
	delete(args, "_truncated")

	if spec := lookupToolSpec(name); spec != nil {
		res, err := e.runSpec(ctx, spec, args, name)
		if err == nil && truncated && (name == "write_file" || name == "read_file") {
			res += "\n(note: recovered from a truncated tool call — the content may be incomplete; offer to continue it)"
		}
		return res, err
	}
	for _, s := range skillRegistry {
		if s.Name == name {
			return runSkillCommand(ctx, s)
		}
	}
	return "", fmt.Errorf("unknown tool %q", name)
}

// runSpec validates parameters and runs a built-in tool spec.
func (e *toolExecutor) runSpec(ctx context.Context, spec *toolSpec, args map[string]string, name string) (string, error) {
	for _, p := range spec.parameters {
		if p.required && strings.TrimSpace(args[p.name]) == "" {
			return "", fmt.Errorf("missing required parameter %q", p.name)
		}
	}
	if name == "run_command" && !toolSafetyBypass {
		cmdline := strings.TrimSpace(args["command"])
		ok, err := e.ui.Confirm("run command: " + cmdline + " — allow? [y/N]")
		if err != nil {
			return "", err
		}
		if !ok {
			return "run_command is not available in the web UI. Do not retry it — continue without it: answer from what you know, or use another tool like read_file.", nil
		}
	}
	return spec.run(ctx, args)
}

// result wraps a tool outcome as a seed tool message.
func (e *toolExecutor) result(call core.ToolCall, content string, status string) core.Message {
	if call.ID == "" {
		call.ID = "call"
	}
	return core.Message{
		Role:       "tool",
		Content:    content,
		ToolCallID: call.ID,
		Meta:       map[string]string{"tool_name": call.Function.Name, "status": status},
	}
}

// compile-time interface check
var _ core.ToolExecutor = (*toolExecutor)(nil)

package main

// replEvents renders seed agent events as REPL status lines:
//
//   tool → read_file            (on tool_start)
//   ← result: the secret is…    (on tool_end)
//
// Everything else (query lifecycle, metrics, compaction) stays silent —
// the streamed answer is the display, not the event log.

import (
	"fmt"

	"github.com/sprout-foundry/seed/core"

	"github.com/sprout-foundry/sprout-local/internal/mdterm"
)

type replEvents struct{}

// Publish implements core.EventPublisher (any Publish(string, any)).
func (e *replEvents) Publish(eventType string, data interface{}) {
	switch eventType {
	case core.EventTypeToolStart:
		name := eventString(data, "tool_name")
		fmt.Printf("%s %s\n", mdterm.AnsiStyle("tool →", mdterm.AnsiCyan), mdterm.AnsiStyle(name, mdterm.AnsiBold))
	case core.EventTypeToolEnd:
		if eventString(data, "status") == core.ToolStatusError {
			fmt.Printf("%s %s: %s\n", mdterm.AnsiStyle("← error:", mdterm.AnsiRed),
				eventString(data, "tool_name"),
				firstLine(eventString(data, "result")))
			return
		}
		fmt.Printf("%s %s\n", mdterm.AnsiStyle("← result:", mdterm.AnsiCyan),
			truncateResultForDisplay(eventString(data, "result")))
	}
}

// eventString extracts a string field from event payload maps.
func eventString(data interface{}, key string) string {
	if m, ok := data.(map[string]interface{}); ok {
		if v, ok := m[key].(string); ok {
			return v
		}
	}
	return ""
}

// eventJSON extracts an arbitrary JSON field (e.g. tool arguments) from
// event payload maps, or nil when absent.
func eventJSON(data interface{}, key string) interface{} {
	if m, ok := data.(map[string]interface{}); ok {
		return m[key]
	}
	return nil
}

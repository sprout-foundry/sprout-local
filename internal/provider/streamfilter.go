package provider

// Streaming display filter: shared by the provider (web + REPL paths get
// it via ChatStream) so tool-call markup never reaches the user's screen
// while seed still sees the full text for parsing.

import (
	"strings"

	"github.com/sprout-foundry/sprout-local/internal/mdterm"
)

// toolStreamFilter suppresses tool-call markup from the streamed display
// while the raw text still accumulates for parsing. Without tools enabled
// it is a pass-through. Clean deltas go to both the terminal printer and
// (optionally) seed's stream handler, so the agent's content buffer
// matches what the user saw. Implementation: hold back a tag-sized tail of
// the stream, watch for an opening tag (<tool_call> for the qwen protocol,
// <function name=" for MiniCPM5); once seen, go silent until the matching
// close has passed. Text after the block flows again normally.
type toolStreamFilter struct {
	printer *mdterm.StreamPrinter // terminal display (may be nil)
	onDelta func(string)          // seed stream handler feed (may be nil)
	tools   bool
	tail    string // held-back characters not yet emitted
	inCall  bool   // inside a tool-call block
	// protocol selects the markup pair ("qwen" default, "minicpm5").
	protocol string
}

const toolCallTag = "<tool_call>"
const toolCallTagEnd = "</tool_call>"
const miniCPM5CallTag = "<function name=\""
const miniCPM5CallTagEnd = "</function>"
const streamHoldback = len(toolCallTag) + 8 // tag + slack for split spans

// streamTags returns the (open, close) markup pair for the active protocol.
func (f *toolStreamFilter) streamTags() (string, string) {
	if f.protocol == "minicpm5" {
		return miniCPM5CallTag, miniCPM5CallTagEnd
	}
	return toolCallTag, toolCallTagEnd
}

// emit fans a clean delta out to the terminal and the agent's buffer.
func (f *toolStreamFilter) emit(s string) {
	if s == "" {
		return
	}
	if f.printer != nil {
		f.printer.Write(s)
	}
	if f.onDelta != nil {
		f.onDelta(s)
	}
}

func (f *toolStreamFilter) write(delta string) {
	if !f.tools {
		f.emit(delta)
		return
	}
	openTag, closeTag := f.streamTags()
	f.tail += delta
	for {
		if f.inCall {
			end := strings.Index(f.tail, closeTag)
			if end < 0 {
				// Keep only a holdback in case the closing tag is split.
				if len(f.tail) > streamHoldback {
					f.tail = f.tail[len(f.tail)-streamHoldback:]
				}
				return
			}
			f.tail = f.tail[end+len(closeTag):]
			f.inCall = false
			continue
		}
		start := strings.Index(f.tail, openTag)
		if start < 0 {
			// No opening tag in the buffer: emit all but the holdback.
			if len(f.tail) > streamHoldback {
				clean := f.tail[:len(f.tail)-streamHoldback]
				f.tail = f.tail[len(f.tail)-streamHoldback:]
				f.emit(clean)
			}
			return
		}
		if start > 0 {
			f.emit(f.tail[:start])
		}
		f.tail = f.tail[start+len(openTag):]
		f.inCall = true
	}
}

// close flushes the held-back tail. If a call block is still open, only
// raw call markup is dropped: any prose in the tail (the model's words
// before it spiraled) still reaches the display, minus a partial "<…"
// tag fragment.
func (f *toolStreamFilter) close() {
	if !f.tools {
		return
	}
	if f.inCall {
		// Inside <tool_call>…: drop markup, but the model may have written
		// prose before the block opened — that already streamed. The tail
		// here is call markup (possibly unterminated); show nothing.
		return
	}
	if f.tail != "" {
		f.emit(f.tail)
		f.tail = ""
	}
}

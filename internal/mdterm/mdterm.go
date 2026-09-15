package mdterm

// ---------------------------------------------------------------------------
// Terminal markdown rendering. REPL and one-shot output streams through
// StreamPrinter, which styles each line as it completes: headings, lists,
// block quotes, horizontal rules, fenced code blocks and basic inline
// spans (bold, italic, code, links, strikethrough) via ANSI escapes.
//
// Rendering is line-buffered so styles are applied only to complete lines
// (no half-applied `**` spans mid-stream). Fenced blocks are tracked so
// their contents are never inline-transformed.
//
// Non-TTY stdout (pipes, redirects), NO_COLOR and TERM=dumb bypass the
// renderer entirely — raw deltas, same as before. Pipe consumers of
// -transcript never see escapes.
// ---------------------------------------------------------------------------

import (
	"fmt"
	"io"
	"os"
	"regexp"
	"strconv"
	"strings"
)

// ANSI SGR escapes used for terminal markdown styling. Bright/basic
// colors only (no 256-color or truecolor) for maximum portability.
const (
	AnsiReset   = "\x1b[0m"
	AnsiBold    = "\x1b[1m"
	AnsiItalic  = "\x1b[3m"
	AnsiUnder   = "\x1b[4m"
	AnsiStrike  = "\x1b[9m"
	AnsiRed     = "\x1b[31m"
	AnsiGreen   = "\x1b[32m"
	AnsiYellow  = "\x1b[33m"
	AnsiMagenta = "\x1b[35m"
	AnsiCyan    = "\x1b[36m"
	AnsiGray    = "\x1b[90m"
	AnsiBlue    = "\x1b[94m"
)

// AnsiStyle wraps text in the given SGR codes with a single reset.
func AnsiStyle(text string, codes ...string) string {
	if len(codes) == 0 {
		return text
	}
	return strings.Join(codes, "") + text + AnsiReset
}

// Mode selects how a StreamPrinter renders its output.
//
// Styled: stdout is an interactive terminal — render markdown.
// Raw: pipe, file redirect, NO_COLOR or TERM=dumb — stream untouched.
type Mode int

const (
	Raw Mode = iota
	Styled
)

// detectMDMode picks the output mode from a stdout file.
func detectMDMode(f *os.File) Mode {
	st, err := f.Stat()
	if err != nil || st.Mode()&os.ModeCharDevice == 0 {
		return Raw // pipe or file: consumers parse text, not visuals
	}
	if os.Getenv("NO_COLOR") != "" || os.Getenv("TERM") == "dumb" {
		return Raw
	}
	return Styled
}

// StreamPrinter renders markdown to w line by line as it streams.
type StreamPrinter struct {
	w       io.Writer
	mode    Mode
	inFence bool
	pending string // partial line not yet terminated by \n
}

// NewStreamPrinter builds a printer with an explicit mode (tests).
func NewStreamPrinter(w io.Writer, mode Mode) *StreamPrinter {
	return &StreamPrinter{w: w, mode: mode}
}

// NewTermPrinter builds a printer for stdout, detecting the mode.
func NewTermPrinter(f *os.File) *StreamPrinter {
	return &StreamPrinter{w: f, mode: detectMDMode(f)}
}

// StdoutPrinter is the process-wide output printer for streamed responses.
func StdoutPrinter() *StreamPrinter { return NewTermPrinter(os.Stdout) }

// WriteDelta adapts the printer to an onToken-style callback.
func (p *StreamPrinter) WriteDelta(delta string) { p.Write(delta) }

// Write buffers a delta and renders every line it completes. In raw mode
// deltas pass straight through.
func (p *StreamPrinter) Write(delta string) {
	if p.mode == Raw {
		fmt.Fprint(p.w, delta)
		return
	}
	p.pending += delta
	for {
		i := strings.IndexByte(p.pending, '\n')
		if i < 0 {
			return
		}
		line := p.pending[:i]
		p.pending = p.pending[i+1:]
		fmt.Fprint(p.w, p.render(line), "\n")
	}
}

// Close flushes a trailing partial line (responses need not end in \n).
func (p *StreamPrinter) Close() {
	if p.mode == Raw || p.pending == "" {
		return
	}
	fmt.Fprint(p.w, p.render(p.pending))
	p.pending = ""
}

// render converts one complete markdown line to terminal text.
func (p *StreamPrinter) render(line string) string {
	if p.mode == Raw {
		return line
	}
	trimmed := strings.TrimSpace(line)

	// Fenced code blocks: dim markers, indent contents, never inline-style.
	if strings.HasPrefix(trimmed, "```") {
		p.inFence = !p.inFence
		return AnsiStyle(trimmed, AnsiGray)
	}
	if p.inFence {
		return "  " + highlightCode(line)
	}
	return renderBlock(line)
}

var (
	mdHeadingRe = regexp.MustCompile(`^(#{1,6})\s+(.+)$`)
	mdItemRe    = regexp.MustCompile(`^(\s*)(?:([-*+])|(\d+)[.)])\s+(.+)$`)
	mdQuoteRe   = regexp.MustCompile(`^\s*>\s?(.*)$`)
	mdRuleRe    = regexp.MustCompile(`^\s*(?:-{3,}|\*{3,}|_{3,})\s*$`)
)

// renderBlock styles one line outside fenced code: headings, list items,
// quotes and rules become simple terminal conventions; anything else goes
// through inline span styling.
func renderBlock(line string) string {
	if mdRuleRe.MatchString(line) {
		return AnsiStyle(strings.Repeat("─", 40), AnsiGray)
	}
	if m := mdHeadingRe.FindStringSubmatch(line); m != nil {
		return AnsiStyle(strings.TrimSpace(m[2]), AnsiBold, AnsiBlue)
	}
	if m := mdItemRe.FindStringSubmatch(line); m != nil {
		label := m[2]
		if label == "" {
			label = m[3] + ". "
		} else {
			label = "• " // bullets: normalize - * + to •
		}
		return m[1] + AnsiStyle(label, AnsiMagenta) + renderInline(m[4])
	}
	if m := mdQuoteRe.FindStringSubmatch(line); m != nil {
		return AnsiStyle("▌ ", AnsiMagenta) + AnsiStyle(renderInline(m[1]), AnsiItalic)
	}
	return renderInline(line)
}

var (
	mdInlineCodeRe = regexp.MustCompile("`([^`\n]+)`")
	mdBoldRe       = regexp.MustCompile(`\*\*([^*\n]+)\*\*|__([^_\n]+)__`)
	mdItalicRe     = regexp.MustCompile(`\*([^*\n]+)\*`)
	mdStrikeRe     = regexp.MustCompile(`~~([^~\n]+)~~`)
	mdLinkRe       = regexp.MustCompile(`\[([^\]\n\x1b]+)\]\((https?://[^)\s]+)\)`)
	mdCodePhRe     = regexp.MustCompile("\x00([0-9]+)\x00")
)

// Fenced-code token rules: strings first (a quote-delimited run is atomic,
// so "//" or keywords inside it are data), then comments, then keywords,
// then numbers. Lifted tokens become unary control-byte placeholders
// (\x1e + n×\x1f + \x1e) that no later rule can match — digits or letters
// in placeholders would otherwise be re-tinted by the number/keyword rules.
var codeTokenRules = []struct {
	re  *regexp.Regexp
	col string
}{
	{regexp.MustCompile(`"(?:[^"\\]|\\.)*"|'(?:[^'\\]|\\.)*'|` + "`" + `[^` + "`" + `]*` + "`"), AnsiGreen},
	{regexp.MustCompile(`(?://[^\n]*|#[^\n]*|/\*.*?\*/)`), AnsiGray},
	{regexp.MustCompile(`\b(?:func|return|if|else|for|range|while|import|package|class|def|const|let|var|type|struct|interface|switch|case|default|break|continue|go|defer|new|public|private|static|void|int|string|bool|float|true|false|nil|null|none|True|False|None)\b`), AnsiBlue},
	{regexp.MustCompile(`\b\d+(?:\.\d+)?\b`), AnsiYellow},
}

// phEncode returns the placeholder for lifted-token index n.
func phEncode(n int) string { return "\x1e" + strings.Repeat("\x1f", n+1) + "\x1e" }

var phRe = regexp.MustCompile("\x1e([\x1f]+)\x1e")

// highlightCode tints one fenced-code line.
func highlightCode(line string) string {
	var lifted []string
	keep := func(styled string) string {
		lifted = append(lifted, styled)
		return phEncode(len(lifted) - 1)
	}
	for _, rule := range codeTokenRules {
		line = rule.re.ReplaceAllStringFunc(line, func(m string) string {
			return keep(AnsiStyle(m, rule.col))
		})
	}
	return phRe.ReplaceAllStringFunc(line, func(m string) string {
		i := strings.Count(phRe.FindStringSubmatch(m)[1], "\x1f") - 1
		if i < 0 || i >= len(lifted) {
			return m
		}
		return lifted[i]
	})
}

// renderInline styles inline spans. Links and inline code are lifted out
// first, while the text is pristine: link matching must never see the
// escapes other passes emit (a reset's "[" can pair with a literal "]"
// elsewhere on the line), and code contents must not be reinterpreted.
// Lifted spans come back in place of \x00N\x00 placeholders at the end.
func renderInline(s string) string {
	var lifted []string
	keep := func(styled string) string {
		lifted = append(lifted, styled)
		return "\x00" + strconv.Itoa(len(lifted)-1) + "\x00"
	}
	s = mdLinkRe.ReplaceAllStringFunc(s, func(m string) string {
		sub := mdLinkRe.FindStringSubmatch(m)
		return keep(AnsiStyle(sub[1], AnsiUnder) + AnsiStyle(" ("+sub[2]+")", AnsiGray))
	})
	s = mdInlineCodeRe.ReplaceAllStringFunc(s, func(m string) string {
		return keep(AnsiStyle(strings.Trim(m, "`"), AnsiCyan))
	})
	s = mdBoldRe.ReplaceAllStringFunc(s, func(m string) string {
		sub := mdBoldRe.FindStringSubmatch(m)
		if sub[1] == "" {
			return AnsiStyle(sub[2], AnsiBold)
		}
		return AnsiStyle(sub[1], AnsiBold)
	})
	s = mdStrikeRe.ReplaceAllStringFunc(s, func(m string) string {
		return AnsiStyle(mdStrikeRe.FindStringSubmatch(m)[1], AnsiStrike)
	})
	s = mdItalicRe.ReplaceAllStringFunc(s, func(m string) string {
		return AnsiStyle(mdItalicRe.FindStringSubmatch(m)[1], AnsiItalic)
	})
	return mdCodePhRe.ReplaceAllStringFunc(s, func(m string) string {
		i, err := strconv.Atoi(mdCodePhRe.FindStringSubmatch(m)[1])
		if err != nil || i >= len(lifted) {
			return m
		}
		return lifted[i]
	})
}

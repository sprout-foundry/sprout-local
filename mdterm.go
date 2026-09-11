package main

// ---------------------------------------------------------------------------
// Terminal markdown rendering. REPL and one-shot output streams through
// streamPrinter, which styles each line as it completes: headings, lists,
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
	ansiReset   = "\x1b[0m"
	ansiBold    = "\x1b[1m"
	ansiItalic  = "\x1b[3m"
	ansiUnder   = "\x1b[4m"
	ansiStrike  = "\x1b[9m"
	ansiRed     = "\x1b[31m"
	ansiGreen   = "\x1b[32m"
	ansiYellow  = "\x1b[33m"
	ansiMagenta = "\x1b[35m"
	ansiCyan    = "\x1b[36m"
	ansiGray    = "\x1b[90m"
	ansiBlue    = "\x1b[94m"
)

// ansiStyle wraps text in the given SGR codes with a single reset.
func ansiStyle(text string, codes ...string) string {
	if len(codes) == 0 {
		return text
	}
	return strings.Join(codes, "") + text + ansiReset
}

// mdStyled: stdout is an interactive terminal — render markdown.
// mdRaw: pipe, file redirect, NO_COLOR or TERM=dumb — stream untouched.
type mdMode int

const (
	mdRaw mdMode = iota
	mdStyled
)

// detectMDMode picks the output mode from a stdout file.
func detectMDMode(f *os.File) mdMode {
	st, err := f.Stat()
	if err != nil || st.Mode()&os.ModeCharDevice == 0 {
		return mdRaw // pipe or file: consumers parse text, not visuals
	}
	if os.Getenv("NO_COLOR") != "" || os.Getenv("TERM") == "dumb" {
		return mdRaw
	}
	return mdStyled
}

// streamPrinter renders markdown to w line by line as it streams.
type streamPrinter struct {
	w       io.Writer
	mode    mdMode
	inFence bool
	pending string // partial line not yet terminated by \n
}

// newStreamPrinter builds a printer with an explicit mode (tests).
func newStreamPrinter(w io.Writer, mode mdMode) *streamPrinter {
	return &streamPrinter{w: w, mode: mode}
}

// newTermPrinter builds a printer for stdout, detecting the mode.
func newTermPrinter(f *os.File) *streamPrinter {
	return &streamPrinter{w: f, mode: detectMDMode(f)}
}

// stdoutPrinter is the process-wide output printer for streamed responses.
func stdoutPrinter() *streamPrinter { return newTermPrinter(os.Stdout) }

// writeDelta adapts the printer to an onToken-style callback.
func (p *streamPrinter) writeDelta(delta string) { p.Write(delta) }

// Write buffers a delta and renders every line it completes. In raw mode
// deltas pass straight through.
func (p *streamPrinter) Write(delta string) {
	if p.mode == mdRaw {
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
func (p *streamPrinter) Close() {
	if p.mode == mdRaw || p.pending == "" {
		return
	}
	fmt.Fprint(p.w, p.render(p.pending))
	p.pending = ""
}

// render converts one complete markdown line to terminal text.
func (p *streamPrinter) render(line string) string {
	if p.mode == mdRaw {
		return line
	}
	trimmed := strings.TrimSpace(line)

	// Fenced code blocks: dim markers, indent contents, never inline-style.
	if strings.HasPrefix(trimmed, "```") {
		p.inFence = !p.inFence
		return ansiStyle(trimmed, ansiGray)
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
		return ansiStyle(strings.Repeat("─", 40), ansiGray)
	}
	if m := mdHeadingRe.FindStringSubmatch(line); m != nil {
		return ansiStyle(strings.TrimSpace(m[2]), ansiBold, ansiBlue)
	}
	if m := mdItemRe.FindStringSubmatch(line); m != nil {
		label := m[2]
		if label == "" {
			label = m[3] + ". "
		} else {
			label = "• " // bullets: normalize - * + to •
		}
		return m[1] + ansiStyle(label, ansiMagenta) + renderInline(m[4])
	}
	if m := mdQuoteRe.FindStringSubmatch(line); m != nil {
		return ansiStyle("▌ ", ansiMagenta) + ansiStyle(renderInline(m[1]), ansiItalic)
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
	{regexp.MustCompile(`"(?:[^"\\]|\\.)*"|'(?:[^'\\]|\\.)*'|` + "`" + `[^` + "`" + `]*` + "`"), ansiGreen},
	{regexp.MustCompile(`(?://[^\n]*|#[^\n]*|/\*.*?\*/)`), ansiGray},
	{regexp.MustCompile(`\b(?:func|return|if|else|for|range|while|import|package|class|def|const|let|var|type|struct|interface|switch|case|default|break|continue|go|defer|new|public|private|static|void|int|string|bool|float|true|false|nil|null|none|True|False|None)\b`), ansiBlue},
	{regexp.MustCompile(`\b\d+(?:\.\d+)?\b`), ansiYellow},
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
			return keep(ansiStyle(m, rule.col))
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
		return keep(ansiStyle(sub[1], ansiUnder) + ansiStyle(" ("+sub[2]+")", ansiGray))
	})
	s = mdInlineCodeRe.ReplaceAllStringFunc(s, func(m string) string {
		return keep(ansiStyle(strings.Trim(m, "`"), ansiCyan))
	})
	s = mdBoldRe.ReplaceAllStringFunc(s, func(m string) string {
		sub := mdBoldRe.FindStringSubmatch(m)
		if sub[1] == "" {
			return ansiStyle(sub[2], ansiBold)
		}
		return ansiStyle(sub[1], ansiBold)
	})
	s = mdStrikeRe.ReplaceAllStringFunc(s, func(m string) string {
		return ansiStyle(mdStrikeRe.FindStringSubmatch(m)[1], ansiStrike)
	})
	s = mdItalicRe.ReplaceAllStringFunc(s, func(m string) string {
		return ansiStyle(mdItalicRe.FindStringSubmatch(m)[1], ansiItalic)
	})
	return mdCodePhRe.ReplaceAllStringFunc(s, func(m string) string {
		i, err := strconv.Atoi(mdCodePhRe.FindStringSubmatch(m)[1])
		if err != nil || i >= len(lifted) {
			return m
		}
		return lifted[i]
	})
}

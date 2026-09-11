package main

import (
	"bytes"
	"os"
	"regexp"
	"strings"
	"testing"
)

var ansiRe = regexp.MustCompile("\x1b\\[[0-9;]*m")

func stripANSI(s string) string { return ansiRe.ReplaceAllString(s, "") }

func TestDetectMDModeFileIsRaw(t *testing.T) {
	f, err := os.CreateTemp(t.TempDir(), "out")
	if err != nil {
		t.Fatal(err)
	}
	defer f.Close()
	// Regular files are not character devices → pipes/redirects stay raw.
	if m := detectMDMode(f); m != mdRaw {
		t.Errorf("detectMDMode(file) = %d, want mdRaw", m)
	}
}

func TestRawModePassesThrough(t *testing.T) {
	var buf bytes.Buffer
	p := newStreamPrinter(&buf, mdRaw)
	p.Write("# head **bold**\nsecond")
	if buf.String() != "# head **bold**\nsecond" {
		t.Errorf("raw mode mangled output: %q", buf.String())
	}
	if strings.Contains(buf.String(), "\x1b[") {
		t.Error("raw mode emitted ANSI escapes")
	}
}

func TestStyledBlocks(t *testing.T) {
	var buf bytes.Buffer
	p := newStreamPrinter(&buf, mdStyled)
	p.Write("# Title\n- item one\n> quoted\n---\nplain **bold**\n")

	want := "Title\n• item one\n▌ quoted\n" + strings.Repeat("─", 40) + "\nplain bold\n"
	if got := stripANSI(buf.String()); got != want {
		t.Errorf("stripped output:\n got %q\nwant %q", got, want)
	}
	for _, seq := range []string{ansiBold, ansiBlue, ansiItalic, ansiMagenta, ansiGray} {
		if !strings.Contains(buf.String(), seq) {
			t.Errorf("expected escape %q missing from %q", seq, buf.String())
		}
	}
	if strings.Contains(buf.String(), "# Title") {
		t.Error("heading marker leaked into styled output")
	}
}

func TestOrderedLists(t *testing.T) {
	var buf bytes.Buffer
	p := newStreamPrinter(&buf, mdStyled)
	p.Write("1. first\n2) second\nplain\n")
	want := "1. first\n2. second\nplain\n"
	if got := stripANSI(buf.String()); got != want {
		t.Errorf("stripped output:\n got %q\nwant %q", got, want)
	}
}

func TestStreamingAcrossChunks(t *testing.T) {
	var buf bytes.Buffer
	p := newStreamPrinter(&buf, mdStyled)
	for _, chunk := range []string{"# Ti", "tle\nhel", "lo **wor", "ld**\n"} {
		p.Write(chunk)
	}
	p.Close()
	stripped := stripANSI(buf.String())
	if !strings.Contains(stripped, "Title\n") {
		t.Errorf("heading split across chunks lost: %q", stripped)
	}
	if !strings.Contains(buf.String(), ansiBold+"world"+ansiReset) {
		t.Errorf("bold split across chunks lost: %q", buf.String())
	}
}

func TestCloseFlushesPartialLine(t *testing.T) {
	var buf bytes.Buffer
	p := newStreamPrinter(&buf, mdStyled)
	p.Write("no trailing newline")
	p.Close()
	if stripANSI(buf.String()) != "no trailing newline" {
		t.Errorf("Close lost pending line: %q", buf.String())
	}
}

func TestFenceBlocksNotInlineStyled(t *testing.T) {
	var buf bytes.Buffer
	p := newStreamPrinter(&buf, mdStyled)
	p.Write("```go\nx := **not** bold\n```\nafter **bold**\n")

	if n := strings.Count(stripANSI(buf.String()), "```"); n != 2 {
		t.Errorf("fence markers: got %d, want 2: %q", n, stripANSI(buf.String()))
	}
	if !strings.Contains(stripANSI(buf.String()), "  x := **not** bold") {
		t.Errorf("code line not indented: %q", stripANSI(buf.String()))
	}
	// Exactly one bold span: the "after" line, never the fenced contents.
	if n := strings.Count(buf.String(), ansiBold); n != 1 {
		t.Errorf("got %d bold spans, want 1 (fence contents must stay plain): %q", n, buf.String())
	}
}

func TestInlineCodeProtectedFromEmphasis(t *testing.T) {
	var buf bytes.Buffer
	p := newStreamPrinter(&buf, mdStyled)
	p.Write("use `**x**` here\n")
	if strings.Contains(buf.String(), ansiBold) {
		t.Errorf("inline code contents were bolded: %q", buf.String())
	}
	if !strings.Contains(buf.String(), ansiCyan) {
		t.Errorf("inline code not styled: %q", buf.String())
	}
}

func TestInlineSpans(t *testing.T) {
	var buf bytes.Buffer
	p := newStreamPrinter(&buf, mdStyled)
	p.Write("*soft* ~~gone~~ see [t](https://e.co)\n")
	want := "soft gone see t (https://e.co)\n"
	if got := stripANSI(buf.String()); got != want {
		t.Errorf("stripped output:\n got %q\nwant %q", got, want)
	}
	for _, seq := range []string{ansiItalic, ansiStrike, ansiUnder} {
		if !strings.Contains(buf.String(), seq) {
			t.Errorf("expected escape %q missing from %q", seq, buf.String())
		}
	}
}

func TestHighlightCode(t *testing.T) {
	got := stripANSI(highlightCode(`x := 42 // answer`))
	if got != "x := 42 // answer" {
		t.Errorf("highlightCode changed text: %q", got)
	}
	for name, seq := range map[string]string{
		"number":  ansiYellow,
		"comment": ansiGray,
		"keyword": ansiBlue,
	} {
		if !strings.Contains(highlightCode(sampleFor(name)), seq) {
			t.Errorf("%s token not tinted in %q", name, sampleFor(name))
		}
	}
	// A string containing "//" must not yield a comment tint inside it.
	s := highlightCode(`s := "http://x"`)
	if strings.Count(s, ansiGray) != 0 {
		t.Errorf("comment tint inside string: %q", s)
	}
	if !strings.Contains(s, ansiGreen) {
		t.Errorf("string not tinted: %q", s)
	}
}

func sampleFor(name string) string {
	switch name {
	case "number":
		return "x := 42"
	case "comment":
		return "// note"
	case "keyword":
		return "func f()"
	}
	return ""
}

package main

import (
	"strings"
	"testing"
)

// Repetition inside <tool_call> parameters is legitimate file content and
// must never be flagged; plain-prose spirals must be.
func TestSpiralDetection(t *testing.T) {
	css := strings.Repeat("        color: #333;\n        margin-bottom: 10px;\n", 120)
	goCode := strings.Repeat("if err := os.MkdirAll(dir, 0755); err != nil {\n\treturn err\n}\n", 80)

	// Tool-wrapped repetitive file content: NOT a spiral.
	toolWrapped := "Building your site now.\n\n<tool_call>\n<function=write_file>\n<parameter=content>\n" +
		css + "\n</parameter>\n</function>\n</tool_call>\nDone!"
	if isSpiral(toolWrapped) {
		t.Error("tool-wrapped CSS flagged as spiral")
	}
	if isSpiral("Sure.\n<tool_call>\n<function=write_file>\n<parameter=content>\n" + goCode + "</content>\n</function>\n</tool_call>") {
		t.Error("tool-wrapped Go flagged as spiral")
	}

	// Plain-prose spirals (the observed failure mode): flagged.
	spiral := strings.Repeat("Show me where they make people more dependent and less independent. ", 40)
	if !isSpiral(spiral) {
		t.Error("plain spiral not detected")
	}
	frag := strings.Repeat("over and over the same thing repeats\n", 40)
	if !isSpiral(frag) {
		t.Error("phrase loop not detected")
	}

	// Legit prose with a repeated refrain (refrain does not dominate).
	refrain := strings.Repeat("The ocean covers more than 70% of our planet's surface. ", 10) +
		strings.Repeat("And that is why the ocean matters. ", 3)
	if isSpiral(refrain) {
		t.Error("refrain prose flagged as spiral")
	}
	// Short text: below the minimum sample.
	if isSpiral("hello world hello world") {
		t.Error("tiny text flagged")
	}
}

// Leaked control tokens cancel mid-stream (unambiguous failure).
func TestLeakCancels(t *testing.T) {
	for _, s := range []string{"some text <|endoftext|> more", "x<|im_start|>user"} {
		g := &genGuard{}
		if !g.feed(s) {
			t.Errorf("leak not caught: %q", s)
		}
	}
}

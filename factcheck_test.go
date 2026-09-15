package main

import "testing"

// TestParseYesNo pins the fact-check verdict parsing (factcheck.go):
// confident No fails the check, anything else passes.
func TestParseYesNo(t *testing.T) {
	tests := []struct {
		in      string
		verdict bool
	}{
		{"Yes", true},
		{"yes.", true},
		{"YES", true},
		{"No", false},
		{"NO — the answer is wrong", false},
		{"  yes  ", true},
		{"maybe", true}, // inconclusive counts as pass
		{"", true},
		{"The answer is No.", false},
	}
	for _, tt := range tests {
		v, ok := parseYesNo(tt.in)
		if v != tt.verdict {
			t.Errorf("parseYesNo(%q) = %v (ok=%v), want %v", tt.in, v, ok, tt.verdict)
		}
	}
}

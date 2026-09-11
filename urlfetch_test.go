package main

import (
	"context"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"
)

func TestExtractPromptURLs(t *testing.T) {
	tests := []struct {
		name   string
		prompt string
		want   []string
	}{
		{"none", "just a question", nil},
		{"simple", "summarize #https://example.com/page please",
			[]string{"https://example.com/page"}},
		{"trailing punctuation", "see #https://example.com/a.",
			[]string{"https://example.com/a"}},
		{"dedupe", "#https://a.com #https://a.com #https://b.com",
			[]string{"https://a.com", "https://b.com"}},
		{"fenced code ignored", "```\ncurl #https://x.com\n```", nil},
		{"inline code ignored", "run `#https://x.com` now", nil},
		{"mixed", "#https://a.com and `#https://x.com` plus #https://c.com",
			[]string{"https://a.com", "https://c.com"}},
		{"bare url ignored", "https://example.com without hash", nil},
	}
	for _, tt := range tests {
		got := extractPromptURLs(tt.prompt)
		if len(got) != len(tt.want) {
			t.Errorf("%s: got %v, want %v", tt.name, got, tt.want)
			continue
		}
		for i := range got {
			if got[i] != tt.want[i] {
				t.Errorf("%s: got %v, want %v", tt.name, got, tt.want)
			}
		}
	}
}

func TestHTMLToText(t *testing.T) {
	in := `<html><head><style>p{color:red}</style><title>T</title></head>
	<body><script>evil()</script><h1>Hello</h1><p>World &amp; more</p>
	<div>Second<br>line</div><!-- hidden --></body></html>`
	got := htmlToText(in)
	for _, want := range []string{"Hello", "World & more", "Second", "line"} {
		if !contains(got, want) {
			t.Errorf("htmlToText missing %q in %q", want, got)
		}
	}
	for _, banned := range []string{"evil", "color:red", "hidden", "<"} {
		if contains(got, banned) {
			t.Errorf("htmlToText should not contain %q: %q", banned, got)
		}
	}
}

func contains(s, sub string) bool {
	return strings.Contains(s, sub)
}

func TestFetchReadable(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.URL.Path != "/" {
			http.NotFound(w, r)
			return
		}
		w.Header().Set("Content-Type", "text/html")
		w.Write([]byte("<html><body><p>banana harvest</p></body></html>"))
	}))
	defer srv.Close()

	text, err := fetchReadable(context.Background(), srv.URL)
	if err != nil {
		t.Fatalf("fetchReadable: %v", err)
	}
	if !contains(text, "banana harvest") {
		t.Errorf("fetchReadable = %q, want page text", text)
	}

	// failure surfaces as an error, never a panic
	if _, err := fetchReadable(context.Background(), srv.URL+"/missing"); err == nil {
		t.Error("expected error for HTTP 404")
	}
}

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

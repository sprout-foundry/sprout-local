package urlfetch

// ---------------------------------------------------------------------------
// urlfetch.go — #url prompt enrichment for the web UI.
//
// A prompt containing "#https://…" pulls in the page's readable text so
// the model can answer questions about it (same protocol as the old
// Python server, minus the puppeteer/chromadb stack: straight HTTP fetch,
// tag stripping, and keyword-free truncation). URLs inside code blocks
// are ignored, so pasted snippets don't trigger fetches.
// ---------------------------------------------------------------------------

import (
	"context"
	"fmt"
	"html"
	"io"
	"net/http"
	"regexp"
	"strings"
	"time"
)

var (
	reFencedCode = regexp.MustCompile("(?s)```.*?```")
	reInlineCode = regexp.MustCompile("`[^`\n]+`")
	rePromptURL  = regexp.MustCompile(`#(https?://[^\s]+)`)

	// fetch caps: don't swallow huge pages into the prompt.
	maxFetchBytes int64 = 2 << 20 // 2 MiB raw body
	maxTextChars        = 8000    // readable text kept per URL
)

// ExtractPromptURLs returns deduplicated #URLs from the prompt, skipping
// anything inside ``` fenced or ` inline ` code blocks.
func ExtractPromptURLs(prompt string) []string {
	cleaned := reFencedCode.ReplaceAllString(prompt, " ")
	cleaned = reInlineCode.ReplaceAllString(cleaned, " ")
	matches := rePromptURL.FindAllStringSubmatch(cleaned, -1)
	seen := map[string]bool{}
	var urls []string
	for _, m := range matches {
		u := strings.TrimRight(m[1], ".,;:!?)]}")
		if u != "" && !seen[u] {
			seen[u] = true
			urls = append(urls, u)
		}
	}
	return urls
}

// FetchReadable downloads a URL and returns its readable text. Network
// failures degrade gracefully: the caller notes the failure and chats on
// without the page.
func FetchReadable(ctx context.Context, rawURL string) (string, error) {
	fetchCtx, cancel := context.WithTimeout(ctx, 20*time.Second)
	defer cancel()

	req, err := http.NewRequestWithContext(fetchCtx, http.MethodGet, rawURL, nil)
	if err != nil {
		return "", err
	}
	req.Header.Set("User-Agent", "Mozilla/5.0 (compatible; chatllm/1.0)")
	req.Header.Set("Accept", "text/html,application/xhtml+xml,text/plain;q=0.9,*/*;q=0.5")

	resp, err := http.DefaultClient.Do(req)
	if err != nil {
		return "", err
	}
	defer resp.Body.Close()
	if resp.StatusCode < 200 || resp.StatusCode > 299 {
		return "", fmt.Errorf("HTTP %d", resp.StatusCode)
	}

	body, err := io.ReadAll(io.LimitReader(resp.Body, maxFetchBytes))
	if err != nil {
		return "", err
	}
	text := htmlToText(string(body))
	if len(text) > maxTextChars {
		text = text[:maxTextChars]
	}
	if strings.TrimSpace(text) == "" {
		return "", fmt.Errorf("no readable text found")
	}
	return text, nil
}

var (
	// RE2 has no backreferences, so the closing tag repeats the alternation.
	reNoiseElems = regexp.MustCompile(`(?is)<(script|style|noscript|svg|head|template|iframe)\b[^>]*>.*?</(script|style|noscript|svg|head|template|iframe)>`)
	reComments   = regexp.MustCompile(`(?s)<!--.*?-->`)
	reBlockEnd   = regexp.MustCompile(`(?i)</(p|div|h[1-6]|li|tr|blockquote|section|article)>|<br\s*/?>`)
	reAnyTag     = regexp.MustCompile(`(?s)<[^>]+>`)
	reSpaces     = regexp.MustCompile(`[ \t\r\f\v]+`)
	reBlankLines = regexp.MustCompile(`\n\s*\n+`)
)

// htmlToText strips markup down to readable text: noise elements and
// comments are dropped, block ends become newlines, remaining tags become
// spaces, and entities are unescaped.
func htmlToText(body string) string {
	s := reNoiseElems.ReplaceAllString(body, " ")
	s = reComments.ReplaceAllString(s, " ")
	s = reBlockEnd.ReplaceAllString(s, "\n")
	s = reAnyTag.ReplaceAllString(s, " ")
	s = html.UnescapeString(s)
	s = reSpaces.ReplaceAllString(s, " ")
	s = reBlankLines.ReplaceAllString(s, "\n\n")
	return strings.TrimSpace(s)
}
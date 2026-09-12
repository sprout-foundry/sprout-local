// Command tpseval measures prefill + decode throughput for a model at the
// context lengths the chat workloads care about (system+tools prefix ~500,
// mid-conversation ~2K, long-document ~8K). Same method as TestTPSBenchmark
// (onToken timestamps) so numbers are comparable across models.
package main

import (
	"context"
	"flag"
	"fmt"
	"math/rand"
	"os"
	"path/filepath"
	"strings"
	"time"

	"github.com/sprout-foundry/sinter/llm"
	_ "github.com/sprout-foundry/sinter/llm/all"
)

type ctxCase struct {
	label     string
	fillWords int
}

func main() {
	models := flag.String("models", "", "comma-separated model dirs")
	flag.Parse()
	if *models == "" {
		fmt.Fprintln(os.Stderr, "usage: tpseval -models dir1,dir2")
		os.Exit(2)
	}

	cases := []ctxCase{
		{"chat-turn(~0.5K)", 2000},
		{"mid-conv(~2K)", 8000},
		{"long-doc(~8K)", 32000},
	}

	fmt.Printf("%-28s %22s %22s\n", "model", "prefill tok/s", "decode tok/s")
	for _, dir := range strings.Split(*models, ",") {
		dir = strings.TrimSpace(expand(dir))
		name := filepath.Base(dir)
		m, err := llm.NewModel(dir)
		if err != nil {
			fmt.Fprintf(os.Stderr, "load %s: %v\n", dir, err)
			continue
		}
		for _, c := range cases {
			pp, tp := bench(m, c.fillWords)
			fmt.Printf("%-28s %18.0f p tok/s %18.0f d tok/s\n", name+" "+c.label, pp, tp)
		}
		m.Close()
		fmt.Println()
	}
}

func bench(m *llm.Model, fillWords int) (prefillTPS, decodeTPS float64) {
	words := []string{"func", "return", "error", "nil", "string", "int", "struct",
		"interface", "package", "import", "context", "time", "sync", "mutex",
		"append", "len", "make", "the", "and", "of", "to", "in", "a", "is"}
	r := rand.New(rand.NewSource(42))
	var buf []byte
	buf = append(buf, "Summarize the following text in one sentence.\n\n"...)
	for i := 0; i < fillWords; i++ {
		buf = append(buf, words[r.Intn(len(words))]...)
		buf = append(buf, ' ')
	}
	buf = append(buf, "\n\nReturn ONLY the summary."...)
	prompt := string(buf)
	promptTokens := len(m.TokenizerEncode(prompt)) + 1

	cfg := llm.GenerateConfig{MaxTokens: 40, Temperature: 0, TopP: 1.0, TopK: 1, RepetitionPenalty: 0}

	var first, last time.Time
	n := 0
	start := time.Now()
	_ = m.Generate(context.Background(), prompt, cfg, func(id int) {
		now := time.Now()
		if n == 0 {
			first = now
		}
		last = now
		n++
	})
	_ = start
	prefillElapsed := first.Sub(start)
	decodeElapsed := last.Sub(first)
	if prefillElapsed > 0 {
		prefillTPS = float64(promptTokens) / prefillElapsed.Seconds()
	}
	if decodeElapsed > 0 && n > 1 {
		decodeTPS = float64(n-1) / decodeElapsed.Seconds()
	}
	return
}

func expand(p string) string {
	if strings.HasPrefix(p, "~/") {
		home, _ := os.UserHomeDir()
		return filepath.Join(home, p[2:])
	}
	return p
}

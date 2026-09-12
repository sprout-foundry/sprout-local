// Command modeleval runs a fixed quality + speed battery against local
// sinter models. Deterministic prompts, greedy decode, pass/fail + partial
// credit scoring, tok/s timing per prompt. Results are comparable across
// models because every run uses the same seeds, caps, and graders.
//
// Usage:
//
//	go run ./cmd/modeleval -model ~/dev/llm-models/minicpm5-2b-mlx [flags]
//
// Or multiple models sequentially:
//
//	go run ./cmd/modeleval -models dir1,dir2,dir3
package main

import (
	"context"
	"encoding/json"
	"flag"
	"fmt"
	"os"
	"path/filepath"
	"regexp"
	"sort"
	"strings"
	"time"

	"github.com/sprout-foundry/sinter/llm"
	_ "github.com/sprout-foundry/sinter/llm/all"
)

// task is one evaluation item: a chat problem, a grader, a category, and a
// generation cap.
type task struct {
	id       string
	category string // "knowledge" | "math" | "code" | "instruction" | "tools"
	prompt   string // user message
	system   string // system message ("" = none)
	maxToks  int
	grade    func(output string) score
}

// score is a grader's verdict.
type score struct {
	pass   bool
	frac   float64 // 0..1 partial credit
	reason string
}

// fullPass is a clean pass.
func fullPass() score { return score{pass: true, frac: 1} }

// partial gives fractional credit.
func partial(f float64, why string) score { return score{frac: f, reason: why} }

// fail rejects the output.
func fail(why string) score { return score{reason: why} }

func main() {
	modelFlag := flag.String("model", "", "single model directory")
	modelsFlag := flag.String("models", "", "comma-separated model directories")
	maxTokensDflt := flag.Int("max-tokens", 1024, "per-task generation cap")
	filter := flag.String("category", "", "run only this category")
	outJSON := flag.String("out", "", "write JSON results to this file")
	flag.Parse()

	var models []string
	if *modelFlag != "" {
		models = append(models, *modelFlag)
	}
	if *modelsFlag != "" {
		for _, m := range strings.Split(*modelsFlag, ",") {
			if m = strings.TrimSpace(expandHome(m)); m != "" {
				models = append(models, m)
			}
		}
	}
	if len(models) == 0 {
		fmt.Fprintln(os.Stderr, "usage: modeleval -model dir | -models dir1,dir2 [-category c] [-out results.json]")
		os.Exit(2)
	}

	all := allTasks()
	tasks := all
	if *filter != "" {
		tasks = nil
		for _, t := range all {
			if t.category == *filter {
				tasks = append(tasks, t)
			}
		}
	}

	var allResults []modelResult
	for _, dir := range models {
		r := runModel(dir, tasks, *maxTokensDflt)
		allResults = append(allResults, r)
		printResult(r)
	}

	if *outJSON != "" && len(allResults) > 1 {
		b, _ := json.MarshalIndent(allResults, "", "  ")
		if err := os.WriteFile(*outJSON, b, 0o644); err != nil {
			fmt.Fprintf(os.Stderr, "write results: %v\n", err)
		} else {
			fmt.Printf("\nresults written to %s\n", *outJSON)
		}
	}

	if len(allResults) > 1 {
		printComparison(allResults)
	}
}

// modelResult aggregates one model's run.
type modelResult struct {
	Model     string       `json:"model"`
	Tasks     int          `json:"tasks"`
	Passed    int          `json:"passed"`
	AvgScore  float64      `json:"avg_score"` // mean task frac, 0..1
	AvgTokSec float64      `json:"avg_tok_s"`
	TotalSecs float64      `json:"total_secs"`
	Details   []taskResult `json:"details"`
}

// taskResult is one graded task.
type taskResult struct {
	ID       string  `json:"id"`
	Category string  `json:"category"`
	Pass     bool    `json:"pass"`
	Score    float64 `json:"score"`
	TokSec   float64 `json:"tok_s"`
	Tokens   int     `json:"tokens"`
	Seconds  float64 `json:"seconds"`
	Why      string  `json:"why,omitempty"`
	Output   string  `json:"output,omitempty"`
}

func runModel(dir string, tasks []task, maxToks int) modelResult {
	name := filepath.Base(dir)
	fmt.Printf("\n=== %s (%d tasks) ===\n", name, len(tasks))

	m, err := llm.NewModel(dir)
	if err != nil {
		fmt.Fprintf(os.Stderr, "load %s: %v\n", dir, err)
		return modelResult{Model: name}
	}
	defer m.Close()

	res := modelResult{Model: name, Tasks: len(tasks)}
	for _, t := range tasks {
		tr := runTask(m, t, maxToks)
		res.Details = append(res.Details, tr)
		mark := "PASS"
		if !tr.Pass {
			mark = "FAIL"
		}
		fmt.Printf("  [%s] %-14s %-38s score=%.2f %6.1f tok/s (%4d tok, %5.1fs) %s\n",
			mark, t.category, t.id, tr.Score, tr.TokSec, tr.Tokens, tr.Seconds, tr.Why)
		res.AvgScore += tr.Score
		if tr.TokSec > 0 {
			res.AvgTokSec += tr.TokSec
		}
		res.TotalSecs += tr.Seconds
		if tr.Pass {
			res.Passed++
		}
	}
	if len(tasks) > 0 {
		res.AvgScore /= float64(len(tasks))
		n := 0
		for _, tr := range res.Details {
			if tr.TokSec > 0 {
				n++
			}
		}
		if n > 0 {
			res.AvgTokSec /= float64(n)
		}
	}
	return res
}

func runTask(m *llm.Model, t task, maxToksCap int) taskResult {
	msgs := []llm.ChatMessage{}
	if t.system != "" {
		msgs = append(msgs, llm.ChatMessage{Role: "system", Content: t.system})
	}
	msgs = append(msgs, llm.ChatMessage{Role: "user", Content: t.prompt})
	prompt := m.FormatChat(msgs)

	maxToks := t.maxToks
	if maxToks == 0 || maxToks > maxToksCap {
		maxToks = maxToksCap
	}
	cfg := llm.GenerateConfig{
		MaxTokens:         maxToks,
		Temperature:       0, // greedy: deterministic + the fast argmax path
		TopP:              1.0,
		TopK:              1,
		RepetitionPenalty: 0,
	}

	start := time.Now()
	var toks int
	text, err := m.GenerateText(context.Background(), prompt, cfg)
	secs := time.Since(start).Seconds()
	if err != nil {
		return taskResult{ID: t.id, Category: t.category, Why: "generate error: " + err.Error()}
	}
	toks = len(m.TokenizerEncode(text))

	sc := t.grade(text)
	tps := 0.0
	if secs > 0 {
		tps = float64(toks) / secs
	}
	return taskResult{
		ID: t.id, Category: t.category, Pass: sc.pass, Score: sc.frac,
		TokSec: tps, Tokens: toks, Seconds: secs, Why: sc.reason, Output: text,
	}
}

func printResult(r modelResult) {
	fmt.Printf("--- %s: %d/%d passed, avg score %.3f, avg %.1f tok/s, %.0fs total\n",
		r.Model, r.Passed, r.Tasks, r.AvgScore, r.AvgTokSec, r.TotalSecs)
}

// printComparison ranks models by average score with speed as secondary.
func printComparison(results []modelResult) {
	fmt.Printf("\n=== COMPARISON ===\n%-28s %8s %8s %8s %10s\n", "model", "passed", "avg", "tok/s", "secs")
	sorted := make([]modelResult, len(results))
	copy(sorted, results)
	sort.Slice(sorted, func(i, j int) bool {
		if sorted[i].AvgScore != sorted[j].AvgScore {
			return sorted[i].AvgScore > sorted[j].AvgScore
		}
		return sorted[i].AvgTokSec > sorted[j].AvgTokSec
	})
	for _, r := range sorted {
		fmt.Printf("%-28s %4d/%-3d %8.3f %8.1f %10.0f\n",
			r.Model, r.Passed, r.Tasks, r.AvgScore, r.AvgTokSec, r.TotalSecs)
	}
}

// ─── Task battery ────────────────────────────────────────────────────────

// allTasks returns the fixed evaluation battery. Greedy decode keeps runs
// deterministic; graders avoid exact-output matching (models legitimately
// phrase things differently) and check the property that matters.
func allTasks() []task {
	return []task{
		// ── Knowledge ──
		{
			id: "capital-france", category: "knowledge",
			prompt:  "What is the capital of France? Answer in one short sentence.",
			maxToks: 80,
			grade: func(out string) score {
				if containsAny(lower(out), "paris") {
					return fullPass()
				}
				return fail("no 'paris'")
			},
		},
		{
			id: "primary-colors", category: "knowledge",
			prompt:  "Name the three primary colors of light (additive). Comma-separated list only.",
			maxToks: 60,
			grade: func(out string) score {
				n := 0
				for _, c := range []string{"red", "green", "blue"} {
					if strings.Contains(lower(out), c) {
						n++
					}
				}
				if n == 3 {
					return fullPass()
				}
				return partial(float64(n)/3, fmt.Sprintf("%d/3 colors", n))
			},
		},
		{
			id: "count-letters", category: "knowledge",
			// Letter counting: a classic small-model stress test.
			prompt:  "How many letter R's are in the word 'strawberry'? Reply with just the number.",
			maxToks: 40,
			grade: func(out string) score {
				if strings.Contains(out, "3") {
					return fullPass()
				}
				return fail("answer isn't 3")
			},
		},

		// ── Math ──
		{
			id: "arith-mult", category: "math",
			prompt:  "What is 17 * 23? Give the number only.",
			maxToks: 40,
			grade: func(out string) score {
				if strings.Contains(out, "391") {
					return fullPass()
				}
				return fail("missing 391")
			},
		},
		{
			id: "word-problem", category: "math",
			prompt:  "A train travels 60 km in 45 minutes. What is its average speed in km/h? Show the answer as a number.",
			maxToks: 300,
			grade: func(out string) score {
				if strings.Contains(out, "80") {
					return fullPass()
				}
				return fail("missing 80")
			},
		},
		{
			id: "percent", category: "math",
			prompt:  "A jacket costs $80 and is discounted by 25%. What is the final price in dollars? Number only.",
			maxToks: 300,
			grade: func(out string) score {
				if strings.Contains(out, "60") {
					return fullPass()
				}
				return fail("missing 60")
			},
		},

		// ── Code ──
		{
			id: "reverse-string", category: "code",
			prompt:  "Write a Go function named reverse that takes a string s and returns the string reversed. Output only a fenced code block.",
			maxToks: 400,
			grade: func(out string) score {
				s := 0.0
				if strings.Contains(out, "func reverse") {
					s += 0.3
				}
				if strings.Contains(out, "for") {
					s += 0.2
				}
				// The loop must walk from both ends or build the result reversed.
				if strings.Contains(out, "i, j :=") || strings.Contains(out, "len(r)-1") || strings.Contains(out, "len(s)-1") || strings.Contains(out, "i--") || strings.Contains(out, "j--") {
					s += 0.3
				}
				if strings.Contains(out, "return") {
					s += 0.2
				}
				if s == 1 {
					return fullPass()
				}
				return partial(s, "structure credit")
			},
		},
		{
			id: "fizzbuzz", category: "code",
			// "Code only." phrasing sends the model into an unbounded
			// reasoning run (never emits </think> within budget); a plain
			// ask keeps reasoning short and visible. Model quirk, not a
			// grading problem.
			prompt:  "Write a Python function fizzbuzz(n) that prints numbers 1..n, replacing multiples of 3 with 'Fizz', multiples of 5 with 'Buzz', and multiples of both with 'FizzBuzz'.",
			maxToks: 400,
			grade: func(out string) score {
				s := 0.0
				if strings.Contains(out, "def fizzbuzz") {
					s += 0.3
				}
				if strings.Contains(out, "% 3") && strings.Contains(out, "% 5") {
					s += 0.4
				}
				if strings.Contains(out, "% 15") || (strings.Contains(out, "% 3 == 0") && strings.Contains(out, "% 5 == 0") && strings.Contains(lower(out), "fizzbuzz")) {
					s += 0.3
				}
				if s == 1 {
					return fullPass()
				}
				return partial(s, "structure credit")
			},
		},
		{
			id: "code-fix", category: "code",
			prompt: "This Go snippet has a bug: it declares x := 5 then x = \"hello\". In one sentence, why does it fail to compile?\n\nx := 5\nx = \"hello\"",
			grade: func(out string) score {
				if containsAny(lower(out), "type mismatch", "cannot use", "string", "int", "type") {
					return fullPass()
				}
				return fail("no type-mismatch reasoning")
			},
		},

		// ── Instruction following ──
		{
			id: "format-json", category: "instruction",
			prompt:  `Return ONLY a JSON object with keys "name" (string, value "Alice") and "age" (number, value 30). No other text.`,
			maxToks: 120,
			grade: func(out string) score {
				out = strings.TrimSpace(out)
				out = strings.TrimPrefix(out, "```json")
				out = strings.TrimPrefix(out, "```")
				out = strings.TrimSuffix(out, "```")
				out = strings.TrimSpace(out)
				var v map[string]interface{}
				if err := json.Unmarshal([]byte(out), &v); err != nil {
					return fail("output isn't valid JSON")
				}
				name, _ := v["name"].(string)
				age, _ := v["age"].(float64)
				if name == "Alice" && age == 30 {
					return fullPass()
				}
				return partial(0.5, "JSON parses but fields wrong")
			},
		},
		{
			id: "haiku", category: "instruction",
			prompt:  "Write a haiku (exactly three lines) about the ocean. Output only the three lines.",
			maxToks: 120,
			grade: func(out string) score {
				lines := nonEmptyLines(out)
				if len(lines) != 3 {
					return partial(0.3, fmt.Sprintf("%d non-empty lines, want 3", len(lines)))
				}
				if !containsAny(lower(out), "ocean", "sea", "wave", "tide", "water", "shore", "surf") {
					return partial(0.7, "three lines but off-topic")
				}
				return fullPass()
			},
		},
		{
			id: "no-list", category: "instruction",
			prompt:  "Explain what gravity is in exactly one sentence. Do not use lists, bullet points, or numbered items.",
			maxToks: 150,
			grade: func(out string) score {
				lines := nonEmptyLines(out)
				text := strings.TrimSpace(out)
				hasList := strings.Contains(text, "- ") || strings.Contains(text, "* ") || regexp.MustCompile(`\d+\.`).MatchString(text)
				if len(lines) == 1 && !hasList {
					return fullPass()
				}
				if len(lines) == 1 {
					return partial(0.5, "one line but list-like punctuation")
				}
				return partial(0.3, fmt.Sprintf("%d lines", len(lines)))
			},
		},

		// ── Tool use (text protocol, offline-graded: the model must emit a
		// well-formed call, not execute one) ──
		{
			id: "tool-format", category: "tools",
			system: "You have a tool: get_weather(city). Call tools with this exact format:\n<function name=\"get_weather\"><param name=\"city\">CITY</param></function>\nWhen the user asks about weather, respond with ONLY the function call, no prose.",
			prompt: "What's the weather in Tokyo?",
			grade: func(out string) score {
				if regexp.MustCompile(`(?s)<function\s+name="get_weather">\s*<param\s+name="city">\s*Tokyo\s*</param>`).MatchString(out) {
					return fullPass()
				}
				if containsAny(lower(out), "get_weather", "tokyo") {
					return partial(0.4, "mentions tool/city but wrong format")
				}
				return fail("no tool call")
			},
		},
		{
			id: "tool-noop", category: "tools",
			system: "You have a tool: get_weather(city). Call tools with this format:\n<function name=\"get_weather\"><param name=\"city\">CITY</param></function>\nOnly call the tool when the user asks about weather. Otherwise answer normally.",
			prompt: "What is the capital of Japan?",
			grade: func(out string) score {
				if strings.Contains(out, "<function") {
					return fail("called a tool when it shouldn't")
				}
				if strings.Contains(lower(out), "tokyo") {
					return fullPass()
				}
				return partial(0.5, "no spurious call but wrong answer")
			},
		},
		{
			id: "tool-args", category: "tools",
			system: "You have a tool: search(query, limit). Call tools with this format:\n<function name=\"search\"><param name=\"query\">Q</param><param name=\"limit\">N</param></function>\nWhen the user asks to search, respond with ONLY the function call. Set limit to 5 unless told otherwise.",
			prompt: "Search for Go concurrency patterns, give me 3 results.",
			grade: func(out string) score {
				s := 0.0
				if strings.Contains(out, `<function name="search">`) {
					s += 0.4
				}
				if strings.Contains(out, `concurrency`) || strings.Contains(out, `Go`) {
					s += 0.3
				}
				if strings.Contains(out, ">3<") || strings.Contains(out, "> 3 <") || strings.Contains(out, `"3"`) {
					s += 0.3
				}
				if s == 1 {
					return fullPass()
				}
				return partial(s, "partial call structure")
			},
		},
	}
}

// ─── helpers ─────────────────────────────────────────────────────────────

func lower(s string) string { return strings.ToLower(s) }

func containsAny(s string, subs ...string) bool {
	for _, sub := range subs {
		if strings.Contains(s, sub) {
			return true
		}
	}
	return false
}

func nonEmptyLines(s string) []string {
	var out []string
	for _, l := range strings.Split(s, "\n") {
		if strings.TrimSpace(l) != "" {
			out = append(out, strings.TrimSpace(l))
		}
	}
	return out
}

// expandHome expands a leading ~ to the user's home directory.
func expandHome(p string) string {
	if p == "~" || strings.HasPrefix(p, "~/") {
		if home, err := os.UserHomeDir(); err == nil {
			return filepath.Join(home, strings.TrimPrefix(p, "~"))
		}
	}
	return p
}

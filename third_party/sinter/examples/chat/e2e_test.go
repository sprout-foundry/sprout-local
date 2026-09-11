//go:build darwin || linux

// End-to-end test for the gomlx module: builds examples/chat as a real
// binary and drives it like an implementor would — a local model, a chat
// prompt, streamed tokens. This catches breakage that unit tests can't:
// cgo link errors, backend registration, template/prompt mismatches, the
// whole load -> prefill -> decode -> detokenize path.
//
// Skips (rather than fails) when no model is on disk, so `go test ./...`
// stays green on any machine; CI passes -args -model-dir to run it live.
package main_test

import (
	"fmt"
	"os"
	"os/exec"
	"path/filepath"
	"runtime"
	"strings"
	"testing"
)

// candidateModelDirs returns local model dirs, smallest first so the test
// runs fast on machines with several models downloaded.
func candidateModelDirs(t *testing.T) []string {
	t.Helper()
	home, err := os.UserHomeDir()
	if err != nil {
		return nil
	}
	roots := []string{
		filepath.Join(home, ".cache", "sprout", "models"),
		filepath.Join(home, ".local", "share", "sprout", "models"),
		filepath.Join(home, ".auto-term", "models"),
	}
	var dirs []string
	for _, root := range roots {
		entries, err := os.ReadDir(root)
		if err != nil {
			continue
		}
		for _, e := range entries {
			if !e.IsDir() {
				continue
			}
			d := filepath.Join(root, e.Name())
			if _, err := os.Stat(filepath.Join(d, "model.safetensors")); err == nil {
				dirs = append(dirs, d)
			}
		}
	}
	return dirs
}

// isThinkingModel reports whether the model at dir uses Qwen3-style
// <think> blocks (and therefore needs the empty-think prefix to answer
// directly). Detected by tokenizer special tokens.
func isThinkingModel(dir string) bool {
	data, err := os.ReadFile(filepath.Join(dir, "tokenizer.json"))
	if err != nil {
		return false
	}
	return strings.Contains(string(data), "<think>")
}

func TestChatEndToEnd(t *testing.T) {
	if runtime.GOOS != "darwin" || runtime.GOARCH != "arm64" {
		t.Skip("e2e generation requires the Metal backend (darwin/arm64)")
	}
	dirFlag := os.Getenv("SINTER_E2E_MODEL")
	if dirFlag == "" {
		cands := candidateModelDirs(t)
		if len(cands) == 0 {
			t.Skip("no local model found; set SINTER_E2E_MODEL or download one (see README)")
		}
		dirFlag = cands[0]
		t.Logf("using model: %s", dirFlag)
	}
	args := []string{"-model", dirFlag, "-prompt", "Say exactly: hello world", "-max-tokens", "60"}
	if isThinkingModel(dirFlag) {
		// Thinking models emit a <think> block first; prefilling an empty
		// one makes them answer immediately. Non-thinking models (e.g.
		// Qwen2.5) don't know the tokens, so only add it when present.
		args = append(args, "-thinking")
	}

	bin, err := goBuildExample(t)
	if err != nil {
		t.Fatalf("build example: %v", err)
	}

	// "Say exactly" keeps greedy output short and predictable: the model
	// should echo the phrase then stop (EOS) well before the cap.
	cmd := exec.Command(bin, args...)
	out, err := cmd.CombinedOutput()
	if err != nil {
		t.Fatalf("sinter-chat: %v\n%s", err, out)
	}
	text := string(out)
	if !strings.Contains(strings.ToLower(text), "hello world") {
		t.Fatalf("output did not contain 'hello world':\n%s", text)
	}
	if strings.Contains(text, "tok/s") {
		t.Logf("%s", text[strings.LastIndex(text, "["):])
	}
}

// TestChatUsageExitCode pins the CLI contract: missing -model exits 2 with
// usage on stderr. Runs everywhere (no GPU, no model needed).
func TestChatUsageExitCode(t *testing.T) {
	bin, err := goBuildExample(t)
	if err != nil {
		t.Fatalf("build example: %v", err)
	}
	out, err := exec.Command(bin).CombinedOutput()
	if err == nil {
		t.Fatalf("expected non-zero exit, got 0:\n%s", out)
	}
	if ee, ok := err.(*exec.ExitError); !ok || ee.ExitCode() != 2 {
		t.Fatalf("expected exit code 2, got %v", err)
	}
	if !strings.Contains(string(out), "usage:") {
		t.Fatalf("expected usage on stderr:\n%s", out)
	}
}

func goBuildExample(t *testing.T) (string, error) {
	t.Helper()
	// The test lives in the same dir as the example, so "." is the main
	// package; building via `go build .` from the module root keeps this
	// working regardless of where `go test` was invoked from.
	root, err := findModuleRoot()
	if err != nil {
		return "", err
	}
	bin := filepath.Join(t.TempDir(), "sinter-chat")
	build := exec.Command("go", "build", "-o", bin, "./examples/chat")
	build.Dir = root
	if out, err := build.CombinedOutput(); err != nil {
		return "", fmt.Errorf("go build: %v\n%s", err, out)
	}
	return bin, nil
}

func findModuleRoot() (string, error) {
	dir, err := os.Getwd()
	if err != nil {
		return "", err
	}
	for {
		if _, err := os.Stat(filepath.Join(dir, "go.mod")); err == nil {
			return dir, nil
		}
		parent := filepath.Dir(dir)
		if parent == dir {
			return "", fmt.Errorf("go.mod not found")
		}
		dir = parent
	}
}

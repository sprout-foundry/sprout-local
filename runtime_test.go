package main

import (
	"context"
	"errors"
	"os"
	"runtime"
	"strings"
	"testing"
	"time"
	"unicode/utf8"

	"github.com/sprout-foundry/seed/core"
)

// ─── run_command: /bin/sh in confirm mode, strict single command in yolo ──

func TestToolRunCommandPipesWorkInConfirmMode(t *testing.T) {
	toolSafetyBypass = false
	t.Cleanup(func() { toolSafetyBypass = false })
	res, err := toolRunCommand(context.Background(), map[string]string{
		"command": "echo hello | tr a-z A-Z",
	})
	if err != nil {
		t.Fatalf("piped command rejected in confirm mode: %v", err)
	}
	if strings.TrimSpace(res) != "HELLO" {
		t.Fatalf("want HELLO, got %q", res)
	}
}

func TestToolRunCommandQuotesWorkInConfirmMode(t *testing.T) {
	toolSafetyBypass = false
	t.Cleanup(func() { toolSafetyBypass = false })
	res, err := toolRunCommand(context.Background(), map[string]string{
		"command": `echo "two  spaces"`,
	})
	if err != nil {
		t.Fatalf("quoted command rejected: %v", err)
	}
	if strings.TrimSpace(res) != "two  spaces" {
		t.Fatalf("quoted whitespace collapsed: %q", res)
	}
}

func TestToolRunCommandYoloRefusesSubstitution(t *testing.T) {
	toolSafetyBypass = true
	t.Cleanup(func() { toolSafetyBypass = false })
	for _, cmd := range []string{
		"echo `echo hi`",
		"echo $(echo hi)",
		"echo a; echo b",
	} {
		if _, err := toolRunCommand(context.Background(), map[string]string{"command": cmd}); err == nil {
			t.Fatalf("yolo mode allowed %q", cmd)
		}
	}
}

func TestToolRunCommandYoloAllowsPlainCommand(t *testing.T) {
	toolSafetyBypass = true
	t.Cleanup(func() { toolSafetyBypass = false })
	res, err := toolRunCommand(context.Background(), map[string]string{"command": "echo plain"})
	if err != nil {
		t.Fatalf("plain command refused in yolo: %v", err)
	}
	if strings.TrimSpace(res) != "plain" {
		t.Fatalf("want plain, got %q", res)
	}
}

func TestToolRunCommandEmpty(t *testing.T) {
	if _, err := toolRunCommand(context.Background(), map[string]string{"command": "   "}); err == nil {
		t.Fatal("empty command should error")
	}
}

// ─── approval memory ─────────────────────────────────────────────────────

func TestCommandApproved(t *testing.T) {
	sessionApprovedCommands = nil
	t.Cleanup(func() { sessionApprovedCommands = nil })
	if commandApproved("git status") {
		t.Fatal("nothing approved yet")
	}
	sessionApprovedCommands = append(sessionApprovedCommands, "git")
	if !commandApproved("git status") {
		t.Fatal("git should be approved")
	}
	if commandApproved("gitx status") || commandApproved("ls") {
		t.Fatal("approval must match the exact command word")
	}
	if commandApproved("") {
		t.Fatal("empty command line cannot be approved")
	}
}

// ─── tools preference persistence ────────────────────────────────────────

func TestLoadToolsPreferenceDefaultOn(t *testing.T) {
	t.Setenv("SPROUT_LOCAL_STATE_ROOT", t.TempDir())
	loadToolsPreference()
	if !toolsRequested || toolSafetyBypass {
		t.Fatal("default must be tools on, no bypass")
	}
}

func TestLoadToolsPreferenceStates(t *testing.T) {
	dir := t.TempDir()
	t.Setenv("SPROUT_LOCAL_STATE_ROOT", dir)
	for _, tc := range []struct {
		stored     string
		wantOn     bool
		wantBypass bool
	}{
		{"on", true, false},
		{"off", false, false},
		{"yolo", true, true},
		{"bogus", true, false},
	} {
		writeTestToolsJSON(t, dir, tc.stored)
		loadToolsPreference()
		if toolsRequested != tc.wantOn || toolSafetyBypass != tc.wantBypass {
			t.Fatalf("stored %q: got on=%v bypass=%v", tc.stored, toolsRequested, toolSafetyBypass)
		}
	}
}

func TestSaveToolsPreferenceRoundTrip(t *testing.T) {
	t.Setenv("SPROUT_LOCAL_STATE_ROOT", t.TempDir())
	toolsRequested, toolSafetyBypass = false, false
	saveToolsPreference()
	loadToolsPreference()
	if toolsRequested {
		t.Fatal("off should round-trip")
	}
	toolsRequested, toolSafetyBypass = true, true
	saveToolsPreference()
	loadToolsPreference()
	if !toolsRequested || !toolSafetyBypass {
		t.Fatal("yolo should round-trip")
	}
	toolsRequested, toolSafetyBypass = true, false
}

func writeTestToolsJSON(t *testing.T, dir, state string) {
	t.Helper()
	if err := os.WriteFile(dir+"/tools.json", []byte(`{"tools":"`+state+`"}`), 0o644); err != nil {
		t.Fatal(err)
	}
}

// ─── tunables ────────────────────────────────────────────────────────────

func TestInitTunablesEnvOverrides(t *testing.T) {
	t.Setenv("SPROUT_LOCAL_MAX_STEPS", "12")
	t.Setenv("SPROUT_LOCAL_TOOL_RESULT_CAP", "9999")
	t.Setenv("SPROUT_LOCAL_COMMAND_TIMEOUT", "60")
	t.Setenv("SPROUT_LOCAL_MAX_TOKENS", "8192")
	old := struct {
		s, c, m int
		d       time.Duration
	}{maxToolSteps, toolResultCap, maxTokens, commandTimeout}
	t.Cleanup(func() {
		maxToolSteps, toolResultCap, maxTokens, commandTimeout = old.s, old.c, old.m, old.d
	})
	initTunables()
	if maxToolSteps != 12 || toolResultCap != 9999 || maxTokens != 8192 {
		t.Fatalf("int overrides not applied: %d %d %d", maxToolSteps, toolResultCap, maxTokens)
	}
	if commandTimeout != 60*time.Second {
		t.Fatalf("timeout override not applied: %v", commandTimeout)
	}
}

func TestInitTunablesBadValuesIgnored(t *testing.T) {
	old := maxToolSteps
	t.Cleanup(func() { maxToolSteps = old })
	t.Setenv("SPROUT_LOCAL_MAX_STEPS", "not-a-number")
	initTunables()
	if maxToolSteps != old {
		t.Fatal("bad env value must be ignored")
	}
}

// ─── environment context ─────────────────────────────────────────────────

func TestEnvironmentContextMentionsOS(t *testing.T) {
	ctx := environmentContext()
	switch runtime.GOOS {
	case "darwin":
		if !strings.Contains(ctx, "macOS") || !strings.Contains(ctx, "ifconfig") {
			t.Fatalf("darwin context missing macOS/ifconfig: %q", ctx)
		}
	case "linux":
		if !strings.Contains(ctx, "Linux") || !strings.Contains(ctx, "ip addr") {
			t.Fatalf("linux context missing Linux/ip addr: %q", ctx)
		}
	}
	if !strings.Contains(ctx, "working directory") {
		t.Fatalf("context missing working directory: %q", ctx)
	}
}

func TestEffectiveSystemPrompt(t *testing.T) {
	env := effectiveSystemPrompt("")
	if !strings.Contains(env, "Environment:") {
		t.Fatalf("bare prompt should be the env context, got %q", env)
	}
	both := effectiveSystemPrompt("Be terse.")
	if !strings.HasPrefix(both, "Be terse.") || !strings.Contains(both, "Environment:") {
		t.Fatalf("user prompt + env context expected, got %q", both)
	}
}

// ─── result display preview ──────────────────────────────────────────────

func TestTruncateResultForDisplaySkipsComments(t *testing.T) {
	got := truncateResultForDisplay("##\n# Host Database\n127.0.0.1 localhost")
	if got != "127.0.0.1 localhost" {
		t.Fatalf("comment banner not skipped: %q", got)
	}
	if got := truncateResultForDisplay("##\n# only comments"); got != "##" {
		t.Fatalf("all-comment input should fall back to first line, got %q", got)
	}
	long := strings.Repeat("x", 200)
	if got := truncateResultForDisplay(long); utf8.RuneCountInString(got) != 121 || !strings.HasSuffix(got, "…") {
		t.Fatalf("long first line not truncated: %d runes", utf8.RuneCountInString(got))
	}
}

// ─── graceful exhaustion (guard constants / error matching) ─────────────

func TestGraceFinalQueryMatchesSeedError(t *testing.T) {
	// The REPL/web fallback keys on errors.Is(err, core.ErrMaxIterations);
	// seed wraps it, so errors.Is must see through the wrap.
	wrapped := errors.Join(core.ErrMaxIterations)
	if !errors.Is(wrapped, core.ErrMaxIterations) {
		t.Fatal("errors.Is must detect wrapped ErrMaxIterations")
	}
	if !strings.Contains(graceFinalQuery, "Do not attempt any further tool calls") {
		t.Fatalf("grace prompt lost its no-tools instruction: %q", graceFinalQuery)
	}
}

package config

import (
	"os"
	"runtime"
	"strings"
	"testing"
	"time"
)

// ─── tools preference persistence ────────────────────────────────────────

func TestLoadToolsPreferenceDefaultOn(t *testing.T) {
	t.Setenv("SPROUT_LOCAL_STATE_ROOT", t.TempDir())
	LoadToolsPreference()
	if !ToolsRequested || ToolSafetyBypass {
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
		LoadToolsPreference()
		if ToolsRequested != tc.wantOn || ToolSafetyBypass != tc.wantBypass {
			t.Fatalf("stored %q: got on=%v bypass=%v", tc.stored, ToolsRequested, ToolSafetyBypass)
		}
	}
}

func TestSaveToolsPreferenceRoundTrip(t *testing.T) {
	t.Setenv("SPROUT_LOCAL_STATE_ROOT", t.TempDir())
	ToolsRequested, ToolSafetyBypass = false, false
	SaveToolsPreference()
	LoadToolsPreference()
	if ToolsRequested {
		t.Fatal("off should round-trip")
	}
	ToolsRequested, ToolSafetyBypass = true, true
	SaveToolsPreference()
	LoadToolsPreference()
	if !ToolsRequested || !ToolSafetyBypass {
		t.Fatal("yolo should round-trip")
	}
	ToolsRequested, ToolSafetyBypass = true, false
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
	}{MaxToolSteps, ToolResultCap, MaxTokens, CommandTimeout}
	t.Cleanup(func() {
		MaxToolSteps, ToolResultCap, MaxTokens, CommandTimeout = old.s, old.c, old.m, old.d
	})
	InitTunables()
	if MaxToolSteps != 12 || ToolResultCap != 9999 || MaxTokens != 8192 {
		t.Fatalf("int overrides not applied: %d %d %d", MaxToolSteps, ToolResultCap, MaxTokens)
	}
	if CommandTimeout != 60*time.Second {
		t.Fatalf("timeout override not applied: %v", CommandTimeout)
	}
}

func TestInitTunablesBadValuesIgnored(t *testing.T) {
	old := MaxToolSteps
	t.Cleanup(func() { MaxToolSteps = old })
	t.Setenv("SPROUT_LOCAL_MAX_STEPS", "not-a-number")
	InitTunables()
	if MaxToolSteps != old {
		t.Fatal("bad env value must be ignored")
	}
}

// ─── environment context ─────────────────────────────────────────────────

func TestEnvironmentContextMentionsOS(t *testing.T) {
	ctx := EnvironmentContext()
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
	env := EffectiveSystemPrompt("")
	if !strings.Contains(env, "Environment:") {
		t.Fatalf("bare prompt should be the env context, got %q", env)
	}
	both := EffectiveSystemPrompt("Be terse.")
	if !strings.HasPrefix(both, "Be terse.") || !strings.Contains(both, "Environment:") {
		t.Fatalf("user prompt + env context expected, got %q", both)
	}
}
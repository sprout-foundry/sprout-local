package main

// ---------------------------------------------------------------------------
// runtime.go — session tunables, machine context, and persisted /tools
// state.
//
// The limits that used to be hardcoded (tool round-trips per turn,
// tool-result size, command timeout, generation token cap) are plain vars
// with defaults, so flags (-max-steps, -max-tokens) and environment
// variables can adjust them without code edits. The /tools state persists
// across sessions (tools.json); tools default ON — seed never mentions
// tools unless the executor is attached, so an off session costs nothing
// and an on session is the common case for "help me with my machine".
// ---------------------------------------------------------------------------

import (
	"encoding/json"
	"fmt"
	"log"
	"os"
	"path/filepath"
	"runtime"
	"strconv"
	"strings"
	"time"
)

// ─── Session tunables ────────────────────────────────────────────────────
//
// Defaults live in tools.go (defaultMaxToolSteps etc.); these vars hold
// the session values. SPROUT_LOCAL_* environment variables override the
// defaults at startup, and flags (-max-steps, -max-tokens) override the
// environment:
//
//	SPROUT_LOCAL_MAX_STEPS        tool round-trips per user turn
//	SPROUT_LOCAL_TOOL_RESULT_CAP  per-result character cap
//	SPROUT_LOCAL_COMMAND_TIMEOUT  run_command timeout in seconds
//	SPROUT_LOCAL_MAX_TOKENS       generation token cap

var (
	maxToolSteps   = defaultMaxToolSteps
	toolResultCap  = defaultToolResultCap
	commandTimeout = defaultCommandTimeout * time.Second
	maxTokens      = defaultMaxTokens
)

// toolsRequested is the session switch for tool calling (see /tools and
// -tools). On by default; the choice persists in tools.json.
var toolsRequested = true

// toolSafetyBypass skips the run_command confirmation prompt (yolo).
var toolSafetyBypass = false

// initTunables applies SPROUT_LOCAL_* overrides. Unknown values are
// reported and ignored — a typo should not silently halve the budget.
func initTunables() {
	setPositiveInt("SPROUT_LOCAL_MAX_STEPS", func(n int) { maxToolSteps = n })
	setPositiveInt("SPROUT_LOCAL_TOOL_RESULT_CAP", func(n int) { toolResultCap = n })
	setPositiveSeconds("SPROUT_LOCAL_COMMAND_TIMEOUT", func(d time.Duration) { commandTimeout = d })
	setPositiveInt("SPROUT_LOCAL_MAX_TOKENS", func(n int) { maxTokens = n })
}

// setPositiveInt parses key as a positive integer and calls set with it.
func setPositiveInt(key string, set func(int)) {
	v := strings.TrimSpace(os.Getenv(key))
	if v == "" {
		return
	}
	n, err := strconv.Atoi(v)
	if err != nil || n <= 0 {
		log.Printf("ignoring %s=%q: want a positive integer", key, v)
		return
	}
	set(n)
}

// setPositiveSeconds parses key as a positive integer of seconds.
func setPositiveSeconds(key string, set func(time.Duration)) {
	v := strings.TrimSpace(os.Getenv(key))
	if v == "" {
		return
	}
	n, err := strconv.Atoi(v)
	if err != nil || n <= 0 {
		log.Printf("ignoring %s=%q: want a positive integer (seconds)", key, v)
		return
	}
	set(time.Duration(n) * time.Second)
}

// ─── Machine context for the model ───────────────────────────────────────

// environmentContext describes the machine: OS, architecture, shell, and
// working directory, plus a platform hint or two. Without it, small
// models guess the platform (ip addr on macOS) and burn tool budget
// discovering their mistakes.
func environmentContext() string {
	osName := runtime.GOOS
	hints := ""
	switch runtime.GOOS {
	case "darwin":
		osName = "macOS"
		hints = "Network: ifconfig or ipconfig getifaddr en0. System: sw_vers, sysctl. Packages: brew."
	case "linux":
		osName = "Linux"
		hints = "Network: ip addr. Packages: apt/dnf/pacman."
	}
	shell := os.Getenv("SHELL")
	if shell == "" {
		shell = "/bin/sh"
	}
	return fmt.Sprintf("Environment: %s (%s/%s), shell %s, working directory %s. %s",
		osName, runtime.GOOS, runtime.GOARCH, shell, cwd(), hints)
}

// effectiveSystemPrompt combines the user's -s / /system prompt with the
// machine context; the environment note alone when no user prompt is set.
func effectiveSystemPrompt(user string) string {
	env := environmentContext()
	if strings.TrimSpace(user) == "" {
		return env
	}
	return strings.TrimRight(user, "\n") + "\n\n" + env
}

// ─── Persisted /tools state ──────────────────────────────────────────────

// toolsPreference is the on-disk shape of <stateRoot>/tools.json.
type toolsPreference struct {
	Tools string `json:"tools"` // "on", "off", or "yolo"
}

func toolsPreferencePath() string {
	root := stateRoot()
	if root == "" {
		return ""
	}
	return filepath.Join(root, "tools.json")
}

// loadToolsPreference applies the persisted /tools choice, defaulting to
// tools on. Called at startup when -tools is not given.
func loadToolsPreference() {
	toolsRequested = true
	toolSafetyBypass = false
	p := toolsPreferencePath()
	if p == "" {
		return
	}
	b, err := os.ReadFile(p)
	if err != nil {
		return
	}
	var pref toolsPreference
	if json.Unmarshal(b, &pref) != nil {
		return
	}
	switch strings.ToLower(strings.TrimSpace(pref.Tools)) {
	case "off":
		toolsRequested = false
	case "yolo":
		toolsRequested = true
		toolSafetyBypass = true
	default: // "on" or unknown → on
		toolsRequested = true
	}
}

// saveToolsPreference persists the current /tools state for next launch.
func saveToolsPreference() {
	p := toolsPreferencePath()
	if p == "" {
		return
	}
	if err := os.MkdirAll(filepath.Dir(p), 0o755); err != nil {
		return
	}
	pref := toolsPreference{Tools: "on"}
	switch {
	case !toolsRequested:
		pref.Tools = "off"
	case toolSafetyBypass:
		pref.Tools = "yolo"
	}
	b, err := json.MarshalIndent(pref, "", "  ")
	if err != nil {
		return
	}
	_ = os.WriteFile(p, append(b, '\n'), 0o644)
}

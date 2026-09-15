package config

// ---------------------------------------------------------------------------
// config — session tunables, machine context, and persisted /tools state.
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

	"github.com/sprout-foundry/sprout-local/internal/paths"
)

// ─── Session tunables ────────────────────────────────────────────────────
//
// Defaults below; these vars hold the session values. SPROUT_LOCAL_*
// environment variables override the defaults at startup (InitTunables),
// and flags (-max-steps, -max-tokens) override the environment:
//
//	SPROUT_LOCAL_MAX_STEPS        tool round-trips per user turn
//	SPROUT_LOCAL_TOOL_RESULT_CAP  per-result character cap
//	SPROUT_LOCAL_COMMAND_TIMEOUT  run_command timeout in seconds
//	SPROUT_LOCAL_MAX_TOKENS       generation token cap

const (
	// defaultMaxToolSteps caps tool round-trips per user turn, so a model
	// that keeps calling tools can't loop forever (or burn the token
	// budget). Default raised from 4: with self-correcting errors, a
	// lookup usually needs 1–2 steps and a broken first guess shouldn't
	// kill the turn. 100 = effectively "until the model stops on its
	// own"; the graceful-exhaustion path still bounds a runaway.
	defaultMaxToolSteps = 100
	// defaultToolResultCap bounds each tool result fed back to the model.
	defaultToolResultCap = 6000 // characters
	// defaultCommandTimeout bounds run_command executions, in seconds.
	defaultCommandTimeout = 30
	// defaultMaxTokens is the session generation token cap. Higher
	// temperature/variety settings live with the engine (chatmodel).
	defaultMaxTokens = 4096
)

var (
	MaxToolSteps   = defaultMaxToolSteps
	ToolResultCap  = defaultToolResultCap
	CommandTimeout = defaultCommandTimeout * time.Second
	MaxTokens      = defaultMaxTokens
)

// ToolsRequested is the session switch for tool calling (see /tools and
// -tools). On by default; the choice persists in tools.json.
var ToolsRequested = true

// ToolSafetyBypass skips the run_command confirmation prompt (yolo).
var ToolSafetyBypass = false

// InitTunables applies SPROUT_LOCAL_* overrides. Unknown values are
// reported and ignored — a typo should not silently halve the budget.
func InitTunables() {
	setPositiveInt("SPROUT_LOCAL_MAX_STEPS", func(n int) { MaxToolSteps = n })
	setPositiveInt("SPROUT_LOCAL_TOOL_RESULT_CAP", func(n int) { ToolResultCap = n })
	setPositiveSeconds("SPROUT_LOCAL_COMMAND_TIMEOUT", func(d time.Duration) { CommandTimeout = d })
	setPositiveInt("SPROUT_LOCAL_MAX_TOKENS", func(n int) { MaxTokens = n })
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

// EnvironmentContext describes the machine: OS, architecture, shell, and
// working directory, plus a platform hint or two. Without it, small
// models guess the platform (ip addr on macOS) and burn tool budget
// discovering their mistakes.
func EnvironmentContext() string {
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
	wd, err := os.Getwd()
	if err != nil {
		wd = "."
	}
	return fmt.Sprintf("Environment: %s (%s/%s), shell %s, working directory %s. %s",
		osName, runtime.GOOS, runtime.GOARCH, shell, wd, hints)
}

// EffectiveSystemPrompt combines the user's -s / /system prompt with the
// machine context; the environment note alone when no user prompt is set.
func EffectiveSystemPrompt(user string) string {
	env := EnvironmentContext()
	if strings.TrimSpace(user) == "" {
		return env
	}
	return strings.TrimRight(user, "\n") + "\n\n" + env
}

// ─── Persisted /tools state ──────────────────────────────────────────────

// ToolsPreference is the on-disk shape of <stateRoot>/tools.json.
type ToolsPreference struct {
	Tools string `json:"tools"` // "on", "off", or "yolo"
}

// ToolsPreferencePath is the persisted /tools state location
// (<stateRoot>/tools.json). "" when the state root is unresolvable.
func ToolsPreferencePath() string {
	root := paths.StateRoot()
	if root == "" {
		return ""
	}
	return filepath.Join(root, "tools.json")
}

// LoadToolsPreference applies the persisted /tools choice, defaulting to
// tools on. Called at startup when -tools is not given.
func LoadToolsPreference() {
	ToolsRequested = true
	ToolSafetyBypass = false
	p := ToolsPreferencePath()
	if p == "" {
		return
	}
	b, err := os.ReadFile(p)
	if err != nil {
		return
	}
	var pref ToolsPreference
	if json.Unmarshal(b, &pref) != nil {
		return
	}
	switch strings.ToLower(strings.TrimSpace(pref.Tools)) {
	case "off":
		ToolsRequested = false
	case "yolo":
		ToolsRequested = true
		ToolSafetyBypass = true
	default: // "on" or unknown → on
		ToolsRequested = true
	}
}

// SaveToolsPreference persists the current /tools state for next launch.
func SaveToolsPreference() {
	p := ToolsPreferencePath()
	if p == "" {
		return
	}
	if err := os.MkdirAll(filepath.Dir(p), 0o755); err != nil {
		return
	}
	pref := ToolsPreference{Tools: "on"}
	switch {
	case !ToolsRequested:
		pref.Tools = "off"
	case ToolSafetyBypass:
		pref.Tools = "yolo"
	}
	b, err := json.MarshalIndent(pref, "", "  ")
	if err != nil {
		return
	}
	_ = os.WriteFile(p, append(b, '\n'), 0o644)
}
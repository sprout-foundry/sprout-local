package repl

// ---------------------------------------------------------------------------
// Session logging — preserves the bash tool's behavior of writing every
// exchange to <stateRoot>/sessions/<timestamp>.log (default
// ~/.sprout-local/sessions/).
// ---------------------------------------------------------------------------

import (
	"fmt"
	"os"
	"path/filepath"
	"strings"
	"time"

	"github.com/sprout-foundry/sprout-local/internal/paths"
)

var (
	logEnabled bool
	logPath    string
)

// EnableSessionLog turns on per-exchange logging to
// <stateRoot>/sessions/<timestamp>.log. Called by the entrypoint unless
// -no-log is set.
func EnableSessionLog() {
	logPath = defaultLogPath()
	logEnabled = logPath != ""
}

// defaultLogPath returns <stateRoot>/sessions/<YYYYMMDDHHMMSS>.log,
// creating the directory when needed.
func defaultLogPath() string {
	dir := paths.SessionsDir()
	if dir == "" {
		return ""
	}
	if err := os.MkdirAll(dir, 0o755); err != nil {
		return "" // logging is best-effort; never fatal
	}
	return filepath.Join(dir, time.Now().Format("20060102150405")+".log")
}

// appendLog appends one exchange to the session log. Failures are silent —
// logging must never break the chat.
func appendLog(role, modelPath, systemPrompt, request, response string) {
	if !logEnabled || logPath == "" {
		return
	}
	f, err := os.OpenFile(logPath, os.O_APPEND|os.O_CREATE|os.O_WRONLY, 0o644)
	if err != nil {
		return
	}
	defer f.Close()

	ts := time.Now().Format("2006-01-02 15:04:05")
	fmt.Fprintf(f, "[%s] role=%s model=%s\n", ts, role, filepath.Base(modelPath))
	if systemPrompt != "" {
		fmt.Fprintf(f, "System:\n-----------------------\n%s\n", systemPrompt)
	}
	if request != "" {
		fmt.Fprintf(f, "Request:\n-----------------------\n%s\n", request)
	}
	if response != "" {
		fmt.Fprintf(f, "Response:\n-----------------------\n%s\n", response)
	}
	fmt.Fprintln(f, strings.Repeat("-", 40))
}

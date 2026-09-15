package chatmodel

// ---------------------------------------------------------------------------
// logfilter.go — keep sinter's engine chatter off the console.
//
// sinter logs load progress and warmup details through the standard log
// package ("llm: warmup complete…", "qwen35: 32 layers loaded…", "sinter:
// loaded …"). Useful for debugging, noisy for chatting. The standard
// logger is pointed at a logFilter that drops lines matching known engine
// markers unless -v / CHATLLM_DEBUG is set. Everything else (fatal errors,
// HTTP errors, cache evictions) still reaches stderr, so failures stay
// visible.
// ---------------------------------------------------------------------------

import (
	"io"
	"log"
	"os"
	"strings"
)

// stderrWriter returns the real stderr (kept separate so tests can swap it).
func stderrWriter() io.Writer { return os.Stderr }

// logSetOutput wraps log.SetOutput (kept separate so tests can swap it).
func logSetOutput(w io.Writer) { log.SetOutput(w) }

// engineNoiseMarkers identify sinter engine chatter. Matched as substrings
// after the log timestamp (e.g. "2026/09/10 10:00:00 llm: warmup …").
var engineNoiseMarkers = []string{
	" llm: ",
	" qwen35: ",
	" qwen3: ",
	" gemma4: ",
	" lfm2: ",
	" sinter: loaded ",
}

type logFilter struct {
	w       io.Writer
	verbose bool
}

func (l *logFilter) Write(p []byte) (int, error) {
	if l.verbose || !isEngineNoise(string(p)) {
		return l.w.Write(p)
	}
	return len(p), nil // swallowed
}

func isEngineNoise(line string) bool {
	for _, m := range engineNoiseMarkers {
		if strings.Contains(line, m) {
			return true
		}
	}
	return false
}

// InstallLogFilter routes the standard logger through the noise filter.
func InstallLogFilter(verbose bool) {
	logSetOutput(&logFilter{w: stderrWriter(), verbose: verbose})
}

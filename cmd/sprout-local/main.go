// Command sprout-local — local chat with some extra help.
//
// Thin entrypoint: flags and environment setup, then dispatch to the
// internal packages (repl, webui). All behavior lives in internal/.
package main

import (
	"bufio"
	"flag"
	"fmt"
	"log"
	"os"
	"strings"

	"github.com/sprout-foundry/seed/core"

	"github.com/sprout-foundry/sprout-local/internal/chatmodel"
	"github.com/sprout-foundry/sprout-local/internal/config"
	"github.com/sprout-foundry/sprout-local/internal/download"
	"github.com/sprout-foundry/sprout-local/internal/paths"
	"github.com/sprout-foundry/sprout-local/internal/repl"
	"github.com/sprout-foundry/sprout-local/internal/tools"
	"github.com/sprout-foundry/sprout-local/internal/webui"
)

func main() {
	// ── CLI flags ────────────────────────────────────────────────────────
	//   -m / --model-dir : model directory override (same as SPROUT_LOCAL_MODEL_DIR)
	//   -s / --system    : system prompt for the session
	//   -p / --prompt    : one-shot prompt (print response and exit)
	//   -no-log          : disable session logging
	//   -pull [name]     : download a catalog model, then continue (chat
	//                      starts with the freshly pulled model; bare -pull
	//                      lists available models)
	// ─────────────────────────────────────────────────────────────────────
	fs := flag.NewFlagSet("chatllm", flag.ExitOnError)
	flagModelDir := fs.String("m", "", "Model directory (overrides SPROUT_LOCAL_MODEL_DIR)")
	flagSystem := fs.String("s", "", "System prompt for the session")
	flagPrompt := fs.String("p", "", "One-shot prompt: stream the response and exit")
	flagNoLog := fs.Bool("no-log", false, "Disable session logging")
	flagPull := fs.Bool("pull", false, "Download a catalog model (name as next arg; bare -pull lists)")
	flagTranscript := fs.Bool("transcript", false, "Read a conversation from stdin and stream a reply (pipe mode for scripting)")
	flagEOM := fs.Bool("eom", false, "With -transcript: append an end-of-message marker line")
	flagServe := fs.Bool("serve", false, "Host the embedded web chat UI (WebSocket + sinter in-process)")
	flagAddr := fs.String("addr", "127.0.0.1:8321", "Listen address for -serve")
	flagVerbose := fs.Bool("v", false, "Verbose: show engine load/debug output")
	flagTools := fs.String("tools", "", "Tool calling for this session: on, off, or yolo (overrides tools.json)")
	flagMaxSteps := fs.Int("max-steps", 0, "Tool round-trips per user turn (default 100; env SPROUT_LOCAL_MAX_STEPS)")
	flagMaxTokens := fs.Int("max-tokens", 0, "Generation token cap (default 4096; env SPROUT_LOCAL_MAX_TOKENS)")
	fs.Parse(os.Args[1:])

	config.InitTunables()
	if *flagMaxSteps > 0 {
		config.MaxToolSteps = *flagMaxSteps
	}
	if *flagMaxTokens > 0 {
		config.MaxTokens = *flagMaxTokens
	}
	config.LoadToolsPreference()
	if prefPath := config.ToolsPreferencePath(); prefPath != "" && !config.ToolsRequested {
		// A persisted off is a deliberate earlier choice, but it silently
		// overrides the default-on after every rebuild — surface it.
		log.Printf("tools off (persisted in %s — run /tools on to re-enable)", prefPath)
	}
	if *flagTools != "" {
		switch strings.ToLower(*flagTools) {
		case "on":
			config.ToolsRequested, config.ToolSafetyBypass = true, false
		case "off":
			config.ToolsRequested, config.ToolSafetyBypass = false, false
		case "yolo":
			config.ToolsRequested, config.ToolSafetyBypass = true, true
		default:
			log.Printf("ignoring -tools %q: want on, off, or yolo", *flagTools)
		}
	}

	// Engine chatter (sinter load/warmup lines) is filtered out unless -v
	// or CHATLLM_DEBUG is set. Installed before any model loads.
	chatmodel.InstallLogFilter(*flagVerbose || os.Getenv("CHATLLM_DEBUG") != "")

	// Skills (<stateRoot>/skills/*.json, default ~/.sprout-local/skills)
	// load once at startup; /tools reloads them in the REPL. Load
	// failures are non-fatal noise.
	if n, errs := tools.ReloadSkills(); len(errs) > 0 {
		for _, err := range errs {
			log.Printf("skill: %v", err)
		}
	} else if n > 0 {
		log.Printf("loaded %d skill%s from %s", n, tools.Plural(n), paths.SkillsDir())
	}

	// -serve: warm the default model's system+tools prefix in the
	// background so the first web turn delta-prefills instead of paying
	// the full prefill. Loading the weights is the cost; the server was
	// going to do it on first request anyway.
	if *flagServe {
		executor := core.ToolExecutor(core.NoopExecutor)
		if config.ToolsRequested {
			executor = tools.NewToolExecutor(nil)
		}
		dir := paths.ResolveModelDir()
		if dir != "" {
			go webui.Warm(dir, config.EffectiveSystemPrompt(*flagSystem), config.ToolsRequested, executor)
		}
	}

	// -pull: fetch the model first, then drop into normal startup with the
	// new model selected. A bare -pull lists the catalog and exits.
	if *flagPull {
		name := strings.TrimSpace(strings.Join(fs.Args(), " "))
		if name == "" {
			download.PrintPullList()
			return
		}
		m, err := download.FindCatalogModel(name)
		if err != nil {
			log.Fatalf("Fatal: %v", err)
		}
		dest, err := download.DownloadModel(chatmodel.CtxBg(), m)
		if err != nil {
			log.Fatalf("Fatal: %v", err)
		}
		fmt.Printf("Pulled %s → %s\n", m.Name, dest)
		// Select the freshly pulled model for this run.
		os.Setenv("SPROUT_LOCAL_MODEL_DIR", dest)
	}

	// -m override: applied before the model loads. resolveModelDir reads the
	// env var, so we set it for this process.
	if *flagModelDir != "" {
		os.Setenv("SPROUT_LOCAL_MODEL_DIR", *flagModelDir)
	}
	// -serve: host the embedded web UI + WebSocket API. Session logging is
	// REPL-only, so this path never touches the session log.
	if *flagServe {
		webui.Serve(*flagAddr)
		return
	}

	if !*flagNoLog {
		repl.EnableSessionLog()
	}

	// First-run walkthrough: a bare REPL on a model-less machine would
	// otherwise fatal out in modelBackend. Offer to download a model. Only
	// the interactive REPL reaches this — -p one-shot and -transcript pipe
	// mode bail out of the flow, and -serve / -pull returned earlier.
	replMode := *flagPrompt == "" && !*flagTranscript
	if replMode && paths.ResolveModelDir() == "" {
		repl.FirstRun(chatmodel.CtxBg(), bufio.NewReader(os.Stdin))
	}

	// Startup probe — mirrors gmitllm's chatmodel.ModelBackend(): fail fast with a
	// clear message instead of a deep sinter load error on first message.
	engine, modelPath := chatmodel.ModelBackend()

	repl.StartSignalWatch()

	// One-shot mode: single completion, no REPL.
	if *flagPrompt != "" {
		repl.RunOneShot(config.EffectiveSystemPrompt(*flagSystem), *flagPrompt, engine, modelPath)
		return
	}

	// Transcript mode: read a JSON conversation from stdin, stream the
	// assistant reply, and exit. Used by chat/server.py to drive the same
	// engine over a pipe. With -eom, an end-of-message marker line is
	// appended for consumers that stream without EOF visibility.
	if *flagTranscript {
		if err := repl.RunOneShotPipe(engine, modelPath, *flagNoLog, *flagEOM); err != nil {
			log.Fatalf("Fatal: %v", err)
		}
		return
	}

	repl.Run(engine, modelPath, *flagSystem)
}

// runOneShotPipe reads {"messages":[{role,content}...]} JSON from stdin,
// streams the assistant response to stdout, and (with eom) finishes with
// the chatmodel.EndOfMessage marker line. Diagnostics stay on stderr. When the
// transcript has no messages, nothing is generated — with eom the marker
// is still emitted so pipe consumers see a well-formed (empty) exchange.

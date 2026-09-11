# sprout-local — Local Chat With Some Extra Help

## What It Does

A chat-first REPL over local models only — no API providers, no cloud.
Single Go binary: in-process inference via
[sinter](https://github.com/sprout-foundry/sinter) (no external LLM
server, no GGUF), the [seed](https://github.com/sprout-foundry/seed)
agent loop (the same spine as sprout), and a small set of helper tools
(read files, take notes to files, quick shell lookups, fetch a page) for
when plain chat isn't enough. It is not a coding agent — no repo
awareness, no code-editing loop; tools exist to help everyday chat
questions, not to drive a codebase. Switch models mid-conversation from
the shared local models root (`/models`, `/model`, `/pull`).
`sprout-local -serve` also hosts an embedded web chat UI (single binary, no
runtime dependencies).

## Architecture

Single Go binary. seed owns the conversation loop and tool iteration;
sinter (Go-native engine, Apple Silicon via MLX; Linux GPUs via GGML)
runs inference in-process against an MLX-format model directory. Tool
calling is the qwen3.5 text protocol (the model's own `chat_template.jinja`
format) rendered/parsed by the seed provider adapter — sinter's API has no
structured tools.

### File Layout

| File | Purpose |
|------|---------|
| `main.go` | Entry point, CLI flags, REPL loop, slash commands, `replState` (agent lifecycle), stream filter |
| `seedprovider.go` | sinter as a seed `core.Provider`: render seed messages → qwen tool protocol, parse `<tool_call>` text back into structured calls |
| `seedexecutor.go` | the tool registry as a seed `core.ToolExecutor` (run_command y/N gate via seed UI) |
| `replevents.go` | seed events → REPL status lines (`tool →` / `← result:`) |
| `tools.go` | helper tool registry (read_file, write_file, run_command, web_fetch), path sandbox, qwen tool-prompt/parse helpers, `/tools` command |
| `skills.go` | user skills: JSON-defined fixed-command tools in `~/.chatllm/skills/*.json`, loaded by `/tools` |
| `apiserver.go` | `-serve` OpenAI-compatible API: `/v1/chat/completions` (stream + tool_calls) routed to any installed model, `/v1/models`, `/health` |
| `modelcmd.go` | `/model`, `/models`, `/pull` slash commands: model switching, listing, cache eviction |
| `webui.go` | `-serve`: embedded web UI (go:embed), WebSocket chat on a per-connection seed agent (tools, status/metrics frames), conversation persistence |
| `ui/index.html` | Web chat UI — static, dependency-free, embedded at build time |
| `chatmodel.go` | sinter inference: model load/cache, streaming generation, output hygiene |
| `chatmodel_darwin.go` | Per-platform sinter arch registration — Apple Silicon/MLX (all archs) |
| `chatmodel_linux_ggml.go` | Per-platform sinter arch registration — Linux/GGML (qwen2 + qwen35 only) |
| `backend.go` | Startup model probe (fails fast when no model dir resolves) |
| `signalwatch.go` | Ctrl-C: first signal cancels the in-flight generation, second exits |
| `sessionlog.go` | Session logs to `~/.cmd_chat_llm/<timestamp>.log` |
| `download.go` | `-pull`: HF download via `hf` CLI, sinter catalog metadata, RAM gate |
| `urlfetch.go` | Web `#url` enrichment: fetch + readable-text extraction for prompts |
| `factcheck.go` | Post-answer Yes/No self-check using the loaded model (web UI) |
| `mdterm.go` | Streaming markdown→ANSI renderer for REPL/one-shot output |
| `sysinfo_*.go` | Build-tagged total RAM detection (darwin/linux/other) |

### Skills

User-addable tools: drop a JSON file in `~/.chatllm/skills/` —
`{"name":"motd","description":"…","command":"/bin/echo","args":["hi"]}` —
and `/tools` picks it up (fixed command line, no model-supplied params;
installing the skill is the consent, so no per-call prompt). Skills run
through the seed agent loop like built-ins.

`replState.newAgent()` builds `core.Agent{Provider: sinterProvider,
Executor: toolExecutor|NoopExecutor, UI: termUI, MaxIterations: 4}`.
`/model`, `/pull`, `/system`, `/tools` rebuild the agent with
`ExportState`/`ImportState` carrying the conversation across. Tools off →
`NoopExecutor`, so the model never sees the tool protocol. `CHATLLM_SEED_DEBUG=1`
enables seed's loop trace. go.mod `replace`s seed to the sibling checkout
`~/dev/sprout-foundry/seed`.

### Seed wiring

## Running

```bash
sprout-local             # interactive REPL
sprout-local -p "question"    # one-shot: stream response and exit
sprout-local -s "prompt"      # set a session system prompt
sprout-local -m <model-dir>   # override the model directory
sprout-local -no-log          # disable session logging
sprout-local -pull            # list downloadable catalog models (RAM-tier annotated)
sprout-local -pull <name>     # download from HuggingFace, then chat with it
sprout-local -serve           # web UI + OpenAI-style /v1 API at http://127.0.0.1:8321
sprout-local -serve -addr :8321  # custom listen address
```

### REPL commands

| Command | Purpose |
|---------|---------|
| `/new` | New conversation (drops history, keeps system prompt) |
| `/system [text]` | Show or set the system prompt |
| `/model [ref]` | Show or switch the model (history kept; ref = dir name, catalog name, or path) |
| `/models` | List installed models (`*` marks the session model) |
| `/pull [name]` | Download a catalog model, switch to it, confirm with a greeting (bare `/pull` lists) |
| `/tools [on\|off\|yolo]` | Toggle tool calling (read_file, write_file, run_command, web_fetch); `yolo` skips the run_command confirm |
| `/history` | Show the conversation so far |
| `/exit` | Quit (also `/quit`, `/q`); Ctrl-D also exits |
| `"""` | Open/close a multiline input block (like a heredoc) |

Ctrl-C cancels the current response and keeps the session; press twice quickly to quit.

## Web UI (`-serve`)

- Embedded via `go:embed all:ui`; `UI_CHATLLM_DEV=1 sprout-local -serve` serves `ui/` from disk for live editing.
- WebSocket chat runs the same seed agent as the REPL: per-connection agent, tools/skills via the ⚙ toggle (run_command auto-declines on web), tool chips, stop/retry, conversation sidebar with resume.
- OpenAI-compatible API on the same port: `POST /v1/chat/completions` (SSE streaming, native `tool_calls` via sinter's openaisserver), `GET /v1/models` (all installed models; the request's `model` field selects any of them), `GET /health`.
- `#url` enrichment: `#https://…` in a prompt fetches the page (2 MiB / 8k-char caps, code-block URLs ignored) and appends its readable text to the turn; fetch failures arrive as `{"note":...}` frames.
- Fact self-check: after each web turn the same model re-reads the turn as a reference and answers Yes/No at temperature 0; a confident "No" emits a warning note frame (`factcheck.go`).
- `/models` lists MLX model dirs in the shared models root (default first); the UI can switch models mid-conversation (history is preserved).
- Conversations persist per user (server-side JSON in `~/.chatllm/conversations/`); a disconnect cancels the in-flight generation and the client resumes the conversation on reconnect (partial answers from Stop are kept).
- Palette: One Light / One Dark (from the retired `chat/` UI); theme toggle is persisted in localStorage.

## Model Resolution Order

`resolveModelDir()` checks in this order:
1. `LOCAL_MODEL_DIR` environment variable (or `-m` flag)
2. `~/dev/llm-models/qwen3.5-4b-sprout-tuned-mlx-q5` (shared models root)

## Key Constants

- `maxTokens = 2048` — generation token limit (chat needs more than commit messages)
- `temperature = 0.4` — a little more variety than gmitllm's 0.2
- `maxToolSteps = 4` — seed agent iterations per user turn (tool round-trips)
- `historyLimit = 200` — web-UI history cap; the REPL uses seed compaction instead
- Tool sandbox: file tools stay under cwd + `/tmp`/`$TMPDIR`; run_command blocks shell metacharacters (`|;&\`` and redirects), 30s timeout

## Testing

```bash
make test    # unit tests
make build   # build ./sprout-local
make install # symlink into ~/.local/bin
```

## Building

The Makefile selects the sinter backend per platform:
- **macOS (Apple Silicon)** → MLX backend: `CGO_ENABLED=1 go build` (no extra flags)
- **Linux / Termux** → GGML backend: `CGO_ENABLED=1 go build -tags ggml`.
  On Termux, the Makefile points `CGO_CFLAGS`/`CGO_LDFLAGS` at `$PREFIX`
  (sinter's cgo directives hardcode `/usr/local`), where `libggml`/`libggml-base` live.

**`third_party/sinter`** — a local copy of sinter v0.1.1 wired via `replace`
in go.mod. It carries local patches because upstream v0.1.1/v0.1.2 cannot
compile on Linux+GGML (undefined `compiledDecode` in `llm/qwen35`; the
qwen3 closure stub tag excluded linux+amd64). Tracked in
https://github.com/sprout-foundry/sinter/issues/1.

## Design Decisions

- **seed owns the loop** — sprout-local is a thin UI over seed's agent (same spine as sprout): query → tool calls → response, with compaction, retries, and interrupts. sprout-local contributes the sinter provider, the tool executor, and terminal rendering.
- **Text-protocol tool calls** — sinter has no structured tool API, so the provider renders seed's `Tools` into the qwen3.5 template's native `# Tools`/`<tool_call>`/`<tool_response>` text and parses calls back out; seed sees ordinary structured tool calls.
- **In-process streaming** — sinter's `Generate` takes an onToken callback; each token is decoded and streamed through a filter that suppresses `<tool_call>` markup, while the parsed clean text enters seed state and logs.
- **Full re-render per turn** — the entire message list goes through `FormatChat` each turn; sinter's prefix caching makes repeat prefixes cheap.
- **Tools default off** — chat first, tools when asked. A 4B model spends tokens and attention on the protocol; `/tools on` opts in per session. Off means `NoopExecutor`, so the model never sees the tool prompt.
- **Session logs** — every exchange appends to `~/.cmd_chat_llm/` (the old bash tool's location), so scrollback survives terminal loss.
- **Download stays app-level** — sinter's README keeps the catalog separate from the engine "so apps can keep their own list"; the engine has no download API. The `hf` CLI mechanics (pipe draining, disk-based progress polling) are ported from sprout's `localmodel.EnsureModel`.

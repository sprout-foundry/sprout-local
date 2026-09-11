# sprout-local

Local chat with some extra help. A lightweight sibling of
[sprout](https://github.com/sprout-foundry/sprout): the same
[seed](https://github.com/sprout-foundry/seed) agent spine, in-process
inference via [sinter](https://github.com/sprout-foundry/sinter) (MLX on
Apple Silicon, GGML on Linux), switchable local models, and a small set
of helper tools. Chat-first — not a coding agent.

- **Local only.** No API providers, no cloud. Models come from the
  shared local models root (`~/dev/llm-models`).
- **Chat-first.** Tools exist to help everyday questions — read a file,
  jot a note, quick shell lookup, fetch a page — not to drive a codebase.
- **One binary.** `sprout-local` chats in the terminal; `-serve` hosts a
  web chat UI and an OpenAI-compatible endpoint.

## Running

```bash
sprout-local                # interactive REPL
sprout-local -p "question"  # one-shot: stream response and exit
sprout-local -s "prompt"    # set a session system prompt
sprout-local -m <model-dir> # override the model directory
sprout-local -pull          # list downloadable catalog models
sprout-local -pull <name>   # download from HuggingFace, then chat
sprout-local -serve         # web UI + OpenAI-style /v1 API at http://127.0.0.1:8321
sprout-local -v             # verbose: show engine load/debug output
```

### REPL commands

| Command | Purpose |
|---------|---------|
| `/new` | New conversation |
| `/system [text]` | Show or set the system prompt |
| `/model [ref]` | Show or switch the model (history kept) |
| `/models` | List installed models |
| `/pull [name]` | Download a catalog model, switch to it |
| `/tools [on\|off\|yolo]` | Toggle tools (read_file, write_file, run_command, web_fetch + skills) |
| `/history` | Show the conversation |
| `/exit` | Quit |

`"""` opens a multiline block. Ctrl-C cancels the current response; twice quits.

## Tools and skills

Tools are opt-in per session (`/tools on`) because small models spend
tokens and attention on the protocol. File tools are sandboxed to the
working directory plus `/tmp`; `run_command` blocks shell
metacharacters and asks y/N (`/tools yolo` skips).

User skills are JSON-defined fixed-command tools in
`~/.chatllm/skills/*.json`:

```json
{
  "name": "motd",
  "description": "Print today's message of the day",
  "command": "/bin/echo",
  "args": ["hello"]
}
```

Fixed command lines, no model-supplied parameters — installing the
skill is the consent.

## Web UI and API (`-serve`)

- Chat UI with model picker, conversation sidebar (server-side
  persistence in `~/.chatllm/conversations/`), streaming with status
  and metrics (prompt/gen tokens, t/s, context use), stop/retry, and
  click-to-expand tool chips. Files written as HTML get preview links.
- OpenAI-compatible endpoint on the same port:
  `POST /v1/chat/completions` (SSE streaming, native `tool_calls`),
  `GET /v1/models`, `GET /health`. The request's `model` field selects
  any installed model.

## Architecture

Single Go binary. seed owns the conversation loop (compaction,
interrupts, state export); sinter runs inference in-process. Tool
calling uses qwen3.5's native text protocol — declarations in a
`# Tools` system block, calls as `<tool_call>` markup, results as
`<tool_response>` — rendered and parsed by the seed provider adapter,
since sinter's API has no structured tools. A single `runGeneration`
core serves every surface: generation config, metrics, a leak/repetition
guard, and output hygiene.

See `AGENTS.md` for the file map and design decisions.

## Building

```bash
make build   # ./sprout-local (MLX on macOS; -tags ggml on Linux)
make test
make install # symlink into ~/.local/bin
```

`third_party/sinter` vendors a patched sinter (linux+ggml fix,
`llm/qwen35/compiled_stub.go`); drop it once upstream releases the fix.
`seed` resolves to the sibling checkout `../seed` via a go.mod replace.

# sprout-local

Local chat with some extra help. A lightweight sibling of
[sprout](https://github.com/sprout-foundry/sprout): the same
[seed](https://github.com/sprout-foundry/seed) agent spine, in-process
inference via [sinter](https://github.com/sprout-foundry/sinter) (MLX on
Apple Silicon, GGML on Linux), switchable local models, and a small set
of helper tools. Chat-first — not a coding agent.

- **Local only.** No API providers, no cloud. Models download from
  HuggingFace via `-pull` into the local models root
  (`~/.sprout-local/models`, overridden by `SPROUT_LOCAL_MODELS_ROOT`).
- **Chat-first.** Tools exist to help everyday questions — read a file,
  jot a note, quick shell lookup, fetch a page — not to drive a codebase.
- **One binary.** `sprout-local` chats in the terminal; `-serve` hosts a
  web chat UI and an OpenAI-compatible endpoint.

## Install

One line (macOS, Apple Silicon or Intel; Linux builds included):

```bash
curl -fsSL https://raw.githubusercontent.com/sprout-foundry/sprout-local/main/scripts/install.sh | sh
```

Pin a version, or override install location:

```bash
SPROUT_LOCAL_VERSION=v0.1.0 curl -fsSL …/install.sh | sh
```

The script downloads the release tarball for your platform, verifies the
SHA256, and installs to `~/.local/bin` (`/usr/local/bin` when run as
root). Add `~/.local/bin` to your PATH if it isn't already.

Build from source instead:

```bash
git clone https://github.com/sprout-foundry/sprout-local.git
cd sprout-local && make build && make install
```

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
`~/.sprout-local/skills/*.json`:

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
  persistence in `~/.sprout-local/conversations/`), streaming with status
  and metrics (prompt/gen tokens, t/s, context use), stop/retry, and
  click-to-expand tool chips. Files written as HTML get preview links.
- OpenAI-compatible endpoint on the same port:
  `POST /v1/chat/completions` (SSE streaming, native `tool_calls`),
  `GET /v1/models`, `GET /health`. The request's `model` field selects
  any installed model.

## Architecture

One Go binary built from a standard layout: `cmd/sprout-local` (thin
entrypoint) over `internal/` packages (`repl`, `webui`, `apiserver`,
`provider`, `tools`, `chatmodel`, `config`, `download`, …).
seed owns the conversation loop (compaction,
interrupts, state export); sinter runs inference in-process. Tool
calling uses qwen3.5's native text protocol — declarations in a
`# Tools` system block, calls as `<tool_call>` markup, results as
`<tool_response>` — rendered and parsed by the seed provider adapter,
since sinter's API has no structured tools. A single `RunGeneration`
core serves every surface: generation config, metrics, a leak/repetition
guard, and output hygiene.

See `AGENTS.md` for the file map and design decisions.

## Building

```bash
make build    # ./sprout-local (MLX on macOS; -tags ggml on Linux)
make test
make install  # symlink into ~/.local/bin
```

Releases: push a `v*` tag — `.github/workflows/release.yml` builds
darwin/linux × arm64/amd64 tarballs plus a `SHA256SUMS` manifest and
publishes them to the GitHub Release, which `scripts/install.sh`
consumes for the one-line install.

Both `sinter` and `seed` are ordinary Go module dependencies, resolved
from the module proxy by version — no `replace` directives, nothing
vendored in-tree.

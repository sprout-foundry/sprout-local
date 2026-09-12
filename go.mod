module github.com/sprout-foundry/sprout-local

go 1.25.6

require github.com/sprout-foundry/sinter v0.2.1

require golang.org/x/sys v0.45.0

require github.com/gorilla/websocket v1.5.3

require github.com/sprout-foundry/seed v1.4.0

// Local copy of sinter, pinned verbatim to v0.2.1 (plus .gitignore). The
// original reason for the fork — upstream v0.1.1/v0.1.2 could not compile on
// Linux+GGML (undefined compiledDecode in llm/qwen35) — is fixed upstream as
// of 8561fc0, released in v0.2.1. To drop the fork: delete this replace and
// third_party/, then verify a Linux+GGML build (make build on Linux/Termux).
replace github.com/sprout-foundry/sinter => ./third_party/sinter

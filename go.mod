module github.com/sprout-foundry/sprout-local

go 1.25.6

require github.com/sprout-foundry/sinter v0.1.1

require golang.org/x/sys v0.45.0

require github.com/gorilla/websocket v1.5.3

require github.com/sprout-foundry/seed v0.0.0

// Local seed checkout (agent loop: query → tool calls → response).
replace github.com/sprout-foundry/seed => ../seed

// Local copy of sinter v0.1.1 with a linux+ggml patch (llm/qwen35/compiled_stub.go)
// added because upstream v0.1.2 cannot compile on Linux+GGML (undefined
// compiledDecode in llm/qwen35). See the comment in that stub file. Drop the
// replace (and third_party/) once the upstream fix is released.
replace github.com/sprout-foundry/sinter => ./third_party/sinter

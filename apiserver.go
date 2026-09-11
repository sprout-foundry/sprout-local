package main

// ---------------------------------------------------------------------------
// apiserver.go — OpenAI-compatible endpoint for chatllm -serve.
//
// Wraps sinter's openaisserver (the same contract sprout consumes: SSE
// streaming, [DONE] marker, tool-call text protocol, /v1/models,
// /health) with multi-model routing: the request's "model" field selects
// any installed model in the shared models root, loaded on first use and
// cached in sinter's model cache (with LRU eviction from /model
// switching). Endpoints:
//
//	POST /v1/chat/completions   (stream + tools, OpenAI wire format)
//	GET  /v1/models             (all installed models)
//	GET  /health                (default model + status)
//
// ---------------------------------------------------------------------------

import (
	"bytes"
	"encoding/json"
	"io"
	"net/http"
	"path/filepath"
	"sort"
	"strings"
	"sync"
	"time"

	"github.com/sprout-foundry/sinter/llm/openaisserver"
)

// apiServer routes OpenAI-style requests to per-model openaisserver
// instances. Generation is serialized per model by the inner servers; the
// single-GPU constraint is handled by sinter's model mutex.
type apiServer struct {
	mu    sync.Mutex
	byID  map[string]*openaisserver.Server
	defID string
}

func newAPIServer() *apiServer {
	return &apiServer{byID: map[string]*openaisserver.Server{}, defID: filepath.Base(resolveModelDir())}
}

// serverFor returns (building if needed) the openaisserver for a model.
// The model argument is a bare name from /v1/models or the default when
// empty. Unknown models 400.
func (a *apiServer) serverFor(name string) (*openaisserver.Server, error) {
	a.mu.Lock()
	defer a.mu.Unlock()

	name = strings.TrimSpace(name)
	if name == "" {
		name = a.defID
	}
	if s, ok := a.byID[name]; ok {
		return s, nil
	}
	dir, err := resolveWebModelDir(name)
	if err != nil {
		return nil, err
	}
	m, err := loadModelDir(dir)
	if err != nil {
		return nil, err
	}
	s := openaisserver.New(m, filepath.Base(dir), maxTokens)
	a.byID[filepath.Base(dir)] = s
	return s, nil
}

// HandleChatCompletions adapts the OpenAI route to the per-model server.
// The body is buffered so the model-field probe and openaisserver's own
// decode both see it.
func (a *apiServer) HandleChatCompletions(w http.ResponseWriter, r *http.Request) {
	if r.Method != http.MethodPost {
		http.Error(w, "POST only", http.StatusMethodNotAllowed)
		return
	}
	body, err := io.ReadAll(r.Body)
	if err != nil {
		http.Error(w, "read body: "+err.Error(), http.StatusBadRequest)
		return
	}
	r.Body = io.NopCloser(bytes.NewReader(body)) // replay for HandleChat

	var probe struct {
		Model string `json:"model"`
	}
	if err := json.Unmarshal(body, &probe); err != nil {
		http.Error(w, "bad JSON body: "+err.Error(), http.StatusBadRequest)
		return
	}
	s, err := a.serverFor(probe.Model)
	if err != nil {
		// Match the OpenAI error shape well enough for SDKs to surface it.
		w.Header().Set("Content-Type", "application/json")
		w.WriteHeader(http.StatusNotFound)
		json.NewEncoder(w).Encode(map[string]interface{}{
			"error": map[string]string{
				"message": err.Error(),
				"type":    "invalid_request_error",
				"code":    "model_not_found",
			},
		})
		return
	}
	s.HandleChat(w, r)
}

// HandleModels lists every installed model (default first).
func (a *apiServer) HandleModels(w http.ResponseWriter, r *http.Request) {
	w.Header().Set("Content-Type", "application/json")
	def := a.defID
	names := availableModels()
	if names == nil {
		names = []string{def}
	}
	sort.Slice(names, func(i, j int) bool {
		if (names[i] == def) != (names[j] == def) {
			return names[i] == def
		}
		return names[i] < names[j]
	})
	type mi struct {
		ID            string `json:"id"`
		Object        string `json:"object"`
		Created       int64  `json:"created"`
		OwnedBy       string `json:"owned_by"`
		ContextLength int    `json:"context_length"`
	}
	now := time.Now().Unix()
	list := make([]mi, 0, len(names))
	for _, n := range names {
		// ContextLength only for already-resident models: listing must not
		// trigger a multi-second model load just to fill one field.
		ctxLen := 0
		if dir, err := resolveWebModelDir(n); err == nil && isModelLoaded(dir) {
			if m, err := loadModelDir(dir); err == nil {
				ctxLen = m.ContextLength()
			}
		}
		list = append(list, mi{ID: n, Object: "model", Created: now, OwnedBy: "chatllm-local", ContextLength: ctxLen})
	}
	json.NewEncoder(w).Encode(map[string]interface{}{"object": "list", "data": list})
}

// HandleHealth reports service status and the default model.
func (a *apiServer) HandleHealth(w http.ResponseWriter, r *http.Request) {
	w.Header().Set("Content-Type", "application/json")
	json.NewEncoder(w).Encode(map[string]string{"status": "ok", "model": a.defID})
}

// serveAPI mounts the OpenAI-compatible API on the -serve mux.
func serveAPI(mux *http.ServeMux) {
	a := newAPIServer()
	mux.HandleFunc("/v1/chat/completions", a.HandleChatCompletions)
	mux.HandleFunc("/v1/models", a.HandleModels)
	mux.HandleFunc("/health", a.HandleHealth)
}

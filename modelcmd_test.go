package main

import (
	"bufio"
	"os"
	"path/filepath"
	"strings"
	"testing"

	"github.com/sprout-foundry/sinter/llm"
)

// makeFakeModelDir creates a directory that passes isModelDir (config.json
// + tokenizer.json + a *.safetensors file). Returns the dir path.
func makeFakeModelDir(t *testing.T, root, name string) string {
	t.Helper()
	dir := filepath.Join(root, name)
	if err := os.MkdirAll(dir, 0o755); err != nil {
		t.Fatal(err)
	}
	for _, f := range []string{"config.json", "tokenizer.json", "model.safetensors"} {
		if err := os.WriteFile(filepath.Join(dir, f), []byte("{}"), 0o644); err != nil {
			t.Fatal(err)
		}
	}
	return dir
}

func resetModelCache() {
	modelMu.Lock()
	defer modelMu.Unlock()
	modelCache = map[string]*llm.Model{}
}

func TestModelFromRef(t *testing.T) {
	root := t.TempDir()
	t.Setenv("SPROUT_LOCAL_MODELS_ROOT", root)
	makeFakeModelDir(t, root, "alpha-1b")

	// Bare name under the models root.
	dir, err := modelFromRef("alpha-1b")
	if err != nil || dir != filepath.Join(root, "alpha-1b") {
		t.Errorf("bare name: dir=%q err=%v", dir, err)
	}
	// Absolute path passthrough.
	dir, err = modelFromRef(filepath.Join(root, "alpha-1b"))
	if err != nil || dir != filepath.Join(root, "alpha-1b") {
		t.Errorf("abs path: dir=%q err=%v", dir, err)
	}
	// Unknown name.
	if _, err := modelFromRef("nope"); err == nil {
		t.Error("unknown name should error")
	}
	// Empty ref.
	if _, err := modelFromRef("   "); err == nil {
		t.Error("empty ref should error")
	}
	// Catalog name whose Dir is not installed: error suggests /pull.
	m, err := findCatalogModel("gemma4-e2b")
	if err != nil {
		t.Skip("gemma4-e2b not in catalog")
	}
	_, err = modelFromRef(m.Dir)
	if err == nil {
		t.Error("uninstalled catalog Dir should error")
	} else if !strings.Contains(err.Error(), "/pull") {
		t.Errorf("uninstalled catalog error should suggest /pull, got: %v", err)
	}
}

func TestAvailableModelNames(t *testing.T) {
	root := t.TempDir()
	t.Setenv("SPROUT_LOCAL_MODELS_ROOT", root)
	// Pin the process default inside the root so the expected list is
	// machine-independent (the real default dir may or may not exist).
	alpha := makeFakeModelDir(t, root, "alpha-1b")
	t.Setenv("LOCAL_MODEL_DIR", alpha)
	makeFakeModelDir(t, root, "beta-2b")
	os.MkdirAll(filepath.Join(root, "not-a-model"), 0o755) // fails isModelDir

	names, def := availableModelNames()
	if len(names) != 2 || names[0] != "alpha-1b" || names[1] != "beta-2b" {
		t.Errorf("names = %v, want [alpha-1b beta-2b]", names)
	}
	if def != "alpha-1b" {
		t.Errorf("def = %q, want alpha-1b", def)
	}
}

func TestEvictModels(t *testing.T) {
	root := t.TempDir()
	resetModelCache()
	defer resetModelCache()

	for _, n := range []string{"a", "b", "c"} {
		d := makeFakeModelDir(t, root, n)
		modelCache[d] = nil // entries stand in for loaded models
	}
	recent := filepath.Join(root, "c")

	// A loaded model would be Closed here; nil entries make Close a no-op.
	// The point is the bookkeeping: keep limit honored, recent survives.
	evictModels(2, recent)

	if len(modelCache) != 2 {
		t.Errorf("after evict: %d cached, want 2", len(modelCache))
	}
	if _, ok := modelCache[recent]; !ok {
		t.Error("evictModels dropped the recent model")
	}
	if _, ok := modelCache[filepath.Join(root, "a")]; ok {
		t.Error("evictModels kept the oldest model")
	}

	// Under the limit: no-op.
	evictModels(5, recent)
	if len(modelCache) != 2 {
		t.Errorf("under-limit evict changed the cache: %d cached", len(modelCache))
	}
}

// TestDispatchModelCommands covers the wiring without loading real models:
// show and failed lookups must never switch the session model or exit.
func TestDispatchModelCommands(t *testing.T) {
	root := t.TempDir()
	t.Setenv("SPROUT_LOCAL_MODELS_ROOT", root)
	resetModelCache()
	defer resetModelCache()

	st := &replState{ui: &termUI{reader: bufio.NewReader(strings.NewReader(""))}}
	st.newAgent()

	// /model show: no switch, no exit.
	if dispatchCommand("model", "", st) {
		t.Error("/model must not signal exit")
	}
	if st.model == filepath.Join(root, "bogus") {
		t.Errorf("/model show switched the model: %q", st.model)
	}
	// /model with an unknown ref: error path, still no switch.
	before := st.model
	if dispatchCommand("model", "bogus-model", st) {
		t.Error("/model must not signal exit")
	}
	if st.model != before {
		t.Errorf("failed switch changed the active model: %q", st.model)
	}
	// /models listing: no exit.
	if dispatchCommand("models", "", st) {
		t.Error("/models must not signal exit")
	}
	// /pull with an unknown name: error path, no switch.
	if dispatchCommand("pull", "bogus-model", st) {
		t.Error("/pull must not signal exit")
	}
	if st.model != before {
		t.Errorf("failed pull changed the active model: %q", st.model)
	}
}

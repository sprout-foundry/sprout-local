package chatmodel

import (
	"os"
	"path/filepath"
	"testing"

	"github.com/sprout-foundry/sinter/llm"
)

// makeFakeModelDir creates a directory that passes paths.IsModelDir
// (config.json + tokenizer.json + a *.safetensors file). Returns the dir.
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
	Evict(2, recent)

	if len(modelCache) != 2 {
		t.Errorf("after evict: %d cached, want 2", len(modelCache))
	}
	if _, ok := modelCache[recent]; !ok {
		t.Error("Evict dropped the recent model")
	}
	if _, ok := modelCache[filepath.Join(root, "a")]; ok {
		t.Error("Evict kept the oldest model")
	}

	// Under the limit: no-op.
	Evict(5, recent)
	if len(modelCache) != 2 {
		t.Errorf("under-limit evict changed the cache: %d cached", len(modelCache))
	}
}
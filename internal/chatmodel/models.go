package chatmodel

// Model listing and name→directory resolution shared by the web UI and
// the OpenAI-compatible API server: both surfaces expose "installed
// models by name" and accept a name or absolute path per request.

import (
	"fmt"
	"os"
	"path/filepath"
	"sort"
	"strings"

	"github.com/sprout-foundry/sprout-local/internal/paths"
)

// AvailableModels lists the MLX model directories under the shared models
// root (names only, sorted). A pinned SPROUT_LOCAL_MODEL_DIR means no
// listing: the process has exactly one model.
func AvailableModels() []string {
	if _, pinned := paths.ModelDirEnv(); pinned != "" {
		return nil // pinned: no listing
	}
	root := paths.ModelsRoot()
	entries, err := os.ReadDir(root)
	if err != nil {
		return nil
	}
	var names []string
	for _, e := range entries {
		if e.IsDir() && paths.IsModelDir(filepath.Join(root, e.Name())) {
			names = append(names, e.Name())
		}
	}
	sort.Strings(names)
	return names
}

// ResolveModelName maps a UI/API model name to a model directory. Bare
// names resolve against the shared models root; absolute paths pass
// through. Empty name → the process default (paths.ResolveModelDir).
func ResolveModelName(name string) (string, error) {
	name = strings.TrimSpace(name)
	if name == "" {
		dir := paths.ResolveModelDir()
		if dir == "" {
			return "", fmt.Errorf("no model configured (set SPROUT_LOCAL_MODEL_DIR or -m)")
		}
		return dir, nil
	}
	if filepath.IsAbs(name) {
		if paths.IsModelDir(name) {
			return name, nil
		}
		return "", fmt.Errorf("%s is not an MLX-format model directory", name)
	}
	cand := filepath.Join(paths.ModelsRoot(), name)
	if paths.IsModelDir(cand) {
		return cand, nil
	}
	return "", fmt.Errorf("model %q not found under %s", name, paths.ModelsRoot())
}

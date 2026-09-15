package paths

// ---------------------------------------------------------------------------
// modeldir.go — pure model-directory resolution: the env-var override, the
// models-root scan, and the RAM-aware catalog pick. No model loading or
// cache state — the chatmodel package owns that layer.
// ---------------------------------------------------------------------------

import (
	"log"
	"os"
	"path/filepath"
	"sort"

	"github.com/sprout-foundry/sinter/llm/catalog"

	"github.com/sprout-foundry/sprout-local/internal/sysinfo"
)

// preferredModelName is the first model picked when scanning the models
// root: the q5 tuned 4B export. Below ~5-bit the 4B tier loses too much
// to follow tool-call format and multi-step instructions reliably; the q5
// tuned export is the smallest quant that stays cogent at this size.
const preferredModelName = "qwen3.5-4b-sprout-tuned-mlx-q5"

// ModelDirEnv returns the explicit model-directory override:
// SPROUT_LOCAL_MODEL_DIR first, the legacy LOCAL_MODEL_DIR as fallback.
// The variable name that supplied it is returned so error messages can
// point at the right one; ("", "") when neither is set.
func ModelDirEnv() (varName, dir string) {
	if v := os.Getenv("SPROUT_LOCAL_MODEL_DIR"); v != "" {
		return "SPROUT_LOCAL_MODEL_DIR", v
	}
	if v := os.Getenv("LOCAL_MODEL_DIR"); v != "" {
		return "LOCAL_MODEL_DIR", v
	}
	return "", ""
}

// BestInstalledModel scans the models root for MLX-format model
// directories and returns the best installed one: the tuned 4B export
// when present (preserves the historical default where it lives),
// otherwise the alphabetically-first model dir. A missing or empty root
// yields "" — the caller surfaces the actionable error.
func BestInstalledModel() string {
	root := ModelsRoot()
	if root == "" {
		return ""
	}
	entries, err := os.ReadDir(root)
	if err != nil {
		return ""
	}
	var names []string
	for _, e := range entries {
		if !e.IsDir() || !IsModelDir(filepath.Join(root, e.Name())) {
			continue
		}
		if e.Name() == preferredModelName {
			return filepath.Join(root, e.Name())
		}
		names = append(names, e.Name())
	}
	if len(names) == 0 {
		return ""
	}
	sort.Strings(names)
	return filepath.Join(root, names[0])
}

// ResolveModelDir returns the MLX-format model directory for sinter, or
// "" when no model can be found (the caller surfaces the error).
//
// Resolution order:
//  1. SPROUT_LOCAL_MODEL_DIR, or the legacy LOCAL_MODEL_DIR (MLX-format
//     directory: config.json + tokenizer.json + *.safetensors)
//  2. the RAM-aware catalog pick for the models root
//     (SPROUT_LOCAL_MODELS_ROOT or <stateRoot>/models):
//     catalog.SelectModelForRAM walks the suggested tier down to smaller
//     ones plus one stretch tier, preferring installed sprout-tuned
//     variants and honoring the memory gate
//  3. the BestInstalledModel scan as a fallback, which also covers
//     manually-added non-catalog model dirs that SelectModelForRAM cannot
//     see
func ResolveModelDir() string {
	if varName, envDir := ModelDirEnv(); envDir != "" {
		if IsModelDir(envDir) {
			return envDir
		}
		log.Fatalf("%s=%s does not look like a model directory (need config.json + tokenizer.json + *.safetensors)", varName, envDir)
	}
	root := ModelsRoot()
	if root != "" {
		if m, err := catalog.SelectModelForRAM(root, sysinfo.TotalSystemRAM()); err == nil && m != nil {
			if IsModelDir(m.Dir) {
				return m.Dir
			}
		}
	}
	return BestInstalledModel()
}

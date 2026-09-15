package main

import (
	"os"
	"path/filepath"
	"testing"
)

// TestResolveModelDirEnvOrder covers the model-dir override precedence:
// SPROUT_LOCAL_MODEL_DIR beats legacy LOCAL_MODEL_DIR, and both beat the
// scanned models root; with neither set the best installed model wins.
func TestResolveModelDirEnvOrder(t *testing.T) {
	root := t.TempDir()
	rootTuned := makeFakeModelDir(t, root, preferredModelName)
	alpha := makeFakeModelDir(t, root, "alpha-1b")
	t.Setenv("SPROUT_LOCAL_MODELS_ROOT", root)
	t.Setenv("SPROUT_LOCAL_STATE_ROOT", t.TempDir())

	// SPROUT_LOCAL_MODEL_DIR wins over both the legacy var and the root.
	t.Setenv("SPROUT_LOCAL_MODEL_DIR", alpha)
	t.Setenv("LOCAL_MODEL_DIR", rootTuned)
	if got := resolveModelDir(); got != alpha {
		t.Errorf("SPROUT_LOCAL_MODEL_DIR lost: got %q, want %q", got, alpha)
	}
	// Legacy LOCAL_MODEL_DIR alone is still honored.
	t.Setenv("SPROUT_LOCAL_MODEL_DIR", "")
	t.Setenv("LOCAL_MODEL_DIR", rootTuned)
	if got := resolveModelDir(); got != rootTuned {
		t.Errorf("legacy LOCAL_MODEL_DIR: got %q, want %q", got, rootTuned)
	}
	// Neither set: best installed model from the root (tuned preferred).
	t.Setenv("LOCAL_MODEL_DIR", "")
	if got := resolveModelDir(); got != rootTuned {
		t.Errorf("scanned root: got %q, want %q", got, rootTuned)
	}
}

// TestBestInstalledModel covers the models-root selection: the tuned 4B
// export is preferred, the alphabetically-first model dir is the fallback,
// and a missing or empty root yields "" (backend.go turns that into the
// actionable error).
func TestBestInstalledModel(t *testing.T) {
	t.Setenv("SPROUT_LOCAL_STATE_ROOT", t.TempDir())
	t.Setenv("SPROUT_LOCAL_MODEL_DIR", "")
	t.Setenv("LOCAL_MODEL_DIR", "")

	// Tuned export preferred over alphabetically-earlier models.
	root := t.TempDir()
	makeFakeModelDir(t, root, "alpha-1b")
	tuned := makeFakeModelDir(t, root, preferredModelName)
	t.Setenv("SPROUT_LOCAL_MODELS_ROOT", root)
	if got := bestInstalledModel(); got != tuned {
		t.Errorf("preferred: got %q, want %q", got, tuned)
	}

	// No tuned export: alphabetically-first model dir (non-model dirs skipped).
	root2 := t.TempDir()
	beta := makeFakeModelDir(t, root2, "beta-2b")
	makeFakeModelDir(t, root2, "gamma-1b")
	os.MkdirAll(filepath.Join(root2, "not-a-model"), 0o755)
	t.Setenv("SPROUT_LOCAL_MODELS_ROOT", root2)
	if got := bestInstalledModel(); got != beta {
		t.Errorf("alphabetical: got %q, want %q", got, beta)
	}

	// Empty root: no model dirs at all.
	root3 := t.TempDir()
	os.MkdirAll(filepath.Join(root3, "empty-dir"), 0o755)
	t.Setenv("SPROUT_LOCAL_MODELS_ROOT", root3)
	if got := bestInstalledModel(); got != "" {
		t.Errorf("empty root: got %q, want empty", got)
	}

	// Missing root: unreadable, same result.
	t.Setenv("SPROUT_LOCAL_MODELS_ROOT", filepath.Join(t.TempDir(), "does-not-exist"))
	if got := bestInstalledModel(); got != "" {
		t.Errorf("missing root: got %q, want empty", got)
	}
}

// makeFakeModelDir is defined in modelcmd_test.go (package main).
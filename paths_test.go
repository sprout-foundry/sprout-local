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

// legacySkillFile writes one good skill into dir and returns its path.
func legacySkillFile(t *testing.T, dir string) {
	t.Helper()
	if err := os.MkdirAll(dir, 0o755); err != nil {
		t.Fatal(err)
	}
	content := `{"name":"motd","description":"Daily note","command":"/bin/echo","args":["hi"]}`
	if err := os.WriteFile(filepath.Join(dir, "motd.json"), []byte(content), 0o644); err != nil {
		t.Fatal(err)
	}
}

// TestSkillsDirLegacyCompat: the pre-migration ~/.chatllm state root is
// still served when the new directory does not exist yet, and the new
// directory wins once it exists (newer wins over the legacy fallback).
func TestSkillsDirLegacyCompat(t *testing.T) {
	home := t.TempDir()
	t.Setenv("HOME", home) // legacy location derives from the user home
	state := t.TempDir()   // new state root, distinct from the home tree
	t.Setenv("SPROUT_LOCAL_STATE_ROOT", state)
	t.Setenv("SPROUT_LOCAL_SKILLS_DIR", "")

	legacy := filepath.Join(home, ".chatllm", "skills")
	legacySkillFile(t, legacy)
	fresh := filepath.Join(state, "skills")

	// New dir absent + legacy present → legacy serves.
	if got := skillsDir(); got != legacy {
		t.Errorf("skillsDir = %q, want legacy %q", got, legacy)
	}
	// New dir exists → it wins over the legacy fallback.
	os.MkdirAll(fresh, 0o755)
	if got := skillsDir(); got != fresh {
		t.Errorf("skillsDir = %q, want %q", got, fresh)
	}
	// Explicit override beats both.
	override := t.TempDir()
	t.Setenv("SPROUT_LOCAL_SKILLS_DIR", override)
	if got := skillsDir(); got != override {
		t.Errorf("skillsDir = %q, want override %q", got, override)
	}
}

// TestConversationsDirLegacyCompat mirrors TestSkillsDirLegacyCompat for
// conversation persistence.
func TestConversationsDirLegacyCompat(t *testing.T) {
	home := t.TempDir()
	t.Setenv("HOME", home)
	state := t.TempDir()
	t.Setenv("SPROUT_LOCAL_STATE_ROOT", state)

	legacy := filepath.Join(home, ".chatllm", "conversations")
	if err := os.MkdirAll(legacy, 0o755); err != nil {
		t.Fatal(err)
	}
	fresh := filepath.Join(state, "conversations")

	if got := conversationsDir(); got != legacy {
		t.Errorf("conversationsDir = %q, want legacy %q", got, legacy)
	}
	os.MkdirAll(fresh, 0o755)
	if got := conversationsDir(); got != fresh {
		t.Errorf("conversationsDir = %q, want %q", got, fresh)
	}
	// Neither present: the new directory is the answer (created on demand
	// by the writers, not by the dir resolver). A fresh home with no
	// legacy state rules out the compat fallback.
	t.Setenv("HOME", t.TempDir())
	t.Setenv("SPROUT_LOCAL_STATE_ROOT", t.TempDir())
	want := filepath.Join(stateRoot(), "conversations")
	if got := conversationsDir(); got != want {
		t.Errorf("conversationsDir = %q, want %q", got, want)
	}
}
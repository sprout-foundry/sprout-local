package paths

import (
	"os"
	"path/filepath"
	"testing"
)

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
	if got := SkillsDir(); got != legacy {
		t.Errorf("SkillsDir = %q, want legacy %q", got, legacy)
	}
	// New dir exists → it wins over the legacy fallback.
	os.MkdirAll(fresh, 0o755)
	if got := SkillsDir(); got != fresh {
		t.Errorf("SkillsDir = %q, want %q", got, fresh)
	}
	// Explicit override beats both.
	override := t.TempDir()
	t.Setenv("SPROUT_LOCAL_SKILLS_DIR", override)
	if got := SkillsDir(); got != override {
		t.Errorf("SkillsDir = %q, want override %q", got, override)
	}
}
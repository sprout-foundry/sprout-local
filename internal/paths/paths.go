package paths

// ---------------------------------------------------------------------------
// paths — sprout-local's on-disk state layout.
//
// All per-user state lives under the state root: SPROUT_LOCAL_STATE_ROOT,
// or ~/.sprout-local when unset. Sub-locations derive from it unless they
// have their own override:
//
//	sessions     <stateRoot>/sessions   (session logs + raw dumps)
//	conversations <stateRoot>/conversations (legacy ~/.chatllm compat,
//	                                  resolved in the conversations package)
//	skills       <stateRoot>/skills     (SPROUT_LOCAL_SKILLS_DIR override,
//	                                  legacy ~/.chatllm compat)
//	models       <stateRoot>/models     (SPROUT_LOCAL_MODELS_ROOT override)
// ---------------------------------------------------------------------------

import (
	"os"
	"path/filepath"
)

// HomeDir returns the user's home directory, or "" when it cannot be
// resolved.
func HomeDir() string {
	h, err := os.UserHomeDir()
	if err != nil {
		return ""
	}
	return h
}

// StateRoot is sprout-local's per-user state directory:
// SPROUT_LOCAL_STATE_ROOT, or ~/.sprout-local when unset.
func StateRoot() string {
	if root := os.Getenv("SPROUT_LOCAL_STATE_ROOT"); root != "" {
		return root
	}
	h := HomeDir()
	if h == "" {
		return ""
	}
	return filepath.Join(h, ".sprout-local")
}

// SessionsDir is the session-log directory (<StateRoot>/sessions).
// "" when no home directory can be resolved.
func SessionsDir() string {
	root := StateRoot()
	if root == "" {
		return ""
	}
	return filepath.Join(root, "sessions")
}

// DirExists reports whether dir exists and is a directory.
func DirExists(dir string) bool {
	st, err := os.Stat(dir)
	return err == nil && st.IsDir()
}

// ModelsRoot is the on-device models directory: SPROUT_LOCAL_MODELS_ROOT,
// or <StateRoot>/models (default ~/.sprout-local/models) when unset.
// Models land here from -pull and are listed by /models.
func ModelsRoot() string {
	if root := os.Getenv("SPROUT_LOCAL_MODELS_ROOT"); root != "" {
		return root
	}
	state := StateRoot()
	if state == "" {
		return ""
	}
	return filepath.Join(state, "models")
}

// SkillsDir is the skills directory: SPROUT_LOCAL_SKILLS_DIR, else
// <StateRoot>/skills (default ~/.sprout-local/skills). Legacy compat:
// when the new directory does not exist but the pre-migration
// ~/.chatllm/skills does, the legacy directory is used.
func SkillsDir() string {
	if dir := os.Getenv("SPROUT_LOCAL_SKILLS_DIR"); dir != "" {
		return dir
	}
	state := StateRoot()
	if state == "" {
		return ""
	}
	newDir := filepath.Join(state, "skills")
	if DirExists(newDir) {
		return newDir
	}
	if legacy := filepath.Join(HomeDir(), ".chatllm", "skills"); DirExists(legacy) {
		return legacy
	}
	return newDir
}

// IsModelDir reports whether dir contains the files sinter needs to load a
// model directly (config + tokenizer + at least one weights file).
func IsModelDir(dir string) bool {
	if _, err := os.Stat(filepath.Join(dir, "config.json")); err != nil {
		return false
	}
	if _, err := os.Stat(filepath.Join(dir, "tokenizer.json")); err != nil {
		return false
	}
	matches, _ := filepath.Glob(filepath.Join(dir, "*.safetensors"))
	return len(matches) > 0
}

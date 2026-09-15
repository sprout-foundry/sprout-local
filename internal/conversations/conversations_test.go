package conversations

import (
	"os"
	"path/filepath"
	"testing"

	"github.com/sprout-foundry/sprout-local/internal/paths"
)

// TestConversationsDirLegacyCompat: the pre-migration ~/.chatllm state
// root is still served when the new directory does not exist yet, and the
// new directory wins once it exists (newer wins over the legacy
// fallback).
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

	if got := ConversationsDir(); got != legacy {
		t.Errorf("ConversationsDir = %q, want legacy %q", got, legacy)
	}
	os.MkdirAll(fresh, 0o755)
	if got := ConversationsDir(); got != fresh {
		t.Errorf("ConversationsDir = %q, want %q", got, fresh)
	}
	// Neither present: the new directory is the answer (created on demand
	// by the writers, not by the dir resolver). A fresh home with no
	// legacy state rules out the compat fallback.
	t.Setenv("HOME", t.TempDir())
	t.Setenv("SPROUT_LOCAL_STATE_ROOT", t.TempDir())
	want := filepath.Join(paths.StateRoot(), "conversations")
	if got := ConversationsDir(); got != want {
		t.Errorf("ConversationsDir = %q, want %q", got, want)
	}
}

package all_test

import (
	"testing"

	_ "github.com/sprout-foundry/sinter/llm/all"

	"github.com/sprout-foundry/sinter/llm"
)

func TestAllRegisters(t *testing.T) {
	for _, name := range []string{"qwen2", "qwen3", "qwen3_5_text", "qwen3_5_moe_text"} {
		if !llmHasArch(t, name) {
			t.Errorf("architecture %q not registered after blank import of llm/all", name)
		}
	}
}

func llmHasArch(t *testing.T, name string) bool {
	t.Helper()
	_, err := llm.ArchFactory(name)
	return err == nil
}

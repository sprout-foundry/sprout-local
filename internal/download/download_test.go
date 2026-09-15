package download

import (
	"os"
	"path/filepath"
	"testing"

	"github.com/sprout-foundry/sinter/llm/catalog"
)

func TestFindCatalogModel(t *testing.T) {
	// Empty name: error listing models
	if _, err := FindCatalogModel(""); err == nil {
		t.Error("empty name should error")
	}
	// Exact match
	m, err := FindCatalogModel("qwen3.5-4b")
	if err != nil || m.Name != "qwen3.5-4b" {
		t.Errorf("exact match: got %q, err %v", m.Name, err)
	}
	// Unique prefix
	m, err = FindCatalogModel("gemma4")
	if err != nil || m.Name != "gemma4-e2b" {
		t.Errorf("prefix match: got %q, err %v", m.Name, err)
	}
	// Ambiguous prefix (qwen3.5 matches both 4b and 9b)
	if _, err := FindCatalogModel("qwen3.5"); err == nil {
		t.Error("ambiguous prefix should error")
	}
	// Unknown
	if _, err := FindCatalogModel("llama-7b"); err == nil {
		t.Error("unknown model should error")
	}
}

func TestBuildHFArgs(t *testing.T) {
	// No include: straight into dest
	m := catalog.CatalogModel{Name: "a", HFRepo: "org/repo", Dir: "a"}
	args := BuildHFArgs(m, "/models/a")
	want := []string{"download", "org/repo", "--local-dir", "/models/a"}
	if strEq(args, want) != true {
		t.Errorf("no-include args = %v, want %v", args, want)
	}
	// With include: download into parent so the include subdir lands right
	m.HFInclude = "5bit/*"
	args = BuildHFArgs(m, "/models/a")
	want = []string{"download", "org/repo", "--include", "5bit/*", "--local-dir", "/models"}
	if strEq(args, want) != true {
		t.Errorf("include args = %v, want %v", args, want)
	}
}

func TestHumanBytes(t *testing.T) {
	tests := []struct {
		in   uint64
		want string
	}{
		{512, "512 B"},
		{2048, "2.0 KB"},
		{5 << 20, "5.0 MB"},
		{3 << 30, "3.0 GB"},
	}
	for _, tt := range tests {
		if got := HumanBytes(tt.in); got != tt.want {
			t.Errorf("HumanBytes(%d) = %q, want %q", tt.in, got, tt.want)
		}
	}
}

func TestDirSize(t *testing.T) {
	dir := t.TempDir()
	os.WriteFile(filepath.Join(dir, "a.bin"), make([]byte, 100), 0o644)
	os.WriteFile(filepath.Join(dir, "b.bin"), make([]byte, 50), 0o644)
	os.Mkdir(filepath.Join(dir, "sub"), 0o755)
	os.WriteFile(filepath.Join(dir, "sub", "c.bin"), make([]byte, 25), 0o644)
	// DirSize is non-recursive: 150, not 175
	if got := DirSize(dir); got != 150 {
		t.Errorf("DirSize = %d, want 150", got)
	}
	if got := DirSize(filepath.Join(dir, "missing")); got != 0 {
		t.Errorf("missing dir should give 0, got %d", got)
	}
}

func TestCheckRAMGate(t *testing.T) {
	// No gate configured: always passes
	m := catalog.CatalogModel{Name: "free"}
	if err := CheckRAMGate(m, 16<<30); err != nil {
		t.Errorf("no-gate model should pass, got %v", err)
	}
	// Unknown RAM (0): don't block
	m = catalog.CatalogModel{Name: "gated", MinRAMSelect: 1 << 40, MinRAMSuggested: 1 << 41}
	if err := CheckRAMGate(m, 0); err != nil {
		t.Errorf("unknown RAM should pass, got %v", err)
	}
	// Under MinRAMSelect: refused
	if err := CheckRAMGate(m, 16<<30); err == nil {
		t.Error("model over RAM budget should be refused")
	}
	// Between Select and Suggested: passes (caller prints its own warning path)
	m2 := catalog.CatalogModel{Name: "tight", MinRAMSelect: 8 << 30, MinRAMSuggested: 32 << 30}
	if err := CheckRAMGate(m2, 16<<30); err != nil {
		t.Errorf("tight-fit model should pass, got %v", err)
	}
	// Overweight override: passes despite being under MinRAMSelect
	t.Setenv("SINTER_ALLOW_OVERWEIGHT", "1")
	if err := CheckRAMGate(m, 16<<30); err != nil {
		t.Errorf("SINTER_ALLOW_OVERWEIGHT should pass, got %v", err)
	}
}

func strEq(a, b []string) bool {
	if len(a) != len(b) {
		return false
	}
	for i := range a {
		if a[i] != b[i] {
			return false
		}
	}
	return true
}

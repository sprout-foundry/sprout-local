package catalog

import (
	"os"
	"path/filepath"
	"testing"
)

const testGB = 1024 * 1024 * 1024

// TestSelectModelForRAM checks auto-selection picks the largest installed,
// fitting SUGGESTED model — never a risky stretch pick. Stretch models are
// only reachable via explicit user selection (SelectableForRAM), not
// automatic RAM-based resolution... except as graceful degradation: when
// NOTHING at or below the suggested tier is installed, the walk may use a
// fitting installed stretch model rather than fail outright.
func TestSelectModelForRAM(t *testing.T) {
	root := t.TempDir()
	for _, d := range []string{"gemma-4-e2b-it-5bit", "qwen3.5-4b-4bit", "qwen3.5-9b-4bit"} {
		if err := os.MkdirAll(filepath.Join(root, d), 0o755); err != nil {
			t.Fatal(err)
		}
	}

	cases := []struct {
		name string
		ram  uint64
		want string // expected Dir basename, or "" for error
	}{
		{"1gb", 1 * testGB, "gemma-4-e2b-it-5bit"},
		{"4gb", 4 * testGB, "gemma-4-e2b-it-5bit"},
		// 8GB: qwen3.5-4b is only a STRETCH pick here (MinRAMSelect=8GB,
		// MinRAMSuggested=16GB) — auto-selection must not pick it.
		{"8gb", 8 * testGB, "gemma-4-e2b-it-5bit"},
		{"16gb", 16 * testGB, "qwen3.5-4b-4bit"},
		{"24gb", 24 * testGB, "qwen3.5-9b-4bit"},
		{"32gb", 32 * testGB, "qwen3.5-9b-4bit"}, // 35b-a3b never auto-selected
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			m, err := SelectModelForRAM(root, tc.ram)
			if err != nil {
				t.Fatalf("unexpected error: %v", err)
			}
			if filepath.Base(m.Dir) != tc.want {
				t.Fatalf("got %s, want %s", filepath.Base(m.Dir), tc.want)
			}
		})
	}
}

// TestSelectModelForRAMStretchOnlyInstalled guards the graceful-degradation
// path: a machine with ONLY a stretch-tier model installed still gets it
// picked (over an outright error), and — since the minicpm5-2b catalog
// addition — the walk must reach stretch tiers beyond suggestedIdx+1: at
// 8GB with only a qwen3.5-4b tuned variant installed, the +1 slot belongs
// to minicpm5-2b (not installed), and the old two-step walk gave up even
// though a fitting model was present.
func TestSelectModelForRAMStretchOnlyInstalled(t *testing.T) {
	root := t.TempDir()
	dir := filepath.Join(root, "qwen3.5-4b-sprout-tuned-mlx-q5")
	if err := os.MkdirAll(dir, 0o755); err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(filepath.Join(dir, "model.safetensors"), []byte("x"), 0o644); err != nil {
		t.Fatal(err)
	}

	m, err := SelectModelForRAM(root, 8*testGB)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if filepath.Base(m.Dir) != "qwen3.5-4b-sprout-tuned-mlx-q5" {
		t.Fatalf("got %s, want the installed tuned 4b variant", filepath.Base(m.Dir))
	}
}

// TestCatalogThresholdsMonotonic ensures each catalog entry's MinRAMSelect
// and MinRAMSuggested are both >= the previous entry's, when sorted by
// MinRAMSuggested — the tier walk in TieredCatalogForRAM depends on this.
func TestCatalogThresholdsMonotonic(t *testing.T) {
	sorted := sortedCatalog()
	for i := 1; i < len(sorted); i++ {
		if sorted[i].MinRAMSelect < sorted[i-1].MinRAMSelect {
			t.Fatalf("MinRAMSelect not monotonic: %s (%d) after %s (%d)",
				sorted[i].Name, sorted[i].MinRAMSelect, sorted[i-1].Name, sorted[i-1].MinRAMSelect)
		}
	}
}

// TestMiniCPM5InCatalog pins MiniCPM5-2B's catalog placement: a 2B 4-bit
// model fits anywhere (MinRAM 0 like gemma4-e2b), and the "largest fitting
// suggested" rule must not let it shadow larger models on big machines —
// with equal MinRAMSuggested (0), sortedCatalog keeps input order, so
// gemma4-e2b (listed first) stays the suggested pick for small RAM and
// qwen3.5-4b takes over from 16GB.
func TestMiniCPM5InCatalog(t *testing.T) {
	m := RecommendModelForRAM(4 * testGB)
	if m.Name != "gemma4-e2b" && m.Name != "minicpm5-2b" {
		t.Fatalf("small-RAM suggestion unexpectedly %s", m.Name)
	}
	// minicpm5-2b is known and selectable everywhere, including 1GB machines
	// (as a warned stretch alternative to the gemma4 default).
	status, known := SelectableForRAM("minicpm5-2b", 1*testGB)
	if !known || status == TierBlocked {
		t.Fatalf("minicpm5-2b at 1GB: known=%v status=%v, want known + selectable", known, status)
	}
	// At 128GB it must never be the suggested default (bigger models win).
	if m := RecommendModelForRAM(128 * testGB); m.Name == "minicpm5-2b" {
		t.Fatal("minicpm5-2b should not be the suggested default on a 128GB machine")
	}
}

// TestRecommendModelForRAM checks the pure-RAM recommendation (the safe
// "suggested" tier only).
func TestRecommendModelForRAM(t *testing.T) {
	cases := []struct {
		ram  uint64
		want string
	}{
		{1 * testGB, "gemma4-e2b"},
		{4 * testGB, "gemma4-e2b"},
		{8 * testGB, "gemma4-e2b"},
		{16 * testGB, "qwen3.5-4b"},
		{24 * testGB, "qwen3.5-9b"},
		{32 * testGB, "qwen3.5-9b"},
		{128 * testGB, "qwen3.5-9b"}, // 35b-a3b is never the unwarned default
	}
	for _, tc := range cases {
		m := RecommendModelForRAM(tc.ram)
		if m == nil || m.Name != tc.want {
			t.Fatalf("RecommendModelForRAM(%d) = %+v, want %s", tc.ram, m, tc.want)
		}
		if m.HFRepo == "" {
			t.Fatalf("catalog entry %s missing HFRepo", m.Name)
		}
	}
}

// TestTieredCatalogForRAM locks in the exact suggested/eligible/stretch/
// blocked matrix requested: <8GB suggested=gemma4-e2b no stretch; 8-16GB
// suggested=gemma4-e2b stretch=qwen3.5-4b; 16-24GB suggested=qwen3.5-4b
// (gemma4-e2b now a safe downgrade, not blocked) stretch=qwen3.5-9b;
// 24-32GB suggested=qwen3.5-9b no stretch; 32GB+ suggested=qwen3.5-9b
// stretch=qwen3.6-35b-a3b. Anything smaller than suggested is always a
// selectable (eligible) downgrade, never blocked — only tiers beyond the
// one-up stretch are genuinely blocked.
func TestTieredCatalogForRAM(t *testing.T) {
	cases := []struct {
		name   string
		ram    uint64
		expect map[string]TierStatus
	}{
		{"4gb", 4 * testGB, map[string]TierStatus{
			"gemma4-e2b": TierSuggested, "qwen3.5-4b": TierBlocked, "qwen3.5-9b": TierBlocked, "qwen3.6-35b-a3b": TierBlocked,
		}},
		{"8gb", 8 * testGB, map[string]TierStatus{
			"gemma4-e2b": TierSuggested, "qwen3.5-4b": TierStretch, "qwen3.5-9b": TierBlocked, "qwen3.6-35b-a3b": TierBlocked,
		}},
		{"12gb", 12 * testGB, map[string]TierStatus{
			"gemma4-e2b": TierSuggested, "qwen3.5-4b": TierStretch, "qwen3.5-9b": TierBlocked, "qwen3.6-35b-a3b": TierBlocked,
		}},
		{"16gb", 16 * testGB, map[string]TierStatus{
			"gemma4-e2b": TierEligible, "qwen3.5-4b": TierSuggested, "qwen3.5-9b": TierStretch, "qwen3.6-35b-a3b": TierBlocked,
		}},
		{"20gb", 20 * testGB, map[string]TierStatus{
			"gemma4-e2b": TierEligible, "qwen3.5-4b": TierSuggested, "qwen3.5-9b": TierStretch, "qwen3.6-35b-a3b": TierBlocked,
		}},
		{"24gb", 24 * testGB, map[string]TierStatus{
			"gemma4-e2b": TierEligible, "qwen3.5-4b": TierEligible, "qwen3.5-9b": TierSuggested, "qwen3.6-35b-a3b": TierBlocked,
		}},
		{"28gb", 28 * testGB, map[string]TierStatus{
			"gemma4-e2b": TierEligible, "qwen3.5-4b": TierEligible, "qwen3.5-9b": TierSuggested, "qwen3.6-35b-a3b": TierBlocked,
		}},
		{"32gb", 32 * testGB, map[string]TierStatus{
			"gemma4-e2b": TierEligible, "qwen3.5-4b": TierEligible, "qwen3.5-9b": TierSuggested, "qwen3.6-35b-a3b": TierStretch,
		}},
		{"128gb", 128 * testGB, map[string]TierStatus{
			"gemma4-e2b": TierEligible, "qwen3.5-4b": TierEligible, "qwen3.5-9b": TierSuggested, "qwen3.6-35b-a3b": TierStretch,
		}},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			tiered := TieredCatalogForRAM(tc.ram)
			got := map[string]TierStatus{}
			for _, tm := range tiered {
				got[tm.Model.Name] = tm.Status
			}
			for name, want := range tc.expect {
				if got[name] != want {
					t.Errorf("%s at %s: got %v, want %v", name, tc.name, got[name], want)
				}
			}
		})
	}
}

// TestSelectableForRAM exercises the direct per-model gate check used by
// /model selection.
func TestSelectableForRAM(t *testing.T) {
	// On an 8GB machine: 4b is a warned stretch, 9b is blocked outright.
	status, known := SelectableForRAM("qwen3.5-4b", 8*testGB)
	if !known || status != TierStretch {
		t.Fatalf("qwen3.5-4b at 8GB: got known=%v status=%v, want known=true status=stretch", known, status)
	}
	status, known = SelectableForRAM("qwen3.5-9b", 8*testGB)
	if !known || status != TierBlocked {
		t.Fatalf("qwen3.5-9b at 8GB: got known=%v status=%v, want known=true status=blocked", known, status)
	}
	if _, known := SelectableForRAM("not-a-real-model", 128*testGB); known {
		t.Fatal("expected unknown model name to report known=false")
	}
}

package main

// ---------------------------------------------------------------------------
// Model download: `chatllm -pull [name]` fetches an MLX-format model from
// HuggingFace into the shared models root, using sinter's llm/catalog for
// metadata (repo, include globs, RAM tiers).
//
// Deliberately app-level glue, not part of sinter: the engine stays
// download-free (its README keeps the catalog separate "so apps can keep
// their own list"), and the mechanics live here — shell out to the hf CLI
// and poll the destination directory for progress. The hf CLI prints no
// progress to pipes (tqdm is isatty-gated), so bytes-on-disk is the only
// reliable progress signal; the same approach sprout's localmodel uses.
// ---------------------------------------------------------------------------

import (
	"context"
	"fmt"
	"io"
	"os"
	"os/exec"
	"path/filepath"
	"strings"
	"time"

	"github.com/sprout-foundry/sinter/llm/catalog"
)

// modelsRoot is the shared on-device models directory (same root gmitllm,
// sprout, and auto-term use). CHATLLM_MODELS_ROOT overrides it.
func modelsRoot() string {
	if root := os.Getenv("CHATLLM_MODELS_ROOT"); root != "" {
		return root
	}
	return filepath.Join(homeDir(), "dev", "llm-models")
}

// findCatalogModel resolves a catalog entry by canonical name, accepting a
// unique prefix (e.g. "qwen3.5-4" → "qwen3.5-4b"). Empty name returns an
// error listing the available models.
func findCatalogModel(name string) (catalog.CatalogModel, error) {
	models := catalog.ModelCatalog
	if name == "" {
		var names []string
		for _, m := range models {
			names = append(names, m.Name)
		}
		return catalog.CatalogModel{}, fmt.Errorf("no model specified — available: %s", strings.Join(names, ", "))
	}
	var matches []catalog.CatalogModel
	for _, m := range models {
		if m.Name == name {
			return m, nil
		}
		if strings.HasPrefix(m.Name, name) {
			matches = append(matches, m)
		}
	}
	switch len(matches) {
	case 1:
		return matches[0], nil
	case 0:
		return catalog.CatalogModel{}, fmt.Errorf("unknown model %q (try: chatllm -pull for the list)", name)
	default:
		var names []string
		for _, m := range matches {
			names = append(names, m.Name)
		}
		return catalog.CatalogModel{}, fmt.Errorf("ambiguous %q — matches: %s", name, strings.Join(names, ", "))
	}
}

// checkRAMGate refuses downloads the machine cannot run, warns on tight
// fits. Honors SINTER_ALLOW_OVERWEIGHT (the same override the sinter engine
// gate uses) to skip the hard refusal. ramBytes of 0 means unknown: no gate.
func checkRAMGate(m catalog.CatalogModel, ramBytes uint64) error {
	if m.MinRAMSelect == 0 || ramBytes == 0 {
		return nil
	}
	ram := ramBytes
	switch {
	case ram < m.MinRAMSelect:
		if os.Getenv("SINTER_ALLOW_OVERWEIGHT") != "" {
			fmt.Printf("warning: %s needs %s RAM; this machine has %s (SINTER_ALLOW_OVERWEIGHT set — proceeding)\n",
				m.Name, humanBytes(m.MinRAMSelect), humanBytes(ram))
			return nil
		}
		return fmt.Errorf("%s needs at least %s of RAM; this machine has %s (set SINTER_ALLOW_OVERWEIGHT to override)",
			m.Name, humanBytes(m.MinRAMSelect), humanBytes(ram))
	case ram < m.MinRAMSuggested:
		fmt.Printf("warning: %s runs best with %s RAM; this machine has %s — expect tight memory\n",
			m.Name, humanBytes(m.MinRAMSuggested), humanBytes(ram))
	}
	return nil
}

// buildHFArgs constructs the hf download command line. Split out for tests.
func buildHFArgs(m catalog.CatalogModel, dest string) []string {
	// When HFInclude is set, files land with their repo-path prefix
	// preserved, so download into the parent of dest and the include
	// subdir completes the path (sprout's EnsureModel approach).
	localDir := dest
	if m.HFInclude != "" {
		localDir = filepath.Dir(dest)
	}
	args := []string{"download", m.HFRepo}
	if m.HFInclude != "" {
		args = append(args, "--include", m.HFInclude)
	}
	return append(args, "--local-dir", localDir)
}

// downloadModel runs the hf download for a catalog entry into the models
// root, streaming disk-based progress to stdout. Returns the model
// directory on success.
func downloadModel(ctx context.Context, m catalog.CatalogModel) (string, error) {
	if err := checkRAMGate(m, totalSystemRAM()); err != nil {
		return "", err
	}

	bin := "hf"
	if _, err := exec.LookPath(bin); err != nil {
		bin = "huggingface-cli"
		if _, err := exec.LookPath(bin); err != nil {
			return "", fmt.Errorf("huggingface CLI not found — install with: pip install -U huggingface_hub")
		}
	}

	dest := filepath.Join(modelsRoot(), m.Dir)
	if isModelDir(dest) {
		return dest, nil // already present: hf will no-op, but skip the noise
	}
	if err := os.MkdirAll(filepath.Dir(dest), 0o755); err != nil {
		return "", fmt.Errorf("create models dir: %w", err)
	}

	cmd := exec.CommandContext(ctx, bin, buildHFArgs(m, dest)...)
	stderr, err := cmd.StderrPipe()
	if err != nil {
		return "", fmt.Errorf("pipe stderr: %w", err)
	}
	stdout, err := cmd.StdoutPipe()
	if err != nil {
		return "", fmt.Errorf("pipe stdout: %w", err)
	}
	if err := cmd.Start(); err != nil {
		return "", fmt.Errorf("start download: %w", err)
	}

	// Drain both pipes so the subprocess never blocks on a full buffer.
	go io.Copy(io.Discard, stderr)
	go io.Copy(io.Discard, stdout)

	stopPoll := make(chan struct{})
	pollDone := make(chan struct{})
	go func() {
		defer close(pollDone)
		pollDownloadProgress(dest, stopPoll)
	}()

	waitErr := cmd.Wait()
	close(stopPoll)
	<-pollDone
	if waitErr != nil {
		return "", fmt.Errorf("download failed: %w", waitErr)
	}
	if !isModelDir(dest) {
		return "", fmt.Errorf("download finished but %s does not look like a model directory", dest)
	}
	return dest, nil
}

// pollDownloadProgress reports bytes landed in dest every 2 seconds until
// stop is closed, then prints a final newline. See the file-level comment
// for why this beats parsing hf's output.
func pollDownloadProgress(dest string, stop <-chan struct{}) {
	ticker := time.NewTicker(2 * time.Second)
	defer ticker.Stop()
	var last int64
	for {
		select {
		case <-stop:
			if last > 0 {
				fmt.Printf("\r  downloaded: %s          \n", humanBytes(uint64(last)))
			}
			return
		case <-ticker.C:
			size := dirSize(dest)
			if size != last {
				fmt.Printf("\r  downloaded: %s          ", humanBytes(uint64(size)))
				last = size
			}
		}
	}
}

// dirSize sums the byte size of regular files under dir (non-recursive into
// unreadable entries). Mirrors sinter catalog's dirSize.
func dirSize(dir string) int64 {
	entries, err := os.ReadDir(dir)
	if err != nil {
		return 0
	}
	var total int64
	for _, e := range entries {
		if e.IsDir() {
			continue
		}
		if info, err := e.Info(); err == nil {
			total += info.Size()
		}
	}
	return total
}

// humanBytes renders a byte count as B/MB/GB.
func humanBytes(n uint64) string {
	switch {
	case n >= 1<<30:
		return fmt.Sprintf("%.1f GB", float64(n)/(1<<30))
	case n >= 1<<20:
		return fmt.Sprintf("%.1f MB", float64(n)/(1<<20))
	case n >= 1<<10:
		return fmt.Sprintf("%.1f KB", float64(n)/(1<<10))
	default:
		return fmt.Sprintf("%d B", n)
	}
}

// printPullList shows the catalog with RAM-tier guidance for this machine.
func printPullList() {
	ram := totalSystemRAM()
	fmt.Println("Models available to pull (chatllm -pull <name>):")
	for _, m := range catalog.ModelCatalog {
		gate := ""
		switch {
		case ram != 0 && m.MinRAMSelect != 0 && ram < m.MinRAMSelect:
			gate = "  [needs more RAM]"
		case ram != 0 && m.MinRAMSuggested != 0 && ram < m.MinRAMSuggested && m.MinRAMSuggested != ^uint64(0):
			gate = "  [tight fit]"
		}
		fmt.Printf("  %-16s %-40s%s\n", m.Name, m.HFRepo, gate)
	}
}

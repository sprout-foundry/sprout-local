package main

// ---------------------------------------------------------------------------
// Slash-command model management:
//
//   /model [ref]   show or switch the active model
//   /models        list installed models (shared models root)
//   /pull [name]   download a catalog model, switch to it, confirm it runs
//
// A ref is a bare directory name under the shared models root, a catalog
// name (mapped to its Dir), or an absolute path. Switching loads the new
// model before the active model is committed, so a failed load changes
// nothing; conversation history is kept across switches (the web UI
// already allows this — sinter's prefix cache is per-model).
// ---------------------------------------------------------------------------

import (
	"fmt"
	"os"
	"path/filepath"
	"sort"
	"strings"

	"github.com/sprout-foundry/sinter/llm"
	"github.com/sprout-foundry/sinter/llm/catalog"
)

// currentModelDir returns the active model directory, falling back to the
// process default when no switch has happened yet.
func currentModelDir(active *string) string {
	if active != nil && *active != "" {
		return *active
	}
	return resolveModelDir()
}

// handleModelCommand shows (no args) or switches (args) the active model.
// The model is loaded (and over-limit older models evicted) before the
// switch is committed, so a failed load leaves the session unchanged.
func handleModelCommand(ref string, active *string) {
	if ref == "" {
		fmt.Printf("Model: %s\n", filepath.Base(currentModelDir(active)))
		return
	}
	dir, err := modelFromRef(ref)
	if err != nil {
		fmt.Printf("%s %v\n", ansiStyle("Error:", ansiRed), err)
		return
	}
	fmt.Printf("Loading %s …\n", filepath.Base(dir))
	if _, err := loadModelDir(dir); err != nil {
		fmt.Printf("%s %v\n", ansiStyle("Error:", ansiRed), err)
		return
	}
	evictModels(residentLimit(), dir)
	*active = dir
	setSessionModelProtocol(dir)
	fmt.Printf("Switched to %s — history kept; /new starts fresh.\n", filepath.Base(dir))
}

// handlePullCommand downloads a catalog model (exact name or unique
// prefix) and switches to it. With no args it prints the catalog with
// RAM-tier guidance. After switching, a short generated greeting confirms
// the model actually runs before the user commits to a turn.
func handlePullCommand(name string, active *string) {
	if name == "" {
		printPullList()
		return
	}
	m, err := findCatalogModel(name)
	if err != nil {
		fmt.Printf("%s %v\n", ansiStyle("Error:", ansiRed), err)
		return
	}
	dest, err := downloadModel(ctxBg(), m)
	if err != nil {
		fmt.Printf("%s %v\n", ansiStyle("Error:", ansiRed), err)
		return
	}
	fmt.Printf("Pulled %s → %s\n", m.Name, dest)

	if _, err := loadModelDir(dest); err != nil {
		fmt.Printf("%s %v\n", ansiStyle("Error:", ansiRed), err)
		return
	}
	evictModels(residentLimit(), dest)
	*active = dest
	setSessionModelProtocol(dest)
	fmt.Printf("Switched to %s.\n", filepath.Base(dest))

	printer := stdoutPrinter()
	_, _ = streamChatModel(ctxBg(), dest, []llm.ChatMessage{
		{Role: "user", Content: "Introduce yourself in one short sentence."},
	}, printer.writeDelta)
	printer.Close()
	fmt.Println()
}

// modelFromRef resolves a model reference to an MLX-format directory:
// absolute or ~/ path, bare directory name under the shared models root,
// then catalog name (its Dir).
func modelFromRef(ref string) (string, error) {
	ref = strings.TrimSpace(ref)
	if ref == "" {
		return "", fmt.Errorf("no model specified")
	}
	if ref == "~" || strings.HasPrefix(ref, "~/") {
		ref = filepath.Join(homeDir(), strings.TrimPrefix(strings.TrimPrefix(ref, "~"), "/"))
	}
	if filepath.IsAbs(ref) {
		if isModelDir(ref) {
			return ref, nil
		}
		return "", fmt.Errorf("%s is not an MLX-format model directory (need config.json + tokenizer.json + *.safetensors)", ref)
	}
	cand := filepath.Join(modelsRoot(), ref)
	if isModelDir(cand) {
		return cand, nil
	}
	for _, m := range catalog.ModelCatalog {
		if m.Dir == ref {
			return "", fmt.Errorf("%s is in the catalog but not installed — try /pull %s", m.Name, m.Name)
		}
	}
	return "", fmt.Errorf("no installed model %q — /models lists what's available, /pull downloads from the catalog", ref)
}

// availableModelNames lists installed models: MLX-format directories under
// the shared models root, plus the process default when it lives elsewhere
// (e.g. SPROUT_LOCAL_MODEL_DIR pointing outside the root). Sorted; the
// active default is prepended so it can be listed first.
func availableModelNames() (names []string, def string) {
	root := modelsRoot()
	entries, err := os.ReadDir(root)
	if err == nil {
		for _, e := range entries {
			if e.IsDir() && isModelDir(filepath.Join(root, e.Name())) {
				names = append(names, e.Name())
			}
		}
	}
	sort.Strings(names)
	if dir := resolveModelDir(); dir != "" {
		def = filepath.Base(dir)
		found := false
		for _, n := range names {
			if n == def {
				found = true
				break
			}
		}
		if !found {
			names = append([]string{def}, names...)
		}
	}
	return names, def
}

// printModelList lists installed models with the session model marked.
func printModelList(active string) {
	names, def := availableModelNames()
	if active == "" {
		active = resolveModelDir()
	}
	mark := func(name string) string {
		if active != "" && filepath.Base(active) == name {
			return " *"
		}
		return ""
	}
	if len(names) == 0 {
		fmt.Println("No installed models — /pull <name> downloads from the catalog.")
		return
	}
	fmt.Println("Installed models (* = this session):")
	for _, n := range names {
		fmt.Printf("  %s%s\n", n, mark(n))
	}
	if def != "" {
		fmt.Printf("Process default: %s\n", def)
	}
}

// evictModels frees and drops cache entries so at most keep models stay
// resident (kept for the slash-command call sites; the shared cap also
// runs inside loadModelDir, so every load path is covered).
func evictModels(keep int, recent string) {
	modelMu.Lock()
	defer modelMu.Unlock()
	evictModelsLocked(keep, recent)
}

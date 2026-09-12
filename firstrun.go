package main

// ---------------------------------------------------------------------------
// firstrun.go — first-run model walkthrough for the bare REPL.
//
// A fresh machine has no models in its models root, so resolveModelDir()
// returns "" and the REPL would otherwise fatal out. runFirstRun turns that
// dead end into a guided flow: it lists the catalog with RAM-tier guidance
// for this machine, asks which model to download (a number, a name, or
// Enter for the RAM-recommended default), and — on success — points
// SPROUT_LOCAL_MODEL_DIR at the freshly pulled directory so the rest of
// startup proceeds as normal.
//
// Trigger: only from the interactive REPL path, and only when
// resolveModelDir() == "". One-shot (-p), transcript, -serve, and -pull all
// bail before this; explicit/legacy model-dir envs already resolve a dir so
// the walkthrough never fires. "later"/"skip" exits 0; EOF (a non-
// interactive stdin) exits 1 with a pointer to -pull.
// ---------------------------------------------------------------------------

import (
	"bufio"
	"context"
	"fmt"
	"os"
	"strconv"
	"strings"

	"github.com/sprout-foundry/sinter/llm/catalog"
)

// firstRunPrompt is the prompt shown above the input line.
const firstRunPrompt = "Model to download [1-N | name | Enter=default]: "

// firstRunAttempts caps how many times a RAM-gate refusal re-prompts before
// the walkthrough gives up and exits.
const firstRunAttempts = 3

// runFirstRun walks a model-less machine through downloading its first
// model. It returns true when a model was selected and pulled (the caller
// continues startup; SPROUT_LOCAL_MODEL_DIR now resolves a dir). It exits
// the process (os.Exit) for every terminal outcome: success is signaled via
// the returned true before any exit, "later/skip" exits 0, and EOF or
// exhausting the attempt budget exits 1.
func runFirstRun(ctx context.Context, r *bufio.Reader) bool {
	root := modelsRoot()
	ram := totalSystemRAM()
	suggested := catalog.SuggestedForRAM(ram)

	fmt.Println("No models found in " + root + ".")
	fmt.Println("Models available to pull (RAM tier for this machine):")
	entries := catalog.ModelCatalog
	printFirstRunList(entries, ram, suggested)

	for attempt := 1; attempt <= firstRunAttempts; attempt++ {
		raw := readFirstRunLine(r)
		if raw == nil { // EOF: non-interactive stdin can't drive this flow.
			fmt.Fprintln(os.Stderr, "non-interactive: run 'sprout-local -pull <name>'")
			os.Exit(1)
		}
		line := strings.TrimSpace(*raw)

		// Defer the download: the user will pick a model later.
		if line == "" {
			line = suggested.Name // Enter = RAM-recommended default
		}
		if isFirstRunSkip(line) {
			fmt.Println("ok — set SPROUT_LOCAL_MODEL_DIR later or run 'sprout-local -pull <name>'")
			os.Exit(0)
		}

		m, err := resolveFirstRunChoice(line, entries)
		if err != nil {
			fmt.Printf("  %v\n", err)
			continue // re-prompt (counts against the attempt budget)
		}

		dest, derr := downloadModel(ctx, m)
		if derr != nil {
			// A RAM-gate refusal (or an equally unsalvageable refusal)
			// re-prompts; anything that isn't a gate refusal is terminal.
			if strings.Contains(errString(derr), "needs at least") ||
				strings.Contains(errString(derr), "set SINTER_ALLOW_OVERWEIGHT") {
				fmt.Printf("  %v\n", derr)
				continue
			}
			fmt.Printf("  download failed: %v\n", derr)
			fmt.Fprintln(os.Stderr, "check your connection and retry, or run 'sprout-local -pull <name>' later")
			os.Exit(1)
		}

		fmt.Printf("Pulled %s → %s\n", m.Name, dest)
		// Select the freshly pulled model for this run; resolveModelDir
		// reads the env var, so startup proceeds from here.
		os.Setenv("SPROUT_LOCAL_MODEL_DIR", dest)
		return true
	}

	// Exhausted the attempt budget on repeated gate refusals / bad picks.
	fmt.Fprintln(os.Stderr, "non-interactive: run 'sprout-local -pull <name>'")
	os.Exit(1)
	return false // unreachable; satisfies the compiler
}

// printFirstRunList renders the catalog with each model's RAM-tier tag for
// this machine. It mirrors printPullList's tier annotations, then overrides
// the RAM-recommended entry's tag with "(recommended for this machine)".
func printFirstRunList(entries []catalog.CatalogModel, ram uint64, suggested catalog.CatalogModel) {
	for i, m := range entries {
		tag := ""
		switch {
		case ram != 0 && m.MinRAMSelect != 0 && ram < m.MinRAMSelect:
			tag = "  [needs more RAM]"
		case ram != 0 && m.MinRAMSuggested != 0 && ram < m.MinRAMSuggested && m.MinRAMSuggested != ^uint64(0):
			tag = "  [tight fit]"
		}
		if m.Name == suggested.Name {
			tag = "  (recommended for this machine)"
		}
		fmt.Printf("  %d. %-16s %-40s%s\n", i+1, m.Name, m.HFRepo, tag)
	}
}

// resolveFirstRunChoice maps a prompt reply to a catalog entry: a plain
// number indexes the listed catalog (1-based), anything else is a name
// resolved via findCatalogModel (unique prefix allowed).
func resolveFirstRunChoice(line string, entries []catalog.CatalogModel) (catalog.CatalogModel, error) {
	if n, err := strconv.Atoi(line); err == nil {
		if n < 1 || n > len(entries) {
			return catalog.CatalogModel{}, fmt.Errorf("number out of range (1-%d)", len(entries))
		}
		return entries[n-1], nil
	}
	m, err := findCatalogModel(line)
	if err != nil {
		return catalog.CatalogModel{}, fmt.Errorf("%v", err)
	}
	return m, nil
}

// isFirstRunSkip reports whether the user wants to defer the download.
func isFirstRunSkip(line string) bool {
	switch strings.ToLower(line) {
	case "later", "skip", "quit", "exit", "not now", "q":
		return true
	}
	return false
}

// readFirstRunLine reads one input line from the walkthrough's reader. It
// returns nil on EOF (a non-interactive stdin) so the caller can take the
// exit-1 path; otherwise it returns the raw line.
func readFirstRunLine(r *bufio.Reader) *string {
	line, err := r.ReadString('\n')
	if err != nil {
		return nil // EOF: no terminal to prompt on
	}
	return &line
}
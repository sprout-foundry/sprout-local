package chatmodel

import (
	"log"

	"github.com/sprout-foundry/sprout-local/internal/paths"
)

// ModelBackend probes the sinter model at startup and returns
// (engineLabel, modelLabel) for the status line. Fatals with an actionable
// message when no MLX-format model directory resolves — sinter in-process
// inference is the only engine. Callers set the session tool protocol for
// the returned directory (chatmodel must not depend on the tools package).
func ModelBackend() (engine, model string) {
	dir := paths.ResolveModelDir()
	if dir == "" {
		log.Fatalf("No local model found.\n"+
			"Checked the models root: %s (no model directories there).\n"+
			"Point SPROUT_LOCAL_MODEL_DIR at an MLX-format model directory, or run\n"+
			"'sprout-local -pull' to list and download a model.\n", paths.ModelsRoot())
	}
	if _, err := loadSinterModel(); err != nil {
		log.Fatalf("Sinter failed to load %s: %v", dir, err)
	}
	return "sinter", dir
}

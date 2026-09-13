package main

import "log"

// modelBackend probes the sinter model at startup and returns
// (engineLabel, modelLabel) for the status line. Fatals with an actionable
// message when no MLX-format model directory resolves — sinter in-process
// inference is the only engine.
func modelBackend() (engine, model string) {
	dir := resolveModelDir()
	if dir == "" {
		log.Fatalf("No local model found.\n"+
			"Checked the models root: %s (no model directories there).\n"+
			"Point SPROUT_LOCAL_MODEL_DIR at an MLX-format model directory, or run\n"+
			"'sprout-local -pull' to list and download a model.\n", modelsRoot())
	}
	if _, err := loadSinterModel(); err != nil {
		log.Fatalf("Sinter failed to load %s: %v", dir, err)
	}
	setSessionModelProtocol(dir)
	return "sinter", dir
}

package main

import "log"

// modelBackend probes the sinter model at startup and returns
// (engineLabel, modelLabel) for the status line. Fatals with a clear
// message when no MLX-format model directory resolves — sinter
// in-process inference is the only engine. Mirrors gmitllm's backend.go.
func modelBackend() (engine, model string) {
	dir := resolveModelDir()
	if dir == "" {
		log.Fatalf("No local model found.\n"+
			"Set LOCAL_MODEL_DIR to an MLX-format model directory\n"+
			"(default: %s).\n", defaultModelDir)
	}
	if _, err := loadSinterModel(); err != nil {
		log.Fatalf("Sinter failed to load %s: %v", dir, err)
	}
	return "sinter", dir
}

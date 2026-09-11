package tensor

import (
	"os"
	"runtime"
)

// DetectBackend returns the best available tensor backend for this machine,
// or nil if no GPU backend is available. SINTER_BACKEND forces a specific
// backend by name (e.g. "ggml", "metal") for testing and benchmarking —
// returns nil when the named backend isn't registered.
//
// Detection order:
//  1. metal — Apple Silicon (darwin/arm64) with MLX build tag
//  2. Future: cuda, rocm, vulkan via GGML
//  3. nil — no backend (caller falls back to cloud providers)
func DetectBackend() Backend {
	if want := os.Getenv("SINTER_BACKEND"); want != "" {
		for _, b := range registeredBackends {
			// Match on the logical name (metal/ggml), not the device name
			// GGML reports post-init (MTL0, CPU, ...). Registered names are
			// stable, so a flat map of registered order is enough.
			if b.Available() && backendLogicalName(b) == want {
				return b
			}
		}
		return nil
	}
	// The metal backend is registered via init() in pkg/mlx/backend.go
	// (build tag: darwin && arm64 && cgo && mlx). On other platforms the
	// registry is empty and this returns nil.
	for _, b := range registeredBackends {
		if b.Available() {
			return b
		}
	}
	return nil
}

// backendLogicalName returns the stable backend family name for backend
// selection. MLX's Name() is already the logical "metal"; GGML's Name()
// reports the selected device (e.g. "MTL0" on Metal, "CPU" on plain CPU)
// after lazy init, so map its prefix back to "ggml".
func backendLogicalName(b Backend) string {
	n := b.Name()
	switch {
	case n == "metal":
		return "metal"
	case n == "ggml", n == "MTL0", n == "CPU", n == "CUDA0", n == "Vulkan0":
		return "ggml"
	default:
		return n
	}
}

// registeredBackends is populated by init() in each backend package.
var registeredBackends []Backend

// RegisterBackend adds a backend to the detection list. Called by init()
// in metal/cuda/rocm/vulkan packages.
func RegisterBackend(b Backend) {
	registeredBackends = append(registeredBackends, b)
}

// PlatformSupported reports whether the current platform could support any
// GPU backend at all. Used by the UI to decide whether to show "Local (Offline)".
func PlatformSupported() bool {
	switch runtime.GOOS {
	case "darwin":
		return runtime.GOARCH == "arm64"
	case "linux", "windows":
		return true // could have CUDA/ROCm/Vulkan
	default:
		return false
	}
}

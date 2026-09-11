//go:build !linux && !darwin

package main

// totalSystemRAM returns total physical RAM in bytes, or 0 if unknown.
func totalSystemRAM() uint64 {
	return 0
}

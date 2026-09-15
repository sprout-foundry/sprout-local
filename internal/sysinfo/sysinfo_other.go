//go:build !linux && !darwin

package sysinfo

// TotalSystemRAM returns total physical RAM in bytes, or 0 if unknown.
func TotalSystemRAM() uint64 {
	return 0
}
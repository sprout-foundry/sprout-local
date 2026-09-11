//go:build linux

package main

import "syscall"

// totalSystemRAM returns total physical RAM in bytes, or 0 if unknown.
func totalSystemRAM() uint64 {
	var si syscall.Sysinfo_t
	if err := syscall.Sysinfo(&si); err != nil {
		return 0
	}
	return uint64(si.Totalram) * uint64(si.Unit)
}

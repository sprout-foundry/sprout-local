//go:build linux

package sysinfo

import "syscall"

// TotalSystemRAM returns total physical RAM in bytes, or 0 if unknown.
func TotalSystemRAM() uint64 {
	var si syscall.Sysinfo_t
	if err := syscall.Sysinfo(&si); err != nil {
		return 0
	}
	return uint64(si.Totalram) * uint64(si.Unit)
}

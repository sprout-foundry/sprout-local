//go:build darwin

package sysinfo

import "golang.org/x/sys/unix"

// TotalSystemRAM returns total physical RAM in bytes, or 0 if unknown.
func TotalSystemRAM() uint64 {
	mem, err := unix.SysctlUint64("hw.memsize")
	if err != nil {
		return 0
	}
	return mem
}

//go:build darwin

package main

import "golang.org/x/sys/unix"

// totalSystemRAM returns total physical RAM in bytes, or 0 if unknown.
func totalSystemRAM() uint64 {
	mem, err := unix.SysctlUint64("hw.memsize")
	if err != nil {
		return 0
	}
	return mem
}

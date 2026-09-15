package repl

// ---------------------------------------------------------------------------
// Ctrl-C handling: the first signal cancels the in-flight generation; a
// second signal within the grace window exits the process. The watcher runs
// for the whole session, not per-generation — one goroutine, one channel.
// ---------------------------------------------------------------------------

import (
	"fmt"
	"os"
	"os/signal"
	"syscall"
)

var signalCh = make(chan os.Signal, 1)

// startSignalWatch installs the SIGINT handler once and runs a goroutine
// that cancels the active generation context on the first Ctrl-C and exits
// on the second within the grace window.
func StartSignalWatch() {
	signal.Notify(signalCh, os.Interrupt, syscall.SIGTERM)
	go func() {
		for range signalCh {
			if cancelActive() {
				fmt.Println("\n(cancelling… Ctrl-C again to quit)")
				continue
			}
			fmt.Println("\nBye!")
			os.Exit(0)
		}
	}()
}

// cancelActive cancels the in-flight generation, if any. Reports whether
// a generation was active.
func cancelActive() bool {
	mu.Lock()
	defer mu.Unlock()
	if currentGen == nil {
		return false
	}
	currentGen.cancel()
	currentGen = nil
	return true
}

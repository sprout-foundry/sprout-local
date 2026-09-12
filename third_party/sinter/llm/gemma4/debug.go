//go:build cgo && ((darwin && arm64) || (linux && ggml && (arm64 || amd64)))

package gemma4

import (
	"math"
	"path/filepath"
	"encoding/binary"
	"strings"
	"fmt"
	"os"

	"github.com/sprout-foundry/sinter/tensor"
)

// dumpArrayF32 writes an array's first few elements to stderr for debugging.
func dumpArrayF32(a tensor.Array, name string, backend tensor.Backend, s tensor.Stream) {
	if a.Dtype() != tensor.Float32 {
		f32, err := backend.AsType(a, tensor.Float32, s)
		if err != nil {
			return
		}
		defer f32.Free()
		a = f32
	}
	data, err := a.Float32Data()
	if err != nil {
		return
	}
	shape := a.Shape()
	total := len(data)
	n := 10
	if total < n {
		n = total
	}
	fmt.Fprintf(os.Stderr, "[gemma4] %s shape=%v first=%v\n", name, shape, data[:n])
	if dir := os.Getenv("GEMMA4_LAYERS_DUMP"); dir != "" {
		_ = os.MkdirAll(dir, 0o755)
		safe := strings.NewReplacer("/", "_", " ", "_").Replace(name)
		if fw := os.Getenv("GEMMA4_DUMP_FW"); fw != "" {
			safe = fw + "_" + safe
		}
		buf := make([]byte, 0, total*4+16)
		for _, v := range data {
			buf = binary.LittleEndian.AppendUint32(buf, math.Float32bits(v))
		}
		_ = os.WriteFile(filepath.Join(dir, safe+".f32"), buf, 0o644)
	}
}

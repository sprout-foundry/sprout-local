//go:build cgo && ((darwin && arm64) || (linux && ggml && (arm64 || amd64)))

package llm

import "golang.org/x/text/unicode/norm"

// nfcNormalize returns the NFC canonical form of s. Qwen3.8-style
// tokenizer pipelines run Unicode NFC before the pre-tokenization split
// (tokenizer.json: "normalizer": {"type": "NFC"}), so encoding must too,
// or decomposed accents (e + U+0301) land in different pre-tokens than
// the model's training data and token IDs diverge.
func nfcNormalize(s string) string {
	return norm.NFC.String(s)
}

package webui

// ---------------------------------------------------------------------------
// factcheck.go — post-answer self-check for the web UI.
//
// After each web turn, the same loaded model is asked to verify its own
// answer against the question (Yes/No, low temperature). A confident "No"
// sends the old UI's warning — "a bit sus, you might want to double
// check" — as a note frame. This replaces the old Ollama
// bespoke-minicheck dependency; failures degrade silently so the check
// can never break the chat.
// ---------------------------------------------------------------------------

import (
	"strings"

	"github.com/sprout-foundry/sinter/llm"

	"github.com/sprout-foundry/sprout-local/internal/chatmodel"
)

// factCheck asks the model whether answer is correct with respect to
// reference (the user turn, including any #url page text). Returns true
// when the answer passes (or when the check is inconclusive — the chat
// must never nag on a failed parse).
func factCheck(modelDir, reference, answer string) (bool, error) {
	m, err := chatmodel.LoadModelDir(modelDir)
	if err != nil {
		return true, err
	}

	messages := []llm.ChatMessage{
		{Role: "system", Content: "You are a precise fact checker. Answer only Yes or No."},
		{Role: "user", Content: "Reference:\n" + reference +
			"\n\nProposed answer: " + answer +
			"\nIs the proposed answer correct according to the reference? Answer with only Yes or No."},
	}

	rendered := m.FormatChat(messages)
	cfg := llm.DefaultGenerateConfig()
	cfg.MaxTokens = 16
	cfg.Temperature = 0
	cfg.ThinkingTokens = false

	var sb strings.Builder
	chatmodel.SerializeGeneration(func() {
		err = m.Generate(chatmodel.CtxBg(), rendered, cfg, func(id int) {
			sb.WriteString(m.DecodeToken(id))
		})
	})
	if err != nil {
		return true, err
	}
	verdict, _ := parseYesNo(sb.String()) // inconclusive parses count as a pass
	return verdict, nil
}

// parseYesNo interprets a Yes/No verdict by scanning for the first
// yes/no word; ok is false when no confident verdict can be read (callers
// treat that as a pass — the chat must never nag on a failed parse).
func parseYesNo(text string) (verdict bool, ok bool) {
	for _, f := range strings.Fields(chatmodel.StripOutputNoise(text)) {
		w := strings.ToLower(strings.Trim(f, ".,;:!?—-"))
		if strings.HasPrefix(w, "yes") {
			return true, true
		}
		if strings.HasPrefix(w, "no") {
			return false, true
		}
	}
	return true, false
}

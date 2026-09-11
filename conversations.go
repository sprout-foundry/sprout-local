package main

// ---------------------------------------------------------------------------
// conversations.go — server-side conversation persistence for the web UI.
//
// Each conversation is a JSON file in ~/.chatllm/conversations/: id,
// title (first user message), model, and the message list. The web UI's
// sidebar lists them; loading one restores the per-connection history so
// the chat continues with full context. Writes happen after each completed
// turn; the store is tiny and local, matching chatllm's local-only stance.
// ---------------------------------------------------------------------------

import (
	"encoding/json"
	"fmt"
	"os"
	"path/filepath"
	"sort"
	"strings"
	"time"

	"github.com/sprout-foundry/sinter/llm"
)

// conversation is one persisted chat.
type conversation struct {
	ID        string      `json:"id"`
	Title     string      `json:"title"`
	Model     string      `json:"model"`
	Messages  []storedMsg `json:"messages"`
	CreatedAt time.Time   `json:"created_at"`
	UpdatedAt time.Time   `json:"updated_at"`
}

// storedMsg is the JSON-safe message shape (sinter's llm.ChatMessage has
// no json tags, so it would serialize as {"Role":…,"Content":…}).
type storedMsg struct {
	Role    string `json:"role"`
	Content string `json:"content"`
}

// toStored converts sinter messages for persistence.
func toStored(msgs []llm.ChatMessage) []storedMsg {
	out := make([]storedMsg, 0, len(msgs))
	for _, m := range msgs {
		out = append(out, storedMsg{Role: m.Role, Content: m.Content})
	}
	return out
}

// conversationsDir is ~/.chatllm/conversations.
func conversationsDir() string {
	home, err := os.UserHomeDir()
	if err != nil {
		return ""
	}
	return filepath.Join(home, ".chatllm", "conversations")
}

// newConversationID is a timestamp-based, collision-safe-enough id.
func newConversationID() string {
	return fmt.Sprintf("%d", time.Now().UnixNano())
}

// saveConversation writes the conversation file (best effort).
func saveConversation(c *conversation) {
	dir := conversationsDir()
	if dir == "" || c.ID == "" {
		return
	}
	if err := os.MkdirAll(dir, 0o755); err != nil {
		return
	}
	b, err := json.MarshalIndent(c, "", "  ")
	if err != nil {
		return
	}
	_ = os.WriteFile(filepath.Join(dir, c.ID+".json"), b, 0o644)
}

// loadConversation reads one conversation by id. IDs are sanitized to a
// plain digit string, so the path join stays inside the directory.
//
// Legacy files (written before storedMsg had json tags) carry capitalized
// "Role"/"Content" keys; Go's JSON unmarshal matches those to the tagged
// fields case-insensitively, so no special-casing is needed.
func loadConversation(id string) (*conversation, error) {
	id = strings.TrimSpace(id)
	if id == "" || strings.ContainsAny(id, "/\\.") {
		return nil, fmt.Errorf("bad conversation id")
	}
	b, err := os.ReadFile(filepath.Join(conversationsDir(), id+".json"))
	if err != nil {
		return nil, err
	}
	var c conversation
	if err := json.Unmarshal(b, &c); err != nil {
		return nil, err
	}
	return &c, nil
}

// conversationMeta is the sidebar listing entry (no messages).
type conversationMeta struct {
	ID        string `json:"id"`
	Title     string `json:"title"`
	Model     string `json:"model"`
	UpdatedAt int64  `json:"updated_at"`
}

// listConversations returns every stored conversation, newest first.
// Never nil: the client treats a null list as an error shape.
func listConversations() []conversationMeta {
	metas := []conversationMeta{}
	dir := conversationsDir()
	entries, err := os.ReadDir(dir)
	if err != nil {
		return metas
	}
	for _, e := range entries {
		if e.IsDir() || !strings.HasSuffix(e.Name(), ".json") {
			continue
		}
		c, err := loadConversation(strings.TrimSuffix(e.Name(), ".json"))
		if err != nil {
			continue // unreadable: skip, don't break the listing
		}
		metas = append(metas, conversationMeta{
			ID: c.ID, Title: c.Title, Model: c.Model, UpdatedAt: c.UpdatedAt.Unix(),
		})
	}
	sort.Slice(metas, func(i, j int) bool { return metas[i].UpdatedAt > metas[j].UpdatedAt })
	return metas
}

// deleteConversation removes one conversation file. Idempotent: deleting
// an already-gone conversation (double-click, stale sidebar) succeeds.
func deleteConversation(id string) error {
	id = strings.TrimSpace(id)
	if id == "" || strings.ContainsAny(id, "/\\.") {
		return fmt.Errorf("bad conversation id")
	}
	err := os.Remove(filepath.Join(conversationsDir(), id+".json"))
	if err != nil && os.IsNotExist(err) {
		return nil
	}
	return err
}

// conversationTitle derives a sidebar title from the first user message.
func conversationTitle(messages []llm.ChatMessage) string {
	for _, m := range messages {
		if m.Role == "user" {
			t := strings.TrimSpace(m.Content)
			t = strings.ReplaceAll(t, "\n", " ")
			if len(t) > 48 {
				t = t[:48] + "…"
			}
			return t
		}
	}
	return "Untitled"
}

// conversationTitleFromStored is conversationTitle over persisted messages.
func conversationTitleFromStored(messages []storedMsg) string {
	return conversationTitle(toLLMMessages(messages))
}

// toLLMMessages converts stored messages back to sinter ChatMessages.
func toLLMMessages(msgs []storedMsg) []llm.ChatMessage {
	out := make([]llm.ChatMessage, 0, len(msgs))
	for _, m := range msgs {
		out = append(out, llm.ChatMessage{Role: m.Role, Content: m.Content})
	}
	return out
}

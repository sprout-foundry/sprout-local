package conversations

// ---------------------------------------------------------------------------
// conversations.go — server-side conversation persistence for the web UI.
//
// Each conversation is a JSON file in <stateRoot>/conversations (default
// ~/.sprout-local/conversations; SPROUT_LOCAL_STATE_ROOT moves the root,
// see the paths package): id, title (first user message), model, and the
// message list. The web UI's sidebar lists them; loading one restores the
// per-connection history so the chat continues with full context. Writes
// happen after each completed turn; the store is tiny and local, matching
// sprout-local's local-only stance.
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

	"github.com/sprout-foundry/sprout-local/internal/paths"
)

// Conversation is one persisted chat.
type Conversation struct {
	ID        string      `json:"id"`
	Title     string      `json:"title"`
	Model     string      `json:"model"`
	Messages  []StoredMsg `json:"messages"`
	CreatedAt time.Time   `json:"created_at"`
	UpdatedAt time.Time   `json:"updated_at"`
}

// StoredMsg is the JSON-safe message shape (sinter's llm.ChatMessage has
// no json tags, so it would serialize as {"Role":…,"Content":…}).
type StoredMsg struct {
	Role    string `json:"role"`
	Content string `json:"content"`
}

// ToStored converts sinter messages for persistence.
func ToStored(msgs []llm.ChatMessage) []StoredMsg {
	out := make([]StoredMsg, 0, len(msgs))
	for _, m := range msgs {
		out = append(out, StoredMsg{Role: m.Role, Content: m.Content})
	}
	return out
}

// ConversationsDir is <stateRoot>/conversations (default
// ~/.sprout-local/conversations). Legacy compat: when the new directory
// does not exist but the pre-migration ~/.chatllm/conversations does,
// the legacy directory keeps serving reads and writes (no copying).
func ConversationsDir() string {
	state := paths.StateRoot()
	if state == "" {
		return ""
	}
	newDir := filepath.Join(state, "conversations")
	if paths.DirExists(newDir) {
		return newDir
	}
	if legacy := filepath.Join(paths.HomeDir(), ".chatllm", "conversations"); paths.DirExists(legacy) {
		return legacy
	}
	return newDir
}

// NewConversationID is a timestamp-based, collision-safe-enough id.
func NewConversationID() string {
	return fmt.Sprintf("%d", time.Now().UnixNano())
}

// SaveConversation writes the conversation file (best effort).
func SaveConversation(c *Conversation) {
	dir := ConversationsDir()
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

// LoadConversation reads one conversation by id. IDs are sanitized to a
// plain digit string, so the path join stays inside the directory.
//
// Legacy files (written before StoredMsg had json tags) carry capitalized
// "Role"/"Content" keys; Go's JSON unmarshal matches those to the tagged
// fields case-insensitively, so no special-casing is needed.
func LoadConversation(id string) (*Conversation, error) {
	id = strings.TrimSpace(id)
	if id == "" || strings.ContainsAny(id, "/\\.") {
		return nil, fmt.Errorf("bad conversation id")
	}
	b, err := os.ReadFile(filepath.Join(ConversationsDir(), id+".json"))
	if err != nil {
		return nil, err
	}
	var c Conversation
	if err := json.Unmarshal(b, &c); err != nil {
		return nil, err
	}
	return &c, nil
}

// ConversationMeta is the sidebar listing entry (no messages).
type ConversationMeta struct {
	ID        string `json:"id"`
	Title     string `json:"title"`
	Model     string `json:"model"`
	UpdatedAt int64  `json:"updated_at"`
}

// ListConversations returns every stored conversation, newest first.
// Never nil: the client treats a null list as an error shape.
func ListConversations() []ConversationMeta {
	metas := []ConversationMeta{}
	dir := ConversationsDir()
	entries, err := os.ReadDir(dir)
	if err != nil {
		return metas
	}
	for _, e := range entries {
		if e.IsDir() || !strings.HasSuffix(e.Name(), ".json") {
			continue
		}
		c, err := LoadConversation(strings.TrimSuffix(e.Name(), ".json"))
		if err != nil {
			continue // unreadable: skip, don't break the listing
		}
		metas = append(metas, ConversationMeta{
			ID: c.ID, Title: c.Title, Model: c.Model, UpdatedAt: c.UpdatedAt.Unix(),
		})
	}
	sort.Slice(metas, func(i, j int) bool { return metas[i].UpdatedAt > metas[j].UpdatedAt })
	return metas
}

// DeleteConversation removes one conversation file. Idempotent: deleting
// an already-gone conversation (double-click, stale sidebar) succeeds.
func DeleteConversation(id string) error {
	id = strings.TrimSpace(id)
	if id == "" || strings.ContainsAny(id, "/\\.") {
		return fmt.Errorf("bad conversation id")
	}
	err := os.Remove(filepath.Join(ConversationsDir(), id+".json"))
	if err != nil && os.IsNotExist(err) {
		return nil
	}
	return err
}

// ConversationTitle derives a sidebar title from the first user message.
func ConversationTitle(messages []llm.ChatMessage) string {
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

// ConversationTitleFromStored is ConversationTitle over persisted messages.
func ConversationTitleFromStored(messages []StoredMsg) string {
	return ConversationTitle(ToLLMMessages(messages))
}

// ToLLMMessages converts stored messages back to sinter ChatMessages.
func ToLLMMessages(msgs []StoredMsg) []llm.ChatMessage {
	out := make([]llm.ChatMessage, 0, len(msgs))
	for _, m := range msgs {
		out = append(out, llm.ChatMessage{Role: m.Role, Content: m.Content})
	}
	return out
}
package webui

// eventString extracts a string field from event payload maps.
func eventString(data interface{}, key string) string {
	if m, ok := data.(map[string]interface{}); ok {
		if s, ok := m[key].(string); ok {
			return s
		}
	}
	return ""
}

// eventJSON extracts an arbitrary field from event payload maps.
func eventJSON(data interface{}, key string) interface{} {
	if m, ok := data.(map[string]interface{}); ok {
		if v, ok := m[key]; ok {
			return v
		}
	}
	return nil
}

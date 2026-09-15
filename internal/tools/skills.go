package tools

// ---------------------------------------------------------------------------
// skills.go — user-addable tools as "skills".
//
// A skill is a JSON file in <stateRoot>/skills/*.json (default
// ~/.sprout-local/skills; SPROUT_LOCAL_SKILLS_DIR overrides the
// directory outright, SPROUT_LOCAL_STATE_ROOT moves the root):
//
//   {
//     "name": "sysinfo",
//     "description": "Read system load, memory and disk from the local sysinfo script",
//     "command": "/Users/alanp/bin/sysinfo.sh",
//     "args": ["-brief"]
//   }
//
// The tool takes no parameters from the model (the JSON pins the full
// command line), so a small model can't fumble arguments; the model only
// decides *whether* to call it. Skills join the registry between /tools
// toggles via ReloadSkills(); built-ins (tools.go) are always available.
// Skills run through the seed agent loop like any other tool when /tools
// is on.
// ---------------------------------------------------------------------------

import (
	"context"
	"encoding/json"
	"fmt"
	"os"
	"os/exec"
	"path/filepath"
	"sort"
	"strings"

	"github.com/sprout-foundry/sprout-local/internal/config"
	"github.com/sprout-foundry/sprout-local/internal/paths"
)

// skill is a user-defined no-parameter tool backed by a fixed command.
type skill struct {
	Name        string   `json:"name"`
	Description string   `json:"description"`
	Command     string   `json:"command"`
	Args        []string `json:"args"`
}

// skillRegistry holds the loaded skills (nil before first load).
var skillRegistry []skill

// ReloadSkills reads every *.json in the skills directory. Returns the
// count and an error list (bad files are skipped, not fatal).
func ReloadSkills() (int, []error) {
	skillRegistry = nil
	dir := paths.SkillsDir()
	entries, err := os.ReadDir(dir)
	if err != nil {
		return 0, nil // no skill dir: zero skills, not an error
	}
	var errs []error
	for _, e := range entries {
		if e.IsDir() || !strings.HasSuffix(e.Name(), ".json") {
			continue
		}
		b, err := os.ReadFile(filepath.Join(dir, ``, e.Name()))
		if err != nil {
			errs = append(errs, fmt.Errorf("%s: %w", e.Name(), err))
			continue
		}
		var s skill
		if err := json.Unmarshal(b, &s); err != nil {
			errs = append(errs, fmt.Errorf("%s: %w", e.Name(), err))
			continue
		}
		s.Name = strings.TrimSpace(s.Name)
		if s.Name == "" || s.Command == "" {
			errs = append(errs, fmt.Errorf("%s: name and command are required", e.Name()))
			continue
		}
		if t := lookupToolSpec(s.Name); t != nil {
			errs = append(errs, fmt.Errorf("%s: name %q collides with a built-in tool", e.Name(), s.Name))
			continue
		}
		skillRegistry = append(skillRegistry, s)
	}
	sort.Slice(skillRegistry, func(i, j int) bool { return skillRegistry[i].Name < skillRegistry[j].Name })
	return len(skillRegistry), errs
}

// lookupToolSpec finds a built-in tool by name (nil if absent).
func lookupToolSpec(name string) *ToolSpec {
	for i := range toolRegistry {
		if toolRegistry[i].Name == name {
			return &toolRegistry[i]
		}
	}
	return nil
}

// skillAsToolSpec converts a skill into the registry's ToolSpec shape.
func skillAsToolSpec(s skill) ToolSpec {
	return ToolSpec{
		Name:        s.Name,
		Description: s.Description + " (a personal skill — no parameters)",
		Parameters:  nil,
		Run: func(ctx context.Context, args map[string]string) (string, error) {
			return runSkillCommand(ctx, s)
		},
	}
}

// activeToolSpecs returns the full tool list the model may call: built-ins
// plus skills. Skills appear only when /tools is on (the executor is only
// wired then).
func activeToolSpecs() []ToolSpec {
	specs := make([]ToolSpec, len(toolRegistry), len(toolRegistry)+len(skillRegistry))
	copy(specs, toolRegistry)
	for _, s := range skillRegistry {
		specs = append(specs, skillAsToolSpec(s))
	}
	return specs
}

// runSkillCommand executes a skill's fixed command line (same gate rules
// as run_command: no shell, no metacharacters, timeout).
func runSkillCommand(ctx context.Context, s skill) (string, error) {
	// Skills are user-installed fixed command lines — no per-call prompt.
	// They run shell-free (exec argv, no /bin/sh), so substitution
	// characters are refused up front, same as yolo-mode run_command.
	for _, a := range append([]string{s.Command}, s.Args...) {
		for _, r := range a {
			if name, bad := commandDenyNames[r]; bad {
				return "", fmt.Errorf("skill %s: refusing argument with %s", s.Name, name)
			}
		}
	}
	ctx, cancel := context.WithTimeout(ctx, config.CommandTimeout)
	defer cancel()
	cmd := exec.CommandContext(ctx, s.Command, s.Args...)
	cmd.Dir = cwd()
	out, err := cmd.CombinedOutput()
	text := strings.TrimRight(string(out), "\n")
	if len(text) > config.ToolResultCap {
		text = text[:config.ToolResultCap] + "\n…[truncated]"
	}
	if err != nil {
		if text == "" {
			return "", err
		}
		return text + "\n(exit status: " + err.Error() + ")", nil
	}
	if text == "" {
		return "(no output)", nil
	}
	return text, nil
}

package workflows

import (
	"context"
	"encoding/json"
	"fmt"
	"os"
	"strings"
	"testing"
	"time"

	"github.com/fancl20/akasha/pkg/models"
	"github.com/fancl20/akasha/pkg/models/deepseek"
	"github.com/fancl20/akasha/pkg/notes"
	"go.temporal.io/sdk/activity"
	"go.temporal.io/sdk/testsuite"
)

func newTestModel(t *testing.T) models.Model {
	t.Helper()
	key := os.Getenv("DEEPSEEK_API_KEY")
	if key == "" {
		t.Skip("DEEPSEEK_API_KEY not set")
	}
	return deepseek.New(key)
}

func newTestRefs(t *testing.T) notes.Refs {
	t.Helper()
	s, err := notes.Open(":memory:")
	if err != nil {
		t.Fatalf("open db: %v", err)
	}
	return s.Refs()
}

func newTestEnv(t *testing.T) (*testsuite.TestWorkflowEnvironment, notes.Refs, models.Model) {
	t.Helper()
	suite := &testsuite.WorkflowTestSuite{}
	env := suite.NewTestWorkflowEnvironment()
	refs := newTestRefs(t)
	model := newTestModel(t)
	return env, refs, model
}

func registerActivities(env *testsuite.TestWorkflowEnvironment, refs notes.Refs, model models.Model, tools []models.Tool) {
	acts := &Activities{Refs: refs, Model: model, Tools: tools}
	env.RegisterActivityWithOptions(acts.AppendNote, activity.RegisterOptions{Name: ActivityAppendNote})
	env.RegisterActivityWithOptions(acts.GetNotes, activity.RegisterOptions{Name: ActivityGetNotes})
	env.RegisterActivityWithOptions(acts.Generate, activity.RegisterOptions{Name: ActivityGenerate})
	env.RegisterActivityWithOptions(acts.ToolCall, activity.RegisterOptions{Name: ActivityToolExecution})
}

func TestE2E_SimpleResponse(t *testing.T) {
	env, refs, model := newTestEnv(t)
	registerActivities(env, refs, model, nil)

	env.ExecuteWorkflow(AgentWorkflow, WorkflowInput{
		RefID:       "session-1",
		UserMessage: "What is 2 + 3? Reply with just the number.",
		Config:      &deepseek.Config{MaxTokens: 16},
	})

	requireNoWorkflowError(t, env)

	var result string
	requireNoError(t, env.GetWorkflowResult(&result))

	if !strings.Contains(result, "5") {
		t.Fatalf("expected response to contain 5, got %q", result)
	}

	// Verify notes were persisted.
	ref, err := refs.Get("session-1")
	requireNoError(t, err)
	count := countNotes(t, ref)
	if count < 2 {
		t.Fatalf("expected at least 2 notes (user + assistant), got %d", count)
	}
}

func TestE2E_NotesAreValidJSON(t *testing.T) {
	env, refs, model := newTestEnv(t)
	registerActivities(env, refs, model, nil)

	env.ExecuteWorkflow(AgentWorkflow, WorkflowInput{
		RefID:       "json-test",
		UserMessage: "Say hi",
		Config:      &deepseek.Config{MaxTokens: 16},
	})

	requireNoWorkflowError(t, env)

	ref, err := refs.Get("json-test")
	requireNoError(t, err)

	ns := ref.Notes()
	for n := range ns.Iter() {
		var m models.Message
		if err := json.Unmarshal([]byte(n.Content), &m); err != nil {
			t.Fatalf("note is not valid Message JSON: %v, content: %q", err, n.Content)
		}
		if m.Role == "" {
			t.Fatalf("message has empty role: %q", n.Content)
		}
	}
	if err := ns.Err(); err != nil {
		t.Fatalf("iterate notes: %v", err)
	}
}

func TestE2E_ToolUse(t *testing.T) {
	env, refs, model := newTestEnv(t)
	env.SetTestTimeout(time.Second * 5)

	weatherTool := models.NewFunctionTool("get_weather", "Get the current weather for a location", func(_ context.Context, in struct {
		Location string `json:"location" jsonschema_description:"City name"`
	}) (string, error) {
		return fmt.Sprintf("Sunny, 22°C in %s", in.Location), nil
	})

	registerActivities(env, refs, model, []models.Tool{weatherTool})

	env.ExecuteWorkflow(AgentWorkflow, WorkflowInput{
		RefID:       "tool-session",
		UserMessage: "What is the weather in Paris?",
		Config: &deepseek.Config{
			MaxTokens: 256,
		},
	})

	requireNoWorkflowError(t, env)

	var result string
	requireNoError(t, env.GetWorkflowResult(&result))

	if !strings.Contains(result, "22") {
		t.Fatalf("expected response to mention 22°C, got %q", result)
	}
}

func TestE2E_WithExistingHistory(t *testing.T) {
	env, refs, model := newTestEnv(t)

	// Pre-populate a user note.
	ref, err := refs.Get("history-session")
	requireNoError(t, err)
	prev := &models.Message{Role: "user", Content: models.Content{Text: "My name is Alice."}}
	requireNoError(t, ref.Append(&notes.Note{Content: prev.ToJSON()}))

	registerActivities(env, refs, model, nil)

	env.ExecuteWorkflow(AgentWorkflow, WorkflowInput{
		RefID:       "history-session",
		UserMessage: "What is my name? Reply with just the name.",
		Config:      &deepseek.Config{MaxTokens: 16},
	})

	requireNoWorkflowError(t, env)

	var result string
	requireNoError(t, env.GetWorkflowResult(&result))

	if !strings.Contains(strings.ToLower(result), "alice") {
		t.Fatalf("expected response to mention Alice, got %q", result)
	}

	// 1 pre-existing + 1 user + at least 1 assistant.
	ref, err = refs.Get("history-session")
	requireNoError(t, err)
	count := countNotes(t, ref)
	if count < 3 {
		t.Fatalf("expected at least 3 notes, got %d", count)
	}
}

// --- Helpers ---

func countNotes(t *testing.T, ref notes.Ref) int {
	t.Helper()
	ns := ref.Notes()
	var count int
	for range ns.Iter() {
		count++
	}
	if err := ns.Err(); err != nil {
		t.Fatalf("iterate notes: %v", err)
	}
	return count
}

func requireNoWorkflowError(t *testing.T, env *testsuite.TestWorkflowEnvironment) {
	t.Helper()
	if err := env.GetWorkflowError(); err != nil {
		t.Fatalf("workflow error: %v", err)
	}
}

func requireNoError(t *testing.T, err error) {
	t.Helper()
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
}

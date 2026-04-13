package workflows

import (
	"fmt"
	"strings"
	"time"

	"github.com/fancl20/akasha/pkg/models"
	"github.com/fancl20/akasha/pkg/notes"
	"go.temporal.io/sdk/temporal"
	"go.temporal.io/sdk/workflow"
)

const (
	TaskQueue = "agent-session"

	ActivityAppendNote    = "ActivityAppendNote"
	ActivityGetNotes      = "ActivityGetNotes"
	ActivityGenerate      = "ActivityGenerate"
	ActivityToolExecution = "ActivityToolExecution"
)

// WorkflowInput is the input to the AgentWorkflow.
type WorkflowInput struct {
	RefID       string
	UserMessage string
	Config      any
	Tools       map[string]any
}

// AgentWorkflow orchestrates an LLM agent session with thinking mode and tool use.
// Each invocation handles one user turn: loads history, calls the LLM (with potential
// tool-call loops), and persists all notes.
func AgentWorkflow(ctx workflow.Context, input WorkflowInput) (string, error) {
	ctx = workflow.WithActivityOptions(ctx, workflow.ActivityOptions{
		StartToCloseTimeout: time.Minute * 15,
		RetryPolicy: &temporal.RetryPolicy{
			MaximumAttempts: 3,
		},
	})

	// 1. Load persistent conversation history (before appending current user message).
	var msgs []*models.Message
	if err := workflow.ExecuteActivity(ctx, ActivityGetNotes, input.RefID).Get(ctx, &msgs); err != nil {
		return "", fmt.Errorf("get notes: %w", err)
	}

	// 2. Append user note.
	userMsg := &models.Message{
		Role:    "user",
		Content: models.Content{Text: input.UserMessage},
	}
	if err := workflow.ExecuteActivity(ctx, ActivityAppendNote, input.RefID, &notes.Note{
		Content:   userMsg.ToJSON(),
		CreatedAt: time.Now(),
	}).Get(ctx, nil); err != nil {
		return "", fmt.Errorf("append user note: %w", err)
	}

	// 3. Build initial message list for current turn.
	msgs = append(msgs, userMsg)

	// 4. Tool-call + thinking loop.
	var answer strings.Builder
	for {
		var output []*models.Message
		if err := workflow.ExecuteActivity(ctx, ActivityGenerate, &models.Request{
			Contents: msgs,
			Config:   input.Config,
		}).Get(ctx, &output); err != nil {
			return "", fmt.Errorf("llm activity: %w", err)
		}

		var pending bool
		for _, msg := range output {
			if err := workflow.ExecuteActivity(ctx, ActivityAppendNote, input.RefID, &notes.Note{
				Content:   msg.ToJSON(),
				CreatedAt: time.Now(),
			}).Get(ctx, nil); err != nil {
				return "", fmt.Errorf("append user note: %w", err)
			}
			msgs = append(msgs, msg)

			if tc := msg.Content.ToolCall; tc != nil {
				pending = true

				var res string
				if err := workflow.ExecuteActivity(
					workflow.WithActivityOptions(ctx, workflow.ActivityOptions{
						ScheduleToCloseTimeout: time.Hour,
						RetryPolicy:            &temporal.RetryPolicy{MaximumAttempts: 1},
					}), ActivityToolExecution, tc).Get(ctx, &res); err != nil {
					res = fmt.Sprintf("tool call: %v", err)
				}

				toolMsg := &models.Message{
					Role: "tool",
					Content: models.Content{
						Text:     res,
						ToolCall: &models.ToolCall{ID: tc.ID},
					},
				}
				if err := workflow.ExecuteActivity(ctx, ActivityAppendNote, input.RefID, &notes.Note{
					Content:   toolMsg.ToJSON(),
					CreatedAt: time.Now(),
				}).Get(ctx, nil); err != nil {
					return "", fmt.Errorf("append user note: %w", err)
				}
				msgs = append(msgs, toolMsg)
			}
		}

		if !pending {
			for _, msg := range output {
				if !msg.Content.Thought {
					_, _ = answer.WriteString(msg.Content.Text)
				}
			}
			break
		}
	}

	return answer.String(), nil
}

package workflows

import (
	"fmt"
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

	ActivitySendMessage    = "ActivitySendMessage"
	ActivityReceiveMessage = "ActivityReceiveMessage"
)

// SessionTurnParam is the param to the SessionTurnWorkflow.
type SessionTurnParam struct {
	RefID       string
	UserMessage string
	Config      any
	Tools       map[string]any
}

// AgentWorkflow orchestrates an LLM agent session with thinking mode and tool use.
// Each invocation handles one user turn: loads history, calls the LLM (with potential
// tool-call loops), and persists all notes.
func SessionTurnWorkflow(ctx workflow.Context, param SessionTurnParam) ([]*notes.Note, error) {
	ctx = workflow.WithActivityOptions(ctx, workflow.ActivityOptions{
		StartToCloseTimeout: time.Minute * 15,
		RetryPolicy: &temporal.RetryPolicy{
			MaximumAttempts: 3,
		},
	})

	// 1. Load persistent conversation history (before appending current user message).
	var msgs []*models.Message
	if err := workflow.ExecuteActivity(ctx, ActivityGetNotes, param.RefID).Get(ctx, &msgs); err != nil {
		return nil, fmt.Errorf("get notes: %w", err)
	}

	// 2. Append user note.
	userMsg := &models.Message{
		Role:    "user",
		Content: models.Content{Text: param.UserMessage},
	}
	if err := workflow.ExecuteActivity(ctx, ActivityAppendNote, param.RefID, &notes.Note{
		Content:   userMsg.ToJSON(),
		CreatedAt: time.Now(),
	}).Get(ctx, nil); err != nil {
		return nil, fmt.Errorf("append user note: %w", err)
	}

	// 3. Build initial message list for current turn.
	msgs = append(msgs, userMsg)

	// 4. Tool-call + thinking loop.
	var answer []*notes.Note
	for {
		var output []*models.Message
		if err := workflow.ExecuteActivity(ctx, ActivityGenerate, &models.Request{
			Contents: msgs,
			Config:   param.Config,
		}).Get(ctx, &output); err != nil {
			return nil, fmt.Errorf("llm activity: %w", err)
		}

		var pending bool
		for _, msg := range output {
			if err := workflow.ExecuteActivity(ctx, ActivityAppendNote, param.RefID, &notes.Note{
				Content:   msg.ToJSON(),
				CreatedAt: time.Now(),
			}).Get(ctx, nil); err != nil {
				return nil, fmt.Errorf("append model note: %w", err)
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
				if err := workflow.ExecuteActivity(ctx, ActivityAppendNote, param.RefID, &notes.Note{
					Content:   toolMsg.ToJSON(),
					CreatedAt: time.Now(),
				}).Get(ctx, nil); err != nil {
					return nil, fmt.Errorf("append tool note: %w", err)
				}
				msgs = append(msgs, toolMsg)
			}
		}

		if !pending {
			for _, msg := range output {
				if !msg.Content.Thought {
					answer = append(answer, &notes.Note{
						Content:   msg.ToJSON(),
						CreatedAt: time.Now(),
					})
				}
			}
			break
		}
	}

	return answer, nil
}

// SessionParam is the param to the SessionWorkflow.
type SessionParam struct {
	RefID     string
	ChannelID int64
}

func SessionWorkflow(ctx workflow.Context, param SessionParam) (string, error) {
	ctx = workflow.WithActivityOptions(ctx, workflow.ActivityOptions{
		StartToCloseTimeout: time.Minute * 15,
		RetryPolicy: &temporal.RetryPolicy{
			MaximumAttempts: 3,
		},
	})

	childOpts := workflow.ChildWorkflowOptions{TaskQueue: TaskQueue}

	var offset int64
	for {
		var msgs []*notes.Note
		if err := workflow.ExecuteActivity(ctx, ActivityReceiveMessage, offset).Get(ctx, &msgs); err != nil {
			return "", fmt.Errorf("receive message: %w", err)
		}

		if len(msgs) == 0 {
			continue
		}

		for _, msg := range msgs {
			offset++

			var result []*notes.Note
			if err := workflow.ExecuteChildWorkflow(
				workflow.WithChildOptions(ctx, childOpts),
				SessionTurnWorkflow,
				SessionTurnParam{
					RefID:       param.RefID,
					UserMessage: msg.Content,
				},
			).Get(ctx, &result); err != nil {
				return "", fmt.Errorf("session turn: %w", err)
			}

			if len(result) > 0 {
				if err := workflow.ExecuteActivity(ctx, ActivitySendMessage, result).Get(ctx, nil); err != nil {
					return "", fmt.Errorf("send message: %w", err)
				}
			}
		}
	}
}

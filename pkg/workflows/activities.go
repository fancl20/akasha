package workflows

import (
	"context"
	"encoding/json"
	"fmt"
	"slices"

	"github.com/fancl20/akasha/pkg/messages"
	"github.com/fancl20/akasha/pkg/models"
	"github.com/fancl20/akasha/pkg/notes"
	"go.temporal.io/sdk/temporal"
)

// ModelActivities holds dependencies for all Temporal activities.
type ModelActivities struct {
	Refs  notes.Refs
	Model models.Model
	Tools []models.Tool
}

// AppendNote serializes a Message and appends it as a Note to the session ref.
func (a *ModelActivities) AppendNote(ctx context.Context, refID string, n *notes.Note) error {
	if refID == "" {
		return temporal.NewNonRetryableApplicationError("empty ref id", "InvalidArgument", nil)
	}

	ref, err := a.Refs.Get(refID)
	if err != nil {
		return fmt.Errorf("get ref: %w", err)
	}

	return ref.Append(n)
}

// GetNotes loads notes for a session.
func (a *ModelActivities) GetNotes(ctx context.Context, refID string) ([]*models.Message, error) {
	ref, err := a.Refs.Get(refID)
	if err != nil {
		return nil, fmt.Errorf("get ref: %w", err)
	}

	ns := ref.Notes()
	var msgs []*models.Message
	for n := range ns.Iter() {
		var m models.Message
		if err := m.FromJSON(n.Content); err != nil {
			return nil, fmt.Errorf("unmarshal note: %v, %w", n, err)
		}
		msgs = append(msgs, &m)
	}
	if err := ns.Err(); err != nil {
		return nil, fmt.Errorf("iterate notes: %w", err)
	}
	slices.Reverse(msgs)

	return msgs, nil
}

// Generate calls the model and accumulates all responses into a single LLMOutput.
// It consumes the full iterator (streaming or non-streaming) and merges reasoning,
// content, and tool calls.
func (a *ModelActivities) Generate(ctx context.Context, req *models.Request) ([]*models.Message, error) {
	var msgs []*models.Message
	for resp, err := range a.Model.Generate(ctx, a.Tools, req) {
		if err != nil {
			return nil, fmt.Errorf("generate: %w", err)
		}
		msgs = append(msgs, resp.Messages...)
	}
	return msgs, nil
}

// ToolCall looks up a tool by name and executes it.
func (a *ModelActivities) ToolCall(ctx context.Context, tc *models.ToolCall) (string, error) {
	var tool models.Tool
	for _, t := range a.Tools {
		if t.Name() == tc.Name {
			tool = t
			break
		}
	}
	if tool == nil {
		return "", fmt.Errorf("tool not found: %q", tc.Name)
	}

	res, err := tool.Call(ctx, tc.Args)
	if err != nil {
		return "", err
	}
	b, err := json.Marshal(res)
	if err != nil {
		return "", fmt.Errorf("marshal tool output: %w", err)
	}
	return string(b), err
}

// MessageActivities holds dependencies for channel-based send/receive activities.
type MessageActivities struct {
	Channel messages.Channel
}

// SendMessage sends notes through the channel.
func (a *MessageActivities) SendMessage(ctx context.Context, ns []*notes.Note) error {
	return a.Channel.Send(ctx, ns)
}

// ReceiveMessage receives notes from the channel.
func (a *MessageActivities) ReceiveMessage(ctx context.Context, offset int64) ([]*notes.Note, error) {
	return a.Channel.Recieve(ctx, offset)
}

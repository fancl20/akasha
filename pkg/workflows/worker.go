package workflows

import (
	"fmt"

	"github.com/fancl20/akasha/pkg/messages"
	"github.com/fancl20/akasha/pkg/models"
	"github.com/fancl20/akasha/pkg/notes"
	"go.temporal.io/sdk/client"
	"go.temporal.io/sdk/worker"
)

// WorkerConfig holds the configuration for creating a Temporal worker.
type WorkerConfig struct {
	TaskQueue string
	Refs      notes.Refs
	Model     models.Model
	Tools     []models.Tool
	Channel   messages.Channel
}

// NewWorker creates a Temporal worker that registers the SessionTurnWorkflow and all activities.
// The caller is responsible for managing the temporalClient lifecycle.
func NewWorker(temporalClient client.Client, cfg WorkerConfig) (worker.Worker, error) {
	if cfg.TaskQueue == "" {
		cfg.TaskQueue = TaskQueue
	}
	if cfg.Refs == nil {
		return nil, fmt.Errorf("refs is required")
	}
	if cfg.Model == nil {
		return nil, fmt.Errorf("model is required")
	}
	if cfg.Channel == nil {
		return nil, fmt.Errorf("channel is required")
	}

	w := worker.New(temporalClient, cfg.TaskQueue, worker.Options{})

	w.RegisterWorkflow(SessionWorkflow)
	w.RegisterWorkflow(SessionTurnWorkflow)

	acts := &ModelActivities{
		Refs:  cfg.Refs,
		Model: cfg.Model,
		Tools: cfg.Tools,
	}
	w.RegisterActivity(acts.AppendNote)
	w.RegisterActivity(acts.GetNotes)
	w.RegisterActivity(acts.Generate)
	w.RegisterActivity(acts.ToolCall)

	msgActs := &MessageActivities{Channel: cfg.Channel}
	w.RegisterActivity(msgActs.SendMessage)
	w.RegisterActivity(msgActs.ReceiveMessage)

	return w, nil
}

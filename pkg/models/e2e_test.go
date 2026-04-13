package models_test

import (
	"context"
	"encoding/json"
	"fmt"
	"os"
	"strings"
	"testing"

	"github.com/fancl20/akasha/pkg/models"
	"github.com/fancl20/akasha/pkg/models/deepseek"
)

func newTestModel(t *testing.T) models.Model {
	t.Helper()
	key := os.Getenv("DEEPSEEK_API_KEY")
	if key == "" {
		t.Skip("DEEPSEEK_API_KEY not set")
	}
	return deepseek.New(key)
}

func TestE2E_BasicGenerate(t *testing.T) {
	m := newTestModel(t)
	req := &models.Request{
		Contents: []*models.Message{
			{Role: "user", Content: models.Content{Text: "What is 2 + 3? Reply with just the number."}},
		},
		Config: &deepseek.Config{MaxTokens: 16},
	}

	var text string
	for resp, err := range m.Generate(context.Background(), nil, req) {
		if err != nil {
			t.Fatalf("Generate: %v", err)
		}
		for _, msg := range resp.Messages {
			text += msg.Content.Text
		}
	}

	if !strings.Contains(text, "5") {
		t.Fatalf("expected response to contain 5, got %q", text)
	}
}

func TestE2E_StreamingGenerate(t *testing.T) {
	m := newTestModel(t)
	req := &models.Request{
		Contents: []*models.Message{
			{Role: "user", Content: models.Content{Text: "Say exactly: hello world"}},
		},
		Config: &deepseek.Config{Stream: true},
	}

	var chunks int
	var text string
	for resp, err := range m.Generate(context.Background(), nil, req) {
		if err != nil {
			t.Fatalf("Generate: %v", err)
		}
		chunks++
		for _, msg := range resp.Messages {
			text += msg.Content.Text
		}
	}

	if chunks < 2 {
		t.Fatalf("expected multiple streaming chunks, got %d", chunks)
	}
	if !strings.Contains(strings.ToLower(text), "hello") {
		t.Fatalf("unexpected response: %q", text)
	}
}

type weatherInput struct {
	Location string `json:"location" jsonschema_description:"City name"`
}

func TestE2E_ToolUse(t *testing.T) {
	m := newTestModel(t)

	// Define a tool using FunctionTool
	weatherTool := models.NewFunctionTool("get_weather", "Get the current weather for a location", func(ctx context.Context, in weatherInput) (string, error) {
		return fmt.Sprintf("Sunny, 22°C in %s", in.Location), nil
	})

	// Build the request with tool declared
	req := &models.Request{
		Contents: []*models.Message{
			{Role: "user", Content: models.Content{Text: "What is the weather in Paris?"}},
		},
		Config: &deepseek.Config{
			MaxTokens:  256,
			ToolChoice: json.RawMessage(`"required"`),
		},
	}

	// Step 1: Get the model's tool call
	var toolCalls []*models.ToolCall
	for resp, err := range m.Generate(context.Background(), []models.Tool{weatherTool}, req) {
		if err != nil {
			t.Fatalf("Generate: %v", err)
		}
		for _, msg := range resp.Messages {
			if tc := msg.Content.ToolCall; tc.ID != "" {
				toolCalls = append(toolCalls, tc)
			}
		}
	}

	if len(toolCalls) == 0 {
		t.Fatal("expected at least one tool call")
	}
	tc := toolCalls[0]
	if tc.Name != "get_weather" {
		t.Fatalf("expected tool get_weather, got %q", tc.Name)
	}
	loc, _ := tc.Args["location"].(string)
	if !strings.Contains(strings.ToLower(loc), "paris") {
		t.Fatalf("expected location to contain paris, got %v", loc)
	}

	// Step 2: Execute the tool locally
	result, err := weatherTool.Call(context.Background(), tc.Args)
	if err != nil {
		t.Fatalf("tool Call: %v", err)
	}
	resultStr := fmt.Sprint(result)

	// Step 3: Feed the tool result back to the model
	req2 := &models.Request{
		Contents: append(req.Contents,
			&models.Message{
				Role: "assistant",
				Content: models.Content{
					ToolCall: tc,
				},
			},
			&models.Message{
				Role: "tool",
				Content: models.Content{
					ToolCall: &models.ToolCall{
						ID: tc.ID,
					},
					Text: resultStr,
				},
			},
		),
		Config: &deepseek.Config{MaxTokens: 128},
	}

	var finalText string
	for resp, err := range m.Generate(context.Background(), []models.Tool{weatherTool}, req2) {
		if err != nil {
			t.Fatalf("second Generate: %v", err)
		}
		for _, msg := range resp.Messages {
			finalText += msg.Content.Text
		}
	}

	if !strings.Contains(strings.ToLower(finalText), "22") {
		t.Fatalf("expected final response to mention 22°C, got %q", finalText)
	}
}

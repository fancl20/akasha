package deepseek

import (
	"context"
	"encoding/json"
	"fmt"
	"os"
	"strings"
	"testing"

	"github.com/fancl20/akasha/pkg/models"
)

func newTestClient(t *testing.T) *DeepSeek {
	t.Helper()
	key := os.Getenv("DEEPSEEK_API_KEY")
	if key == "" {
		t.Skip("DEEPSEEK_API_KEY not set")
	}
	return New(key)
}

func TestGenerateNonStream(t *testing.T) {
	m := newTestClient(t)
	req := &models.Request{
		Contents: []*models.Message{
			{Role: "user", Content: models.Content{Text: "Say exactly: hello world"}},
		},
		Config: &Config{MaxTokens: 32},
	}

	var text string
	for resp, err := range m.Generate(context.Background(), req) {
		if err != nil {
			t.Fatalf("Generate: %v", err)
		}
		if resp == nil {
			t.Fatal("got nil response")
		}
		if len(resp.Messages) == 0 {
			t.Fatal("got empty messages")
		}
		for _, msg := range resp.Messages {
			text += msg.Content.Text
		}
	}

	if text == "" {
		t.Fatal("got empty response text")
	}
	if !strings.Contains(strings.ToLower(text), "hello") {
		t.Fatalf("unexpected response: %q", text)
	}
}

func TestGenerateStream(t *testing.T) {
	m := newTestClient(t)
	req := &models.Request{
		Contents: []*models.Message{
			{Role: "user", Content: models.Content{Text: "Say exactly: ping"}},
		},
		Config: &Config{
			Stream:        true,
			StreamOptions: &StreamOptions{IncludeUsage: true},
		},
	}

	var chunks int
	var text string
	for resp, err := range m.Generate(context.Background(), req) {
		if err != nil {
			t.Fatalf("Generate: %v", err)
		}
		if resp == nil {
			t.Fatal("got nil response")
		}
		if len(resp.Messages) == 0 {
			t.Fatal("got empty messages")
		}
		chunks++
		for _, msg := range resp.Messages {
			text += msg.Content.Text
		}
	}

	if chunks == 0 {
		t.Fatal("expected at least one chunk")
	}
	if !strings.Contains(strings.ToLower(text), "ping") {
		t.Fatalf("unexpected response: %q", text)
	}
}

func TestName(t *testing.T) {
	m := New("test-key")
	if m.Name() != "deepseek-chat" {
		t.Fatalf("expected deepseek-chat, got %q", m.Name())
	}

	m2 := New("test-key", WithModel("deepseek-reasoner"))
	if m2.Name() != "deepseek-reasoner" {
		t.Fatalf("expected deepseek-reasoner, got %q", m2.Name())
	}
}

func TestGenerateContextCancel(t *testing.T) {
	m := newTestClient(t)
	ctx, cancel := context.WithCancel(context.Background())
	cancel()

	req := &models.Request{
		Contents: []*models.Message{
			{Role: "user", Content: models.Content{Text: "hello"}},
		},
	}

	for _, err := range m.Generate(ctx, req) {
		if err != nil {
			return // expected: cancelled context
		}
	}
}

func TestGenerateToolCallRequired(t *testing.T) {
	m := newTestClient(t)
	req := &models.Request{
		Contents: []*models.Message{
			{Role: "user", Content: models.Content{Text: "What is the weather in Tokyo?"}},
		},
		Tools: map[string]any{
			"get_weather": map[string]any{
				"type": "object",
				"properties": map[string]any{
					"location": map[string]any{
						"type":        "string",
						"description": "City name",
					},
				},
				"required": []string{"location"},
			},
		},
		Config: &Config{
			MaxTokens:  128,
			ToolChoice: json.RawMessage(`"required"`),
		},
	}

	var toolCalls []models.ToolCall
	for resp, err := range m.Generate(context.Background(), req) {
		if err != nil {
			t.Fatalf("Generate: %v", err)
		}
		if resp == nil {
			t.Fatal("got nil response")
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
		t.Fatalf("expected tool call get_weather, got %q", tc.Name)
	}
	if tc.Args == nil {
		t.Fatal("expected tool call args")
	}
	loc, ok := tc.Args["location"]
	if !ok {
		t.Fatalf("expected location arg, got %v", tc.Args)
	}
	if !strings.Contains(strings.ToLower(fmt.Sprint(loc)), "tokyo") {
		t.Fatalf("expected location to contain tokyo, got %v", loc)
	}
}

package models

import (
	"context"
	"encoding/json"
	"iter"
)

type Model interface {
	Name() string
	Generate(ctx context.Context, tool []Tool, req *Request) iter.Seq2[*Response, error]
}

type Request struct {
	Model    string
	Contents []*Message
	Config   any
}

type Response struct {
	Messages []*Message
}

type Message struct {
	Role    string  `json:"role"`
	Content Content `json:"content"`
}

func (m *Message) ToJSON() string {
	if data, err := json.Marshal(m); err != nil {
		panic(err)
	} else {
		return string(data)
	}
}

func (m *Message) FromJSON(data string) error {
	return json.Unmarshal([]byte(data), m)
}

type Content struct {
	Text     string    `json:"text,omitempty"`
	Thought  bool      `json:"thought,omitempty"`
	ToolCall *ToolCall `json:"tool_call,omitempty"`
}

type ToolCall struct {
	ID   string         `json:"id,omitempty"`
	Name string         `json:"name,omitempty"`
	Args map[string]any `json:"args,omitempty"`
}

package models

import (
	"context"
	"iter"
)

type Model interface {
	Name() string
	Generate(ctx context.Context, req *Request) iter.Seq2[*Response, error]
}

type Request struct {
	Model    string
	Contents []*Message
	Config   any

	Tools map[string]any
}

type Response struct {
	Messages []*Message
}

type Message struct {
	Content Content
	Role    string
}

type Content struct {
	Text     string
	Thought  bool
	ToolCall ToolCall
}

type ToolCall struct {
	ID   string
	Name string
	Args map[string]any
}

package models

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"reflect"

	"github.com/google/jsonschema-go/jsonschema"
)

type Tool interface {
	Name() string
	Description() string
	ParameterSchema() map[string]any
	Call(ctx context.Context, args map[string]any) (any, error)
}

func Declaration(t Tool) map[string]any {
	return map[string]any{
		"type": "function",
		"function": map[string]any{
			"name":        t.Name(),
			"description": t.Description(),
			"parameters":  t.ParameterSchema(),
		},
	}
}

type FunctionTool[In, Out any] struct {
	name        string
	description string
	handler     func(ctx context.Context, input In) (Out, error)
}

func NewFunctionTool[In, Out any](
	name, description string,
	handler func(ctx context.Context, input In) (Out, error),
) (*FunctionTool[In, Out], error) {
	// Validate handler signature (already enforced by generics, but double-check)
	if handler == nil {
		return nil, errors.New("handler cannot be nil")
	}
	return &FunctionTool[In, Out]{
		name:        name,
		description: description,
		handler:     handler,
	}, nil
}

func (ft *FunctionTool[In, Out]) Name() string        { return ft.name }
func (ft *FunctionTool[In, Out]) Description() string { return ft.description }

func (ft *FunctionTool[In, Out]) ParameterSchema() map[string]any {
	var zero In
	t := reflect.TypeOf(zero)

	schema, err := jsonschema.ForType(t, nil)
	if err != nil {
		return map[string]any{"type": "object"}
	}

	data, err := json.Marshal(schema)
	if err != nil {
		return map[string]any{"type": "object"}
	}
	var result map[string]any
	if err := json.Unmarshal(data, &result); err != nil {
		return map[string]any{"type": "object"}
	}
	return result
}

func (ft *FunctionTool[In, Out]) Call(ctx context.Context, args map[string]any) (any, error) {
	var input In
	data, err := json.Marshal(args)
	if err != nil {
		return nil, fmt.Errorf("marshal args: %w", err)
	}
	if err := json.Unmarshal(data, &input); err != nil {
		return nil, fmt.Errorf("unmarshal to input: %w", err)
	}
	return ft.handler(ctx, input)
}

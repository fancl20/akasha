package models

import (
	"context"
	"errors"
	"testing"
)

// testInput is a simple struct for testing ParameterSchema and Call.
type testInput struct {
	Name string `json:"name"`
	Age  int    `json:"age"`
}

func TestNewFunctionTool_NilHandler(t *testing.T) {
	_, err := NewFunctionTool[testInput, string]("test", "desc", nil)
	if err == nil {
		t.Fatal("expected error for nil handler")
	}
	if err.Error() != "handler cannot be nil" {
		t.Fatalf("unexpected error: %v", err)
	}
}

func TestNewFunctionTool_Valid(t *testing.T) {
	handler := func(ctx context.Context, in testInput) (string, error) {
		return in.Name, nil
	}
	tool, err := NewFunctionTool("echo", "echoes name", handler)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if tool.Name() != "echo" {
		t.Fatalf("expected name echo, got %s", tool.Name())
	}
	if tool.Description() != "echoes name" {
		t.Fatalf("expected description echoes name, got %s", tool.Description())
	}
}

func TestDeclaration(t *testing.T) {
	handler := func(ctx context.Context, in testInput) (string, error) {
		return "", nil
	}
	tool, _ := NewFunctionTool("mytool", "my desc", handler)

	decl := Declaration(tool)
	if decl["type"] != "function" {
		t.Fatalf("expected type function, got %v", decl["type"])
	}
	fn := decl["function"].(map[string]any)
	if fn["name"] != "mytool" {
		t.Fatalf("expected name mytool, got %v", fn["name"])
	}
	if fn["description"] != "my desc" {
		t.Fatalf("expected description my desc, got %v", fn["description"])
	}
	params, ok := fn["parameters"].(map[string]any)
	if !ok {
		t.Fatal("expected parameters to be a map")
	}
	props, ok := params["properties"].(map[string]any)
	if !ok {
		t.Fatal("expected properties in schema")
	}
	if _, ok := props["name"]; !ok {
		t.Fatal("expected name property in schema")
	}
	if _, ok := props["age"]; !ok {
		t.Fatal("expected age property in schema")
	}
}

func TestParameterSchema_Struct(t *testing.T) {
	handler := func(ctx context.Context, in testInput) (string, error) {
		return "", nil
	}
	tool, _ := NewFunctionTool("t", "d", handler)
	schema := tool.ParameterSchema()

	props, ok := schema["properties"].(map[string]any)
	if !ok {
		t.Fatal("expected properties in schema")
	}
	if len(props) != 2 {
		t.Fatalf("expected 2 properties, got %d", len(props))
	}
}

func TestParameterSchema_Primitive(t *testing.T) {
	handler := func(ctx context.Context, in string) (string, error) {
		return in, nil
	}
	tool, _ := NewFunctionTool[string, string]("t", "d", handler)
	schema := tool.ParameterSchema()
	// For a non-struct zero value, jsonschema may still return something valid.
	// At minimum it should not panic and should return a map.
	if schema == nil {
		t.Fatal("schema should not be nil")
	}
}

func TestCall_Success(t *testing.T) {
	handler := func(ctx context.Context, in testInput) (string, error) {
		return in.Name, nil
	}
	tool, _ := NewFunctionTool("t", "d", handler)

	result, err := tool.Call(context.Background(), map[string]any{
		"name": "alice",
		"age":  30,
	})
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if result != "alice" {
		t.Fatalf("expected alice, got %v", result)
	}
}

func TestCall_HandlerError(t *testing.T) {
	handler := func(ctx context.Context, in testInput) (string, error) {
		return "", errors.New("boom")
	}
	tool, _ := NewFunctionTool("t", "d", handler)

	_, err := tool.Call(context.Background(), map[string]any{
		"name": "alice",
		"age":  30,
	})
	if err == nil {
		t.Fatal("expected error")
	}
	if err.Error() != "boom" {
		t.Fatalf("expected boom, got %v", err)
	}
}

func TestCall_BadArgs(t *testing.T) {
	handler := func(ctx context.Context, in testInput) (string, error) {
		return "", nil
	}
	tool, _ := NewFunctionTool("t", "d", handler)

	// channels are not JSON-marshalable, so this should fail
	_, err := tool.Call(context.Background(), map[string]any{
		"ch": make(chan int),
	})
	if err == nil {
		t.Fatal("expected error for unmarshalable args")
	}
}

func TestCall_TypeMismatch(t *testing.T) {
	handler := func(ctx context.Context, in testInput) (string, error) {
		return in.Name, nil
	}
	tool, _ := NewFunctionTool("t", "d", handler)

	// age is a string instead of int — should fail on unmarshal
	_, err := tool.Call(context.Background(), map[string]any{
		"name": "alice",
		"age":  "not-a-number",
	})
	if err == nil {
		t.Fatal("expected error for type mismatch")
	}
}

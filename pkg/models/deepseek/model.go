package deepseek

import (
	"bufio"
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"io"
	"iter"
	"net/http"
	"strings"

	"github.com/fancl20/akasha/pkg/models"
)

// Config holds DeepSeek-specific generation parameters.
type Config struct {
	Stream           bool            `json:"stream,omitempty"`
	StreamOptions    *StreamOptions  `json:"stream_options,omitempty"`
	Temperature      *float64        `json:"temperature,omitempty"`
	TopP             *float64        `json:"top_p,omitempty"`
	MaxTokens        int             `json:"max_tokens,omitempty"`
	FrequencyPenalty *float64        `json:"frequency_penalty,omitempty"`
	PresencePenalty  *float64        `json:"presence_penalty,omitempty"`
	Thinking         *Thinking       `json:"thinking,omitempty"`
	ToolChoice       json.RawMessage `json:"tool_choice,omitempty"`
}

// DeepSeek implements models.Model for the DeepSeek chat completions API.
type DeepSeek struct {
	apiKey     string
	baseURL    string
	model      string
	httpClient *http.Client
}

// Option configures a DeepSeek instance.
type Option func(*DeepSeek)

func WithBaseURL(url string) Option {
	return func(d *DeepSeek) { d.baseURL = url }
}

func WithModel(model string) Option {
	return func(d *DeepSeek) { d.model = model }
}

func WithHTTPClient(c *http.Client) Option {
	return func(d *DeepSeek) { d.httpClient = c }
}

// New creates a new DeepSeek model client.
func New(apiKey string, opts ...Option) *DeepSeek {
	d := &DeepSeek{
		apiKey:     apiKey,
		baseURL:    "https://api.deepseek.com",
		model:      "deepseek-chat",
		httpClient: &http.Client{},
	}
	for _, opt := range opts {
		opt(d)
	}
	return d
}

// Name returns the model identifier.
func (m *DeepSeek) Name() string { return m.model }

// Generate sends a request to the DeepSeek API and yields responses.
// If Config.Stream is true, it yields SSE chunks as they arrive.
// Otherwise, it yields a single response.
func (m *DeepSeek) Generate(ctx context.Context, tools []models.Tool, req *models.Request) iter.Seq2[*models.Response, error] {
	return func(yield func(*models.Response, error) bool) {
		dsReq, err := convertRequest(m.model, tools, req)
		if err != nil {
			yield(nil, fmt.Errorf("convert request: %w", err))
			return
		}

		body, err := json.Marshal(dsReq)
		if err != nil {
			yield(nil, fmt.Errorf("marshal request: %w", err))
			return
		}

		httpReq, err := http.NewRequestWithContext(ctx, http.MethodPost, m.baseURL+"/chat/completions", bytes.NewReader(body))
		if err != nil {
			yield(nil, fmt.Errorf("create request: %w", err))
			return
		}
		httpReq.Header.Set("Content-Type", "application/json")
		httpReq.Header.Set("Authorization", "Bearer "+m.apiKey)

		resp, err := m.httpClient.Do(httpReq)
		if err != nil {
			yield(nil, fmt.Errorf("execute request: %w", err))
			return
		}
		defer resp.Body.Close()

		if resp.StatusCode != http.StatusOK {
			respBody, _ := io.ReadAll(resp.Body)
			yield(nil, fmt.Errorf("deepseek API error %d: %s", resp.StatusCode, respBody))
			return
		}

		if dsReq.Stream {
			m.streamResponse(resp.Body, yield)
		} else {
			m.nonStreamResponse(resp.Body, yield)
		}
	}
}

func (m *DeepSeek) streamResponse(body io.Reader, yield func(*models.Response, error) bool) {
	type pendingToolCall struct {
		id      string
		name    string
		rawArgs strings.Builder
	}
	pending := make(map[int]*pendingToolCall)

	scanner := bufio.NewScanner(body)
	buf := make([]byte, 0, 1024*1024)
	scanner.Buffer(buf, 1024*1024)

	for scanner.Scan() {
		line := scanner.Text()

		if line == "" || strings.HasPrefix(line, ":") {
			continue
		}

		if !strings.HasPrefix(line, "data: ") {
			continue
		}
		data := strings.TrimPrefix(line, "data: ")

		if data == "[DONE]" {
			// Yield any remaining accumulated tool calls
			for _, tc := range pending {
				args := parseToolArgs(tc.rawArgs.String())
				r := &models.Response{
					Messages: []*models.Message{{
						Role: "assistant",
						Content: models.Content{
							ToolCall: &models.ToolCall{
								ID:   tc.id,
								Name: tc.name,
								Args: args,
							},
						},
					}},
				}
				if !yield(r, nil) {
					return
				}
			}
			return
		}

		var chunk ChatCompletionChunk
		if err := json.Unmarshal([]byte(data), &chunk); err != nil {
			yield(nil, fmt.Errorf("parse chunk: %w", err))
			return
		}

		for _, choice := range chunk.Choices {
			delta := choice.Delta

			// Yield text content
			if delta.Content != "" {
				r := &models.Response{
					Messages: []*models.Message{{
						Role:    firstNonEmpty(delta.Role, "assistant"),
						Content: models.Content{Text: delta.Content},
					}},
				}
				if !yield(r, nil) {
					return
				}
			}

			// Yield reasoning/thought content
			if delta.ReasoningContent != "" {
				r := &models.Response{
					Messages: []*models.Message{{
						Role:    firstNonEmpty(delta.Role, "assistant"),
						Content: models.Content{Text: delta.ReasoningContent, Thought: true},
					}},
				}
				if !yield(r, nil) {
					return
				}
			}

			// Accumulate tool calls
			for _, tc := range delta.ToolCalls {
				idx := tc.Index
				if tc.ID != "" {
					pending[idx] = &pendingToolCall{
						id:   tc.ID,
						name: tc.Function.Name,
					}
				}
				if p, ok := pending[idx]; ok {
					p.rawArgs.WriteString(tc.Function.Arguments)
				}
			}

			// On finish, yield accumulated tool calls
			if choice.FinishReason != nil && *choice.FinishReason == "tool_calls" {
				for _, tc := range pending {
					args := parseToolArgs(tc.rawArgs.String())
					r := &models.Response{
						Messages: []*models.Message{{
							Role: "assistant",
							Content: models.Content{
								ToolCall: &models.ToolCall{
									ID:   tc.id,
									Name: tc.name,
									Args: args,
								},
							},
						}},
					}
					if !yield(r, nil) {
						return
					}
				}
				pending = make(map[int]*pendingToolCall)
			}
		}
	}

	if err := scanner.Err(); err != nil {
		yield(nil, fmt.Errorf("read stream: %w", err))
	}
}

func (m *DeepSeek) nonStreamResponse(body io.Reader, yield func(*models.Response, error) bool) {
	respBody, err := io.ReadAll(body)
	if err != nil {
		yield(nil, fmt.Errorf("read response: %w", err))
		return
	}

	var apiResp ChatCompletionResponse
	if err := json.Unmarshal(respBody, &apiResp); err != nil {
		yield(nil, fmt.Errorf("parse response: %w", err))
		return
	}

	for _, choice := range apiResp.Choices {
		r := convertMessage(choice.Message)
		if r != nil {
			if !yield(r, nil) {
				return
			}
		}
	}
}

func convertRequest(defaultModel string, tools []models.Tool, req *models.Request) (*ChatCompletionRequest, error) {
	model := defaultModel
	if req.Model != "" {
		model = req.Model
	}

	ts, err := convertTools(tools)
	if err != nil {
		return nil, err
	}

	dsReq := &ChatCompletionRequest{
		Model:    model,
		Messages: convertMessages(req.Contents),
		Tools:    ts,
	}

	if req.Config != nil {
		var cfg Config
		configBytes, err := json.Marshal(req.Config)
		if err != nil {
			return nil, fmt.Errorf("marshal config: %w", err)
		}
		if err := json.Unmarshal(configBytes, &cfg); err != nil {
			return nil, fmt.Errorf("unmarshal config: %w", err)
		}
		dsReq.Stream = cfg.Stream
		dsReq.StreamOptions = cfg.StreamOptions
		dsReq.Temperature = cfg.Temperature
		dsReq.TopP = cfg.TopP
		dsReq.MaxTokens = cfg.MaxTokens
		if cfg.FrequencyPenalty != nil {
			dsReq.FrequencyPenalty = *cfg.FrequencyPenalty
		}
		if cfg.PresencePenalty != nil {
			dsReq.PresencePenalty = *cfg.PresencePenalty
		}
		dsReq.Thinking = cfg.Thinking
		if cfg.ToolChoice != nil {
			dsReq.ToolChoice = cfg.ToolChoice
		}
	}

	return dsReq, nil
}

func convertMessages(msgs []*models.Message) []Message {
	result := make([]Message, 0, len(msgs))
	for _, msg := range msgs {
		m := Message{Role: msg.Role}

		if msg.Content.Thought {
			m.ReasoningContent = msg.Content.Text
		} else if msg.Content.Text != "" {
			m.Content = msg.Content.Text
		}

		if tc := msg.Content.ToolCall; tc != nil {
			argsJSON, _ := json.Marshal(tc.Args)
			m.ToolCalls = append(m.ToolCalls, ToolCall{
				ID:   tc.ID,
				Type: "function",
				Function: FunctionCall{
					Name:      tc.Name,
					Arguments: string(argsJSON),
				},
			})
		}

		// Handle tool result messages
		if msg.Role == "tool" && msg.Content.ToolCall.ID != "" {
			m.ToolCallID = msg.Content.ToolCall.ID
		}

		result = append(result, m)
	}
	return result
}

func convertTools(tools []models.Tool) ([]Tool, error) {
	result := make([]Tool, 0, len(tools))
	for _, t := range tools {
		schema, err := json.Marshal(t.ParameterSchema())
		if err != nil {
			return nil, fmt.Errorf("marshal tool: %q, %w", t.Name(), err)
		}
		result = append(result, Tool{
			Type: "function",
			Function: FunctionDefinition{
				Name:        t.Name(),
				Description: t.Description(),
				Parameters:  schema,
			},
		})
	}
	return result, nil
}

func convertMessage(msg Message) *models.Response {
	var msgs []*models.Message

	if msg.Content != "" {
		msgs = append(msgs, &models.Message{
			Role:    msg.Role,
			Content: models.Content{Text: msg.Content},
		})
	}

	if msg.ReasoningContent != "" {
		msgs = append(msgs, &models.Message{
			Role:    msg.Role,
			Content: models.Content{Text: msg.ReasoningContent, Thought: true},
		})
	}

	for _, tc := range msg.ToolCalls {
		args := parseToolArgs(tc.Function.Arguments)
		msgs = append(msgs, &models.Message{
			Role: msg.Role,
			Content: models.Content{
				ToolCall: &models.ToolCall{
					ID:   tc.ID,
					Name: tc.Function.Name,
					Args: args,
				},
			},
		})
	}

	if len(msgs) == 0 {
		return nil
	}
	return &models.Response{Messages: msgs}
}

func parseToolArgs(raw string) map[string]any {
	var args map[string]any
	if err := json.Unmarshal([]byte(raw), &args); err != nil {
		return map[string]any{"raw": raw}
	}
	return args
}

func firstNonEmpty(s, fallback string) string {
	if s != "" {
		return s
	}
	return fallback
}

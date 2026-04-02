package deepseek

import "encoding/json"

// ChatCompletionRequest is the request body for POST /chat/completions.
type ChatCompletionRequest struct {
	Model            string          `json:"model"`
	Messages         []Message       `json:"messages"`
	Thinking         *Thinking       `json:"thinking,omitempty"`
	FrequencyPenalty float64         `json:"frequency_penalty,omitempty"`
	MaxTokens        int             `json:"max_tokens,omitempty"`
	PresencePenalty  float64         `json:"presence_penalty,omitempty"`
	ResponseFormat   *ResponseFormat `json:"response_format,omitempty"`
	Stop             []string        `json:"stop,omitempty"`
	Stream           bool            `json:"stream,omitempty"`
	StreamOptions    *StreamOptions  `json:"stream_options,omitempty"`
	Temperature      *float64        `json:"temperature,omitempty"`
	TopP             *float64        `json:"top_p,omitempty"`
	Tools            []Tool          `json:"tools,omitempty"`
	ToolChoice       json.RawMessage `json:"tool_choice,omitempty"`
	Logprobs         bool            `json:"logprobs,omitempty"`
	TopLogprobs      int             `json:"top_logprobs,omitempty"`
}

type Message struct {
	Role             string     `json:"role"`
	Content          string     `json:"content,omitempty"`
	ReasoningContent string     `json:"reasoning_content,omitempty"`
	Name             string     `json:"name,omitempty"`
	ToolCalls        []ToolCall `json:"tool_calls,omitempty"`
	ToolCallID       string     `json:"tool_call_id,omitempty"`
}

type Thinking struct {
	Type string `json:"type"` // "enabled" or "disabled"
}

type ResponseFormat struct {
	Type string `json:"type"` // "text" or "json_object"
}

type StreamOptions struct {
	IncludeUsage bool `json:"include_usage"`
}

type Tool struct {
	Type     string             `json:"type"` // "function"
	Function FunctionDefinition `json:"function"`
}

type FunctionDefinition struct {
	Name        string          `json:"name"`
	Description string          `json:"description,omitempty"`
	Parameters  json.RawMessage `json:"parameters,omitempty"`
	Strict      bool            `json:"strict,omitempty"`
}

// ChatCompletionResponse is the response body for POST /chat/completions.
type ChatCompletionResponse struct {
	ID                string   `json:"id"`
	Object            string   `json:"object"`
	Created           int64    `json:"created"`
	Model             string   `json:"model"`
	SystemFingerprint string   `json:"system_fingerprint"`
	Choices           []Choice `json:"choices"`
	Usage             Usage    `json:"usage"`
}

type Choice struct {
	Index        int             `json:"index"`
	Message      Message         `json:"message"`
	FinishReason string          `json:"finish_reason"`
	LogProbs     *ChoiceLogProbs `json:"logprobs"`
}

type ToolCall struct {
	Index    int          `json:"index,omitempty"`
	ID       string       `json:"id"`
	Type     string       `json:"type"` // "function"
	Function FunctionCall `json:"function"`
}

type FunctionCall struct {
	Name      string `json:"name"`
	Arguments string `json:"arguments"`
}

type ChoiceLogProbs struct {
	Content          []LogProbToken `json:"content"`
	ReasoningContent []LogProbToken `json:"reasoning_content,omitempty"`
}

type LogProbToken struct {
	Token       string       `json:"token"`
	LogProb     float64      `json:"logprob"`
	Bytes       []byte       `json:"bytes,omitempty"`
	TopLogProbs []TopLogProb `json:"top_logprobs"`
}

type TopLogProb struct {
	Token   string  `json:"token"`
	LogProb float64 `json:"logprob"`
	Bytes   []byte  `json:"bytes,omitempty"`
}

type Usage struct {
	PromptTokens            int                      `json:"prompt_tokens"`
	CompletionTokens        int                      `json:"completion_tokens"`
	TotalTokens             int                      `json:"total_tokens"`
	PromptCacheHitTokens    int                      `json:"prompt_cache_hit_tokens"`
	PromptCacheMissTokens   int                      `json:"prompt_cache_miss_tokens"`
	CompletionTokensDetails *CompletionTokensDetails `json:"completion_tokens_details,omitempty"`
}

type CompletionTokensDetails struct {
	ReasoningTokens int `json:"reasoning_tokens"`
}

// ChatCompletionChunk is a single SSE chunk from a streaming response.
type ChatCompletionChunk struct {
	ID                string        `json:"id"`
	Object            string        `json:"object"`
	Created           int64         `json:"created"`
	Model             string        `json:"model"`
	SystemFingerprint string        `json:"system_fingerprint"`
	Choices           []ChunkChoice `json:"choices"`
	Usage             *Usage        `json:"usage,omitempty"`
}

type ChunkChoice struct {
	Index        int          `json:"index"`
	Delta        DeltaMessage `json:"delta"`
	FinishReason *string      `json:"finish_reason"`
}

type DeltaMessage struct {
	Role             string     `json:"role,omitempty"`
	Content          string     `json:"content,omitempty"`
	ReasoningContent string     `json:"reasoning_content,omitempty"`
	ToolCalls        []ToolCall `json:"tool_calls,omitempty"`
}

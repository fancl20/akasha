use async_trait::async_trait;
use bytes::Bytes;
use futures::{SinkExt, Stream, StreamExt};
use reqwest::Client;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::pin::Pin;

use crate::core::providers::{Model, Provider, ProviderError, StreamResponse, StreamResponseStream};
use crate::core::types::{ContentBlock, Message, TextContent, TokenUsage, ToolCall, ToolDefinition, ToolResultContent};

const DEFAULT_BASE_URL: &str = "https://api.deepseek.com";

pub struct DeepSeekProvider {
    client: Client,
    api_key: String,
}

impl DeepSeekProvider {
    pub fn new(api_key: impl Into<String>) -> Self {
        Self { client: Client::new(), api_key: api_key.into() }
    }
}

#[derive(Serialize)]
#[serde(tag = "role")]
enum ApiMessage {
    #[serde(rename = "system")]
    System { content: String },
    #[serde(rename = "user")]
    User { content: String },
    #[serde(rename = "assistant")]
    Assistant {
        #[serde(skip_serializing_if = "Option::is_none")]
        content: Option<String>,
        #[serde(skip_serializing_if = "Option::is_none")]
        reasoning_content: Option<String>,
        #[serde(skip_serializing_if = "Option::is_none")]
        tool_calls: Option<Vec<ApiToolCall>>,
    },
    #[serde(rename = "tool")]
    Tool { content: String, tool_call_id: Option<String> },
}

#[derive(Serialize)]
struct ApiToolCall {
    id: String,
    r#type: String,
    function: ApiFunctionCall,
}

#[derive(Serialize)]
struct ApiFunctionCall {
    name: String,
    arguments: String,
}

#[derive(Serialize)]
struct ApiTool {
    r#type: String,
    function: ApiFunction,
}

#[derive(Serialize)]
struct ApiFunction {
    name: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    description: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    parameters: Option<serde_json::Value>,
}

#[derive(Deserialize)]
struct ChatChunk {
    #[serde(default)]
    choices: Vec<ChunkChoice>,
    #[serde(default)]
    usage: Option<ChunkUsage>,
}

#[derive(Deserialize)]
struct ChunkChoice {
    #[serde(default)]
    delta: ChunkDelta,
    finish_reason: Option<String>,
}

#[derive(Deserialize, Default)]
struct ChunkDelta {
    #[serde(default)]
    content: Option<String>,
    #[serde(default)]
    reasoning_content: Option<String>,
    #[serde(default)]
    tool_calls: Option<Vec<ChunkToolCall>>,
}

#[derive(Deserialize)]
struct ChunkToolCall {
    index: usize,
    #[serde(default)]
    id: Option<String>,
    #[allow(dead_code)]
    #[serde(default)]
    r#type: Option<String>,
    #[serde(default)]
    function: ChunkFunction,
}

#[derive(Deserialize, Default)]
struct ChunkFunction {
    #[serde(default)]
    name: Option<String>,
    #[serde(default)]
    arguments: Option<String>,
}

#[derive(Deserialize)]
struct ChunkUsage {
    prompt_tokens: u64,
    completion_tokens: u64,
    #[serde(default)]
    prompt_cache_hit_tokens: Option<u64>,
    #[allow(dead_code)]
    #[serde(default)]
    prompt_cache_miss_tokens: Option<u64>,
}

fn build_messages<'a>(messages: impl Iterator<Item = &'a Message>) -> Vec<ApiMessage> {
    let mut out = Vec::new();

    for msg in messages {
        match msg.role.as_str() {
            "system" => {
                let text = extract_text(&msg.content);
                out.push(ApiMessage::System { content: text });
            }
            "user" => {
                let text = extract_text(&msg.content);
                out.push(ApiMessage::User { content: text });
            }
            "assistant" => {
                let text = extract_text(&msg.content);
                let reasoning = extract_reasoning(&msg.content);
                let api_tcs: Vec<ApiToolCall> = msg
                    .content
                    .iter()
                    .filter_map(|b| match b {
                        ContentBlock::ToolCall(tc) => Some(ApiToolCall {
                            id: tc.id.clone(),
                            r#type: "function".to_string(),
                            function: ApiFunctionCall { name: tc.name.clone(), arguments: tc.arguments.to_string() },
                        }),
                        _ => None,
                    })
                    .collect();
                out.push(ApiMessage::Assistant {
                    content: if text.is_empty() { None } else { Some(text) },
                    reasoning_content: if reasoning.is_empty() { None } else { Some(reasoning) },
                    tool_calls: if api_tcs.is_empty() { None } else { Some(api_tcs) },
                });
            }
            "tool" => {
                for block in &msg.content {
                    if let ContentBlock::ToolResult(result) = block {
                        let text: String = result
                            .content
                            .iter()
                            .filter_map(|c| match c {
                                ToolResultContent::Text(t) => Some(t.content.as_str()),
                                _ => None,
                            })
                            .collect();
                        out.push(ApiMessage::Tool { content: text, tool_call_id: result.tool_call_id.clone() });
                    }
                }
            }
            _ => {}
        }
    }

    out
}

fn build_tools(tools: &Vec<ToolDefinition>) -> Option<Vec<ApiTool>> {
    if tools.is_empty() {
        return None;
    }
    Some(
        tools
            .iter()
            .map(|t| ApiTool {
                r#type: "function".to_string(),
                function: ApiFunction {
                    name: t.name.clone(),
                    description: Some(t.description.clone()),
                    parameters: Some(t.parameters.clone()),
                },
            })
            .collect(),
    )
}

fn extract_text(blocks: &[ContentBlock]) -> String {
    blocks
        .iter()
        .filter_map(|b| match b {
            ContentBlock::Text(t) => Some(t.content.as_str()),
            _ => None,
        })
        .collect::<Vec<_>>()
        .join("")
}

fn extract_reasoning(blocks: &[ContentBlock]) -> String {
    blocks
        .iter()
        .filter_map(|b| match b {
            ContentBlock::Reasoning(t) => Some(t.content.as_str()),
            _ => None,
        })
        .collect::<Vec<_>>()
        .join("")
}

struct PartialToolCall {
    id: String,
    name: String,
    arguments: String,
}

async fn process_sse(
    mut byte_stream: Pin<Box<dyn Stream<Item = Result<Bytes, reqwest::Error>> + Send>>,
    mut tx: futures::channel::mpsc::Sender<StreamResponse>,
) {
    let mut buffer = String::new();
    let mut partial_tools: HashMap<usize, PartialToolCall> = HashMap::new();
    let mut last_usage: Option<TokenUsage> = None;

    while let Some(chunk) = byte_stream.next().await {
        let bytes = match chunk {
            Ok(b) => b,
            Err(_) => {
                return;
            }
        };

        buffer.push_str(&String::from_utf8_lossy(&bytes));

        while let Some(pos) = buffer.find('\n') {
            let line = buffer[..pos].trim_end().to_owned();
            buffer = buffer[pos + 1..].to_owned();

            let data = match line.strip_prefix("data: ") {
                Some(d) => d.trim(),
                None => continue,
            };

            if data == "[DONE]" {
                emit_done(&mut tx, &last_usage).await;
                return;
            }

            let chunk: ChatChunk = match serde_json::from_str(data) {
                Ok(c) => c,
                Err(_) => continue,
            };

            if let Some(usage) = chunk.usage {
                last_usage = Some(TokenUsage {
                    input_tokens: usage.prompt_tokens,
                    output_tokens: usage.completion_tokens,
                    cache_read_tokens: usage.prompt_cache_hit_tokens,
                    cache_write_tokens: None,
                });
            }

            for choice in chunk.choices {
                if let Some(content) = choice.delta.content {
                    if !content.is_empty() {
                        let _ = tx
                            .send(StreamResponse {
                                message: Message {
                                    role: "assistant".to_string(),
                                    content: vec![ContentBlock::Text(TextContent { content })],
                                },
                                usage: TokenUsage {
                                    input_tokens: 0,
                                    output_tokens: 0,
                                    cache_read_tokens: None,
                                    cache_write_tokens: None,
                                },
                                stop_reason: None,
                            })
                            .await;
                    }
                }

                if let Some(reasoning) = choice.delta.reasoning_content {
                    if !reasoning.is_empty() {
                        let _ = tx
                            .send(StreamResponse {
                                message: Message {
                                    role: "assistant".to_string(),
                                    content: vec![ContentBlock::Reasoning(TextContent { content: reasoning })],
                                },
                                usage: TokenUsage {
                                    input_tokens: 0,
                                    output_tokens: 0,
                                    cache_read_tokens: None,
                                    cache_write_tokens: None,
                                },
                                stop_reason: None,
                            })
                            .await;
                    }
                }

                if let Some(tc_deltas) = choice.delta.tool_calls {
                    for tc in tc_deltas {
                        let entry = partial_tools.entry(tc.index).or_insert_with(|| PartialToolCall {
                            id: String::new(),
                            name: String::new(),
                            arguments: String::new(),
                        });
                        if let Some(id) = tc.id {
                            entry.id = id;
                        }
                        if let Some(name) = tc.function.name {
                            entry.name = name;
                        }
                        if let Some(args) = tc.function.arguments {
                            entry.arguments.push_str(&args);
                        }
                    }
                }

                if choice.finish_reason.as_deref() == Some("tool_calls") {
                    let indices: Vec<usize> = partial_tools.keys().copied().collect();
                    for idx in indices {
                        if let Some(ptc) = partial_tools.remove(&idx) {
                            let args = serde_json::from_str(&ptc.arguments).unwrap_or(serde_json::Value::Null);
                            let tc = ContentBlock::ToolCall(ToolCall { id: ptc.id, name: ptc.name, arguments: args });
                            let _ = tx
                                .send(StreamResponse {
                                    message: Message { role: "assistant".to_string(), content: vec![tc] },
                                    usage: TokenUsage {
                                        input_tokens: 0,
                                        output_tokens: 0,
                                        cache_read_tokens: None,
                                        cache_write_tokens: None,
                                    },
                                    stop_reason: Some("tool_calls".to_string()),
                                })
                                .await;
                        }
                    }
                }
            }
        }
    }

    emit_done(&mut tx, &last_usage).await;
}

async fn emit_done(tx: &mut futures::channel::mpsc::Sender<StreamResponse>, usage: &Option<TokenUsage>) {
    let usage = usage.clone().unwrap_or(TokenUsage {
        input_tokens: 0,
        output_tokens: 0,
        cache_read_tokens: None,
        cache_write_tokens: None,
    });

    let _ = tx
        .send(StreamResponse {
            message: Message { role: "assistant".to_string(), content: vec![] },
            usage,
            stop_reason: Some("stop".to_string()),
        })
        .await;
}

#[async_trait]
impl Provider for DeepSeekProvider {
    async fn stream<'a>(
        &self,
        model: &Model,
        messages: Box<dyn Iterator<Item = &'a Message> + Send + 'a>,
        tools: &Vec<ToolDefinition>,
    ) -> Result<StreamResponseStream, ProviderError> {
        let api_messages = build_messages(messages);
        let api_tools = build_tools(tools);

        let mut body = serde_json::json!({
            "model": model.id,
            "messages": api_messages,
            "stream": true,
            "stream_options": { "include_usage": true },
        });
        if let Some(tools) = api_tools {
            body["tools"] = serde_json::to_value(tools).unwrap();
        }

        let base_url = if model.base_url.is_empty() { DEFAULT_BASE_URL } else { &model.base_url };
        let url = format!("{}/chat/completions", base_url);

        let mut req = self.client.post(&url).header("Authorization", format!("Bearer {}", self.api_key)).json(&body);

        for (key, value) in &model.headers {
            req = req.header(key.as_str(), value.as_str());
        }

        let response = req.send().await.map_err(|e| ProviderError::RequestFailed(e.to_string()))?;

        match response.status() {
            s if s == reqwest::StatusCode::TOO_MANY_REQUESTS => {
                return Err(ProviderError::RateLimited);
            }
            s if s == reqwest::StatusCode::NOT_FOUND => {
                return Err(ProviderError::ModelNotFound(model.id.clone()));
            }
            s if !s.is_success() => {
                let body = response.text().await.unwrap_or_default();
                return Err(ProviderError::RequestFailed(format!("{}: {}", s, body)));
            }
            _ => {}
        }

        let byte_stream: Pin<Box<dyn Stream<Item = Result<Bytes, reqwest::Error>> + Send>> =
            Box::pin(response.bytes_stream());

        let (tx, rx) = futures::channel::mpsc::channel(64);

        tokio::spawn(async move {
            process_sse(byte_stream, tx).await;
        });

        Ok(Box::pin(rx))
    }

    fn name(&self) -> &str {
        "deepseek"
    }
}

#[cfg(test)]
mod tests {

    use super::*;
    use crate::core::extensions::NoopExtension;
    use crate::core::providers::{Model, Provider};
    use crate::core::session::Session;
    use crate::core::tools::{ToolHandler, ToolRegistry};
    use crate::core::types::{ContentBlock, Message, TextContent, ToolDefinition, ToolResult, ToolResultContent};
    use futures::StreamExt;

    fn api_key() -> Option<String> {
        std::env::var("DEEPSEEK_API_KEY").ok()
    }

    fn test_model() -> Model {
        Model {
            id: "deepseek-v4-flash".to_string(),
            provider: "deepseek".to_string(),
            context_window: 64000,
            base_url: String::new(),
            headers: HashMap::new(),
        }
    }

    fn simple_messages(prompt: &str) -> Vec<Message> {
        vec![Message {
            role: "user".to_string(),
            content: vec![ContentBlock::Text(TextContent { content: prompt.to_string() })],
        }]
    }

    async fn collect_stream(stream: StreamResponseStream) -> Vec<StreamResponse> {
        stream.collect::<Vec<_>>().await
    }

    // --- Unit tests for build_messages ---

    #[test]
    fn test_build_messages_user_and_system() {
        let messages = vec![
            Message {
                role: "system".to_string(),
                content: vec![ContentBlock::Text(TextContent { content: "You are helpful.".to_string() })],
            },
            Message {
                role: "user".to_string(),
                content: vec![ContentBlock::Text(TextContent { content: "Hello".to_string() })],
            },
        ];
        let api_messages = build_messages(messages.iter());
        assert_eq!(api_messages.len(), 2);
    }

    #[test]
    fn test_build_messages_with_tool_result() {
        let messages = vec![
            Message {
                role: "user".to_string(),
                content: vec![ContentBlock::Text(TextContent { content: "What's the weather?".to_string() })],
            },
            Message {
                role: "assistant".to_string(),
                content: vec![ContentBlock::ToolCall(ToolCall {
                    id: "call_1".to_string(),
                    name: "get_weather".to_string(),
                    arguments: serde_json::json!({"city": "Tokyo"}),
                })],
            },
            Message {
                role: "tool".to_string(),
                content: vec![ContentBlock::ToolResult(ToolResult {
                    tool_call_id: Some("call_1".to_string()),
                    content: vec![ToolResultContent::Text(TextContent { content: "Sunny, 22C".to_string() })],
                    is_error: false,
                })],
            },
        ];
        let api_messages = build_messages(messages.iter());
        assert_eq!(api_messages.len(), 3);
    }

    #[test]
    fn test_build_tools_empty() {
        let tools: Vec<ToolDefinition> = vec![];
        assert!(build_tools(&tools).is_none());
    }

    #[test]
    fn test_build_tools_some() {
        let tools = vec![ToolDefinition {
            name: "get_weather".to_string(),
            description: "Get weather".to_string(),
            parameters: serde_json::json!({
                "type": "object",
                "properties": {
                    "city": { "type": "string" }
                }
            }),
        }];
        let api_tools = build_tools(&tools).unwrap();
        assert_eq!(api_tools.len(), 1);
        assert_eq!(api_tools[0].function.name, "get_weather");
    }

    #[test]
    fn test_extract_text() {
        let blocks = vec![
            ContentBlock::Text(TextContent { content: "Hello ".to_string() }),
            ContentBlock::Text(TextContent { content: "world".to_string() }),
            ContentBlock::ToolCall(ToolCall {
                id: "1".to_string(),
                name: "foo".to_string(),
                arguments: serde_json::Value::Null,
            }),
        ];
        assert_eq!(extract_text(&blocks), "Hello world");
    }

    #[test]
    fn test_provider_name() {
        let p = DeepSeekProvider::new("test-key");
        assert_eq!(p.name(), "deepseek");
    }

    // --- Integration tests (require DEEPSEEK_API_KEY) ---

    #[tokio::test]
    async fn test_basic_chat() {
        let key = match api_key() {
            Some(k) => k,
            None => return,
        };
        let provider = DeepSeekProvider::new(key);
        let model = test_model();
        let messages = simple_messages("Reply with exactly: PONG");
        let tools: Vec<ToolDefinition> = vec![];

        let stream = provider.stream(&model, Box::new(messages.iter()), &tools).await.unwrap();
        let responses = collect_stream(stream).await;

        assert!(!responses.is_empty(), "should receive at least one response");

        let combined_text: String = responses
            .iter()
            .flat_map(|r| {
                r.message.content.iter().filter_map(|b| match b {
                    ContentBlock::Text(t) => Some(t.content.as_str()),
                    _ => None,
                })
            })
            .collect();

        assert!(combined_text.to_lowercase().contains("pong"), "response should contain 'pong', got: {combined_text}");
    }

    #[tokio::test]
    async fn test_streaming_yields_usage() {
        let key = match api_key() {
            Some(k) => k,
            None => return,
        };
        let provider = DeepSeekProvider::new(key);
        let model = test_model();
        let messages = simple_messages("Say hi in one word.");
        let tools: Vec<ToolDefinition> = vec![];

        let stream = provider.stream(&model, Box::new(messages.iter()), &tools).await.unwrap();
        let responses = collect_stream(stream).await;

        let has_usage = responses.iter().any(|r| r.usage.input_tokens > 0);
        assert!(has_usage, "at least one chunk should report token usage");
    }

    #[tokio::test]
    async fn test_streaming_finishes_with_stop_reason() {
        let key = match api_key() {
            Some(k) => k,
            None => return,
        };
        let provider = DeepSeekProvider::new(key);
        let model = test_model();
        let messages = simple_messages("Say hello.");
        let tools: Vec<ToolDefinition> = vec![];

        let stream = provider.stream(&model, Box::new(messages.iter()), &tools).await.unwrap();
        let responses = collect_stream(stream).await;

        let last = responses.last().expect("should have responses");
        assert!(last.stop_reason.is_some(), "final response should have a stop_reason");
    }

    #[tokio::test]
    async fn test_tool_use() {
        let key = match api_key() {
            Some(k) => k,
            None => return,
        };
        let provider = DeepSeekProvider::new(key);
        let model = test_model();
        let messages = vec![Message {
            role: "user".to_string(),
            content: vec![ContentBlock::Text(TextContent {
                content: "What is the weather in Tokyo? Use the get_weather tool.".to_string(),
            })],
        }];
        let tools = vec![ToolDefinition {
            name: "get_weather".to_string(),
            description: "Get the current weather for a city".to_string(),
            parameters: serde_json::json!({
                "type": "object",
                "properties": {
                    "city": { "type": "string", "description": "City name" }
                },
                "required": ["city"]
            }),
        }];

        let stream: Pin<Box<dyn Stream<Item = StreamResponse> + Send>> =
            provider.stream(&model, Box::new(messages.iter()), &tools).await.unwrap();
        let responses = collect_stream(stream).await;

        let has_tool_call =
            responses.iter().any(|r| r.message.content.iter().any(|b| matches!(b, ContentBlock::ToolCall { .. })));
        assert!(has_tool_call, "model should return a tool call");
    }

    #[tokio::test]
    async fn test_invalid_api_key() {
        let provider = DeepSeekProvider::new("sk-invalid-key-12345");
        let model = test_model();
        let messages = simple_messages("Hi");
        let tools: Vec<ToolDefinition> = vec![];

        let result = provider.stream(&model, Box::new(messages.iter()), &tools).await;
        assert!(result.is_err(), "invalid API key should return an error");
    }

    #[tokio::test]
    async fn test_model_not_found() {
        let key = match api_key() {
            Some(k) => k,
            None => return,
        };
        let provider = DeepSeekProvider::new(key);
        let model = Model {
            id: "deepseek-nonexistent-model-xyz".to_string(),
            provider: "deepseek".to_string(),
            context_window: 4096,
            base_url: String::new(),
            headers: HashMap::new(),
        };
        let messages = simple_messages("Hi");
        let tools: Vec<ToolDefinition> = vec![];

        let result = provider.stream(&model, Box::new(messages.iter()), &tools).await;
        assert!(result.is_err(), "unknown model should return an error");
    }

    // --- E2E test: agent run with tool-use chain ---

    struct GetCurrentLocationTool;

    #[async_trait]
    impl ToolHandler for GetCurrentLocationTool {
        fn definition(&self) -> ToolDefinition {
            ToolDefinition {
                name: "get_current_location".to_string(),
                description: "Get the user's current city and country.".to_string(),
                parameters: serde_json::json!({
                    "type": "object",
                    "properties": {},
                    "required": []
                }),
            }
        }

        async fn execute(
            &self,
            _cancel: futures::channel::oneshot::Receiver<bool>,
            _params: serde_json::Value,
        ) -> Result<crate::core::types::ToolResult, crate::core::tools::ToolError> {
            Ok(crate::core::types::ToolResult {
                tool_call_id: None,
                content: vec![crate::core::types::ToolResultContent::Text(crate::core::types::TextContent {
                    content: serde_json::json!({
                        "city": "San Francisco",
                        "country": "US"
                    })
                    .to_string(),
                })],
                is_error: false,
            })
        }
    }

    struct GetWeatherByLocationTool;

    #[async_trait]
    impl ToolHandler for GetWeatherByLocationTool {
        fn definition(&self) -> ToolDefinition {
            ToolDefinition {
                name: "get_weather_by_location".to_string(),
                description: "Get the current weather for a given location.".to_string(),
                parameters: serde_json::json!({
                    "type": "object",
                    "properties": {
                        "city": { "type": "string", "description": "City name" },
                        "country": { "type": "string", "description": "Country code" }
                    },
                    "required": ["city"]
                }),
            }
        }

        async fn execute(
            &self,
            _cancel: futures::channel::oneshot::Receiver<bool>,
            _params: serde_json::Value,
        ) -> Result<crate::core::types::ToolResult, crate::core::tools::ToolError> {
            Ok(crate::core::types::ToolResult {
                tool_call_id: None,
                content: vec![crate::core::types::ToolResultContent::Text(crate::core::types::TextContent {
                    content: serde_json::json!({
                        "temperature": "18°C",
                        "condition": "Foggy",
                        "humidity": "82%"
                    })
                    .to_string(),
                })],
                is_error: false,
            })
        }
    }

    #[tokio::test]
    async fn test_e2e_agent_tool_chain() {
        let key = match api_key() {
            Some(k) => k,
            None => return,
        };

        let provider: std::sync::Arc<dyn Provider> = std::sync::Arc::new(DeepSeekProvider::new(key));

        let mut tools = ToolRegistry::new();
        tools.register(GetCurrentLocationTool.into());
        tools.register(GetWeatherByLocationTool.into());

        let model = test_model();

        let agent_state = crate::core::agent::AgentState {
            model,
            tools,
            session: crate::core::session::InMemorySession::new().arc(),
        };
        let mut agent = crate::core::agent::Agent {
            state: agent_state,
            provider,
            extension: NoopExtension.into(),
        };

        let user_msg = Message {
            role: "user".into(),
            content: vec![ContentBlock::Text(TextContent {
                content: "What's the weather at my current location? Use the available tools to find out.".to_string(),
            })],
        };

        agent.prompt(user_msg).await.expect("agent run should succeed");

        let output: Vec<_> = agent.state.session.lock().unwrap().messages().cloned().collect();

        assert!(
            output.len() >= 3,
            "agent should produce multiple messages (tool calls + result), got {}",
            output.len()
        );

        let called_get_location = output.iter().any(|msg| {
            msg.content.iter().any(|b| {
                matches!(
                    b,
                    ContentBlock::ToolCall(tc) if tc.name == "get_current_location"
                )
            })
        });
        assert!(called_get_location, "agent should have called get_current_location");

        let called_get_weather = output.iter().any(|msg| {
            msg.content.iter().any(|b| {
                matches!(
                    b,
                    ContentBlock::ToolCall(tc) if tc.name == "get_weather_by_location"
                )
            })
        });
        assert!(called_get_weather, "agent should have called get_weather_by_location");

        let last = output.last().expect("should have output");
        let has_text = last.content.iter().any(|b| matches!(b, ContentBlock::Text { .. }));
        assert!(has_text && last.role == "assistant", "final message should be assistant text with weather info");
    }
}

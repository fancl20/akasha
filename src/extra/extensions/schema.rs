//! Schema verification extension.
//!
//! [`SchemaVerification`] is a cross-cutting [`Extension`] that checks every
//! tool call against the JSON Schemas its tools declare. It centralizes the
//! verification that previously lived inside individual tools (notably the skill
//! tool), so a tool only has to *declare* its schemas — not validate them.
//!
//! The hooks map directly onto the agent loop's tool lifecycle:
//!
//! - `on_turn_start` — snapshots each tool's input + output schemas from the
//!   registry and compiles validators for them.
//! - `on_tool_execution_start` — verifies the call's arguments against the
//!   recorded input schema; on failure it [`Deny`](ToolCallDecision::Deny)s the
//!   call (the agent loop turns a deny into an `is_error` tool result, so the
//!   tool never runs).
//! - `tool_execution_end` — verifies a successful result against the recorded
//!   output schema; on failure (including unparseable JSON) it replaces the
//!   result with an `is_error` one carrying the reason.
//!
//! A tool with no output schema is passed through unchecked; a tool whose
//! `parameters` is not a compilable schema is skipped for input verification
//! (neither is fatal).

use std::collections::HashMap;

use async_trait::async_trait;

use crate::core::agent::AgentState;
use crate::core::extensions::{Extension, ExtensionError, ToolCallDecision};
use crate::core::tools::ToolError;
use crate::core::types::{TextContent, ToolResult, ToolResultContent};

/// Compiles and holds the input + output validators for a single tool. Either
/// may be `None` when the tool declares no such schema or the schema fails to
/// compile (in which case verification is silently skipped for that side).
struct ToolValidators {
    input: Option<jsonschema::Validator>,
    output: Option<jsonschema::Validator>,
}

/// Verifies tool I/O against declared JSON Schemas.
///
/// Construct one and attach it to an agent (compose it with other extensions via
/// [`And`](super::combinator::And) when needed). It is stateless across turns:
/// `on_turn_start` rebuilds its snapshot from the live registry, so tools added
/// or changed between turns are picked up automatically.
pub struct SchemaVerification {
    /// `tool name → compiled validators`, snapshotted at turn start.
    schemas: HashMap<String, ToolValidators>,
    /// `tool_call_id → tool name`, tracked from `on_tool_execution_start` to
    /// `tool_execution_end` within a single turn (cleared each turn start).
    pending: HashMap<String, String>,
}

impl Default for SchemaVerification {
    fn default() -> Self {
        Self { schemas: HashMap::new(), pending: HashMap::new() }
    }
}

impl SchemaVerification {
    pub fn new() -> Self {
        Self::default()
    }

    /// Rebuilds the `name → validators` snapshot from the registry's current
    /// schemas. Compilation failures are skipped per-tool (logged on stderr),
    /// so one malformed schema cannot disable verification for the rest.
    fn snapshot(
        schemas: HashMap<String, (serde_json::Value, Option<serde_json::Value>)>,
    ) -> HashMap<String, ToolValidators> {
        let mut out = HashMap::new();
        for (name, (input, output)) in schemas {
            let input = match &input {
                v if compilable(v) => match jsonschema::validator_for(v) {
                    Ok(v) => Some(v),
                    Err(e) => {
                        eprintln!("[schema] skipping input schema for '{name}': {e}");
                        None
                    }
                },
                _ => None,
            };
            let output = output.as_ref().filter(|v| compilable(v)).and_then(|v| match jsonschema::validator_for(v) {
                Ok(v) => Some(v),
                Err(e) => {
                    eprintln!("[schema] skipping output schema for '{name}': {e}");
                    None
                }
            });
            out.insert(name, ToolValidators { input, output });
        }
        out
    }
}

/// Only object-typed schemas are worth compiling; this cheaply filters out the
/// common `null`/non-schema placeholders some tools emit as `parameters`.
fn compilable(value: &serde_json::Value) -> bool {
    value.is_object()
}

/// Validates `value` against `validator`, joining all errors into one string.
fn validate_value(validator: &jsonschema::Validator, value: &serde_json::Value) -> Result<(), String> {
    let errors: Vec<String> = validator.iter_errors(value).map(|e| e.to_string()).collect();
    if errors.is_empty() { Ok(()) } else { Err(errors.join("; ")) }
}

/// Concatenates the `Text` blocks of a tool result — the shape output-schema
/// verification parses as JSON.
fn result_text(result: &ToolResult) -> String {
    result
        .content
        .iter()
        .filter_map(|c| match c {
            ToolResultContent::Text(t) => Some(t.content.as_str()),
            _ => None,
        })
        .collect::<Vec<_>>()
        .join("")
}

#[async_trait]
impl Extension for SchemaVerification {
    fn name(&self) -> &str {
        "schema"
    }

    async fn on_turn_start(&mut self, state: AgentState) -> Result<AgentState, ExtensionError> {
        self.schemas = Self::snapshot(state.tools.schemas());
        self.pending.clear();
        Ok(state)
    }

    async fn on_tool_execution_start(
        &mut self,
        tool_call_id: &str,
        name: &str,
        args: &serde_json::Value,
    ) -> Result<ToolCallDecision, ExtensionError> {
        self.pending.insert(tool_call_id.to_string(), name.to_string());

        if let Some(ToolValidators { input: Some(validator), .. }) = self.schemas.get(name) {
            if let Err(reason) = validate_value(validator, args) {
                return Ok(ToolCallDecision::Deny(format!("tool '{name}': invalid input: {reason}")));
            }
        }
        Ok(ToolCallDecision::Allow)
    }

    async fn tool_execution_end(
        &mut self,
        tool_call_id: &str,
        result: Result<ToolResult, ToolError>,
    ) -> Result<Result<ToolResult, ToolError>, ExtensionError> {
        let name = match self.pending.remove(tool_call_id) {
            Some(name) => name,
            None => return Ok(result),
        };
        let Some(ToolValidators { output: Some(validator), .. }) = self.schemas.get(&name) else {
            return Ok(result);
        };
        let result = match result {
            Ok(r) => r,
            err @ Err(_) => return Ok(err),
        };

        let text = result_text(&result);
        let value: serde_json::Value = match serde_json::from_str(&text) {
            Ok(v) => v,
            Err(e) => {
                return Ok(Ok(ToolResult {
                    tool_call_id: result.tool_call_id.clone(),
                    content: vec![ToolResultContent::Text(TextContent {
                        content: format!(
                            "tool '{name}': output is not valid JSON and cannot be checked against its output schema ({e})"
                        ),
                    })],
                    is_error: true,
                }));
            }
        };
        if let Err(reason) = validate_value(validator, &value) {
            return Ok(Ok(ToolResult {
                tool_call_id: result.tool_call_id.clone(),
                content: vec![ToolResultContent::Text(TextContent {
                    content: format!("tool '{name}': invalid output: {reason}"),
                })],
                is_error: true,
            }));
        }
        Ok(Ok(result))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::core::agent::{Agent, AgentState};
    use crate::core::providers::{Model, Provider, ProviderError, StreamResponse, StreamResponseStream};
    use crate::core::session::{InMemorySession, Session};
    use crate::core::tools::{ToolHandler, ToolRegistry};
    use crate::core::types::{ContentBlock, Message, TextContent, TokenUsage, ToolCall, ToolDefinition};
    use futures::channel::oneshot;
    use futures::stream;
    use std::collections::HashMap;
    use std::sync::{Arc, Mutex};

    /// A handler with explicit input + output schemas and a scripted result.
    struct SchemedTool {
        def: ToolDefinition,
        output: Option<serde_json::Value>,
        result_text: String,
    }

    #[async_trait]
    impl ToolHandler for SchemedTool {
        fn definition(&self) -> ToolDefinition {
            self.def.clone()
        }
        fn schema(&self) -> (serde_json::Value, Option<serde_json::Value>) {
            (self.def.parameters.clone(), self.output.clone())
        }
        async fn execute(&self, _cancel: oneshot::Receiver<bool>, _params: serde_json::Value) -> Result<ToolResult, ToolError> {
            Ok(ToolResult {
                tool_call_id: None,
                content: vec![ToolResultContent::Text(TextContent { content: self.result_text.clone() })],
                is_error: false,
            })
        }
    }

    fn input_schema() -> serde_json::Value {
        serde_json::json!({ "type": "object", "required": ["path"], "properties": { "path": { "type": "string" } } })
    }

    fn output_schema() -> serde_json::Value {
        serde_json::json!({ "type": "object", "required": ["x"], "properties": { "x": { "type": "integer" } } })
    }

    fn registry_with(tool: SchemedTool) -> ToolRegistry {
        let mut tools = ToolRegistry::new();
        tools.register(tool.into());
        tools
    }

    fn state(tools: ToolRegistry) -> AgentState {
        AgentState {
            model: Model {
                id: "m".into(),
                provider: "p".into(),
                context_window: 0,
                base_url: String::new(),
                headers: HashMap::new(),
            },
            tools,
            session: InMemorySession::new().arc(),
        }
    }

    #[tokio::test]
    async fn records_schemas_at_turn_start() {
        let tools = registry_with(SchemedTool {
            def: ToolDefinition {
                name: "t".into(),
                description: "d".into(),
                parameters: input_schema(),
            },
            output: Some(output_schema()),
            result_text: r#"{"x":1}"#.into(),
        });
        let mut ext = SchemaVerification::new();
        ext.on_turn_start(state(tools)).await.unwrap();
        let v = ext.schemas.get("t").expect("schema recorded");
        assert!(v.input.is_some(), "input validator compiled");
        assert!(v.output.is_some(), "output validator compiled");
    }

    #[tokio::test]
    async fn denies_invalid_input() {
        let tools = registry_with(SchemedTool {
            def: ToolDefinition {
                name: "t".into(),
                description: "d".into(),
                parameters: input_schema(),
            },
            output: None,
            result_text: "ok".into(),
        });
        let mut ext = SchemaVerification::new();
        ext.on_turn_start(state(tools)).await.unwrap();

        let decision = ext.on_tool_execution_start("c1", "t", &serde_json::json!({})).await.unwrap();
        match decision {
            ToolCallDecision::Deny(r) => assert!(r.contains("invalid input"), "got: {r}"),
            ToolCallDecision::Allow => panic!("invalid input must be denied"),
        }
    }

    #[tokio::test]
    async fn allows_valid_input() {
        let tools = registry_with(SchemedTool {
            def: ToolDefinition {
                name: "t".into(),
                description: "d".into(),
                parameters: input_schema(),
            },
            output: None,
            result_text: "ok".into(),
        });
        let mut ext = SchemaVerification::new();
        ext.on_turn_start(state(tools)).await.unwrap();

        let decision = ext.on_tool_execution_start("c1", "t", &serde_json::json!({"path":"/tmp"})).await.unwrap();
        assert!(matches!(decision, ToolCallDecision::Allow));
    }

    fn ok_result(text: &str) -> Result<ToolResult, ToolError> {
        Ok(ToolResult {
            tool_call_id: None,
            content: vec![ToolResultContent::Text(TextContent { content: text.into() })],
            is_error: false,
        })
    }

    #[tokio::test]
    async fn flags_invalid_output() {
        let tools = registry_with(SchemedTool {
            def: ToolDefinition {
                name: "t".into(),
                description: "d".into(),
                parameters: serde_json::json!({"type":"object"}),
            },
            output: Some(output_schema()),
            result_text: r#"{"x":"not-an-int"}"#.into(),
        });
        let mut ext = SchemaVerification::new();
        ext.on_turn_start(state(tools)).await.unwrap();
        ext.on_tool_execution_start("c1", "t", &serde_json::json!({})).await.unwrap();

        let out = ext.tool_execution_end("c1", ok_result(r#"{"x":"not-an-int"}"#)).await.unwrap().unwrap();
        assert!(out.is_error, "invalid output must become an error result");
    }

    #[tokio::test]
    async fn flags_non_json_output() {
        let tools = registry_with(SchemedTool {
            def: ToolDefinition {
                name: "t".into(),
                description: "d".into(),
                parameters: serde_json::json!({"type":"object"}),
            },
            output: Some(output_schema()),
            result_text: "just text".into(),
        });
        let mut ext = SchemaVerification::new();
        ext.on_turn_start(state(tools)).await.unwrap();
        ext.on_tool_execution_start("c1", "t", &serde_json::json!({})).await.unwrap();

        let out = ext.tool_execution_end("c1", ok_result("just text")).await.unwrap().unwrap();
        assert!(out.is_error, "non-JSON output must become an error result");
    }

    #[tokio::test]
    async fn passes_valid_output() {
        let tools = registry_with(SchemedTool {
            def: ToolDefinition {
                name: "t".into(),
                description: "d".into(),
                parameters: serde_json::json!({"type":"object"}),
            },
            output: Some(output_schema()),
            result_text: r#"{"x":7}"#.into(),
        });
        let mut ext = SchemaVerification::new();
        ext.on_turn_start(state(tools)).await.unwrap();
        ext.on_tool_execution_start("c1", "t", &serde_json::json!({})).await.unwrap();

        let out = ext.tool_execution_end("c1", ok_result(r#"{"x":7}"#)).await.unwrap().unwrap();
        assert!(!out.is_error, "valid output passes through");
    }

    #[tokio::test]
    async fn passthrough_when_no_output_schema() {
        let tools = registry_with(SchemedTool {
            def: ToolDefinition {
                name: "t".into(),
                description: "d".into(),
                parameters: serde_json::json!({"type":"object"}),
            },
            output: None,
            result_text: "free text".into(),
        });
        let mut ext = SchemaVerification::new();
        ext.on_turn_start(state(tools)).await.unwrap();
        ext.on_tool_execution_start("c1", "t", &serde_json::json!({})).await.unwrap();

        let out = ext.tool_execution_end("c1", ok_result("free text")).await.unwrap().unwrap();
        assert!(!out.is_error, "no output schema → unchecked passthrough");
    }

    #[tokio::test]
    async fn resnapshots_each_turn() {
        // Start with one tool, then swap the registry for two tools and confirm turn_start picks it up.
        let one = registry_with(SchemedTool {
            def: ToolDefinition { name: "a".into(), description: "d".into(), parameters: serde_json::json!({"type":"object"}) },
            output: None,
            result_text: "ok".into(),
        });
        let mut ext = SchemaVerification::new();
        ext.on_turn_start(state(one)).await.unwrap();
        assert!(ext.schemas.contains_key("a"));
        assert!(!ext.schemas.contains_key("b"));

        let two = {
            let mut r = ToolRegistry::new();
            r.register(
                SchemedTool {
                    def: ToolDefinition { name: "a".into(), description: "d".into(), parameters: serde_json::json!({"type":"object"}) },
                    output: None,
                    result_text: "ok".into(),
                }
                .into(),
            );
            r.register(
                SchemedTool {
                    def: ToolDefinition { name: "b".into(), description: "d".into(), parameters: serde_json::json!({"type":"object"}) },
                    output: None,
                    result_text: "ok".into(),
                }
                .into(),
            );
            r
        };
        ext.on_turn_start(state(two)).await.unwrap();
        assert!(ext.schemas.contains_key("a"));
        assert!(ext.schemas.contains_key("b"), "turn_start resnapshots the registry");
    }

    // --- E2E: a real agent loop driving a schemed tool through the extension ---

    /// A provider that replays scripted assistant messages, one per stream() call.
    struct ScriptedProvider {
        turns: Arc<Mutex<std::collections::VecDeque<Message>>>,
    }

    impl ScriptedProvider {
        fn new(turns: Vec<Message>) -> Self {
            Self { turns: Arc::new(Mutex::new(turns.into())) }
        }
    }

    #[async_trait]
    impl Provider for ScriptedProvider {
        async fn stream<'a>(
            &self,
            _model: &Model,
            _messages: Box<dyn Iterator<Item = &'a Message> + Send + 'a>,
            _tools: &Vec<ToolDefinition>,
        ) -> Result<StreamResponseStream, ProviderError> {
            let msg = self.turns.lock().unwrap().pop_front().unwrap_or_else(|| Message {
                role: "assistant".to_string(),
                content: vec![ContentBlock::Text(TextContent { content: String::new() })],
            });
            let stop =
                if msg.content.iter().any(|b| matches!(b, ContentBlock::ToolCall(_))) { "tool_calls" } else { "stop" };
            Ok(Box::pin(stream::iter(vec![StreamResponse {
                message: msg,
                usage: TokenUsage { input_tokens: 0, output_tokens: 0, cache_read_tokens: None, cache_write_tokens: None },
                stop_reason: Some(stop.to_string()),
            }])))
        }
        fn name(&self) -> &str {
            "scripted"
        }
    }

    fn tool_call_msg(id: &str, name: &str, args: serde_json::Value) -> Message {
        Message {
            role: "assistant".to_string(),
            content: vec![ContentBlock::ToolCall(ToolCall { id: id.into(), name: name.into(), arguments: args })],
        }
    }

    fn user_text(text: &str) -> Message {
        Message { role: "user".to_string(), content: vec![ContentBlock::Text(TextContent { content: text.into() })] }
    }

    fn last_tool_result_text(session: &Arc<Mutex<dyn crate::core::session::Session>>) -> Option<(bool, String)> {
        for m in session.lock().unwrap().messages().collect::<Vec<_>>().into_iter().rev() {
            for block in &m.content {
                if let ContentBlock::ToolResult(r) = block {
                    let text = r
                        .content
                        .iter()
                        .filter_map(|c| match c {
                            ToolResultContent::Text(t) => Some(t.content.as_str()),
                            _ => None,
                        })
                        .collect::<Vec<_>>()
                        .join("");
                    return Some((r.is_error, text));
                }
            }
        }
        None
    }

    #[tokio::test]
    async fn e2e_invalid_input_is_denied_via_extension() {
        // Tool requires {path}; the model calls it without path → the extension denies and the
        // skill/handler never runs (its scripted result queue stays untouched).
        let provider = Arc::new(ScriptedProvider::new(vec![
            tool_call_msg("c1", "t", serde_json::json!({})),
            Message { role: "assistant".to_string(), content: vec![ContentBlock::Text(TextContent { content: "done".into() })] },
        ]));
        let tools = registry_with(SchemedTool {
            def: ToolDefinition { name: "t".into(), description: "d".into(), parameters: input_schema() },
            output: None,
            result_text: "should-not-run".into(),
        });

        let mut agent = Agent {
            state: state(tools),
            provider,
            extension: Box::new(SchemaVerification::new()),
        };
        agent.prompt(user_text("go")).await.unwrap();

        let (is_error, text) = last_tool_result_text(&agent.state.session).expect("a tool result was recorded");
        assert!(is_error, "denied call must surface as an error tool result");
        assert!(text.contains("invalid input"), "deny reason must reach the model: {text}");
    }

    #[tokio::test]
    async fn e2e_invalid_output_is_flagged_via_extension() {
        // Tool returns output that violates its output schema; the extension converts it to an error.
        let provider = Arc::new(ScriptedProvider::new(vec![
            tool_call_msg("c1", "t", serde_json::json!({})),
            Message { role: "assistant".to_string(), content: vec![ContentBlock::Text(TextContent { content: "done".into() })] },
        ]));
        let tools = registry_with(SchemedTool {
            def: ToolDefinition { name: "t".into(), description: "d".into(), parameters: serde_json::json!({"type":"object"}) },
            output: Some(output_schema()),
            result_text: r#"{"x":"bad"}"#.into(),
        });

        let mut agent = Agent {
            state: state(tools),
            provider,
            extension: Box::new(SchemaVerification::new()),
        };
        agent.prompt(user_text("go")).await.unwrap();

        let (is_error, text) = last_tool_result_text(&agent.state.session).expect("a tool result was recorded");
        assert!(is_error, "invalid output must be flagged");
        assert!(text.contains("invalid output"), "output reason must reach the model: {text}");
    }

    #[tokio::test]
    async fn e2e_valid_io_runs_cleanly() {
        let provider = Arc::new(ScriptedProvider::new(vec![
            tool_call_msg("c1", "t", serde_json::json!({"path":"/tmp"})),
            Message { role: "assistant".to_string(), content: vec![ContentBlock::Text(TextContent { content: "done".into() })] },
        ]));
        let tools = registry_with(SchemedTool {
            def: ToolDefinition { name: "t".into(), description: "d".into(), parameters: input_schema() },
            output: Some(output_schema()),
            result_text: r#"{"x":5}"#.into(),
        });

        let mut agent = Agent {
            state: state(tools),
            provider,
            extension: Box::new(SchemaVerification::new()),
        };
        agent.prompt(user_text("go")).await.unwrap();

        let (is_error, _text) = last_tool_result_text(&agent.state.session).expect("a tool result was recorded");
        assert!(!is_error, "valid I/O must pass through");
    }
}

//! Subagent engine: **spawn / yield / resume** over a [`SessionManager`].
//!
//! [`SubagentTool`] turns a [`Subagent`] definition into a [`ToolHandler`] the
//! parent agent calls. Each invocation is one of:
//!
//! - **spawn** — [`SessionManager::fork`] branches the *main* session into a
//!   private conversation, then a [`stop`](crate::extra::extensions::control::stop_message)
//!   control message is appended. The branch inherits the main conversation in
//!   raw storage (linked, resumable, auditable) but executes isolated:
//!   [`ControlExtension`] drops the inherited prefix from the subagent's provider
//!   view at [`on_message_start`](Extension::on_message_start).
//! - **yield** — the subagent returns its result through an injected
//!   [`YieldTool`]. The standard tool-call → tool-result flow carries the result
//!   (captured into a slot, exactly like the skill's submit tool);
//!   [`AbortOnYield`] stops the loop the instant a result is captured, so there
//!   is no extra model call after the yield.
//! - **resume** — the parent echoes the `session_id` returned with a prior
//!   result; the engine [`SessionManager::get`]s the fork, appends the new args,
//!   and runs again. The spawn-time `stop` still bounds the view, but the prior
//!   subagent turns stay visible so the thread continues.
//!
//! The result envelope is the yielded value with `session_id` stamped on, so the
//! parent can echo it to resume. Build a subagent with [`SubagentTool::new`] (or
//! [`register`] it in one step) and the parent enables it like any other tool.

use std::sync::{Arc, Mutex};

use async_trait::async_trait;
use futures::channel::oneshot;

use crate::core::agent::{Agent, AgentState};
use crate::core::extensions::{Extension, ExtensionError};
use crate::core::providers::{Model, Provider};
use crate::core::tools::{ToolError, ToolHandler, ToolRegistry};
use crate::core::types::{ContentBlock, Message, TextContent, ToolDefinition, ToolResult, ToolResultContent};
use crate::extra::agents::builder::SessionManager;
use crate::extra::extensions::combinator::And;
use crate::extra::extensions::control::{ControlExtension, stop_message};
use crate::extra::extensions::schema::SchemaVerification;

/// The fixed name of the yield tool injected into every subagent run.
pub const YIELD_TOOL: &str = "akasha_subagent_yield";

/// Orchestration field the engine reads from the parent's call to resume a fork.
const SESSION_ID: &str = "session_id";

/// A subagent definition the [`SubagentTool`] engine drives.
///
/// The trait is *declarative*: it says what the subagent is (its tool
/// definition, schemas, tool pool, and how to seed a run), not how to run it.
/// Session lifecycle (fork/resume), the yield mechanism, and schema validation
/// all live in [`SubagentTool`], so every subagent gets them for free.
pub trait Subagent: Send + Sync {
    fn definition(&self) -> ToolDefinition;

    /// `(input, output)` schemas. The input surfaces to the parent via
    /// [`definition`](Self::definition); the output becomes the [`YieldTool`]'s
    /// parameter schema, so the subagent's structured result is validated by
    /// [`SchemaVerification`] on each yield call.
    fn schema(&self) -> (serde_json::Value, Option<serde_json::Value>) {
        (self.definition().parameters.clone(), None)
    }

    /// The tool pool the subagent runs with. The engine injects the yield tool
    /// on top. Defaults to empty.
    fn tools(&self) -> ToolRegistry {
        ToolRegistry::new()
    }

    /// Seed content appended to the fork before running. `resume` is true when
    /// the parent is returning to an existing fork (so the instruction body is
    /// already in-session and only the new input need be appended).
    fn seed(&self, params: &serde_json::Value, resume: bool) -> Result<Vec<ContentBlock>, ToolError>;
}

/// The shared engine: turns a [`Subagent`] into a [`ToolHandler`] the parent
/// agent calls. Holds the model/provider to drive the inner [`Agent`], the
/// [`SessionManager`] it forks/resumes from, and the subagent definition.
pub struct SubagentTool {
    model: Model,
    provider: Arc<dyn Provider>,
    manager: Arc<Mutex<dyn SessionManager>>,
    subagent: Arc<dyn Subagent>,
}

impl SubagentTool {
    pub fn new(
        model: Model,
        provider: Arc<dyn Provider>,
        manager: Arc<Mutex<dyn SessionManager>>,
        subagent: impl Subagent + 'static,
    ) -> Self {
        Self { model, provider, manager, subagent: Arc::new(subagent) }
    }

    /// Resolve the fork for this call: resume by `session_id` when the parent
    /// echoes one (and it exists), else fork the main session. Returns the id,
    /// the drivable session, and whether this is a resume.
    fn resolve(
        &self,
        session_id: Option<&str>,
    ) -> Result<(String, Arc<Mutex<dyn crate::core::session::Session>>, bool), ToolError> {
        let mut mgr = self.manager.lock().unwrap();
        if let Some(id) = session_id {
            if let Ok(session) = mgr.get(id) {
                return Ok((id.to_string(), session, true));
            }
        }
        let (id, session) =
            mgr.fork().map_err(|e| ToolError::Execution(format!("failed to fork subagent session: {e}")))?;
        Ok((id, session, false))
    }
}

#[async_trait]
impl ToolHandler for SubagentTool {
    fn definition(&self) -> ToolDefinition {
        // Surface `session_id` as a declared parameter so the parent model knows
        // it can echo it to resume; the engine strips it before seeding.
        let mut def = self.subagent.definition();
        def.parameters = with_session_id(def.parameters);
        def
    }

    // Default schema() = (definition().parameters, None): the parent validates the
    // call's args against the input schema but never validates the result (the
    // subagent's own run already validated the yield against the output schema).

    async fn execute(
        &self,
        _cancel: oneshot::Receiver<bool>,
        params: serde_json::Value,
    ) -> Result<ToolResult, ToolError> {
        let session_id = params.get(SESSION_ID).and_then(|v| v.as_str()).map(str::to_string);
        let (id, session, is_resume) = self.resolve(session_id.as_deref())?;

        // Spawn only: a stop control message drops the inherited main prefix from the
        // subagent's provider view, so it runs isolated while the branch keeps the lineage.
        if !is_resume {
            session.lock().unwrap().append(stop_message()).map_err(|e| ToolError::Execution(e.to_string()))?;
        }

        // Strip the orchestration field so the subagent only sees its own input.
        let mut input = params;
        if let Some(obj) = input.as_object_mut() {
            obj.remove(SESSION_ID);
        }
        let seed_blocks = self.subagent.seed(&input, is_resume)?;

        // Tools: the subagent's pool + the injected yield tool.
        let mut tools = self.subagent.tools();
        let (_, output_schema) = self.subagent.schema();
        let yield_schema = output_schema.unwrap_or_else(default_yield_schema);
        let yielded = Arc::new(Mutex::new(None::<serde_json::Value>));
        tools.register(YieldTool::new(yield_schema, yielded.clone()).into());

        // Drive the subagent. ControlExtension renders the stop (drops the prefix);
        // SchemaVerification validates each yield call against the output schema (a bad
        // call is denied so the subagent can retry); AbortOnYield stops after a yield.
        let extension = And::new(
            ControlExtension::new(),
            And::new(SchemaVerification::new(), AbortOnYield { yielded: yielded.clone() }),
        );
        let outcome = Agent {
            state: AgentState { model: self.model.clone(), tools, session: session.clone() },
            provider: self.provider.clone(),
            extension: Box::new(extension),
        }
        .prompt(Message { role: "user".to_string(), content: seed_blocks })
        .await;

        // A captured result ⇒ success: stamp session_id and return. None ⇒ the run
        // ended without yielding, which is an error (the parent must not trust free text).
        let mut value = match yielded.lock().unwrap().take() {
            Some(v) => v,
            None => {
                let detail = match outcome {
                    Ok(()) => "subagent ended without yielding a result".to_string(),
                    Err(e) => e.to_string(),
                };
                return Ok(ToolResult {
                    tool_call_id: None,
                    content: vec![ToolResultContent::Text(TextContent { content: detail })],
                    is_error: true,
                });
            }
        };
        if let Some(obj) = value.as_object_mut() {
            obj.insert(SESSION_ID.to_string(), serde_json::json!(id));
        }
        Ok(ToolResult {
            tool_call_id: None,
            content: vec![ToolResultContent::Text(TextContent {
                content: serde_json::to_string(&value).unwrap_or_else(|_| value.to_string()),
            })],
            is_error: false,
        })
    }
}

/// Stamp a `session_id` property onto an object-typed parameter schema so the
/// parent model sees it can echo it to resume. Leaves non-object schemas alone.
fn with_session_id(mut params: serde_json::Value) -> serde_json::Value {
    let Some(obj) = params.as_object_mut() else { return params };
    let props = obj.entry("properties").or_insert_with(|| serde_json::json!({}));
    if let Some(p) = props.as_object_mut() {
        p.insert(
            SESSION_ID.to_string(),
            serde_json::json!({
                "type": "string",
                "description": "Echo the session_id from a prior result to resume that subagent thread; omit to spawn a fresh one."
            }),
        );
    }
    params
}

/// The default yield schema when a subagent declares no output: a single
/// free-text `result` string (mirrors the skill's default output shape).
fn default_yield_schema() -> serde_json::Value {
    serde_json::json!({
        "type": "object",
        "properties": {
            "result": { "type": "string", "description": "The subagent's result, as free text." }
        },
        "required": ["result"]
    })
}

/// The exit tool injected into every subagent. Its `parameters` are the
/// subagent's output schema (or the default `{result: string}`); the subagent
/// calls it once with its result, and the arguments are captured via `yielded`
/// so [`SubagentTool`] can return them. First call wins.
struct YieldTool {
    schema: serde_json::Value,
    yielded: Arc<Mutex<Option<serde_json::Value>>>,
}

impl YieldTool {
    fn new(schema: serde_json::Value, yielded: Arc<Mutex<Option<serde_json::Value>>>) -> Self {
        Self { schema, yielded }
    }
}

#[async_trait]
impl ToolHandler for YieldTool {
    fn definition(&self) -> ToolDefinition {
        ToolDefinition {
            name: YIELD_TOOL.to_string(),
            description: "Yield this subagent's result back to the parent agent. Call exactly once \
                with the result object as the arguments (conforming to the tool's parameter schema)."
                .to_string(),
            parameters: self.schema.clone(),
        }
    }

    async fn execute(
        &self,
        _cancel: oneshot::Receiver<bool>,
        params: serde_json::Value,
    ) -> Result<ToolResult, ToolError> {
        let mut slot = self.yielded.lock().unwrap();
        if slot.is_none() {
            *slot = Some(params);
        }
        Ok(ToolResult {
            tool_call_id: None,
            content: vec![ToolResultContent::Text(TextContent { content: "yielded".to_string() })],
            is_error: false,
        })
    }
}

/// [`Extension`] that stops the agent loop the instant the subagent has yielded.
///
/// [`on_message_start`](Extension::on_message_start) fires at the start of every
/// model call inside `agent_loop`. Once [`YieldTool`] has captured a result, the
/// next message start returns `ExtensionError::Stopped`, short-circuiting the loop so the subagent
/// makes no further calls after its result is in. The engine treats that abort
/// as the expected end-of-run signal (it checks `yielded`), not a failure.
struct AbortOnYield {
    yielded: Arc<Mutex<Option<serde_json::Value>>>,
}

#[async_trait]
impl Extension for AbortOnYield {
    fn name(&self) -> &str {
        "subagent/abort-on-yield"
    }

    async fn on_message_start(&mut self, messages: Vec<Message>) -> Result<Vec<Message>, ExtensionError> {
        if self.yielded.lock().unwrap().is_some() {
            return Err(ExtensionError::Stopped {
                name: self.name().to_string(),
                message: "subagent yielded; aborting remaining turns".to_string(),
            });
        }
        Ok(messages)
    }
}

/// Build and register a subagent tool in one step. Equivalent to
/// `registry.register(Box::new(SubagentTool::new(model, provider, manager, subagent)))`.
pub fn register(
    registry: &mut ToolRegistry,
    model: Model,
    provider: Arc<dyn Provider>,
    manager: Arc<Mutex<dyn SessionManager>>,
    subagent: impl Subagent + 'static,
) {
    registry.register(Box::new(SubagentTool::new(model, provider, manager, subagent)));
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::core::providers::{ProviderError, StreamResponse, StreamResponseStream};
    use crate::core::session::{InMemorySession, Session};
    use crate::core::types::{TokenUsage, ToolCall};
    use crate::extra::agents::builder::SessionAdapter;
    use futures::stream;
    use std::collections::VecDeque;

    // --- test doubles ---

    /// A provider that replays a queue of scripted assistant messages, one per
    /// `stream()` call, and records the message counts and texts it is handed —
    /// enough to drive the inner agent and prove what the subagent's provider saw.
    struct ScriptedProvider {
        turns: Arc<Mutex<VecDeque<Message>>>,
        seen_counts: Arc<Mutex<Vec<usize>>>,
        seen_texts: Arc<Mutex<Vec<String>>>,
    }

    impl ScriptedProvider {
        fn new(turns: Vec<Message>) -> Self {
            Self {
                turns: Arc::new(Mutex::new(turns.into())),
                seen_counts: Arc::new(Mutex::new(Vec::new())),
                seen_texts: Arc::new(Mutex::new(Vec::new())),
            }
        }
    }

    #[async_trait]
    impl Provider for ScriptedProvider {
        async fn stream<'a>(
            &self,
            _model: &Model,
            messages: Box<dyn Iterator<Item = &'a Message> + Send + 'a>,
            _tools: &Vec<ToolDefinition>,
        ) -> Result<StreamResponseStream, ProviderError> {
            let msgs: Vec<&Message> = messages.collect();
            self.seen_counts.lock().unwrap().push(msgs.len());
            let joined = msgs
                .iter()
                .flat_map(|m| {
                    m.content.iter().filter_map(|b| match b {
                        ContentBlock::Text(t) => Some(t.content.as_str()),
                        _ => None,
                    })
                })
                .collect::<Vec<_>>()
                .join("|");
            self.seen_texts.lock().unwrap().push(joined);
            let msg = self.turns.lock().unwrap().pop_front().unwrap_or_else(|| Message {
                role: "assistant".to_string(),
                content: vec![ContentBlock::Text(TextContent { content: String::new() })],
            });
            let stop =
                if msg.content.iter().any(|b| matches!(b, ContentBlock::ToolCall(_))) { "tool_calls" } else { "stop" };
            Ok(Box::pin(stream::iter(vec![StreamResponse {
                message: msg,
                usage: TokenUsage {
                    input_tokens: 0,
                    output_tokens: 0,
                    cache_read_tokens: None,
                    cache_write_tokens: None,
                },
                stop_reason: Some(stop.to_string()),
            }])))
        }
        fn name(&self) -> &str {
            "scripted"
        }
    }

    /// A minimal subagent: a fixed definition/schema/tool pool and a seed that
    /// records whether it was asked to resume.
    struct EchoSubagent {
        definition: ToolDefinition,
        output: Option<serde_json::Value>,
        seeded_resume: Arc<Mutex<Vec<bool>>>,
    }

    #[async_trait]
    impl Subagent for EchoSubagent {
        fn definition(&self) -> ToolDefinition {
            self.definition.clone()
        }
        fn schema(&self) -> (serde_json::Value, Option<serde_json::Value>) {
            (self.definition.parameters.clone(), self.output.clone())
        }
        fn seed(&self, _params: &serde_json::Value, resume: bool) -> Result<Vec<ContentBlock>, ToolError> {
            self.seeded_resume.lock().unwrap().push(resume);
            Ok(vec![ContentBlock::Text(TextContent { content: "go".to_string() })])
        }
    }

    fn def() -> ToolDefinition {
        ToolDefinition {
            name: "echo".to_string(),
            description: "stub".to_string(),
            parameters: serde_json::json!({ "type": "object", "properties": { "input": { "type": "string" } } }),
        }
    }

    fn model() -> Model {
        Model {
            id: "m".to_string(),
            provider: "p".to_string(),
            context_window: 0,
            base_url: String::new(),
            headers: std::collections::HashMap::new(),
        }
    }

    fn manager() -> Arc<Mutex<dyn SessionManager>> {
        Arc::new(Mutex::new(SessionAdapter::new(InMemorySession::new(), || InMemorySession::new().arc())))
    }

    fn tool_call(name: &str, args: serde_json::Value) -> Message {
        Message {
            role: "assistant".to_string(),
            content: vec![ContentBlock::ToolCall(ToolCall {
                id: "call".to_string(),
                name: name.to_string(),
                arguments: args,
            })],
        }
    }

    fn text_result(result: &ToolResult) -> String {
        result
            .content
            .iter()
            .filter_map(|c| match c {
                ToolResultContent::Text(t) => Some(t.content.clone()),
                _ => None,
            })
            .collect::<Vec<_>>()
            .join("")
    }

    fn parse(result: &ToolResult) -> serde_json::Value {
        serde_json::from_str(&text_result(result)).expect("result is JSON")
    }

    // --- behavior ---

    #[tokio::test]
    async fn spawn_forks_fresh_branch_each_call() {
        let provider = Arc::new(ScriptedProvider::new(vec![
            tool_call(YIELD_TOOL, serde_json::json!({"result": "a"})),
            tool_call(YIELD_TOOL, serde_json::json!({"result": "b"})),
        ]));
        let tool = SubagentTool::new(
            model(),
            provider,
            manager(),
            EchoSubagent { definition: def(), output: None, seeded_resume: Arc::new(Mutex::new(vec![])) },
        );

        let (_, rx) = oneshot::channel();
        let r1 = tool.execute(rx, serde_json::json!({"input": "q1"})).await.unwrap();
        let (_, rx) = oneshot::channel();
        let r2 = tool.execute(rx, serde_json::json!({"input": "q2"})).await.unwrap();

        let id1 = parse(&r1)["session_id"].as_str().unwrap().to_string();
        let id2 = parse(&r2)["session_id"].as_str().unwrap().to_string();
        assert_ne!(id1, id2, "two spawns get distinct session ids");
    }

    #[tokio::test]
    async fn stop_drops_inherited_prefix_from_provider_view() {
        // Seed the main session so a fork inherits "main secret" — the stop must keep
        // it out of the subagent's provider view.
        let mgr = manager();
        mgr.lock()
            .unwrap()
            .append(Message {
                role: "user".to_string(),
                content: vec![ContentBlock::Text(TextContent { content: "main secret".to_string() })],
            })
            .unwrap();

        let provider = Arc::new(ScriptedProvider::new(vec![tool_call(YIELD_TOOL, serde_json::json!({"result": "x"}))]));
        let seen = provider.seen_texts.clone();
        let tool = SubagentTool::new(
            model(),
            provider,
            mgr,
            EchoSubagent { definition: def(), output: None, seeded_resume: Arc::new(Mutex::new(vec![])) },
        );

        let (_, rx) = oneshot::channel();
        tool.execute(rx, serde_json::json!({"input": "q"})).await.unwrap();

        let views = seen.lock().unwrap();
        assert!(!views.is_empty(), "the subagent made at least one provider call");
        assert!(
            !views.iter().any(|v| v.contains("main secret")),
            "inherited main prefix must be dropped from the provider view: {views:?}"
        );
    }

    #[tokio::test]
    async fn yield_captures_result_and_stamps_session_id() {
        let provider =
            Arc::new(ScriptedProvider::new(vec![tool_call(YIELD_TOOL, serde_json::json!({"result": "the answer"}))]));
        let tool = SubagentTool::new(
            model(),
            provider,
            manager(),
            EchoSubagent { definition: def(), output: None, seeded_resume: Arc::new(Mutex::new(vec![])) },
        );

        let (_, rx) = oneshot::channel();
        let result = tool.execute(rx, serde_json::json!({"input": "q"})).await.unwrap();
        assert!(!result.is_error);
        let parsed = parse(&result);
        assert_eq!(parsed["result"], "the answer");
        assert!(parsed["session_id"].as_str().is_some(), "session_id is stamped for resume");
    }

    #[tokio::test]
    async fn output_schema_denial_is_retried() {
        // First yield violates the output schema (x must be an integer); SchemaVerification
        // denies it (nothing captured), the subagent retries with a conforming value.
        let provider = Arc::new(ScriptedProvider::new(vec![
            tool_call(YIELD_TOOL, serde_json::json!({"x": "not-an-int"})),
            tool_call(YIELD_TOOL, serde_json::json!({"x": 7})),
        ]));
        let tool = SubagentTool::new(
            model(),
            provider,
            manager(),
            EchoSubagent {
                definition: def(),
                output: Some(
                    serde_json::json!({"type": "object", "properties": {"x": {"type": "integer"}}, "required": ["x"]}),
                ),
                seeded_resume: Arc::new(Mutex::new(vec![])),
            },
        );

        let (_, rx) = oneshot::channel();
        let result = tool.execute(rx, serde_json::json!({"input": "q"})).await.unwrap();
        assert!(!result.is_error);
        assert_eq!(parse(&result)["x"], 7);
    }

    #[tokio::test]
    async fn ending_without_yielding_is_an_error() {
        let provider = Arc::new(ScriptedProvider::new(vec![Message {
            role: "assistant".to_string(),
            content: vec![ContentBlock::Text(TextContent { content: "I give up".to_string() })],
        }]));
        let tool = SubagentTool::new(
            model(),
            provider,
            manager(),
            EchoSubagent { definition: def(), output: None, seeded_resume: Arc::new(Mutex::new(vec![])) },
        );

        let (_, rx) = oneshot::channel();
        let result = tool.execute(rx, serde_json::json!({"input": "q"})).await.unwrap();
        assert!(result.is_error, "a run with no yield must surface as an error");
        assert!(text_result(&result).contains("without yielding"));
    }

    #[tokio::test]
    async fn resume_reopens_same_branch_and_keeps_context() {
        let provider = Arc::new(ScriptedProvider::new(vec![
            tool_call(YIELD_TOOL, serde_json::json!({"result": "r1"})),
            tool_call(YIELD_TOOL, serde_json::json!({"result": "r2"})),
        ]));
        let counts = provider.seen_counts.clone();
        let seeded = Arc::new(Mutex::new(vec![]));
        let tool = SubagentTool::new(
            model(),
            provider,
            manager(),
            EchoSubagent { definition: def(), output: None, seeded_resume: seeded.clone() },
        );

        // Spawn.
        let (_, rx) = oneshot::channel();
        let r1 = tool.execute(rx, serde_json::json!({"input": "q1"})).await.unwrap();
        let sid = parse(&r1)["session_id"].as_str().unwrap().to_string();

        // Resume by echoing the id.
        let (_, rx) = oneshot::channel();
        let r2 = tool.execute(rx, serde_json::json!({"input": "q2", "session_id": sid})).await.unwrap();
        assert!(!r2.is_error);
        assert_eq!(parse(&r2)["session_id"].as_str().unwrap(), sid, "resume echoes the same session id");
        assert_eq!(parse(&r2)["result"], "r2");

        let counts = counts.lock().unwrap();
        assert!(counts.len() >= 2, "provider streamed at least twice: {:?}", *counts);
        assert!(counts[1] > counts[0], "resume saw prior context: {:?}", *counts);

        let seeded = seeded.lock().unwrap();
        assert_eq!(*seeded, vec![false, true], "first call spawns, second resumes");
    }

    #[tokio::test]
    async fn unknown_session_id_spawns_fresh() {
        let provider = Arc::new(ScriptedProvider::new(vec![
            tool_call(YIELD_TOOL, serde_json::json!({"result": "r1"})),
            tool_call(YIELD_TOOL, serde_json::json!({"result": "r2"})),
        ]));
        let tool = SubagentTool::new(
            model(),
            provider,
            manager(),
            EchoSubagent { definition: def(), output: None, seeded_resume: Arc::new(Mutex::new(vec![])) },
        );

        let (_, rx) = oneshot::channel();
        let r1 = tool.execute(rx, serde_json::json!({"input": "q1"})).await.unwrap();
        let sid1 = parse(&r1)["session_id"].as_str().unwrap().to_string();

        // An id the manager has never seen ⇒ fresh, with a brand-new id.
        let (_, rx) = oneshot::channel();
        let r2 = tool.execute(rx, serde_json::json!({"input": "q2", "session_id": "never-seen"})).await.unwrap();
        let sid2 = parse(&r2)["session_id"].as_str().unwrap().to_string();
        assert_ne!(sid2, sid1);
        assert_ne!(sid2, "never-seen");
    }

    #[test]
    fn definition_merges_session_id() {
        let tool = SubagentTool::new(
            model(),
            Arc::new(ScriptedProvider::new(vec![])),
            manager(),
            EchoSubagent { definition: def(), output: None, seeded_resume: Arc::new(Mutex::new(vec![])) },
        );
        let params = tool.definition().parameters;
        assert!(params["properties"]["session_id"]["type"].is_string(), "session_id is a declared parameter");
        assert_eq!(tool.definition().name, "echo");
    }

    #[test]
    fn registers_as_a_named_handler() {
        let mut registry = ToolRegistry::new();
        register(
            &mut registry,
            model(),
            Arc::new(ScriptedProvider::new(vec![])),
            manager(),
            EchoSubagent { definition: def(), output: None, seeded_resume: Arc::new(Mutex::new(vec![])) },
        );
        assert!(registry.get("echo").is_some());
        assert_eq!(registry.definitions().len(), 1);
    }
}

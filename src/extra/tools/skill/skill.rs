use std::sync::{Arc, Mutex};

use async_trait::async_trait;
use futures::channel::oneshot;

use crate::core::agent::{Agent, AgentState};
use crate::core::extensions::{Extension, ExtensionError};
use crate::core::providers::{Model, Provider};
use crate::core::session::Session;
use crate::core::tools::{ToolError, ToolHandler, ToolRegistry};
use crate::core::types::{ContentBlock, Message, TextContent, ToolDefinition, ToolResult, ToolResultContent};
use crate::extra::extensions::combinator::And;
use crate::extra::extensions::schema::SchemaVerification;
use crate::extra::tools::skill::config::{self, ParsedSkill, SkillConfig};
use crate::extra::tools::subagent::{self, SessionFactory, Subagent};

/// One Agent Skill exposed to the parent agent as a subagent tool.
///
/// When invoked, it uses the skill's pre-parsed `SKILL.md` instruction body,
/// restricts the available tools to those named in `allowed-tools`, and runs an
/// [`Agent`] in a fresh session seeded with the instructions and the caller's
/// input. The sub-agent returns its result through an injected
/// `akasha_skill_submit` tool, which is verified against the skill's output
/// schema — a bad call is rejected so the sub-agent can retry. The loop aborts
/// the instant a valid result is captured (no extra model call), and a run that
/// ends without a valid submission is reported as an error.
///
/// Construct one per discovered skill via [`SkillTool::new`], or use [`register`]
/// to register every skill under a directory at once.
pub struct SkillTool {
    model: Model,
    provider: Arc<dyn Provider>,
    base_tools: ToolRegistry,
    allowed_tools: Vec<String>,
    skill: ParsedSkill,
    input_schema: serde_json::Value,
    output_schema: Option<serde_json::Value>,
}

impl SkillTool {
    /// Resolves the skill's schemas so they can be surfaced to the agent (input
    /// via the tool definition, output via [`Subagent::schema`]). Fails if
    /// a referenced schema file cannot be read. Schema *compilation* is deferred
    /// to the verification extension, so this does not validate the schemas.
    pub fn new(
        model: Model,
        provider: Arc<dyn Provider>,
        base_tools: ToolRegistry,
        skill: ParsedSkill,
    ) -> Result<Self, config::SkillError> {
        let (input_schema, output_schema) = skill.resolve_schema()?;
        let allowed_tools = skill.frontmatter.allowed_tools.clone();
        Ok(Self { model, provider, base_tools, allowed_tools, skill, input_schema, output_schema })
    }
}

#[async_trait]
impl Subagent for SkillTool {
    fn definition(&self) -> ToolDefinition {
        ToolDefinition {
            name: self.skill.frontmatter.name.clone(),
            description: self.skill.frontmatter.description.clone(),
            parameters: self.input_schema.clone(),
        }
    }

    fn schema(&self) -> (serde_json::Value, Option<serde_json::Value>) {
        (self.input_schema.clone(), self.output_schema.clone())
    }

    async fn execute(
        &self,
        session: Arc<Mutex<dyn Session>>,
        _cancel: oneshot::Receiver<bool>,
        params: serde_json::Value,
    ) -> Result<ToolResult, ToolError> {
        // Restrict the sub-agent to the skill's pre-approved tools.
        let mut tools = self.base_tools.subset(&self.allowed_tools);

        // The sub-agent hands its result back through `akasha_skill_submit`. Its parameter
        // schema is the skill's output schema, or a default `{result: string}` shape when the
        // skill declares none — every skill must submit, so a schema is always needed. The
        // captured arguments become the skill's output.
        let submitted = Arc::new(Mutex::new(None::<serde_json::Value>));
        let submit_schema = self.output_schema.clone().unwrap_or_else(config::default_output_schema);
        tools.register(SubmitTool::new(submit_schema, submitted.clone()).into());

        // Drive the sub-agent through the shared `Agent`, seeded with the initial message. The
        // composed extension verifies each submit call against the skill's output schema — a bad
        // call is denied (so the submit tool never runs and the sub-agent sees the error and can
        // retry) and only a valid one is captured — and aborts the loop the moment a result is
        // captured, so no further model call runs after it.
        let outcome = Agent {
            state: AgentState { model: self.model.clone(), tools, session },
            provider: self.provider.clone(),
            extension: And::new(SchemaVerification::new(), AbortAfterSubmit { submitted: submitted.clone() }).into(),
        }
        .prompt(Message {
            role: "user".to_string(),
            content: vec![
                ContentBlock::Text(TextContent { content: self.skill.body.clone() }),
                ContentBlock::Text(TextContent { content: params.to_string() }),
                ContentBlock::Text(TextContent {
                    content: format!(
                        "Your final result MUST be returned by calling the `{SUBMIT_TOOL}` tool, \
                         with the result object as its arguments (conforming to the tool's parameter schema). \
                         Do not emit the result as text."
                    ),
                }),
            ],
        })
        .await;

        // Submit captured a result ⇒ return it (the extension's abort error is the expected
        // end-of-run signal, so `outcome` is ignored here). No submit ⇒ the sub-agent never
        // produced a conforming result, which is an error — the parent must not trust free text.
        let Some(value) = submitted.lock().unwrap().take() else {
            let detail = match outcome {
                Ok(()) => "skill ended without calling the submit tool".to_string(),
                Err(e) => e.to_string(),
            };
            return Ok(ToolResult {
                tool_call_id: None,
                content: vec![ToolResultContent::Text(TextContent { content: detail })],
                is_error: true,
            });
        };

        Ok(ToolResult {
            tool_call_id: None,
            content: vec![ToolResultContent::Text(TextContent {
                content: serde_json::to_string(&value).unwrap_or_else(|_| value.to_string()),
            })],
            is_error: false,
        })
    }
}

/// The fixed name of the structured-output tool injected into a skill sub-agent
/// when the skill declares an output schema.
const SUBMIT_TOOL: &str = "akasha_skill_submit";

/// The exit tool injected into every skill sub-agent. Its `parameters` are the
/// skill's output schema, or a default `{result: string}` shape when the skill
/// declares none; the sub-agent calls it once with its result, and the arguments
/// are captured via `submitted` so [`SkillTool`] can return them as the skill's
/// output.
struct SubmitTool {
    schema: serde_json::Value,
    submitted: Arc<Mutex<Option<serde_json::Value>>>,
}

impl SubmitTool {
    fn new(schema: serde_json::Value, submitted: Arc<Mutex<Option<serde_json::Value>>>) -> Self {
        Self { schema, submitted }
    }
}

#[async_trait]
impl ToolHandler for SubmitTool {
    fn definition(&self) -> ToolDefinition {
        ToolDefinition {
            name: SUBMIT_TOOL.to_string(),
            description: "Submit this skill's structured result. Call exactly once with the \
                result object as the arguments."
                .to_string(),
            parameters: self.schema.clone(),
        }
    }

    async fn execute(
        &self,
        _cancel: oneshot::Receiver<bool>,
        params: serde_json::Value,
    ) -> Result<ToolResult, ToolError> {
        // First call wins; a second call (the model should not make one) is ignored so
        // the skill returns the first submitted result.
        let mut slot = self.submitted.lock().unwrap();
        if slot.is_none() {
            *slot = Some(params);
        }
        Ok(ToolResult {
            tool_call_id: None,
            content: vec![ToolResultContent::Text(TextContent { content: "submitted".to_string() })],
            is_error: false,
        })
    }
}

/// [`Extension`] that stops the agent loop the instant the sub-agent has
/// submitted its result.
///
/// [`on_message_start`](Extension::on_message_start) fires at the start of every
/// model call inside `agent_loop`. Once [`SubmitTool`] has captured a result,
/// the next message start returns `Err`, short-circuiting the loop so the
/// sub-agent makes no further calls after its result is in. The skill treats
/// that abort as the expected end-of-run signal (it checks `submitted`), not a
/// failure.
struct AbortAfterSubmit {
    submitted: Arc<Mutex<Option<serde_json::Value>>>,
}

#[async_trait]
impl Extension for AbortAfterSubmit {
    fn name(&self) -> &str {
        "skill-abort-after-submit"
    }

    async fn on_message_start(&mut self, messages: Vec<Message>) -> Result<Vec<Message>, ExtensionError> {
        if self.submitted.lock().unwrap().is_some() {
            return Err(ExtensionError::ExtensionFailed {
                name: self.name().to_string(),
                message: "skill submitted; aborting remaining turns".to_string(),
            });
        }
        Ok(messages)
    }
}

/// Discovers every valid skill under `config.skills_dir` and registers each as
/// a subagent tool in `registry`.
///
/// `base_tools` is the pool the skills draw from. Pass a registry that does
/// **not** contain the skill tools themselves (e.g. one built before this
/// call) so a skill cannot recursively invoke itself. `session_factory`
/// supplies the fresh, isolated session each skill invocation runs in. A skill
/// whose schemas fail to resolve/compile is skipped with a warning.
pub fn register(
    registry: &mut ToolRegistry,
    config: &SkillConfig,
    model: Model,
    provider: Arc<dyn Provider>,
    base_tools: ToolRegistry,
    session_factory: SessionFactory,
) -> Result<(), config::SkillError> {
    for skill in config::discover(&config.dir)? {
        let label = skill.dir.display().to_string();
        let tool = match SkillTool::new(model.clone(), provider.clone(), base_tools.clone(), skill) {
            Ok(t) => t,
            Err(e) => {
                eprintln!("[skill] skipping {label}: {e}");
                continue;
            }
        };
        // `subagent::register` takes a generic `Fn()`; adapt the shared `SessionFactory`
        // arc into a fresh closure per skill (cheap — a clone of the `Arc`).
        let factory = session_factory.clone();
        subagent::register(registry, tool, move || factory());
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::{HashMap, VecDeque};
    use std::path::PathBuf;
    use std::sync::atomic::{AtomicU64, Ordering};

    use crate::core::providers::{ProviderError, StreamResponse, StreamResponseStream};
    use crate::core::session::InMemorySession;
    use crate::core::tools::ToolHandler;
    use crate::core::types::{TokenUsage, ToolCall};
    use futures::stream;

    /// A provider that replays a queue of scripted assistant messages, one per
    /// `stream()` call, and records the tool definitions it was handed.
    struct ScriptedProvider {
        turns: Arc<Mutex<VecDeque<Message>>>,
        seen_tools: Arc<Mutex<Vec<Vec<ToolDefinition>>>>,
    }

    impl ScriptedProvider {
        fn new(turns: Vec<Message>) -> Self {
            Self { turns: Arc::new(Mutex::new(turns.into())), seen_tools: Arc::new(Mutex::new(Vec::new())) }
        }
    }

    #[async_trait]
    impl Provider for ScriptedProvider {
        async fn stream<'a>(
            &self,
            _model: &Model,
            _messages: Box<dyn Iterator<Item = &'a Message> + Send + 'a>,
            tools: &Vec<ToolDefinition>,
        ) -> Result<StreamResponseStream, ProviderError> {
            self.seen_tools.lock().unwrap().push(tools.clone());
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

    /// A no-op tool used to populate the base registry.
    struct StubTool {
        name: &'static str,
    }

    #[async_trait]
    impl ToolHandler for StubTool {
        fn definition(&self) -> ToolDefinition {
            ToolDefinition {
                name: self.name.to_string(),
                description: "stub".to_string(),
                parameters: serde_json::json!({ "type": "object" }),
            }
        }

        async fn execute(
            &self,
            _cancel: oneshot::Receiver<bool>,
            _params: serde_json::Value,
        ) -> Result<ToolResult, ToolError> {
            Ok(ToolResult {
                tool_call_id: None,
                content: vec![ToolResultContent::Text(TextContent { content: "ok".to_string() })],
                is_error: false,
            })
        }
    }

    fn base_tools() -> ToolRegistry {
        let mut tools = ToolRegistry::new();
        for name in ["a", "b", "c"] {
            tools.register(StubTool { name }.into());
        }
        tools
    }

    fn model() -> Model {
        Model {
            id: "m".to_string(),
            provider: "p".to_string(),
            context_window: 0,
            base_url: String::new(),
            headers: HashMap::new(),
        }
    }

    static COUNTER: AtomicU64 = AtomicU64::new(0);

    struct TempDir(PathBuf);
    impl Drop for TempDir {
        fn drop(&mut self) {
            let _ = std::fs::remove_dir_all(&self.0);
        }
    }

    /// Writes a skill to a unique temp dir. `schema_json`, if given, is written to
    /// `schema.json` (MCP Tool shape: `inputSchema`/`outputSchema`) and referenced
    /// from the frontmatter.
    fn make_skill(name: &str, allowed: Option<&str>, schema_json: Option<&str>) -> (TempDir, ParsedSkill) {
        let n = COUNTER.fetch_add(1, Ordering::SeqCst);
        let dir = std::env::temp_dir().join(format!("akasha-skill-test-{}-{}", std::process::id(), n));
        std::fs::create_dir_all(&dir).unwrap();
        let mut src = format!("---\nname: {name}\ndescription: a test skill\n");
        if let Some(a) = allowed {
            src.push_str(&format!("allowed-tools: \"{a}\"\n"));
        }
        if let Some(json) = schema_json {
            std::fs::write(dir.join("schema.json"), json).unwrap();
            src.push_str("schema: schema.json\n");
        }
        src.push_str("---\n## Instructions\nDo the thing.\n");
        std::fs::write(dir.join("SKILL.md"), &src).unwrap();

        let (frontmatter, body) = config::parse(&src).unwrap();
        (TempDir(dir.clone()), ParsedSkill { frontmatter, dir, body })
    }

    fn assistant_text(text: &str) -> Message {
        Message {
            role: "assistant".to_string(),
            content: vec![ContentBlock::Text(TextContent { content: text.to_string() })],
        }
    }

    fn assistant_tool_call(name: &str, arguments: serde_json::Value) -> Message {
        Message {
            role: "assistant".to_string(),
            content: vec![ContentBlock::ToolCall(ToolCall {
                id: "call".to_string(),
                name: name.to_string(),
                arguments,
            })],
        }
    }

    fn reasoning_only() -> Message {
        Message {
            role: "assistant".to_string(),
            content: vec![ContentBlock::Reasoning(TextContent { content: "thinking...".to_string() })],
        }
    }

    async fn run(
        skill: ParsedSkill,
        turns: Vec<Message>,
        params: serde_json::Value,
    ) -> (ToolResult, Vec<Vec<ToolDefinition>>) {
        let provider = Arc::new(ScriptedProvider::new(turns));
        let seen = provider.seen_tools.clone();
        let tool = SkillTool::new(model(), provider, base_tools(), skill).unwrap();
        let (_, rx) = oneshot::channel();
        let result = tool.execute(InMemorySession::new().arc(), rx, params).await.unwrap();
        (result, seen.lock().unwrap().clone())
    }

    fn tool_names(defs: &[ToolDefinition]) -> Vec<String> {
        let mut names: Vec<String> = defs.iter().map(|d| d.name.clone()).collect();
        names.sort();
        names
    }

    fn text_of(result: &ToolResult) -> String {
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

    #[tokio::test]
    async fn subset_filters_to_allowed_tools() {
        let (_g, skill) = make_skill("s", Some("a c"), None);
        let (result, seen) = run(
            skill,
            vec![assistant_tool_call(SUBMIT_TOOL, serde_json::json!({"result": "done"}))],
            serde_json::json!({"input": "go"}),
        )
        .await;
        assert!(!result.is_error);
        // The subset exposes only the allowed tools, plus the always-injected submit tool.
        for defs in &seen {
            assert_eq!(tool_names(defs), vec!["a", "akasha_skill_submit", "c"]);
        }
    }

    #[tokio::test]
    async fn absent_allowed_tools_gives_only_submit() {
        // With `subset`, an empty allowlist means no base tools (deny-all default); the
        // always-injected submit tool is still available.
        let (_g, skill) = make_skill("s", None, None);
        let (_, seen) = run(skill, vec![assistant_text("done")], serde_json::json!({"input": "go"})).await;
        for defs in &seen {
            assert_eq!(tool_names(defs), vec!["akasha_skill_submit"]);
        }
    }

    #[tokio::test]
    async fn submit_captures_free_text_result() {
        // A no-output-schema skill gets a submit tool with a `{result: string}` schema. The
        // sub-agent calls it after a normal tool turn, and those args become the output; the
        // loop then aborts (no third model call — only two turns are queued).
        let (_g, skill) = make_skill("s", Some("a"), None);
        let (result, seen) = run(
            skill,
            vec![
                assistant_tool_call("a", serde_json::json!({})),
                assistant_tool_call(SUBMIT_TOOL, serde_json::json!({"result": "the answer"})),
            ],
            serde_json::json!({"input": "go"}),
        )
        .await;
        assert!(!result.is_error);
        assert_eq!(text_of(&result), r#"{"result":"the answer"}"#);
        // The provider was invoked twice (tool turn + submit) and no more: the abort fired at
        // the next message start, before a third call.
        assert_eq!(seen.len(), 2);
    }

    #[tokio::test]
    async fn ends_without_submit_is_error() {
        // Any turn that ends without calling the submit tool is an error — including a
        // reasoning-only turn that produces no usable output at all.
        let (_g, skill) = make_skill("s", None, None);
        let (result, _) = run(skill, vec![reasoning_only()], serde_json::json!({"input": "go"})).await;
        assert!(result.is_error);
    }

    #[tokio::test]
    async fn missing_input_runs_skill_without_validating() {
        // The skill no longer validates its own input (the verification extension does). A missing
        // `input` therefore runs through rather than producing a validation error.
        let (_g, skill) = make_skill("s", None, None);
        let (result, _) = run(
            skill,
            vec![assistant_tool_call(SUBMIT_TOOL, serde_json::json!({"result": "done"}))],
            serde_json::json!({}),
        )
        .await;
        assert!(!result.is_error);
    }

    // --- input-schema ---

    #[tokio::test]
    async fn input_schema_surfaces_in_definition() {
        let (_g, skill) = make_skill(
            "s",
            None,
            Some(r#"{"inputSchema":{"type":"object","properties":{"path":{"type":"string"}},"required":["path"]}}"#),
        );
        let provider: Arc<dyn Provider> = Arc::new(ScriptedProvider::new(vec![]));
        let tool = SkillTool::new(model(), provider, base_tools(), skill).unwrap();
        let params = tool.definition().parameters;
        assert_eq!(params["properties"]["path"]["type"], "string");
        assert_eq!(params["required"][0], "path");
    }

    #[tokio::test]
    async fn valid_structured_input_is_accepted() {
        let (_g, skill) = make_skill("s", None, Some(r#"{"inputSchema":{"type":"object","required":["path"]}}"#));
        let (result, _) = run(
            skill,
            vec![assistant_tool_call(SUBMIT_TOOL, serde_json::json!({"result": "done"}))],
            serde_json::json!({"path": "/tmp"}),
        )
        .await;
        assert!(!result.is_error);
    }

    // --- output-schema (declared here; verified by the SchemaVerification extension) ---

    const OUT_SCHEMA: &str =
        r#"{"outputSchema":{"type":"object","properties":{"x":{"type":"integer"}},"required":["x"]}}"#;

    #[tokio::test]
    async fn output_schema_is_exposed() {
        let (_g, skill) = make_skill("s", None, Some(OUT_SCHEMA));
        let provider: Arc<dyn Provider> = Arc::new(ScriptedProvider::new(vec![]));
        let tool = SkillTool::new(model(), provider, base_tools(), skill).unwrap();
        let (_, output) = tool.schema();
        let output = output.expect("output schema is declared");
        assert_eq!(output["required"][0], "x");
    }

    #[tokio::test]
    async fn output_schema_skill_captures_submit_tool_args() {
        // An output-schema skill injects a `akasha_skill_submit` tool; the sub-agent
        // calls it with its structured result, and those arguments are returned as the
        // skill's output (parsing + validation is the parent extension's job).
        let (_g, skill) = make_skill("s", None, Some(OUT_SCHEMA));
        let (result, seen) = run(
            skill,
            vec![assistant_tool_call("akasha_skill_submit", serde_json::json!({"x": 7}))],
            serde_json::json!({"input": "go"}),
        )
        .await;
        assert!(!result.is_error);
        let parsed: serde_json::Value = serde_json::from_str(&text_of(&result)).expect("result is JSON");
        assert_eq!(parsed["x"], 7);
        // The submit tool is injected into the sub-agent's toolset.
        assert!(seen.iter().any(|defs| defs.iter().any(|d| d.name == "akasha_skill_submit")));
        // The provider is invoked exactly once: after submit captures the result, the abort
        // fires at the next message start, so no second model call is made.
        assert_eq!(seen.len(), 1);
    }

    #[tokio::test]
    async fn output_schema_skill_without_submit_is_error() {
        // An output-schema skill that never calls submit cannot produce a conforming result, so
        // the skill reports an error rather than trusting the sub-agent's free text.
        let (_g, skill) = make_skill("s", None, Some(OUT_SCHEMA));
        let (result, seen) = run(skill, vec![assistant_text(r#"{"x": 7}"#)], serde_json::json!({"input": "go"})).await;
        assert!(result.is_error);
        // The submit tool is still offered, even though the model ignored it.
        assert!(seen.iter().any(|defs| defs.iter().any(|d| d.name == "akasha_skill_submit")));
    }

    #[tokio::test]
    async fn retries_submit_after_schema_violation() {
        // The sub-agent first submits args that violate the output schema. SchemaVerification
        // denies the call (the submit tool never runs, so nothing is captured); the sub-agent
        // sees the error and retries with conforming args, which the skill returns.
        let (_g, skill) = make_skill("s", None, Some(OUT_SCHEMA));
        let (result, seen) = run(
            skill,
            vec![
                assistant_tool_call(SUBMIT_TOOL, serde_json::json!({"x": "not-an-int"})),
                assistant_tool_call(SUBMIT_TOOL, serde_json::json!({"x": 7})),
            ],
            serde_json::json!({"input": "go"}),
        )
        .await;
        assert!(!result.is_error);
        let parsed: serde_json::Value = serde_json::from_str(&text_of(&result)).expect("result is JSON");
        assert_eq!(parsed["x"], 7);
        // Two model turns — the invalid attempt then the valid retry — and no third (the abort
        // fires at the next message start after the valid one).
        assert_eq!(seen.len(), 2);
    }

    #[tokio::test]
    async fn invalid_submit_without_retry_is_error() {
        // The sub-agent submits invalid args and then gives up (a plain text turn) without a
        // valid retry → nothing conforming was captured, so the skill reports an error.
        let (_g, skill) = make_skill("s", None, Some(OUT_SCHEMA));
        let (result, _) = run(
            skill,
            vec![assistant_tool_call(SUBMIT_TOOL, serde_json::json!({"x": "not-an-int"})), assistant_text("I give up")],
            serde_json::json!({"input": "go"}),
        )
        .await;
        assert!(result.is_error);
    }

    // --- schema file resolution ---

    #[tokio::test]
    async fn schema_file_is_resolved_and_surfaces() {
        let (_g, skill) = make_skill("s", None, Some(r#"{"inputSchema":{"type":"object","required":["a"]}}"#));
        let provider: Arc<dyn Provider> = Arc::new(ScriptedProvider::new(vec![assistant_tool_call(
            SUBMIT_TOOL,
            serde_json::json!({"result": "done"}),
        )]));
        let tool = SkillTool::new(model(), provider, base_tools(), skill).unwrap();

        // definition exposes the resolved input schema
        assert_eq!(tool.definition().parameters["required"][0], "a");

        // input is not validated by the skill — a valid call runs through and submits
        let (_, rx) = oneshot::channel();
        let result = tool.execute(InMemorySession::new().arc(), rx, serde_json::json!({"a": 1})).await.unwrap();
        assert!(!result.is_error);
    }

    #[tokio::test]
    async fn registers_each_skill_as_a_named_tool() {
        let n = COUNTER.fetch_add(1, Ordering::SeqCst);
        let dir = std::env::temp_dir().join(format!("akasha-skill-reg-{}-{}", std::process::id(), n));
        std::fs::create_dir_all(dir.join("alpha")).unwrap();
        std::fs::write(dir.join("alpha").join("SKILL.md"), "---\nname: alpha\ndescription: alpha skill\n---\nbody\n")
            .unwrap();
        let guard = TempDir(dir.clone());

        let cfg = SkillConfig { dir };
        let provider: Arc<dyn Provider> = Arc::new(ScriptedProvider::new(vec![]));
        let mut registry = ToolRegistry::new();
        register(&mut registry, &cfg, model(), provider, base_tools(), Arc::new(|| Ok(InMemorySession::new().arc())))
            .unwrap();

        assert!(registry.get("alpha").is_some());
        assert_eq!(registry.get("alpha").unwrap().definition().name, "alpha");
        drop(guard);
    }
}

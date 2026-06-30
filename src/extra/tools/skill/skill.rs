use std::sync::{Arc, Mutex};

use async_trait::async_trait;

use crate::core::providers::{Model, Provider};
use crate::core::tools::{ToolError, ToolRegistry};
use crate::core::types::{ContentBlock, TextContent, ToolDefinition};
use crate::extra::agents::builder::SessionManager;
use crate::extra::tools::skill::config::{self, ParsedSkill, SkillConfig};
use crate::extra::tools::subagent::{self, Subagent};

/// One Agent Skill exposed to the parent agent as a subagent.
///
/// A skill is purely declarative: its parsed `SKILL.md` instruction body, its
/// `allowed-tools` subset, and its input/output schemas. The driving — forking
/// a session, injecting the yield tool, validating the result, resume — is the
/// [`SubagentTool`](crate::extra::tools::subagent::SubagentTool) engine's job;
/// this type just says *what* the skill is.
///
/// Construct one per discovered skill via [`SkillTool::new`], or use [`register`]
/// to register every skill under a directory at once.
pub struct SkillTool {
    base_tools: ToolRegistry,
    allowed_tools: Vec<String>,
    skill: ParsedSkill,
    input_schema: serde_json::Value,
    output_schema: Option<serde_json::Value>,
}

impl SkillTool {
    /// Resolves the skill's schemas so they can be surfaced to the agent (input
    /// via the tool definition, output via [`Subagent::schema`]). Fails if a
    /// referenced schema file cannot be read. Schema *compilation* is deferred
    /// to the verification extension, so this does not validate the schemas.
    pub fn new(base_tools: ToolRegistry, skill: ParsedSkill) -> Result<Self, config::SkillError> {
        let (input_schema, output_schema) = skill.resolve_schema()?;
        let allowed_tools = skill.frontmatter.allowed_tools.clone();
        Ok(Self { base_tools, allowed_tools, skill, input_schema, output_schema })
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

    fn tools(&self) -> ToolRegistry {
        // Restrict the sub-agent to the skill's pre-approved tools.
        self.base_tools.subset(&self.allowed_tools)
    }

    fn seed(&self, params: &serde_json::Value, resume: bool) -> Result<Vec<ContentBlock>, ToolError> {
        // A fresh run gets the full instruction body; a resumed thread already has it
        // in-session, so only the new input is appended. The engine has already stripped
        // the orchestration `session_id`, so `params` is just the skill's input.
        let input_text = serde_json::to_string(params).unwrap_or_default();
        let reminder = format!(
            "Your final result MUST be returned by calling the yield tool, \
             with the result object as its arguments (conforming to the tool's parameter schema). \
             Do not emit the result as text."
        );
        Ok(if resume {
            vec![
                ContentBlock::Text(TextContent { content: input_text }),
                ContentBlock::Text(TextContent { content: reminder }),
            ]
        } else {
            vec![
                ContentBlock::Text(TextContent { content: self.skill.body.clone() }),
                ContentBlock::Text(TextContent { content: input_text }),
                ContentBlock::Text(TextContent { content: reminder }),
            ]
        })
    }
}

/// Discovers every valid skill under `config.dir` and registers each as a
/// subagent tool in `registry`.
///
/// `base_tools` is the pool the skills draw from. Pass a registry that does
/// **not** contain the skill tools themselves (e.g. one built before this call)
/// so a skill cannot recursively invoke itself. `manager` is the
/// [`SessionManager`] each skill forks/resumes through. A skill whose schemas
/// fail to resolve is skipped with a warning.
pub fn register(
    registry: &mut ToolRegistry,
    config: &SkillConfig,
    model: Model,
    provider: Arc<dyn Provider>,
    base_tools: ToolRegistry,
    manager: Arc<Mutex<dyn SessionManager>>,
) -> Result<(), config::SkillError> {
    for skill in config::discover(&config.dir)? {
        let label = skill.dir.display().to_string();
        let tool = match SkillTool::new(base_tools.clone(), skill) {
            Ok(t) => t,
            Err(e) => {
                eprintln!("[skill] skipping {label}: {e}");
                continue;
            }
        };
        subagent::register(registry, model.clone(), provider.clone(), manager.clone(), tool);
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::core::session::{InMemorySession, Session};
    use crate::core::tools::ToolHandler;
    use crate::core::types::{Message, ToolCall, ToolResultContent};
    use std::collections::HashMap;
    use std::path::PathBuf;
    use std::sync::atomic::{AtomicU64, Ordering};

    use crate::core::providers::{ProviderError, StreamResponse, StreamResponseStream};
    use crate::core::types::{TokenUsage, ToolDefinition};
    use futures::channel::oneshot;
    use futures::stream;

    /// A provider that replays a queue of scripted assistant messages, one per
    /// `stream()` call, and records the tool definitions it was handed.
    struct ScriptedProvider {
        turns: Arc<Mutex<std::collections::VecDeque<Message>>>,
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
        ) -> Result<crate::core::types::ToolResult, crate::core::tools::ToolError> {
            Ok(crate::core::types::ToolResult {
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

    fn manager() -> Arc<Mutex<dyn SessionManager>> {
        Arc::new(Mutex::new(crate::extra::agents::builder::SessionAdapter::new(InMemorySession::new(), || {
            InMemorySession::new().arc()
        })))
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

    fn tool_call(name: &str, arguments: serde_json::Value) -> Message {
        Message {
            role: "assistant".to_string(),
            content: vec![ContentBlock::ToolCall(ToolCall {
                id: "call".to_string(),
                name: name.to_string(),
                arguments,
            })],
        }
    }

    /// Drive a skill to completion via the engine and return (result, seen tool defs).
    async fn run(
        skill: ParsedSkill,
        turns: Vec<Message>,
        params: serde_json::Value,
    ) -> (crate::core::types::ToolResult, Vec<Vec<ToolDefinition>>) {
        let provider = Arc::new(ScriptedProvider::new(turns));
        let seen = provider.seen_tools.clone();
        let tool = crate::extra::tools::subagent::SubagentTool::new(
            model(),
            provider,
            manager(),
            SkillTool::new(base_tools(), skill).unwrap(),
        );
        let (_, rx) = oneshot::channel();
        let result = ToolHandler::execute(&tool, rx, params).await.unwrap();
        (result, seen.lock().unwrap().clone())
    }

    fn text_of(result: &crate::core::types::ToolResult) -> String {
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

    // --- tools() subset ---

    #[tokio::test]
    async fn tools_subset_filters_to_allowed() {
        let (_g, skill) = make_skill("s", Some("a c"), None);
        let tool = SkillTool::new(base_tools(), skill).unwrap();
        let mut names: Vec<String> = tool.tools().definitions().into_iter().map(|d| d.name).collect();
        names.sort();
        assert_eq!(names, vec!["a".to_string(), "c".to_string()]);
    }

    #[tokio::test]
    async fn tools_absent_allowed_is_empty() {
        let (_g, skill) = make_skill("s", None, None);
        let tool = SkillTool::new(base_tools(), skill).unwrap();
        assert!(tool.tools().definitions().is_empty(), "no allowed-tools ⇒ deny-all base");
    }

    // --- seed() ---

    #[tokio::test]
    async fn seed_fresh_includes_body_and_reminder() {
        let (_g, skill) = make_skill("s", None, None);
        let tool = SkillTool::new(base_tools(), skill).unwrap();
        let blocks = tool.seed(&serde_json::json!({"input": "go"}), false).unwrap();
        assert_eq!(blocks.len(), 3, "fresh seed = body + input + reminder");
        match &blocks[0] {
            ContentBlock::Text(t) => {
                assert!(t.content.contains("Do the thing"), "fresh seed starts with the instruction body")
            }
            _ => panic!("expected text"),
        }
        let reminder = match &blocks[2] {
            ContentBlock::Text(t) => &t.content,
            _ => panic!("expected text"),
        };
        assert!(reminder.contains("yield tool"), "seed reminds the model to use the yield tool");
    }

    #[tokio::test]
    async fn seed_resume_omits_body() {
        let (_g, skill) = make_skill("s", None, None);
        let tool = SkillTool::new(base_tools(), skill).unwrap();
        let blocks = tool.seed(&serde_json::json!({"input": "again"}), true).unwrap();
        assert_eq!(blocks.len(), 2, "resume seed = input + reminder only");
        // Neither block carries the instruction body.
        assert!(!blocks.iter().any(|b| matches!(b, ContentBlock::Text(t) if t.content.contains("Do the thing"))));
    }

    // --- schemas ---

    #[tokio::test]
    async fn input_schema_surfaces_in_definition() {
        let (_g, skill) = make_skill(
            "s",
            None,
            Some(r#"{"inputSchema":{"type":"object","properties":{"path":{"type":"string"}},"required":["path"]}}"#),
        );
        let tool = SkillTool::new(base_tools(), skill).unwrap();
        let params = tool.definition().parameters;
        assert_eq!(params["properties"]["path"]["type"], "string");
        assert_eq!(params["required"][0], "path");
    }

    const OUT_SCHEMA: &str =
        r#"{"outputSchema":{"type":"object","properties":{"x":{"type":"integer"}},"required":["x"]}}"#;

    #[tokio::test]
    async fn output_schema_surfaces() {
        let (_g, skill) = make_skill("s", None, Some(OUT_SCHEMA));
        let tool = SkillTool::new(base_tools(), skill).unwrap();
        let (_, output) = tool.schema();
        let output = output.expect("output schema is declared");
        assert_eq!(output["required"][0], "x");
    }

    // --- end-to-end through the engine ---

    #[tokio::test]
    async fn engine_drives_skill_and_captures_yield() {
        let (_g, skill) = make_skill("s", Some("a"), None);
        let (result, seen) = run(
            skill,
            vec![tool_call("akasha_subagent_yield", serde_json::json!({"result": "the answer"}))],
            serde_json::json!({"input": "go"}),
        )
        .await;
        assert!(!result.is_error);
        let parsed: serde_json::Value = serde_json::from_str(&text_of(&result)).expect("result is JSON");
        assert_eq!(parsed["result"], "the answer");
        assert!(parsed["session_id"].as_str().is_some(), "engine stamps a session_id");
        // The sub-agent was offered its allowed tool plus the injected yield tool.
        assert!(seen.iter().any(|defs| defs.iter().any(|d| d.name == "a")));
        assert!(seen.iter().any(|defs| defs.iter().any(|d| d.name == "akasha_subagent_yield")));
    }

    #[tokio::test]
    async fn engine_enforces_output_schema() {
        // First yield violates the output schema; the engine denies it and the sub-agent retries.
        let (_g, skill) = make_skill("s", None, Some(OUT_SCHEMA));
        let (result, _) = run(
            skill,
            vec![
                tool_call("akasha_subagent_yield", serde_json::json!({"x": "not-an-int"})),
                tool_call("akasha_subagent_yield", serde_json::json!({"x": 7})),
            ],
            serde_json::json!({"input": "go"}),
        )
        .await;
        assert!(!result.is_error);
        let parsed: serde_json::Value = serde_json::from_str(&text_of(&result)).expect("result is JSON");
        assert_eq!(parsed["x"], 7);
    }

    // --- register ---

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
        register(&mut registry, &cfg, model(), provider, base_tools(), manager()).unwrap();

        assert!(registry.get("alpha").is_some());
        assert_eq!(registry.get("alpha").unwrap().definition().name, "alpha");
        drop(guard);
    }
}

//! Fluent builder for an [`Agent`].
//!
//! [`AgentBuilder`] composes the pieces an [`Agent`] needs — model, provider,
//! session, tools, extensions, subagents, skills, and MCP servers — behind a
//! single declarative chain, so purpose-specific agents (and the inner agents
//! that subagents and skills drive) can be defined without hand-assembling the
//! [`AgentState`] triple and folding extensions with [`And`] by hand.
//!
//! # Tool access model
//!
//! The builder holds a single tool pool, filled by [`.tool()`](AgentBuilder::tool),
//! [`.tools()`](AgentBuilder::tools), [`.subagent()`](AgentBuilder::subagent),
//! [`.mcp()`](AgentBuilder::mcp), and [`.skills()`](AgentBuilder::skills):
//!
//! - **Skills** draw their base tool pool from the agent's own tools — the full
//!   pool, snapshotted at `build()` before the skills themselves are added (so a
//!   skill cannot recurse into itself or a sibling).
//! - **The main agent** sees only the tools it explicitly enables via
//!   [`.tools_enable()`](AgentBuilder::tools_enable). With no `tools_enable`
//!   call the agent can use *no* tools — it must opt each one in. This keeps the
//!   agent's reachable surface explicit while skills still see the whole pool.
//!
//! [`AgentBuilder::new`] wires **no** extensions: a bare
//! `AgentBuilder::new(model, provider, session).build().await` runs a
//! [`NoopExtension`]. For the cross-cutting combo every real agent uses —
//! [`SchemaVerification`] then [`CircuitBreaker`] — start from
//! [`AgentBuilder::base`] instead; it returns a partially-configured builder with
//! those two already chained, and further [`.extension()`](AgentBuilder::extension)
//! calls run after them.
//!
//! [`NoopExtension`]: crate::core::extensions::NoopExtension
//! [`And`]: crate::extra::extensions::combinator::And
//! [`SchemaVerification`]: crate::extra::extensions::schema::SchemaVerification
//! [`CircuitBreaker`]: crate::extra::extensions::circuit_breaker::CircuitBreaker

use std::sync::{Arc, Mutex};

use anyhow::Context;
use tokio::sync::mpsc;

use crate::core::agent::{Agent, AgentState};
use crate::core::extensions::{Extension, NoopExtension};
use crate::core::providers::{Model, Provider};
use crate::core::session::Session;
use crate::core::tools::{ToolHandler, ToolRegistry};
use crate::core::types::Message;
use crate::extra::agents::builder::SessionManager;
use crate::extra::extensions::combinator::And;
use crate::extra::extensions::io::{IoExtension, OutputEvent};
use crate::extra::tools::mcp;
use crate::extra::tools::mcp::config::{McpConfig, StreamableHttpConfig};
use crate::extra::tools::skill::{self, SkillConfig};
use crate::extra::tools::subagent::{Subagent, SubagentTool};

/// A fluent builder for an [`Agent`].
///
/// Methods consume and return `self`, so calls chain. Configuration accumulates
/// synchronously; [`.build().await`](AgentBuilder::build) does the only async
/// work (connecting MCP servers) and assembles the [`Agent`].
///
/// # Defaults
///
/// - **Main-agent tools**: none — opt tools in with
///   [`.tools_enable()`](AgentBuilder::tools_enable). Skills still see the full
///   pool regardless.
/// - **Extensions**: none — a bare `new().build()` runs a
///   [`NoopExtension`](crate::core::extensions::NoopExtension). Start from
///   [`AgentBuilder::base`] for the [`SchemaVerification`](crate::extra::extensions::schema::SchemaVerification)
///   + [`CircuitBreaker`](crate::extra::extensions::circuit_breaker::CircuitBreaker)
///   combo, then [`.extension()`](AgentBuilder::extension) appends after them.
///
/// # Example
///
/// ```ignore
/// use std::sync::Arc;
/// use akasha::core::providers::{Model, Provider};
/// use akasha::core::session::{InMemorySession, Session};
/// use akasha::extra::agents::builder::AgentBuilder;
///
/// async fn make(model: Model, provider: Arc<dyn Provider>) {
///     let session = InMemorySession::new().arc();
///     let agent = AgentBuilder::new(model, provider, session)
///         .tools_enable(["read_file", "search"])
///         .build()
///         .await
///         .unwrap();
/// }
/// ```
pub struct AgentBuilder {
    model: Model,
    provider: Arc<dyn Provider>,
    session: Arc<Mutex<dyn Session>>,
    extensions: Vec<Box<dyn Extension>>,
    tools: ToolRegistry,
    /// Tools the *main* agent may call (deny-all until opted in via
    /// `tools_enable`). Skills draw from the full pool, not this subset.
    enabled: Vec<String>,
    /// Deferred skill configs paired with the [`SessionManager`] they fork from;
    /// each dir's skills draw from the agent's full tool pool (snapshotted at
    /// build, before skill registration).
    skills: Vec<(Arc<Mutex<dyn SessionManager>>, SkillConfig)>,
    /// Deferred single-server configs; connected at `build()`.
    mcps: Vec<StreamableHttpConfig>,
    /// Deferred multi-server configs; flattened at `build()`.
    mcp_configs: Vec<McpConfig>,
}

impl AgentBuilder {
    pub fn new(model: Model, provider: Arc<dyn Provider>, session: Arc<Mutex<dyn Session>>) -> Self {
        Self {
            model,
            provider,
            session,
            extensions: Vec::new(),
            tools: ToolRegistry::new(),
            enabled: Vec::new(),
            skills: Vec::new(),
            mcps: Vec::new(),
            mcp_configs: Vec::new(),
        }
    }

    /// Append `ext` to the extension chain. Extensions run in addition order —
    /// the first added runs first and gates the rest (its `on_turn_end` can
    /// short-circuit later ones). To chain after the standard
    /// [`SchemaVerification`] + [`CircuitBreaker`] combo, start from
    /// [`AgentBuilder::base`]: a blocking extension added here (e.g. a mux
    /// fallback) then ends up innermost, matching `And(Schema, And(Circuit, ext))`.
    ///
    /// [`SchemaVerification`]: crate::extra::extensions::schema::SchemaVerification
    /// [`CircuitBreaker`]: crate::extra::extensions::circuit_breaker::CircuitBreaker
    pub fn extension(mut self, ext: impl Into<Box<dyn Extension>>) -> Self {
        self.extensions.push(ext.into());
        self
    }

    /// Replace the tool pool with `registry`. Tools added afterwards via
    /// [`.tool()`](AgentBuilder::tool) are appended to it. Calling this twice
    /// discards the pool set by the first call. Tools land in the pool
    /// unconditionally; the main agent still needs [`.tools_enable()`](AgentBuilder::tools_enable)
    /// to call them.
    pub fn tools(mut self, registry: ToolRegistry) -> Self {
        self.tools = registry;
        self
    }

    /// Add a single tool to the pool. Repeatable. The main agent can only call
    /// it once [`.tools_enable()`](AgentBuilder::tools_enable) names it;
    /// skills see it as base regardless.
    pub fn tool(mut self, handler: impl Into<Box<dyn ToolHandler>>) -> Self {
        self.tools.register(handler.into());
        self
    }

    /// Enable `names` for the **main** agent. Only enabled tools are callable by
    /// the agent the builder produces; every other tool in the pool is hidden
    /// from it (but still available to skills as base). With no `tools_enable`
    /// call the agent can use no tools (deny-all). Repeatable: calls accumulate.
    pub fn tools_enable<I, S>(mut self, names: I) -> Self
    where
        I: IntoIterator<Item = S>,
        S: Into<String>,
    {
        self.enabled.extend(names.into_iter().map(Into::into));
        self
    }

    /// Wrap a [`Subagent`] as a tool in the pool, driven by the
    /// [`SubagentTool`](crate::extra::tools::subagent::SubagentTool) engine. Each
    /// invocation forks `manager` (or resumes a fork by id), so the subagent
    /// inherits that conversation then runs isolated. The main agent must
    /// [enable](AgentBuilder::tools_enable) it to call it.
    pub fn subagent<S: Subagent + 'static>(mut self, manager: Arc<Mutex<dyn SessionManager>>, subagent: S) -> Self {
        let tool = SubagentTool::new(self.model.clone(), self.provider.clone(), manager, subagent);
        self.tools.register(Box::new(tool));
        self
    }

    /// Discover every valid skill under `config.dir` and register each as a
    /// subagent tool in the pool. Skills fork from `manager` and draw their base
    /// tool pool from the agent's own tools — every tool registered so far,
    /// snapshotted at `build()` before the skills themselves are added (so a
    /// skill cannot recursively invoke itself or a sibling). Discovery happens
    /// at `build()`.
    pub fn skills(mut self, manager: Arc<Mutex<dyn SessionManager>>, config: SkillConfig) -> Self {
        self.skills.push((manager, config));
        self
    }

    /// Queue one MCP server (Streamable HTTP) for connection at `build()`.
    /// Its tools join the pool; enable them for the main agent with
    /// [`.tools_enable()`](AgentBuilder::tools_enable).
    pub fn mcp(mut self, server: StreamableHttpConfig) -> Self {
        self.mcps.push(server);
        self
    }

    /// Queue every server in a parsed [`McpConfig`] for connection at
    /// `build()`. An unsupported entry surfaces as an error from `build()`.
    pub fn mcp_servers(mut self, config: McpConfig) -> Self {
        self.mcp_configs.push(config);
        self
    }

    /// Materialize the [`Agent`]: connect MCP servers, discover skills, narrow
    /// the main agent's tools to its enabled subset, compose the extension
    /// chain, and assemble the state. MCP and skill failures surface here as
    /// errors.
    pub async fn build(mut self) -> anyhow::Result<Agent> {
        // MCP servers — a connection failure aborts the build before any agent runs.
        for server in std::mem::take(&mut self.mcps) {
            mcp::register(&mut self.tools, &server)
                .await
                .with_context(|| format!("connecting MCP server '{}'", server.url))?;
        }
        for config in std::mem::take(&mut self.mcp_configs) {
            for (name, entry) in config.mcp_servers {
                let server = entry.into_config().with_context(|| format!("parsing MCP server entry '{name}'"))?;
                mcp::register(&mut self.tools, &server)
                    .await
                    .with_context(|| format!("connecting MCP server '{name}' ({})", server.url))?;
            }
        }

        // Skills draw from the agent's *full* pool. Snapshot it before the skill
        // tools themselves are registered so no skill can recurse into itself
        // (or a sibling). `tools_enable` does not narrow this base — only the
        // main agent's own view, applied below.
        let base = self.tools.clone();
        for (manager, config) in std::mem::take(&mut self.skills) {
            skill::register(&mut self.tools, &config, self.model.clone(), self.provider.clone(), base.clone(), manager)
                .with_context(|| format!("registering skills under '{}'", config.dir.display()))?;
        }

        // The main agent sees only its explicitly enabled tools. With no
        // `tools_enable` call this is the empty set (deny-all); skills already
        // captured the full pool above.
        self.tools = self.tools.subset(&self.enabled);

        let session = self.session.clone();
        let extension = compose(self.extensions);

        Ok(Agent {
            state: AgentState { model: self.model, tools: self.tools, session },
            provider: self.provider,
            extension,
        })
    }

    /// Wire an [`IoExtension`] into the chain and return the transport channels
    /// as `(rx, tx)`:
    ///
    /// - `rx` — outbound [`OutputEvent`]s (streamed content, tool notices, turn
    ///   finishes) the transport renders.
    /// - `tx` — the inbound [`Message`] sender the transport feeds to drive the
    ///   next turn.
    ///
    /// The extension is appended like any other (via
    /// [`.extension()`](AgentBuilder::extension)), so call `bind_io` **last** —
    /// after every other extension — so io's turn-end input gating runs after
    /// them. Returns the builder (for further chaining / `.build()`) alongside
    /// the channels. The bridge owns no task; the caller spawns
    /// `agent.prompt(first)` itself at `build()` time.
    ///
    /// [`IoExtension`]: crate::extra::extensions::io::IoExtension
    /// [`OutputEvent`]: crate::extra::extensions::io::OutputEvent
    /// [`Message`]: crate::core::types::Message
    pub fn bind_io(self) -> (Self, mpsc::UnboundedReceiver<OutputEvent>, mpsc::UnboundedSender<Message>) {
        let (io, tx, rx) = IoExtension::new();
        (self.extension(io), rx, tx)
    }
}

/// Fold a chain of extensions into a single [`Box<dyn Extension>`].
///
/// Built left-to-right with [`And`]: `compose([a, b, c]) == And(a, And(b, c))`,
/// so `a` runs first and gates the rest (its `on_turn_end` can short-circuit
/// `b`/`c`). An empty chain yields a [`NoopExtension`].
///
/// [`And`]: crate::extra::extensions::combinator::And
fn compose(mut extensions: Vec<Box<dyn Extension>>) -> Box<dyn Extension> {
    match extensions.len() {
        0 => Box::new(NoopExtension),
        1 => extensions.remove(0),
        _ => {
            let mut acc = extensions.remove(0);
            for next in extensions {
                acc = And::new(acc, next).into();
            }
            acc
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    use std::collections::HashMap;
    use std::path::PathBuf;
    use std::sync::atomic::{AtomicU64, Ordering};
    use std::time::Duration;

    use crate::core::session::{InMemorySession, Session};
    use crate::extra::agents::builder::SessionAdapter;
    use crate::extra::sessions::sqlite::SqliteSession;

    use async_trait::async_trait;
    use futures::stream;
    use tokio::time::timeout;

    use crate::core::providers::{ProviderError, StreamResponse, StreamResponseStream};
    use crate::core::tools::ToolError;
    use crate::core::types::{
        ContentBlock, Message, TextContent, TokenUsage, ToolDefinition, ToolResult, ToolResultContent,
    };

    static COUNTER: AtomicU64 = AtomicU64::new(0);

    /// A provider that streams exactly one scripted assistant message per
    /// `stream()` call and records the tool definitions it was handed — enough
    /// to drive one turn of `agent_loop` (or one skill sub-agent run) without an
    /// LLM, and to prove which tools were offered.
    struct OneShotProvider {
        message: Message,
        seen_tools: Arc<Mutex<Vec<Vec<ToolDefinition>>>>,
    }

    impl OneShotProvider {
        fn new(message: Message) -> Self {
            Self { message, seen_tools: Arc::new(Mutex::new(Vec::new())) }
        }
    }

    #[async_trait]
    impl crate::core::providers::Provider for OneShotProvider {
        async fn stream<'a>(
            &self,
            _model: &Model,
            _messages: Box<dyn Iterator<Item = &'a Message> + Send + 'a>,
            tools: &Vec<ToolDefinition>,
        ) -> Result<StreamResponseStream, ProviderError> {
            self.seen_tools.lock().unwrap().push(tools.clone());
            let resp = StreamResponse {
                message: self.message.clone(),
                usage: TokenUsage {
                    input_tokens: 0,
                    output_tokens: 0,
                    cache_read_tokens: None,
                    cache_write_tokens: None,
                },
                stop_reason: Some(
                    if self.message.content.iter().any(|b| matches!(b, ContentBlock::ToolCall(_))) {
                        "tool_calls"
                    } else {
                        "stop"
                    }
                    .to_string(),
                ),
            };
            Ok(Box::pin(stream::iter(vec![resp])))
        }

        fn name(&self) -> &str {
            "oneshot"
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
            _cancel: futures::channel::oneshot::Receiver<bool>,
            _params: serde_json::Value,
        ) -> Result<ToolResult, ToolError> {
            Ok(ToolResult {
                tool_call_id: None,
                content: vec![ToolResultContent::Text(TextContent { content: "ok".to_string() })],
                is_error: false,
            })
        }
    }

    /// An extension that reports a fixed name — enough to inspect composition.
    struct NamedExt {
        name: &'static str,
    }

    #[async_trait]
    impl Extension for NamedExt {
        fn name(&self) -> &str {
            self.name
        }
    }

    /// A subagent stub; its only job is to be wrappable as a named tool.
    struct EchoSubagent {
        definition: ToolDefinition,
    }

    #[async_trait]
    impl Subagent for EchoSubagent {
        fn definition(&self) -> ToolDefinition {
            self.definition.clone()
        }
        fn seed(&self, _params: &serde_json::Value, _resume: bool) -> Result<Vec<ContentBlock>, ToolError> {
            Ok(vec![ContentBlock::Text(TextContent { content: "go".to_string() })])
        }
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

    fn provider(msg: &str) -> Arc<dyn Provider> {
        Arc::new(OneShotProvider::new(assistant_text(msg)))
    }

    /// A fresh plain session — the main session most tests hand to `new`.
    fn session() -> Arc<Mutex<dyn Session>> {
        InMemorySession::new().arc()
    }

    /// A fresh `SessionAdapter` manager — the fork source for `subagent`/`skills`.
    fn manager() -> Arc<Mutex<dyn SessionManager>> {
        Arc::new(Mutex::new(SessionAdapter::new(InMemorySession::new(), || InMemorySession::new().arc())))
    }

    fn assistant_text(text: &str) -> Message {
        Message {
            role: "assistant".to_string(),
            content: vec![ContentBlock::Text(TextContent { content: text.to_string() })],
        }
    }

    fn def(name: &str) -> ToolDefinition {
        ToolDefinition {
            name: name.to_string(),
            description: "stub".to_string(),
            parameters: serde_json::json!({ "type": "object" }),
        }
    }

    /// Removes the wrapped directory tree on drop (best-effort).
    struct TempDir(PathBuf);
    impl Drop for TempDir {
        fn drop(&mut self) {
            let _ = std::fs::remove_dir_all(&self.0);
        }
    }

    /// Writes a single valid skill `name` under a fresh temp dir, returning the
    /// guard and a [`SkillConfig`] pointing at it.
    fn skill_dir(name: &str) -> (TempDir, SkillConfig) {
        skill_dir_with_allowed(name, None)
    }

    /// Like [`skill_dir`] but also sets the skill's `allowed-tools` frontmatter,
    /// restricting the sub-agent to the named base tools.
    fn skill_dir_with_allowed(name: &str, allowed: Option<&str>) -> (TempDir, SkillConfig) {
        let n = COUNTER.fetch_add(1, Ordering::SeqCst);
        let root = std::env::temp_dir().join(format!("akasha-builder-{}-{n}", std::process::id()));
        let dir = root.join(name);
        std::fs::create_dir_all(&dir).unwrap();
        let mut src = format!("---\nname: {name}\ndescription: d\n");
        if let Some(a) = allowed {
            src.push_str(&format!("allowed-tools: \"{a}\"\n"));
        }
        src.push_str("---\n## Instructions\ndo the thing.\n");
        std::fs::write(dir.join("SKILL.md"), &src).unwrap();
        (TempDir(root.clone()), SkillConfig { dir: root })
    }

    #[tokio::test]
    async fn minimal_build_is_bare() {
        let agent = AgentBuilder::new(model(), provider("hi"), session()).build().await.unwrap();
        // new() wires no extensions → NoopExtension; no tools enabled; model threaded through.
        assert_eq!(agent.extension.name(), "noop");
        assert!(agent.state.tools.definitions().is_empty(), "deny-all: nothing enabled");
        assert_eq!(agent.state.model.id, "m");
    }

    #[tokio::test]
    async fn single_extension_used_directly() {
        // A single opted-in extension needs no And wrapper.
        let agent = AgentBuilder::new(model(), provider("hi"), session())
            .extension(NamedExt { name: "custom" })
            .build()
            .await
            .unwrap();
        assert_eq!(agent.extension.name(), "custom");
    }

    #[tokio::test]
    async fn multiple_extensions_fold_via_and() {
        // Two or more extensions fold left-to-right via And.
        let agent = AgentBuilder::new(model(), provider("hi"), session())
            .extension(NamedExt { name: "a" })
            .extension(NamedExt { name: "b" })
            .build()
            .await
            .unwrap();
        assert_eq!(agent.extension.name(), "and");
    }

    #[tokio::test]
    async fn main_agent_denies_all_tools_by_default() {
        // Tools registered but never enabled => the main agent sees none.
        let agent = AgentBuilder::new(model(), provider("hi"), session())
            .tool(StubTool { name: "a" })
            .tool(StubTool { name: "b" })
            .build()
            .await
            .unwrap();
        assert!(agent.state.tools.definitions().is_empty(), "no tools_enable => main agent has no tools");
    }

    #[tokio::test]
    async fn tools_enable_exposes_only_named_tools() {
        let agent = AgentBuilder::new(model(), provider("hi"), session())
            .tool(StubTool { name: "a" })
            .tool(StubTool { name: "b" })
            .tool(StubTool { name: "c" })
            .tools_enable(["a", "c"])
            .build()
            .await
            .unwrap();
        let mut names: Vec<String> = agent.state.tools.definitions().into_iter().map(|d| d.name).collect();
        names.sort();
        assert_eq!(names, vec!["a".to_string(), "c".to_string()]);
    }

    #[tokio::test]
    async fn tools_enable_accumulates_across_calls() {
        let agent = AgentBuilder::new(model(), provider("hi"), session())
            .tool(StubTool { name: "a" })
            .tool(StubTool { name: "b" })
            .tools_enable(["a"])
            .tools_enable(["b"])
            .build()
            .await
            .unwrap();
        let mut names: Vec<String> = agent.state.tools.definitions().into_iter().map(|d| d.name).collect();
        names.sort();
        assert_eq!(names, vec!["a".to_string(), "b".to_string()]);
    }

    #[tokio::test]
    async fn tools_enable_with_seeded_registry() {
        let mut seed = ToolRegistry::new();
        seed.register(StubTool { name: "a" }.into());
        let agent = AgentBuilder::new(model(), provider("hi"), session())
            .tools(seed)
            .tool(StubTool { name: "b" })
            .tools_enable(["a", "b"])
            .build()
            .await
            .unwrap();
        let mut names: Vec<String> = agent.state.tools.definitions().into_iter().map(|d| d.name).collect();
        names.sort();
        assert_eq!(names, vec!["a".to_string(), "b".to_string()]);
    }

    #[tokio::test]
    async fn subagent_registers_but_needs_enabling() {
        // Registered into the pool, but invisible to the main agent until enabled.
        let agent = AgentBuilder::new(model(), provider("hi"), session())
            .subagent(manager(), EchoSubagent { definition: def("echo") })
            .build()
            .await
            .unwrap();
        assert!(agent.state.tools.get("echo").is_none(), "not enabled => hidden");

        let agent = AgentBuilder::new(model(), provider("hi"), session())
            .subagent(manager(), EchoSubagent { definition: def("echo") })
            .tools_enable(["echo"])
            .build()
            .await
            .unwrap();
        assert!(agent.state.tools.get("echo").is_some(), "enabled => callable");
    }

    #[tokio::test]
    async fn skills_register_and_are_callable_when_enabled() {
        let (guard, cfg) = skill_dir("alpha");
        let agent = AgentBuilder::new(model(), provider("hi"), session())
            .skills(manager(), cfg)
            .tools_enable(["alpha"])
            .build()
            .await
            .unwrap();
        assert!(agent.state.tools.get("alpha").is_some(), "enabled skill is callable");
        drop(guard);
    }

    #[tokio::test]
    async fn session_override_is_honored() {
        // Any Session plugs in directly — here a SqliteSession. Provided to `new`.
        let mut main = SqliteSession::new(":memory:", "main").unwrap();
        main.append(Message {
            role: "system".to_string(),
            content: vec![ContentBlock::Text(TextContent { content: "seed".to_string() })],
        })
        .unwrap();
        let main = main.arc();

        let agent = AgentBuilder::new(model(), provider("hi"), main).build().await.unwrap();
        let count = agent.state.session.lock().unwrap().messages().count();
        assert_eq!(count, 1, "the seeded session is preserved, not replaced");
    }

    #[tokio::test]
    async fn unsupported_mcp_entry_surfaces_build_error() {
        // An unknown entry fails `into_config()` deterministically (no network).
        let cfg: McpConfig = serde_json::from_str(r#"{"mcpServers":{"s":{"foo":"bar"}}}"#).unwrap();
        let err = AgentBuilder::new(model(), provider("hi"), session())
            .mcp_servers(cfg)
            .build()
            .await
            .err()
            .expect("build should fail on an unsupported entry");
        assert!(err.to_string().contains("MCP"), "error should name the server: {err}");
    }

    #[tokio::test]
    async fn unreachable_mcp_server_surfaces_build_error() {
        // Port 1 is unbound → connection refused → the build fails rather than hanging.
        let bad = StreamableHttpConfig {
            url: "http://127.0.0.1:1/mcp".to_string(),
            headers: HashMap::new(),
            allow: vec![],
            deny: vec![],
        };
        let err = AgentBuilder::new(model(), provider("hi"), session())
            .mcp(bad)
            .build()
            .await
            .err()
            .expect("build should fail on an unreachable server");
        assert!(err.to_string().contains("MCP"), "error should name the server: {err}");
    }

    #[tokio::test]
    async fn build_then_prompt_runs_one_turn() {
        let mut agent = AgentBuilder::new(model(), provider("hello there"), session()).build().await.unwrap();
        agent
            .prompt(Message {
                role: "user".to_string(),
                content: vec![ContentBlock::Text(TextContent { content: "hi".to_string() })],
            })
            .await
            .unwrap();

        // A turn that produced assistant text ends with an assistant message.
        let last_role = agent.state.session.lock().unwrap().messages().last().map(|m| m.role.clone());
        assert_eq!(last_role.as_deref(), Some("assistant"));
    }

    /// Receive the next outbound event, failing (with a timeout) if the bridge stalls.
    async fn next_io_event(rx: &mut mpsc::UnboundedReceiver<OutputEvent>) -> OutputEvent {
        timeout(Duration::from_secs(2), rx.recv()).await.expect("event within timeout").expect("event channel open")
    }

    fn user_msg(text: &str) -> Message {
        Message {
            role: "user".to_string(),
            content: vec![ContentBlock::Text(TextContent { content: text.to_string() })],
        }
    }

    /// `bind_io` appends an `IoExtension` (as a regular extension) and returns
    /// `(builder, rx, tx)`; `build()` then wires it as the agent's extension.
    /// Driving the agent streams output through `rx`, a `Finish` handshake gates
    /// the turn, and a message on `tx` advances to the next turn.
    #[tokio::test]
    async fn bind_io_returns_drivable_agent_and_channels() {
        let (builder, mut rx, tx) = AgentBuilder::new(model(), provider("reply"), session()).bind_io();
        let mut agent = builder.build().await.unwrap();

        // io is the sole extension → used directly.
        assert_eq!(agent.extension.name(), "io");

        // The caller starts the agent task itself; the bridge carries the first prompt.
        let task = tokio::spawn(async move { agent.prompt(user_msg("hello")).await });

        // Turn 1: the streamed reply, then a finish handshake.
        match next_io_event(&mut rx).await {
            OutputEvent::Append(ContentBlock::Text(t)) => assert_eq!(t.content, "reply"),
            _ => panic!("expected Append(Text)"),
        }
        let ack = match next_io_event(&mut rx).await {
            OutputEvent::Finish(ack) => ack,
            _ => panic!("expected Finish after the turn"),
        };
        ack.send(Ok(())).unwrap();

        // Turn 2: a message on tx drives a fresh turn — same contract repeats.
        tx.send(user_msg("again")).unwrap();
        match next_io_event(&mut rx).await {
            OutputEvent::Append(ContentBlock::Text(t)) => assert_eq!(t.content, "reply"),
            _ => panic!("expected Append(Text) on turn 2"),
        }
        assert!(matches!(next_io_event(&mut rx).await, OutputEvent::Finish(_)), "turn 2 also finishes");

        task.abort();
    }
}

use std::sync::{Arc, Mutex};

use async_trait::async_trait;
use futures::channel::oneshot;

use crate::core::session::Session;
use crate::core::tools::{ToolError, ToolHandler, ToolRegistry};
use crate::core::types::{ToolDefinition, ToolResult};

/// Builds a fresh, isolated [`Session`] for a single subagent invocation.
///
/// The result is `Arc<Mutex<dyn Session>>` — the same shape an [`Agent`]
/// consumes — so the subagent can drive it directly. The [`Session::arc`]
/// helper turns any concrete session into this form, e.g.
/// `|| Ok(InMemorySession::new().arc())`.
pub type SessionFactory = Arc<dyn Fn() -> anyhow::Result<Arc<Mutex<dyn Session>>> + Send + Sync + 'static>;

/// A subagent exposed to a parent agent as a single tool.
///
/// Unlike a plain [`ToolHandler`], a subagent receives the [`Session`] it
/// should run in. The parent agent supplies that session (through a
/// [`SessionFactory`] on [`SubagentTool`]) so the subagent gets a private
/// conversation context, isolated from the parent's own session.
#[async_trait]
pub trait Subagent: Send + Sync {
    /// Tool definition surfaced to the parent agent's model.
    fn definition(&self) -> ToolDefinition;

    /// Run the subagent to completion inside `session`.
    ///
    /// `params` are the tool arguments the parent's model supplied; `cancel`
    /// notifies the subagent when the parent aborts the call.
    async fn execute(
        &self,
        session: Arc<Mutex<dyn Session>>,
        cancel: oneshot::Receiver<bool>,
        params: serde_json::Value,
    ) -> Result<ToolResult, ToolError>;

    /// Wrap this subagent as a standard [`ToolHandler`], creating a fresh
    /// session per invocation via `session_factory`. This is how a subagent is
    /// turned into a tool the parent agent can call.
    fn tool<F>(self, session_factory: F) -> Box<dyn ToolHandler>
    where
        Self: Sized + 'static,
        F: Fn() -> anyhow::Result<Arc<Mutex<dyn Session>>> + Send + Sync + 'static,
    {
        Box::new(SubagentTool::new(self, session_factory))
    }
}

/// Adapts a [`Subagent`] into a [`ToolHandler`].
///
/// On every invocation it calls the [`SessionFactory`] to obtain a fresh
/// session, then delegates to the wrapped subagent. This is the bridge that
/// lets a subagent live alongside ordinary tools in a
/// [`ToolRegistry`](crate::core::tools::ToolRegistry).
pub struct SubagentTool {
    subagent: Arc<dyn Subagent>,
    session_factory: SessionFactory,
}

impl SubagentTool {
    pub fn new<S, F>(subagent: S, session_factory: F) -> Self
    where
        S: Subagent + 'static,
        F: Fn() -> anyhow::Result<Arc<Mutex<dyn Session>>> + Send + Sync + 'static,
    {
        Self { subagent: Arc::new(subagent), session_factory: Arc::new(session_factory) }
    }
}

#[async_trait]
impl ToolHandler for SubagentTool {
    fn definition(&self) -> ToolDefinition {
        self.subagent.definition()
    }

    async fn execute(
        &self,
        cancel: oneshot::Receiver<bool>,
        params: serde_json::Value,
    ) -> Result<ToolResult, ToolError> {
        let session = (self.session_factory)()
            .map_err(|e| ToolError::Execution(format!("failed to create subagent session: {e}")))?;
        self.subagent.execute(session, cancel, params).await
    }
}

/// Convenience: build and register a subagent tool in one step. Equivalent to
/// `registry.register(subagent.tool(session_factory))`.
pub fn register<S, F>(registry: &mut ToolRegistry, subagent: S, session_factory: F)
where
    S: Subagent + 'static,
    F: Fn() -> anyhow::Result<Arc<Mutex<dyn Session>>> + Send + Sync + 'static,
{
    registry.register(subagent.tool(session_factory));
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::core::session::InMemorySession;
    use crate::core::types::{TextContent, ToolResultContent};

    /// A stub subagent that records its inputs and reports whether the session
    /// it was handed was fresh, so the plumbing can be tested without an LLM.
    struct EchoSubagent {
        definition: ToolDefinition,
        seen: Arc<Mutex<Vec<serde_json::Value>>>,
    }

    #[async_trait]
    impl Subagent for EchoSubagent {
        fn definition(&self) -> ToolDefinition {
            self.definition.clone()
        }

        async fn execute(
            &self,
            session: Arc<Mutex<dyn Session>>,
            _cancel: oneshot::Receiver<bool>,
            params: serde_json::Value,
        ) -> Result<ToolResult, ToolError> {
            self.seen.lock().unwrap().push(params.clone());
            let prior = session.lock().unwrap().messages().count();
            Ok(ToolResult {
                tool_call_id: None,
                content: vec![ToolResultContent::Text(TextContent {
                    content: format!("params={params}; prior messages={prior}"),
                })],
                is_error: false,
            })
        }
    }

    fn def(name: &str) -> ToolDefinition {
        ToolDefinition {
            name: name.to_string(),
            description: "stub".to_string(),
            parameters: serde_json::json!({ "type": "object" }),
        }
    }

    fn echo() -> (EchoSubagent, Arc<Mutex<Vec<serde_json::Value>>>) {
        let seen = Arc::new(Mutex::new(Vec::new()));
        (EchoSubagent { definition: def("echo"), seen: seen.clone() }, seen)
    }

    fn text_of(result: &ToolResult) -> &str {
        match &result.content[0] {
            ToolResultContent::Text(t) => &t.content,
            _ => panic!("expected text content"),
        }
    }

    #[tokio::test]
    async fn forwards_definition_and_delegates_params() {
        let (sub, seen) = echo();

        let created = Arc::new(Mutex::new(0u32));
        let c = created.clone();
        let tool = sub.tool(move || {
            *c.lock().unwrap() += 1;
            Ok(InMemorySession::new().arc())
        });

        // The adapter forwards the subagent's definition unchanged.
        assert_eq!(tool.definition().name, "echo");

        let (_, rx) = oneshot::channel();
        let result = tool.execute(rx, serde_json::json!({ "k": 1 })).await.unwrap();
        assert!(!result.is_error);
        assert_eq!(seen.lock().unwrap().len(), 1, "subagent execute called once");
        assert_eq!(*created.lock().unwrap(), 1, "session factory called once");
        assert!(text_of(&result).contains("prior messages=0"));
    }

    #[tokio::test]
    async fn creates_fresh_session_each_call() {
        let (sub, _) = echo();
        let tool = sub.tool(|| Ok(InMemorySession::new().arc()));

        for _ in 0..3 {
            let (_, rx) = oneshot::channel();
            let result = tool.execute(rx, serde_json::json!({})).await.unwrap();
            // Every invocation must start from an empty session.
            assert!(text_of(&result).contains("prior messages=0"));
        }
    }

    #[tokio::test]
    async fn propagates_factory_error_as_execution() {
        let (sub, _) = echo();
        let tool =
            sub.tool(|| -> anyhow::Result<Arc<Mutex<dyn Session>>> { anyhow::bail!("session backend unavailable") });

        let (_, rx) = oneshot::channel();
        let err = tool.execute(rx, serde_json::json!({})).await.unwrap_err();
        assert!(matches!(err, ToolError::Execution(_)), "factory failure should surface as Execution, got {err:?}");
    }

    #[test]
    fn registers_as_a_standard_handler() {
        let (sub, _) = echo();
        let mut registry = ToolRegistry::new();
        register(&mut registry, sub, || Ok(InMemorySession::new().arc()));

        assert!(registry.get("echo").is_some(), "subagent should be addressable by name");
        assert_eq!(registry.definitions().len(), 1);
    }
}

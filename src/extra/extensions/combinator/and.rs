use async_trait::async_trait;

use crate::core::agent::AgentState;
use crate::core::extensions::{Extension, ExtensionError, ToolCallDecision};
use crate::core::providers::StreamResponse;
use crate::core::tools::ToolError;
use crate::core::types::{Message, ToolResult};

/// Runs both extensions sequentially. Both must succeed.
/// For state-transforming hooks, output of A feeds into B.
/// For tool decisions, short-circuits on the first `Deny`.
pub struct And {
    a: Box<dyn Extension>,
    b: Box<dyn Extension>,
}

impl And {
    pub fn new<A: Into<Box<dyn Extension>>, B: Into<Box<dyn Extension>>>(a: A, b: B) -> Self {
        Self { a: a.into(), b: b.into() }
    }
}

#[async_trait]
impl Extension for And {
    fn name(&self) -> &str {
        "and"
    }

    async fn on_agent_start(&mut self, state: AgentState) -> Result<AgentState, ExtensionError> {
        let state = self.a.on_agent_start(state).await?;
        self.b.on_agent_start(state).await
    }

    async fn on_agent_end(&mut self, state: AgentState) -> Result<AgentState, ExtensionError> {
        let state = self.a.on_agent_end(state).await?;
        self.b.on_agent_end(state).await
    }

    async fn on_turn_start(&mut self, state: AgentState) -> Result<AgentState, ExtensionError> {
        let state = self.a.on_turn_start(state).await?;
        self.b.on_turn_start(state).await
    }

    async fn on_turn_end(&mut self, state: AgentState) -> Result<AgentState, ExtensionError> {
        let state = self.a.on_turn_end(state).await?;
        if state.session.lock().unwrap().messages().last().map_or(false, |msg| msg.role != "assistant") {
            return Ok(state);
        }
        self.b.on_turn_end(state).await
    }

    async fn on_message_start(&mut self, messages: Vec<Message>) -> Result<Vec<Message>, ExtensionError> {
        let messages = self.a.on_message_start(messages).await?;
        self.b.on_message_start(messages).await
    }

    async fn on_message_update(&mut self, resp: &StreamResponse) -> Result<(), ExtensionError> {
        self.a.on_message_update(resp).await?;
        self.b.on_message_update(resp).await
    }

    async fn on_message_end(&mut self, resp: &StreamResponse) -> Result<(), ExtensionError> {
        self.a.on_message_end(resp).await?;
        self.b.on_message_end(resp).await
    }

    async fn on_tool_execution_start(
        &mut self,
        tool_call_id: &str,
        name: &str,
        args: &serde_json::Value,
    ) -> Result<ToolCallDecision, ExtensionError> {
        match self.a.on_tool_execution_start(tool_call_id, name, args).await? {
            ToolCallDecision::Allow => self.b.on_tool_execution_start(tool_call_id, name, args).await,
            deny => Ok(deny),
        }
    }

    async fn tool_execution_end(
        &mut self,
        tool_call_id: &str,
        result: Result<ToolResult, ToolError>,
    ) -> Result<Result<ToolResult, ToolError>, ExtensionError> {
        let result = self.a.tool_execution_end(tool_call_id, result).await?;
        self.b.tool_execution_end(tool_call_id, result).await
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::core::extensions::NoopExtension;
    use crate::core::providers::Model;
    use crate::core::session::{InMemorySession, Session};
    use crate::core::tools::ToolRegistry;
    use crate::core::types::{ContentBlock, TextContent};
    use std::collections::HashMap;
    use std::sync::atomic::{AtomicBool, Ordering};
    use std::sync::{Arc, Mutex};

    use super::super::test_helpers::*;

    /// Appends a message of `role` to the session in `on_turn_end`, simulating an extension
    /// (like the mux fallback) that leaves the turn ending in a non-assistant message.
    struct TurnEndAppend {
        role: &'static str,
    }

    #[async_trait]
    impl Extension for TurnEndAppend {
        fn name(&self) -> &str {
            "append"
        }

        async fn on_turn_end(&mut self, state: AgentState) -> Result<AgentState, ExtensionError> {
            state
                .session
                .lock()
                .unwrap()
                .append(Message {
                    role: self.role.to_string(),
                    content: vec![ContentBlock::Text(TextContent { content: self.role.into() })],
                })
                .map_err(|e| ExtensionError::ExtensionFailed { name: "append".into(), message: e.to_string() })?;
            Ok(state)
        }
    }

    /// Records whether its `on_turn_end` ran, standing in for an extension that blocks for the
    /// next user input (such as telegram).
    struct TurnEndRecorder {
        called: Arc<AtomicBool>,
    }

    #[async_trait]
    impl Extension for TurnEndRecorder {
        fn name(&self) -> &str {
            "recorder"
        }

        async fn on_turn_end(&mut self, state: AgentState) -> Result<AgentState, ExtensionError> {
            self.called.store(true, Ordering::SeqCst);
            Ok(state)
        }
    }

    /// Builds an `AgentState` over a session pre-seeded with a single assistant message, so the
    /// turn is in the same state `agent_loop` leaves it before `on_turn_end` runs.
    fn turn_end_state() -> (AgentState, Arc<Mutex<dyn crate::core::session::Session>>) {
        let mut session = InMemorySession::new();
        session
            .append(Message {
                role: "assistant".into(),
                content: vec![ContentBlock::Text(TextContent { content: "response".into() })],
            })
            .unwrap();
        let session: Arc<Mutex<dyn crate::core::session::Session>> = Arc::new(Mutex::new(session));
        let state = AgentState {
            model: Model {
                id: "m".into(),
                provider: "p".into(),
                context_window: 0,
                base_url: String::new(),
                headers: HashMap::new(),
            },
            tools: ToolRegistry::new(),
            session: session.clone(),
        };
        (state, session)
    }

    #[tokio::test]
    async fn test_and_name() {
        let ext = And::new(NoopExtension, NoopExtension);
        assert_eq!(ext.name(), "and");
    }

    #[tokio::test]
    async fn test_and_on_message_start_chains() {
        let mut ext = And::new(LabelExt::ok("a"), LabelExt::ok("b"));
        let messages = make_session("");
        let result = ext.on_message_start(messages).await.unwrap();
        assert_eq!(result.len(), 2);
        match &result[0].content[0] {
            ContentBlock::Text(t) => assert_eq!(t.content, "a"),
            _ => panic!("expected text"),
        }
        match &result[1].content[0] {
            ContentBlock::Text(t) => assert_eq!(t.content, "b"),
            _ => panic!("expected text"),
        }
    }

    #[tokio::test]
    async fn test_and_on_message_start_fails_on_first() {
        let mut ext = And::new(LabelExt::fail("a"), LabelExt::ok("b"));
        let messages = make_session("hello");
        match ext.on_message_start(messages).await {
            Err(ExtensionError::ExtensionFailed { name, .. }) => assert_eq!(name, "a"),
            Ok(_) => panic!("expected error"),
            Err(other) => panic!("expected ExtensionFailed, got {other:?}"),
        }
    }

    #[tokio::test]
    async fn test_and_on_message_start_fails_on_second() {
        let mut ext = And::new(LabelExt::ok("a"), LabelExt::fail("b"));
        let messages = make_session("");
        match ext.on_message_start(messages).await {
            Err(ExtensionError::ExtensionFailed { name, .. }) => assert_eq!(name, "b"),
            Ok(_) => panic!("expected error"),
            Err(other) => panic!("expected ExtensionFailed, got {other:?}"),
        }
    }

    #[tokio::test]
    async fn test_and_tool_execution_both_allow() {
        let mut ext = And::new(LabelExt::ok("a"), LabelExt::ok("b"));
        let decision = ext.on_tool_execution_start("", "tool", &serde_json::Value::Null).await.unwrap();
        assert!(matches!(decision, ToolCallDecision::Allow));
    }

    #[tokio::test]
    async fn test_and_tool_execution_first_denies() {
        let mut ext = And::new(LabelExt::deny("a", "nope"), LabelExt::ok("b"));
        let decision = ext.on_tool_execution_start("", "tool", &serde_json::Value::Null).await.unwrap();
        match decision {
            ToolCallDecision::Deny(r) => assert_eq!(r, "nope"),
            ToolCallDecision::Allow => panic!("expected Deny"),
        }
    }

    #[tokio::test]
    async fn test_and_with_noop() {
        let mut ext = And::new(NoopExtension, NoopExtension);
        let messages = make_session("hello");
        let result = ext.on_message_start(messages).await.unwrap();
        assert_eq!(
            result[0].content,
            vec![ContentBlock::Text(crate::core::types::TextContent { content: "hello".into() })]
        );
    }

    #[tokio::test]
    async fn test_and_on_turn_end_aborts_when_not_assistant() {
        // `a` (like the mux fallback) leaves the session ending in a user message, so the turn
        // did not cleanly complete — `b` must not run.
        let recorder = Arc::new(AtomicBool::new(false));
        let mut ext = And::new(TurnEndAppend { role: "user" }, TurnEndRecorder { called: recorder.clone() });
        let (state, session) = turn_end_state();

        ext.on_turn_end(state).await.unwrap();

        assert!(
            !recorder.load(Ordering::SeqCst),
            "b must not run when the turn no longer ends in an assistant message"
        );
        let last_role = session.lock().unwrap().messages().last().map(|m| m.role.clone());
        assert_eq!(last_role.as_deref(), Some("user"), "the user message appended by a must be preserved");
    }

    #[tokio::test]
    async fn test_and_on_turn_end_runs_b_when_assistant() {
        // `a` leaves the session still ending in an assistant message, so `b` runs as usual.
        let recorder = Arc::new(AtomicBool::new(false));
        let mut ext = And::new(TurnEndAppend { role: "assistant" }, TurnEndRecorder { called: recorder.clone() });
        let (state, _session) = turn_end_state();

        ext.on_turn_end(state).await.unwrap();

        assert!(recorder.load(Ordering::SeqCst), "b must run when the turn still ends in an assistant message");
    }
}

use async_trait::async_trait;

use crate::core::agent::AgentState;
use crate::core::extensions::{Extension, ExtensionError, ToolCallDecision};
use crate::core::providers::StreamResponse;
use crate::core::tools::ToolError;
use crate::core::types::{Message, ToolResult};

/// Tries extension A first; if A fails, falls back to B.
/// For state-transforming hooks, A is tried on the original state and,
/// on error, B is retried on the same original state.
/// For tool decisions, if A denies, B gets a chance to override.
pub struct Or {
    a: Box<dyn Extension>,
    b: Box<dyn Extension>,
}

impl Or {
    pub fn new<A: Into<Box<dyn Extension>>, B: Into<Box<dyn Extension>>>(a: A, b: B) -> Self {
        Self { a: a.into(), b: b.into() }
    }
}

#[async_trait]
impl Extension for Or {
    fn name(&self) -> &str {
        "or"
    }

    async fn on_agent_start(&mut self, state: AgentState) -> Result<AgentState, ExtensionError> {
        match self.a.on_agent_start(state.clone()).await {
            Ok(s) => Ok(s),
            Err(_) => self.b.on_agent_start(state).await,
        }
    }

    async fn on_agent_end(&mut self, state: AgentState) -> Result<AgentState, ExtensionError> {
        match self.a.on_agent_end(state.clone()).await {
            Ok(s) => Ok(s),
            Err(_) => self.b.on_agent_end(state).await,
        }
    }

    async fn on_turn_start(&mut self, state: AgentState) -> Result<AgentState, ExtensionError> {
        match self.a.on_turn_start(state.clone()).await {
            Ok(s) => Ok(s),
            Err(_) => self.b.on_turn_start(state).await,
        }
    }

    async fn on_turn_end(&mut self, state: AgentState) -> Result<AgentState, ExtensionError> {
        match self.a.on_turn_end(state.clone()).await {
            Ok(s) => Ok(s),
            Err(_) => self.b.on_turn_end(state).await,
        }
    }

    async fn on_message_start(&mut self, messages: Vec<Message>) -> Result<Vec<Message>, ExtensionError> {
        match self.a.on_message_start(messages.clone()).await {
            Ok(m) => Ok(m),
            Err(_) => self.b.on_message_start(messages).await,
        }
    }

    async fn on_message_update(&mut self, resp: &StreamResponse) -> Result<(), ExtensionError> {
        match self.a.on_message_update(resp).await {
            Ok(()) => Ok(()),
            Err(_) => self.b.on_message_update(resp).await,
        }
    }

    async fn on_message_end(&mut self, resp: &StreamResponse) -> Result<(), ExtensionError> {
        match self.a.on_message_end(resp).await {
            Ok(()) => Ok(()),
            Err(_) => self.b.on_message_end(resp).await,
        }
    }

    async fn on_tool_execution_start(
        &mut self,
        tool_call_id: &str,
        name: &str,
        args: &serde_json::Value,
    ) -> Result<ToolCallDecision, ExtensionError> {
        match self.a.on_tool_execution_start(tool_call_id, name, args).await? {
            ToolCallDecision::Allow => Ok(ToolCallDecision::Allow),
            ToolCallDecision::Deny(_) => self.b.on_tool_execution_start(tool_call_id, name, args).await,
        }
    }

    async fn tool_execution_end(
        &mut self,
        tool_call_id: &str,
        result: Result<ToolResult, ToolError>,
    ) -> Result<Result<ToolResult, ToolError>, ExtensionError> {
        match self.a.tool_execution_end(tool_call_id, result.clone()).await {
            Ok(r) => Ok(r),
            Err(_) => self.b.tool_execution_end(tool_call_id, result).await,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::core::extensions::NoopExtension;
    use crate::core::types::ContentBlock;

    use super::super::test_helpers::*;

    #[tokio::test]
    async fn test_or_name() {
        let ext = Or::new(NoopExtension, NoopExtension);
        assert_eq!(ext.name(), "or");
    }

    #[tokio::test]
    async fn test_or_first_succeeds() {
        let mut ext = Or::new(LabelExt::ok("a"), LabelExt::ok("b"));
        let messages = make_session("");
        let result = ext.on_message_start(messages).await.unwrap();
        assert_eq!(result.len(), 1);
        match &result[0].content[0] {
            ContentBlock::Text(t) => assert_eq!(t.content, "a"),
            _ => panic!("expected text"),
        }
    }

    #[tokio::test]
    async fn test_or_fallback_on_failure() {
        let mut ext = Or::new(LabelExt::fail("a"), LabelExt::ok("b"));
        let messages = make_session("");
        let result = ext.on_message_start(messages).await.unwrap();
        assert_eq!(result.len(), 1);
        match &result[0].content[0] {
            ContentBlock::Text(t) => assert_eq!(t.content, "b"),
            _ => panic!("expected text"),
        }
    }

    #[tokio::test]
    async fn test_or_both_fail() {
        let mut ext = Or::new(LabelExt::fail("a"), LabelExt::fail("b"));
        let messages = make_session("hello");
        match ext.on_message_start(messages).await {
            Err(ExtensionError::ExtensionFailed { name, .. }) => assert_eq!(name, "b"),
            Ok(_) => panic!("expected error"),
            Err(other) => panic!("expected ExtensionFailed, got {other:?}"),
        }
    }

    #[tokio::test]
    async fn test_or_tool_execution_first_allows() {
        let mut ext = Or::new(LabelExt::ok("a"), LabelExt::ok("b"));
        let decision = ext.on_tool_execution_start("", "tool", &serde_json::Value::Null).await.unwrap();
        assert!(matches!(decision, ToolCallDecision::Allow));
    }

    #[tokio::test]
    async fn test_or_tool_execution_first_denies_second_overrides() {
        let mut ext = Or::new(LabelExt::deny("a", "nope"), LabelExt::ok("b"));
        let decision = ext.on_tool_execution_start("", "tool", &serde_json::Value::Null).await.unwrap();
        assert!(matches!(decision, ToolCallDecision::Allow));
    }

    #[tokio::test]
    async fn test_or_tool_execution_both_deny() {
        let mut ext = Or::new(LabelExt::deny("a", "nope"), LabelExt::deny("b", "also nope"));
        let decision = ext.on_tool_execution_start("", "tool", &serde_json::Value::Null).await.unwrap();
        match decision {
            ToolCallDecision::Deny(r) => assert_eq!(r, "also nope"),
            ToolCallDecision::Allow => panic!("expected Deny"),
        }
    }

    #[tokio::test]
    async fn test_or_on_message_update_fallback() {
        let mut ext = Or::new(LabelExt::fail("a"), LabelExt::ok("b"));
        let resp = make_response();
        ext.on_message_update(&resp).await.unwrap();
    }
}

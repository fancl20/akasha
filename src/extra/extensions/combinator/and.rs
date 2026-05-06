use async_trait::async_trait;

use crate::core::agent::AgentState;
use crate::core::extensions::{Extension, ExtensionError, ToolCallDecision};
use crate::core::providers::StreamResponse;
use crate::core::tools::ToolError;
use crate::core::types::Request;

/// Runs both extensions sequentially. Both must succeed.
/// For state-transforming hooks, output of A feeds into B.
/// For tool decisions, short-circuits on the first `Deny`.
pub struct And<A: Extension, B: Extension>(pub A, pub B);

#[async_trait]
impl<A: Extension, B: Extension> Extension for And<A, B> {
    fn name(&self) -> &str {
        "and"
    }

    async fn on_agent_start(&self, state: AgentState) -> Result<AgentState, ExtensionError> {
        let state = self.0.on_agent_start(state).await?;
        self.1.on_agent_start(state).await
    }

    async fn on_agent_end(&self, state: AgentState) -> Result<AgentState, ExtensionError> {
        let state = self.0.on_agent_end(state).await?;
        self.1.on_agent_end(state).await
    }

    async fn on_turn_start(&self, state: AgentState) -> Result<AgentState, ExtensionError> {
        let state = self.0.on_turn_start(state).await?;
        self.1.on_turn_start(state).await
    }

    async fn on_turn_end(&self, state: AgentState) -> Result<AgentState, ExtensionError> {
        let state = self.0.on_turn_end(state).await?;
        self.1.on_turn_end(state).await
    }

    async fn on_message_start(&self, req: Request) -> Result<Request, ExtensionError> {
        let req = self.0.on_message_start(req).await?;
        self.1.on_message_start(req).await
    }

    async fn on_message_update(&self, resp: &StreamResponse) -> Result<(), ExtensionError> {
        self.0.on_message_update(resp).await?;
        self.1.on_message_update(resp).await
    }

    async fn on_message_end(&self, resp: &StreamResponse) -> Result<(), ExtensionError> {
        self.0.on_message_end(resp).await?;
        self.1.on_message_end(resp).await
    }

    async fn on_tool_execution_start(
        &self,
        tool_call_id: &str,
        name: &str,
        args: &serde_json::Value,
    ) -> Result<ToolCallDecision, ExtensionError> {
        match self
            .0
            .on_tool_execution_start(tool_call_id, name, args)
            .await?
        {
            ToolCallDecision::Allow => {
                self.1
                    .on_tool_execution_start(tool_call_id, name, args)
                    .await
            }
            deny => Ok(deny),
        }
    }

    async fn tool_execution_end(
        &self,
        tool_call_id: &str,
        result: Result<String, ToolError>,
    ) -> Result<Result<String, ToolError>, ExtensionError> {
        let result = self.0.tool_execution_end(tool_call_id, result).await?;
        self.1.tool_execution_end(tool_call_id, result).await
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::core::extensions::NoopExtension;
    use crate::core::types::ContentBlock;

    use super::super::test_helpers::*;

    #[tokio::test]
    async fn test_and_name() {
        let ext = And(NoopExtension, NoopExtension);
        assert_eq!(ext.name(), "and");
    }

    #[tokio::test]
    async fn test_and_on_message_start_chains() {
        let ext = And(LabelExt::ok("a"), LabelExt::ok("b"));
        let req = make_request("");
        let result = ext.on_message_start(req).await.unwrap();
        let text = match result.messages[0].content.last() {
            Some(ContentBlock::Text { content }) => content.clone(),
            _ => panic!("expected text"),
        };
        assert_eq!(text, "a,b");
    }

    #[tokio::test]
    async fn test_and_on_message_start_fails_on_first() {
        let ext = And(LabelExt::fail("a"), LabelExt::ok("b"));
        let req = make_request("hello");
        let err = ext.on_message_start(req).await.unwrap_err();
        match err {
            ExtensionError::ExtensionFailed { name, .. } => assert_eq!(name, "a"),
        }
    }

    #[tokio::test]
    async fn test_and_on_message_start_fails_on_second() {
        let ext = And(LabelExt::ok("a"), LabelExt::fail("b"));
        let req = make_request("");
        let err = ext.on_message_start(req).await.unwrap_err();
        match err {
            ExtensionError::ExtensionFailed { name, .. } => assert_eq!(name, "b"),
        }
    }

    #[tokio::test]
    async fn test_and_tool_execution_both_allow() {
        let ext = And(LabelExt::ok("a"), LabelExt::ok("b"));
        let decision = ext
            .on_tool_execution_start("", "tool", &serde_json::Value::Null)
            .await
            .unwrap();
        assert!(matches!(decision, ToolCallDecision::Allow));
    }

    #[tokio::test]
    async fn test_and_tool_execution_first_denies() {
        let ext = And(LabelExt::deny("a", "nope"), LabelExt::ok("b"));
        let decision = ext
            .on_tool_execution_start("", "tool", &serde_json::Value::Null)
            .await
            .unwrap();
        match decision {
            ToolCallDecision::Deny(r) => assert_eq!(r, "nope"),
            ToolCallDecision::Allow => panic!("expected Deny"),
        }
    }

    #[tokio::test]
    async fn test_and_with_noop() {
        let ext = And(NoopExtension, NoopExtension);
        let req = make_request("hello");
        let result = ext.on_message_start(req).await.unwrap();
        assert_eq!(
            result.messages[0].content,
            vec![ContentBlock::Text {
                content: "hello".into()
            }]
        );
    }
}

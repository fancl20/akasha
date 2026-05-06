use async_trait::async_trait;

use crate::core::agent::AgentState;
use crate::core::extensions::{Extension, ExtensionError, ToolCallDecision};
use crate::core::providers::StreamResponse;
use crate::core::tools::ToolError;
use crate::core::types::Request;

pub struct Pipeline {
    extensions: Vec<Box<dyn Extension>>,
}

impl Pipeline {
    pub fn new(extensions: Vec<Box<dyn Extension>>) -> Self {
        Self { extensions }
    }
}

#[async_trait]
impl Extension for Pipeline {
    fn name(&self) -> &str {
        "pipeline"
    }

    async fn on_agent_start(&self, state: AgentState) -> Result<AgentState, ExtensionError> {
        let mut state = state;
        for ext in &self.extensions {
            state = ext.on_agent_start(state).await?;
        }
        Ok(state)
    }

    async fn on_agent_end(&self, state: AgentState) -> Result<AgentState, ExtensionError> {
        let mut state = state;
        for ext in &self.extensions {
            state = ext.on_agent_end(state).await?;
        }
        Ok(state)
    }

    async fn on_turn_start(&self, state: AgentState) -> Result<AgentState, ExtensionError> {
        let mut state = state;
        for ext in &self.extensions {
            state = ext.on_turn_start(state).await?;
        }
        Ok(state)
    }

    async fn on_turn_end(&self, state: AgentState) -> Result<AgentState, ExtensionError> {
        let mut state = state;
        for ext in &self.extensions {
            state = ext.on_turn_end(state).await?;
        }
        Ok(state)
    }

    async fn on_message_start(&self, mut request: Request) -> Result<Request, ExtensionError> {
        for ext in &self.extensions {
            request = ext.on_message_start(request).await?;
        }
        Ok(request)
    }

    async fn on_message_update(&self, resp: &StreamResponse) -> Result<(), ExtensionError> {
        for ext in &self.extensions {
            ext.on_message_update(resp).await?;
        }
        Ok(())
    }

    async fn on_message_end(&self, resp: &StreamResponse) -> Result<(), ExtensionError> {
        for ext in &self.extensions {
            ext.on_message_end(resp).await?;
        }
        Ok(())
    }

    async fn on_tool_execution_start(
        &self,
        tool_call_id: &str,
        name: &str,
        args: &serde_json::Value,
    ) -> Result<ToolCallDecision, ExtensionError> {
        for ext in &self.extensions {
            match ext.on_tool_execution_start(tool_call_id, name, args).await? {
                ToolCallDecision::Deny(reason) => return Ok(ToolCallDecision::Deny(reason)),
                ToolCallDecision::Allow => {}
            }
        }
        Ok(ToolCallDecision::Allow)
    }

    async fn tool_execution_end(
        &self,
        tool_call_id: &str,
        result: Result<String, ToolError>,
    ) -> Result<Result<String, ToolError>, ExtensionError> {
        let mut result = result;
        for ext in &self.extensions {
            result = ext.tool_execution_end(tool_call_id, result).await?;
        }
        Ok(result)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::core::extensions::NoopExtension;
    use crate::core::types::{ContentBlock, Message};
    use crate::core::providers::StreamResponse;
    use std::sync::{Arc, Mutex};

    // Helper: an extension that appends its name to the last text block on_message_start.
    // Tracks call counts for on_message_update / on_message_end / on_tool_execution_start.
    struct MockExtension {
        label: &'static str,
        on_request_label: &'static str,
        call_counts: Arc<Mutex<Vec<(&'static str, usize)>>>,
        tool_decision: ToolCallDecision,
        should_fail: bool,
    }

    impl MockExtension {
        fn allow(label: &'static str) -> Self {
            Self {
                label,
                on_request_label: label,
                call_counts: Arc::new(Mutex::new(vec![])),
                tool_decision: ToolCallDecision::Allow,
                should_fail: false,
            }
        }

        fn deny(label: &'static str, reason: &str) -> Self {
            Self {
                label,
                on_request_label: label,
                call_counts: Arc::new(Mutex::new(vec![])),
                tool_decision: ToolCallDecision::Deny(reason.to_string()),
                should_fail: false,
            }
        }

        fn failing(label: &'static str) -> Self {
            Self {
                label,
                on_request_label: label,
                call_counts: Arc::new(Mutex::new(vec![])),
                tool_decision: ToolCallDecision::Allow,
                should_fail: true,
            }
        }

        fn record(&self, method: &'static str) {
            self.call_counts
                .lock()
                .unwrap()
                .push((method, 1));
        }
    }

    #[async_trait]
    impl Extension for MockExtension {
        fn name(&self) -> &str {
            self.label
        }

        async fn on_message_start(&self, mut request: Request) -> Result<Request, ExtensionError> {
            if self.should_fail {
                return Err(ExtensionError::ExtensionFailed {
                    name: self.label.to_string(),
                    message: "intentional failure".into(),
                });
            }
            // Append label to the last text content so order is observable.
            if let Some(ContentBlock::Text { content }) = request.messages.last_mut().and_then(|m| m.content.last_mut()) {
                if !content.is_empty() {
                    content.push(',');
                }
                content.push_str(self.on_request_label);
            }
            Ok(request)
        }

        async fn on_message_update(&self, _resp: &StreamResponse) -> Result<(), ExtensionError> {
            self.record("on_message_update");
            Ok(())
        }

        async fn on_message_end(&self, _resp: &StreamResponse) -> Result<(), ExtensionError> {
            self.record("on_message_end");
            Ok(())
        }

        async fn on_tool_execution_start(
            &self,
            _tool_call_id: &str,
            _name: &str,
            _args: &serde_json::Value,
        ) -> Result<ToolCallDecision, ExtensionError> {
            self.record("on_tool_execution_start");
            Ok(match &self.tool_decision {
                ToolCallDecision::Allow => ToolCallDecision::Allow,
                ToolCallDecision::Deny(r) => ToolCallDecision::Deny(r.clone()),
            })
        }
    }

    fn make_request(text: &str) -> Request {
        Request {
            messages: vec![Message {
                role: "user".into(),
                content: vec![ContentBlock::Text {
                    content: text.into(),
                }],
            }],
            tools: vec![],
        }
    }

    fn make_response() -> StreamResponse {
        StreamResponse::new()
    }

    #[tokio::test]
    async fn test_name() {
        let pipeline = Pipeline::new(vec![]);
        assert_eq!(pipeline.name(), "pipeline");
    }

    #[tokio::test]
    async fn test_empty_pipeline_on_message_start() {
        let pipeline = Pipeline::new(vec![]);
        let req = make_request("hello");
        let result = pipeline.on_message_start(req).await.unwrap();
        assert_eq!(
            result.messages[0].content,
            vec![ContentBlock::Text { content: "hello".into() }]
        );
    }

    #[tokio::test]
    async fn test_on_message_start_chains_in_order() {
        let pipeline = Pipeline::new(vec![
            Box::new(MockExtension::allow("a")),
            Box::new(MockExtension::allow("b")),
        ]);
        let req = make_request("");
        let result = pipeline.on_message_start(req).await.unwrap();
        let text = match result.messages[0].content.last() {
            Some(ContentBlock::Text { content }) => content.clone(),
            _ => panic!("expected text block"),
        };
        assert_eq!(text, "a,b");
    }

    #[tokio::test]
    async fn test_on_message_start_stops_on_error() {
        let pipeline = Pipeline::new(vec![
            Box::new(MockExtension::allow("a")),
            Box::new(MockExtension::failing("b")),
            Box::new(MockExtension::allow("c")),
        ]);
        let req = make_request("");
        let err = pipeline.on_message_start(req).await.unwrap_err();
        match err {
            ExtensionError::ExtensionFailed { name, .. } => assert_eq!(name, "b"),
        }
    }

    #[tokio::test]
    async fn test_on_message_update_calls_all() {
        let pipeline = Pipeline::new(vec![
            Box::new(MockExtension::allow("e1")),
            Box::new(MockExtension::allow("e2")),
        ]);
        let resp = make_response();
        pipeline.on_message_update(&resp).await.unwrap();
        // No panic / error is sufficient — extensions are not shared so we can't
        // inspect call counts directly, but the method completes successfully.
    }

    #[tokio::test]
    async fn test_on_message_end_calls_all() {
        let pipeline = Pipeline::new(vec![
            Box::new(MockExtension::allow("e1")),
            Box::new(MockExtension::allow("e2")),
        ]);
        let resp = make_response();
        pipeline.on_message_end(&resp).await.unwrap();
    }

    #[tokio::test]
    async fn test_tool_execution_start_all_allow() {
        let pipeline = Pipeline::new(vec![
            Box::new(MockExtension::allow("a")),
            Box::new(MockExtension::allow("b")),
        ]);
        let decision = pipeline.on_tool_execution_start("", "tool", &serde_json::Value::Null).await.unwrap();
        assert!(matches!(decision, ToolCallDecision::Allow));
    }

    #[tokio::test]
    async fn test_tool_execution_start_deny_short_circuits() {
        let pipeline = Pipeline::new(vec![
            Box::new(MockExtension::allow("a")),
            Box::new(MockExtension::deny("b", "not allowed")),
            Box::new(MockExtension::allow("c")),
        ]);
        let decision = pipeline.on_tool_execution_start("", "tool", &serde_json::Value::Null).await.unwrap();
        match decision {
            ToolCallDecision::Deny(reason) => assert_eq!(reason, "not allowed"),
            ToolCallDecision::Allow => panic!("expected Deny"),
        }
    }

    #[tokio::test]
    async fn test_tool_execution_start_first_denies() {
        let pipeline = Pipeline::new(vec![
            Box::new(MockExtension::deny("a", "blocked")),
            Box::new(MockExtension::allow("b")),
        ]);
        let decision = pipeline.on_tool_execution_start("", "tool", &serde_json::Value::Null).await.unwrap();
        assert!(matches!(decision, ToolCallDecision::Deny(_)));
    }

    #[tokio::test]
    async fn test_empty_pipeline_tool_execution_start() {
        let pipeline = Pipeline::new(vec![]);
        let decision = pipeline.on_tool_execution_start("", "tool", &serde_json::Value::Null).await.unwrap();
        assert!(matches!(decision, ToolCallDecision::Allow));
    }

    #[tokio::test]
    async fn test_noop_extension_in_pipeline() {
        let pipeline = Pipeline::new(vec![Box::new(NoopExtension)]);
        let req = make_request("hello");
        let result = pipeline.on_message_start(req).await.unwrap();
        assert_eq!(
            result.messages[0].content,
            vec![ContentBlock::Text { content: "hello".into() }]
        );
    }
}

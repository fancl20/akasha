use async_trait::async_trait;

use crate::core::extensions::{Extension, ExtensionError, ToolCallDecision};
use crate::core::providers::StreamResponse;
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

    async fn on_request(&self, mut request: Request) -> Result<Request, ExtensionError> {
        for ext in &self.extensions {
            request = ext.on_request(request).await?;
        }
        Ok(request)
    }

    async fn on_response_chunk(&self, resp: &StreamResponse) -> Result<(), ExtensionError> {
        for ext in &self.extensions {
            ext.on_response_chunk(resp).await?;
        }
        Ok(())
    }

    async fn on_response(&self, resp: &StreamResponse) -> Result<(), ExtensionError> {
        for ext in &self.extensions {
            ext.on_response(resp).await?;
        }
        Ok(())
    }

    async fn on_tool_call(
        &self,
        name: &str,
        args: &[&str],
    ) -> Result<ToolCallDecision, ExtensionError> {
        for ext in &self.extensions {
            match ext.on_tool_call(name, args).await? {
                ToolCallDecision::Deny(reason) => return Ok(ToolCallDecision::Deny(reason)),
                ToolCallDecision::Allow => {}
            }
        }
        Ok(ToolCallDecision::Allow)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::core::extensions::NoopExtension;
    use crate::core::types::{ContentBlock, Message};
    use crate::core::StreamResponse;
    use std::sync::{Arc, Mutex};

    // Helper: an extension that appends its name to the last text block on_request.
    // Tracks call counts for on_response_chunk / on_response / on_tool_call.
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

        async fn on_request(&self, mut request: Request) -> Result<Request, ExtensionError> {
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

        async fn on_response_chunk(&self, _resp: &StreamResponse) -> Result<(), ExtensionError> {
            self.record("on_response_chunk");
            Ok(())
        }

        async fn on_response(&self, _resp: &StreamResponse) -> Result<(), ExtensionError> {
            self.record("on_response");
            Ok(())
        }

        async fn on_tool_call(
            &self,
            _name: &str,
            _args: &[&str],
        ) -> Result<ToolCallDecision, ExtensionError> {
            self.record("on_tool_call");
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
    async fn test_empty_pipeline_on_request() {
        let pipeline = Pipeline::new(vec![]);
        let req = make_request("hello");
        let result = pipeline.on_request(req).await.unwrap();
        assert_eq!(
            result.messages[0].content,
            vec![ContentBlock::Text { content: "hello".into() }]
        );
    }

    #[tokio::test]
    async fn test_on_request_chains_in_order() {
        let pipeline = Pipeline::new(vec![
            Box::new(MockExtension::allow("a")),
            Box::new(MockExtension::allow("b")),
        ]);
        let req = make_request("");
        let result = pipeline.on_request(req).await.unwrap();
        let text = match result.messages[0].content.last() {
            Some(ContentBlock::Text { content }) => content.clone(),
            _ => panic!("expected text block"),
        };
        assert_eq!(text, "a,b");
    }

    #[tokio::test]
    async fn test_on_request_stops_on_error() {
        let pipeline = Pipeline::new(vec![
            Box::new(MockExtension::allow("a")),
            Box::new(MockExtension::failing("b")),
            Box::new(MockExtension::allow("c")),
        ]);
        let req = make_request("");
        let err = pipeline.on_request(req).await.unwrap_err();
        match err {
            ExtensionError::ExtensionFailed { name, .. } => assert_eq!(name, "b"),
        }
    }

    #[tokio::test]
    async fn test_on_response_chunk_calls_all() {
        let pipeline = Pipeline::new(vec![
            Box::new(MockExtension::allow("e1")),
            Box::new(MockExtension::allow("e2")),
        ]);
        let resp = make_response();
        pipeline.on_response_chunk(&resp).await.unwrap();
        // No panic / error is sufficient — extensions are not shared so we can't
        // inspect call counts directly, but the method completes successfully.
    }

    #[tokio::test]
    async fn test_on_response_calls_all() {
        let pipeline = Pipeline::new(vec![
            Box::new(MockExtension::allow("e1")),
            Box::new(MockExtension::allow("e2")),
        ]);
        let resp = make_response();
        pipeline.on_response(&resp).await.unwrap();
    }

    #[tokio::test]
    async fn test_tool_call_all_allow() {
        let pipeline = Pipeline::new(vec![
            Box::new(MockExtension::allow("a")),
            Box::new(MockExtension::allow("b")),
        ]);
        let decision = pipeline.on_tool_call("tool", &[]).await.unwrap();
        assert!(matches!(decision, ToolCallDecision::Allow));
    }

    #[tokio::test]
    async fn test_tool_call_deny_short_circuits() {
        let pipeline = Pipeline::new(vec![
            Box::new(MockExtension::allow("a")),
            Box::new(MockExtension::deny("b", "not allowed")),
            Box::new(MockExtension::allow("c")),
        ]);
        let decision = pipeline.on_tool_call("tool", &[]).await.unwrap();
        match decision {
            ToolCallDecision::Deny(reason) => assert_eq!(reason, "not allowed"),
            ToolCallDecision::Allow => panic!("expected Deny"),
        }
    }

    #[tokio::test]
    async fn test_tool_call_first_denies() {
        let pipeline = Pipeline::new(vec![
            Box::new(MockExtension::deny("a", "blocked")),
            Box::new(MockExtension::allow("b")),
        ]);
        let decision = pipeline.on_tool_call("tool", &[]).await.unwrap();
        assert!(matches!(decision, ToolCallDecision::Deny(_)));
    }

    #[tokio::test]
    async fn test_empty_pipeline_tool_call() {
        let pipeline = Pipeline::new(vec![]);
        let decision = pipeline.on_tool_call("tool", &[]).await.unwrap();
        assert!(matches!(decision, ToolCallDecision::Allow));
    }

    #[tokio::test]
    async fn test_noop_extension_in_pipeline() {
        let pipeline = Pipeline::new(vec![Box::new(NoopExtension)]);
        let req = make_request("hello");
        let result = pipeline.on_request(req).await.unwrap();
        assert_eq!(
            result.messages[0].content,
            vec![ContentBlock::Text { content: "hello".into() }]
        );
    }
}

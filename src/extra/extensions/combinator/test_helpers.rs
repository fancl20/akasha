use async_trait::async_trait;

use crate::core::agent::AgentState;
use crate::core::extensions::{Extension, ExtensionError, ToolCallDecision};
use crate::core::providers::StreamResponse;
use crate::core::types::{ContentBlock, Message, Request, TextContent};
use std::sync::{Arc, Mutex};

pub struct LabelExt {
    label: &'static str,
    should_fail: bool,
    deny_reason: Option<String>,
    #[allow(dead_code)]
    calls: Arc<Mutex<Vec<&'static str>>>,
}

impl LabelExt {
    pub fn ok(label: &'static str) -> Self {
        Self { label, should_fail: false, deny_reason: None, calls: Arc::new(Mutex::new(vec![])) }
    }

    pub fn fail(label: &'static str) -> Self {
        Self { label, should_fail: true, deny_reason: None, calls: Arc::new(Mutex::new(vec![])) }
    }

    pub fn deny(label: &'static str, reason: &str) -> Self {
        Self {
            label,
            should_fail: false,
            deny_reason: Some(reason.to_string()),
            calls: Arc::new(Mutex::new(vec![])),
        }
    }

    fn record(&self, method: &'static str) {
        self.calls.lock().unwrap().push(method);
    }
}

#[async_trait]
impl Extension for LabelExt {
    fn name(&self) -> &str {
        self.label
    }

    async fn on_message_start(&mut self, mut req: Request) -> Result<Request, ExtensionError> {
        if self.should_fail {
            return Err(ExtensionError::ExtensionFailed {
                name: self.label.to_string(),
                message: "fail".into(),
            });
        }
        if let Some(ContentBlock::Text(t)) =
            req.messages.last_mut().and_then(|m| m.content.last_mut())
        {
            if !t.content.is_empty() {
                t.content.push(',');
            }
            t.content.push_str(self.label);
        }
        Ok(req)
    }

    async fn on_agent_start(&mut self, state: AgentState) -> Result<AgentState, ExtensionError> {
        self.record("on_agent_start");
        if self.should_fail {
            return Err(ExtensionError::ExtensionFailed {
                name: self.label.to_string(),
                message: "fail".into(),
            });
        }
        Ok(state)
    }

    async fn on_tool_execution_start(
        &mut self,
        _id: &str,
        _name: &str,
        _args: &serde_json::Value,
    ) -> Result<ToolCallDecision, ExtensionError> {
        self.record("on_tool_execution_start");
        if let Some(reason) = &self.deny_reason {
            Ok(ToolCallDecision::Deny(reason.clone()))
        } else {
            Ok(ToolCallDecision::Allow)
        }
    }

    async fn on_message_update(&mut self, _resp: &StreamResponse) -> Result<(), ExtensionError> {
        self.record("on_message_update");
        if self.should_fail {
            return Err(ExtensionError::ExtensionFailed {
                name: self.label.to_string(),
                message: "fail".into(),
            });
        }
        Ok(())
    }

    async fn on_message_end(&mut self, _resp: &StreamResponse) -> Result<(), ExtensionError> {
        self.record("on_message_end");
        Ok(())
    }
}

pub fn make_request(text: &str) -> Request {
    Request {
        messages: vec![Message {
            role: "user".into(),
            content: vec![ContentBlock::Text(TextContent { content: text.into() })],
        }],
        tools: vec![],
    }
}

pub fn make_response() -> StreamResponse {
    StreamResponse::new()
}

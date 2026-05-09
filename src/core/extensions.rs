use async_trait::async_trait;

use crate::core::agent::AgentState;
use crate::core::providers::StreamResponse;
use crate::core::tools::ToolError;
use crate::core::types::Request;

pub enum ToolCallDecision {
    Allow,
    Deny(String),
}

#[derive(Debug, thiserror::Error)]
pub enum ExtensionError {
    #[error("extension '{name}': {message}")]
    ExtensionFailed { name: String, message: String },
}

#[async_trait]
pub trait Extension: Send + 'static {
    fn name(&self) -> &str;

    async fn on_agent_start(&mut self, _state: AgentState) -> Result<AgentState, ExtensionError> {
        Ok(_state)
    }

    async fn on_agent_end(&mut self, _state: AgentState) -> Result<AgentState, ExtensionError> {
        Ok(_state)
    }

    async fn on_turn_start(&mut self, _state: AgentState) -> Result<AgentState, ExtensionError> {
        Ok(_state)
    }

    async fn on_turn_end(&mut self, _state: AgentState) -> Result<AgentState, ExtensionError> {
        Ok(_state)
    }

    async fn on_message_start(&mut self, _req: Request) -> Result<Request, ExtensionError> {
        Ok(_req)
    }

    async fn on_message_update(&mut self, _resp: &StreamResponse) -> Result<(), ExtensionError> {
        Ok(())
    }

    async fn on_message_end(&mut self, _resp: &StreamResponse) -> Result<(), ExtensionError> {
        Ok(())
    }

    async fn on_tool_execution_start(
        &mut self,
        _tool_call_id: &str,
        _name: &str,
        _args: &serde_json::Value,
    ) -> Result<ToolCallDecision, ExtensionError> {
        Ok(ToolCallDecision::Allow)
    }

    async fn tool_execution_end(
        &mut self,
        _tool_call_id: &str,
        _result: Result<String, ToolError>,
    ) -> Result<Result<String, ToolError>, ExtensionError> {
        Ok(_result)
    }
}

pub struct NoopExtension;

#[async_trait]
impl Extension for NoopExtension {
    fn name(&self) -> &str {
        "noop"
    }
}

impl<T: Extension> From<T> for Box<dyn Extension> {
    fn from(ext: T) -> Self {
        Box::new(ext)
    }
}

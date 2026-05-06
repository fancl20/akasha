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
pub trait Extension: Send + Sync {
    fn name(&self) -> &str;

    async fn on_agent_start(&self, _state: AgentState) -> Result<AgentState, ExtensionError> {
        Ok(_state)
    }

    #[allow(unused_mut)]
    async fn on_agent_end(&self, _state: AgentState) -> Result<AgentState, ExtensionError> {
        Ok(_state)
    }

    #[allow(unused_mut)]
    async fn on_turn_start(&self, _state: AgentState) -> Result<AgentState, ExtensionError> {
        Ok(_state)
    }

    #[allow(unused_mut)]
    async fn on_turn_end(&self, _state: AgentState) -> Result<AgentState, ExtensionError> {
        Ok(_state)
    }

    #[allow(unused_mut)]
    async fn on_message_start(&self, _req: Request) -> Result<Request, ExtensionError> {
        Ok(_req)
    }

    async fn on_message_update(&self, _resp: &StreamResponse) -> Result<(), ExtensionError> {
        Ok(())
    }

    async fn on_message_end(&self, _resp: &StreamResponse) -> Result<(), ExtensionError> {
        Ok(())
    }

    async fn on_tool_execution_start(
        &self,
        _tool_call_id: &str,
        _name: &str,
        _args: &serde_json::Value,
    ) -> Result<ToolCallDecision, ExtensionError> {
        Ok(ToolCallDecision::Allow)
    }

    async fn tool_execution_end(
        &self,
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

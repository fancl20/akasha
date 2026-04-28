use async_trait::async_trait;

use crate::core::StreamResponse;
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

    #[allow(unused_mut)]
    async fn on_request(&self, mut request: Request) -> Result<Request, ExtensionError> {
        Ok(request)
    }

    async fn on_response_chunk(&self, _resp: &StreamResponse) -> Result<(), ExtensionError> {
        Ok(())
    }

    async fn on_response(&self, _resp: &StreamResponse) -> Result<(), ExtensionError> {
        Ok(())
    }

    async fn on_tool_call(
        &self,
        _name: String,
        _arguments: String,
    ) -> Result<ToolCallDecision, ExtensionError> {
        Ok(ToolCallDecision::Allow)
    }
}

pub struct NoopExtension;

#[async_trait]
impl Extension for NoopExtension {
    fn name(&self) -> &str {
        "noop"
    }
}

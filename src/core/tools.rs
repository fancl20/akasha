use async_trait::async_trait;
use std::collections::HashMap;
use std::sync::Arc;

use crate::core::types::{ToolDefinition, ToolResult};

#[derive(Debug, Clone, thiserror::Error)]
pub enum ToolError {
    #[error("validation error: {0}")]
    Validation(String),
    #[error("execution error: {0}")]
    Execution(String),
    #[error("aborted")]
    Aborted,
}

#[async_trait]
pub trait ToolHandler: Send + Sync {
    fn definition(&self) -> ToolDefinition;

    async fn execute(
        &self,
        cancel: tokio::sync::watch::Receiver<bool>,
        params: serde_json::Value,
    ) -> Result<ToolResult, ToolError>;
}

impl<T: ToolHandler + 'static> From<T> for Box<dyn ToolHandler> {
    fn from(tool: T) -> Self {
        Box::new(tool)
    }
}

#[derive(Clone)]
pub struct ToolRegistry {
    tools: HashMap<String, Arc<dyn ToolHandler>>,
}

impl ToolRegistry {
    pub fn new() -> Self {
        Self { tools: HashMap::new() }
    }

    pub fn register(&mut self, handler: Box<dyn ToolHandler>) {
        let def = handler.definition();
        self.tools.insert(def.name, Arc::from(handler));
    }

    pub fn get(&self, name: &str) -> Option<&dyn ToolHandler> {
        self.tools.get(name).map(|h| h.as_ref())
    }

    pub fn definitions(&self) -> Vec<ToolDefinition> {
        self.tools.values().map(|h| h.definition()).collect()
    }
}

impl Default for ToolRegistry {
    fn default() -> Self {
        Self::new()
    }
}

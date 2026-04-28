use async_trait::async_trait;
use std::collections::HashMap;

use crate::core::types::ToolDefinition;

#[derive(Debug, thiserror::Error)]
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
    ) -> Result<String, ToolError>;
}

pub struct ToolRegistry {
    tools: HashMap<String, Box<dyn ToolHandler>>,
}

impl ToolRegistry {
    pub fn new() -> Self {
        Self {
            tools: HashMap::new(),
        }
    }

    pub fn register(&mut self, handler: Box<dyn ToolHandler>) {
        let def = handler.definition();
        self.tools.insert(def.name, handler);
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

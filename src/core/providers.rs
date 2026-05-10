use async_trait::async_trait;
use futures::Stream;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::pin::Pin;

use crate::core::types::{ContentBlock, Message, Request, TokenUsage};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Model {
    pub id: String,
    pub provider: String,
    pub context_window: usize,
    pub base_url: String,
    pub headers: HashMap<String, String>,
}

#[derive(Debug, thiserror::Error)]
pub enum ProviderError {
    #[error("request failed: {0}")]
    RequestFailed(String),
    #[error("rate limited")]
    RateLimited,
    #[error("context window exceeded: {0} tokens")]
    ContextExceeded(usize),
    #[error("model not found: {0}")]
    ModelNotFound(String),
    #[error("aborted")]
    Aborted,
}

#[derive(Debug, Clone)]
pub struct StreamResponse {
    pub message: Message,
    pub usage: TokenUsage,
    pub stop_reason: Option<String>,
}

impl StreamResponse {
    /// Start with an empty assistant response.
    pub fn new() -> Self {
        Self {
            message: Message {
                role: "assistant".to_string(),
                content: Vec::new(),
            },
            usage: TokenUsage {
                input_tokens: 0,
                output_tokens: 0,
                cache_read_tokens: None,
                cache_write_tokens: None,
            },
            stop_reason: None,
        }
    }

    /// Merge a streaming chunk into this response.
    ///
    /// Text and reasoning blocks are concatenated; tool calls are appended;
    /// usage and stop_reason are taken from the last chunk that carries them.
    pub fn merge(&mut self, chunk: StreamResponse) {
        for block in chunk.message.content {
            match block {
                ContentBlock::Text(t) => {
                    if let Some(ContentBlock::Text(existing)) = self.message.content.last_mut() {
                        existing.content.push_str(&t.content);
                    } else {
                        self.message.content.push(ContentBlock::Text(t));
                    }
                }
                ContentBlock::Reasoning(t) => {
                    if let Some(ContentBlock::Reasoning(existing)) = self.message.content.last_mut()
                    {
                        existing.content.push_str(&t.content);
                    } else {
                        self.message.content.push(ContentBlock::Reasoning(t));
                    }
                }
                other => self.message.content.push(other),
            }
        }
        if chunk.usage.input_tokens > 0 || chunk.usage.output_tokens > 0 {
            self.usage = chunk.usage;
        }
        if chunk.stop_reason.is_some() {
            self.stop_reason = chunk.stop_reason;
        }
    }
}

pub type StreamResponseStream = Pin<Box<dyn Stream<Item = StreamResponse> + Send>>;

#[async_trait]
pub trait Provider: Send + Sync {
    async fn stream(
        &self,
        model: &Model,
        request: &Request,
    ) -> Result<StreamResponseStream, ProviderError>;

    fn name(&self) -> &str;
}

pub struct Registry {
    providers: HashMap<String, Box<dyn Provider>>,
}

impl Registry {
    pub fn new() -> Self {
        Self {
            providers: HashMap::new(),
        }
    }

    pub fn register(&mut self, name: impl Into<String>, provider: Box<dyn Provider>) {
        self.providers.insert(name.into(), provider);
    }

    pub fn get(&self, name: &str) -> Option<&dyn Provider> {
        self.providers.get(name).map(|p| p.as_ref())
    }
}

impl Default for Registry {
    fn default() -> Self {
        Self::new()
    }
}

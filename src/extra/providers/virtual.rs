use std::collections::HashMap;
use std::sync::Arc;

use async_trait::async_trait;

use crate::core::providers::{Model, Provider, ProviderError, StreamResponseStream};
use crate::core::types::{Message, ToolDefinition};

/// A virtual provider that fronts a registry of concrete providers and forwards each
/// stream call to the backend named by [`Model::provider`], passing the model through
/// unchanged.
///
/// Unlike [`crate::extra::providers::tier::TierProvider`], it performs no model
/// remapping: the `Model` the caller streams with is the exact `Model` the backend
/// receives. The virtual provider's only job is to select the backend via
/// `model.provider` and delegate. This lets a single `Arc<dyn Provider>` (the virtual
/// one) serve many backends, with the active one chosen entirely by the model
/// configuration the agent holds.
pub struct VirtualProvider {
    name: String,
    providers: HashMap<String, Arc<dyn Provider>>,
}

impl VirtualProvider {
    pub fn new(name: impl Into<String>) -> Self {
        Self { name: name.into(), providers: HashMap::new() }
    }

    /// Register a backend provider under a name. The name is matched against
    /// [`Model::provider`] at stream time to select which backend receives the call.
    pub fn provider(mut self, name: impl Into<String>, provider: Arc<dyn Provider>) -> Self {
        self.providers.insert(name.into(), provider);
        self
    }
}

#[async_trait]
impl Provider for VirtualProvider {
    async fn stream<'a>(
        &self,
        model: &Model,
        messages: Box<dyn Iterator<Item = &'a Message> + Send + 'a>,
        tools: &Vec<ToolDefinition>,
    ) -> Result<StreamResponseStream, ProviderError> {
        let provider = self.providers.get(&model.provider).ok_or_else(|| {
            ProviderError::RequestFailed(format!(
                "no provider '{}' registered with virtual provider '{}'",
                model.provider, self.name
            ))
        })?;
        // Bypass: forward the exact same model to the selected backend.
        provider.stream(model, messages, tools).await
    }

    fn name(&self) -> &str {
        &self.name
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::Mutex;

    use crate::core::providers::StreamResponse;
    use crate::core::types::{ContentBlock, TextContent, TokenUsage};
    use futures::{StreamExt, stream};

    /// A backend that records the full model it was asked to stream and returns a
    /// single deterministic chunk. Shares its log via `Arc` so the test can read it
    /// after the call, even though the provider is registered into the virtual provider.
    struct RecordingProvider {
        name: &'static str,
        seen: Arc<Mutex<Vec<Model>>>,
    }

    #[async_trait]
    impl Provider for RecordingProvider {
        async fn stream<'a>(
            &self,
            model: &Model,
            _messages: Box<dyn Iterator<Item = &'a Message> + Send + 'a>,
            _tools: &Vec<ToolDefinition>,
        ) -> Result<StreamResponseStream, ProviderError> {
            self.seen.lock().unwrap().push(model.clone());
            let chunk = StreamResponse {
                message: Message {
                    role: "assistant".to_string(),
                    content: vec![ContentBlock::Text(TextContent { content: format!("from:{}", model.id) })],
                },
                usage: TokenUsage {
                    input_tokens: 0,
                    output_tokens: 0,
                    cache_read_tokens: None,
                    cache_write_tokens: None,
                },
                stop_reason: Some("stop".to_string()),
            };
            Ok(Box::pin(stream::iter(vec![chunk])))
        }

        fn name(&self) -> &str {
            self.name
        }
    }

    /// The model the caller streams with — note the non-default `context_window`,
    /// `base_url` and `headers`, which let tests prove the model is forwarded verbatim.
    fn request(provider: &str, id: &str) -> Model {
        Model {
            id: id.to_string(),
            provider: provider.to_string(),
            context_window: 123,
            base_url: "https://example.test".to_string(),
            headers: HashMap::from([("x-custom".to_string(), "v".to_string())]),
        }
    }

    fn backend(seen: Arc<Mutex<Vec<Model>>>) -> Arc<dyn Provider> {
        Arc::new(RecordingProvider { name: "back", seen })
    }

    async fn drain(stream: StreamResponseStream) {
        stream.collect::<Vec<_>>().await;
    }

    #[tokio::test]
    async fn routes_by_provider_and_passes_model_through_unchanged() {
        let seen = Arc::new(Mutex::new(Vec::new()));
        let vp = VirtualProvider::new("virtual").provider("a", backend(seen.clone()));

        drain(vp.stream(&request("a", "model-a"), Box::new(std::iter::empty::<&Message>()), &vec![]).await.unwrap())
            .await;

        // The backend received the exact model the caller streamed with — no remapping.
        let observed = seen.lock().unwrap().clone();
        assert_eq!(observed.len(), 1);
        let m = &observed[0];
        assert_eq!(m.id, "model-a");
        assert_eq!(m.provider, "a");
        assert_eq!(m.context_window, 123);
        assert_eq!(m.base_url, "https://example.test");
        assert_eq!(m.headers.get("x-custom").map(|s| s.as_str()), Some("v"));
    }

    #[tokio::test]
    async fn unknown_provider_is_request_failed() {
        let vp = VirtualProvider::new("virtual").provider("a", backend(Arc::new(Mutex::new(Vec::new()))));

        let result = vp.stream(&request("nope", "model-a"), Box::new(std::iter::empty::<&Message>()), &vec![]).await;
        assert!(matches!(result, Err(ProviderError::RequestFailed(_))));
    }

    #[test]
    fn name_returns_configured() {
        let vp = VirtualProvider::new("virtual");
        assert_eq!(vp.name(), "virtual");
    }

    #[tokio::test]
    async fn two_distinct_backends_no_crosstalk() {
        let seen_a = Arc::new(Mutex::new(Vec::new()));
        let seen_b = Arc::new(Mutex::new(Vec::new()));
        let vp = VirtualProvider::new("virtual")
            .provider("a", Arc::new(RecordingProvider { name: "a", seen: seen_a.clone() }))
            .provider("b", Arc::new(RecordingProvider { name: "b", seen: seen_b.clone() }));

        drain(vp.stream(&request("a", "model-a"), Box::new(std::iter::empty::<&Message>()), &vec![]).await.unwrap())
            .await;
        drain(vp.stream(&request("b", "model-b"), Box::new(std::iter::empty::<&Message>()), &vec![]).await.unwrap())
            .await;

        assert_eq!(seen_a.lock().unwrap().len(), 1);
        assert_eq!(seen_a.lock().unwrap()[0].id, "model-a");
        assert_eq!(seen_b.lock().unwrap().len(), 1);
        assert_eq!(seen_b.lock().unwrap()[0].id, "model-b");
    }
}

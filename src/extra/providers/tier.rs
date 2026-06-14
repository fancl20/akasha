use std::collections::HashMap;
use std::sync::Arc;

use async_trait::async_trait;

use crate::core::providers::{Model, Provider, ProviderError, StreamResponseStream};
use crate::core::types::{Message, ToolDefinition};

/// A generic routing provider that exposes virtual tier names (e.g. "high", "mid")
/// as model ids and delegates each request to a concrete provider + model configured
/// per tier.
///
/// It is self-contained: backends are registered with [`TierProvider::provider`] and
/// each tier maps to a [`Model`] whose `provider` field names one of those backends.
/// At stream time the tier id (`model.id`) selects the target model, the target's
/// `provider` selects the backend, and the call is delegated. The tier id never
/// reaches a backend.
pub struct TierProvider {
    name: String,
    providers: HashMap<String, Arc<dyn Provider>>,
    tiers: HashMap<String, Model>,
}

impl TierProvider {
    pub fn new(name: impl Into<String>) -> Self {
        Self { name: name.into(), providers: HashMap::new(), tiers: HashMap::new() }
    }

    /// Register a backend provider that tiers can route to by name.
    pub fn provider(mut self, name: impl Into<String>, provider: Arc<dyn Provider>) -> Self {
        self.providers.insert(name.into(), provider);
        self
    }

    /// Map a tier name to a target [`Model`] (whose `provider` names a registered backend).
    pub fn tier(mut self, tier: impl Into<String>, model: Model) -> Self {
        self.tiers.insert(tier.into(), model);
        self
    }
}

#[async_trait]
impl Provider for TierProvider {
    async fn stream(
        &self,
        model: &Model,
        messages: &Vec<Message>,
        tools: &Vec<ToolDefinition>,
    ) -> Result<StreamResponseStream, ProviderError> {
        let target = self
            .tiers
            .get(&model.id)
            .ok_or_else(|| ProviderError::ModelNotFound(model.id.clone()))?;
        let provider = self.providers.get(&target.provider).ok_or_else(|| {
            ProviderError::RequestFailed(format!(
                "no provider '{}' registered for tier '{}'",
                target.provider, model.id
            ))
        })?;
        provider.stream(target, messages, tools).await
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
    use futures::{stream, StreamExt};

    /// A backend that records the model id it was asked to stream and returns a
    /// single deterministic chunk. Shares its log via `Arc` so the test can read it
    /// after the call, even though the provider is registered into the tier.
    struct RecordingProvider {
        name: &'static str,
        seen: Arc<Mutex<Vec<String>>>,
    }

    #[async_trait]
    impl Provider for RecordingProvider {
        async fn stream(
            &self,
            model: &Model,
            _messages: &Vec<Message>,
            _tools: &Vec<ToolDefinition>,
        ) -> Result<StreamResponseStream, ProviderError> {
            self.seen.lock().unwrap().push(model.id.clone());
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

    fn target_model(provider: &str, id: &str) -> Model {
        Model {
            id: id.to_string(),
            provider: provider.to_string(),
            context_window: 0,
            base_url: String::new(),
            headers: HashMap::new(),
        }
    }

    /// A model addressed to the tier provider itself (what the agent would hold).
    fn tier_request(tier: &str) -> Model {
        Model {
            id: tier.to_string(),
            provider: "tier".to_string(),
            context_window: 0,
            base_url: String::new(),
            headers: HashMap::new(),
        }
    }

    fn backend(seen: Arc<Mutex<Vec<String>>>) -> Arc<dyn Provider> {
        Arc::new(RecordingProvider { name: "back", seen })
    }

    async fn drain(stream: StreamResponseStream) {
        stream.collect::<Vec<_>>().await;
    }

    #[tokio::test]
    async fn routes_by_tier_and_passes_target_model_id() {
        let seen = Arc::new(Mutex::new(Vec::new()));
        let tp = TierProvider::new("tier")
            .provider("back", backend(seen.clone()))
            .tier("high", target_model("back", "real-high"))
            .tier("mid", target_model("back", "real-mid"));

        drain(tp.stream(&tier_request("high"), &vec![], &vec![]).await.unwrap()).await;
        drain(tp.stream(&tier_request("mid"), &vec![], &vec![]).await.unwrap()).await;

        // The backend saw the *target* model ids, never the tier names.
        let observed = seen.lock().unwrap().clone();
        assert_eq!(observed, vec!["real-high".to_string(), "real-mid".to_string()]);
    }

    #[tokio::test]
    async fn unknown_tier_is_model_not_found() {
        let seen = Arc::new(Mutex::new(Vec::new()));
        let tp = TierProvider::new("tier")
            .provider("back", backend(seen))
            .tier("high", target_model("back", "real-high"));

        let result = tp.stream(&tier_request("nope"), &vec![], &vec![]).await;
        assert!(matches!(result, Err(ProviderError::ModelNotFound(ref s)) if s == "nope"));
    }

    #[tokio::test]
    async fn missing_target_provider_is_request_failed() {
        // No backends registered: the tier points at a provider nobody registered.
        let tp = TierProvider::new("tier").tier("high", target_model("no-such-provider", "real-high"));

        let result = tp.stream(&tier_request("high"), &vec![], &vec![]).await;
        assert!(matches!(result, Err(ProviderError::RequestFailed(_))));
    }

    #[test]
    fn name_returns_configured() {
        let tp = TierProvider::new("tier");
        assert_eq!(tp.name(), "tier");
    }

    #[tokio::test]
    async fn two_distinct_backends_no_crosstalk() {
        let seen_a = Arc::new(Mutex::new(Vec::new()));
        let seen_b = Arc::new(Mutex::new(Vec::new()));
        let tp = TierProvider::new("tier")
            .provider("a", Arc::new(RecordingProvider { name: "a", seen: seen_a.clone() }))
            .provider("b", Arc::new(RecordingProvider { name: "b", seen: seen_b.clone() }))
            .tier("high", target_model("a", "real-high"))
            .tier("mid", target_model("b", "real-mid"));

        drain(tp.stream(&tier_request("high"), &vec![], &vec![]).await.unwrap()).await;
        drain(tp.stream(&tier_request("mid"), &vec![], &vec![]).await.unwrap()).await;

        assert_eq!(seen_a.lock().unwrap().as_slice(), &["real-high".to_string()]);
        assert_eq!(seen_b.lock().unwrap().as_slice(), &["real-mid".to_string()]);
    }
}

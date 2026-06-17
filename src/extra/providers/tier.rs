use std::sync::Arc;

use async_trait::async_trait;

use crate::core::providers::{Model, Provider, ProviderError, StreamResponseStream};
use crate::core::types::{Message, ToolDefinition};
use crate::extra::providers::r#virtual::VirtualProvider;

/// A routing provider that exposes ordered tier indices (`"0"`, `"1"`, `"2"`, ...) as
/// model ids and delegates each request to a concrete provider + model.
///
/// It is built from an ordered list of tier [`Model`]s `[t0, t1, t2, ...]` — where `t0`
/// is the best (highest-tier) model — and a list of named backends. Each tier model's
/// `provider` field names one of those backends. Routing to the backends is delegated
/// to an internal [`VirtualProvider`].
///
/// At stream time the caller addresses a tier by encoding its index in `model.id`
/// (`"0"` selects `t0`, `"1"` selects `t1`, ...). The index is clamped to the last tier
/// when it exceeds the number of tiers, so any sufficiently large number resolves to the
/// lowest-tier model. The selected tier model is then forwarded — unchanged — to the
/// internal virtual provider, which dispatches by the model's `provider` field. The tier
/// index never reaches a backend.
///
/// As a special case, if the incoming `model.provider` is non-empty the caller is
/// addressing a backend directly and tier selection is bypassed: the model is forwarded
/// unchanged to the internal virtual provider, which routes by `model.provider`. This
/// lets the same provider serve both tier-index lookups (`model.id = "0"` with an empty
/// `provider`) and direct backend access (`model.id = "<real model>"` with
/// `provider = "<backend>"`).
pub struct TierProvider {
    name: String,
    inner: VirtualProvider,
    tiers: Vec<Model>,
}

impl TierProvider {
    /// Create a new tier provider.
    ///
    /// `tiers` is the ordered model list (`tiers[0]` is the best); `providers` is the
    /// set of named backends tiers can route to. Each tier model's `provider` must name
    /// one of the registered backends.
    pub fn new(
        name: impl Into<String>,
        tiers: Vec<Model>,
        providers: impl IntoIterator<Item = (impl Into<String>, Arc<dyn Provider>)>,
    ) -> Self {
        let name = name.into();
        let mut inner = VirtualProvider::new(name.clone());
        for (backend_name, provider) in providers {
            inner = inner.provider(backend_name, provider);
        }
        Self { name, inner, tiers }
    }
}

#[async_trait]
impl Provider for TierProvider {
    async fn stream<'a>(
        &self,
        model: &Model,
        messages: Box<dyn Iterator<Item = &'a Message> + Send + 'a>,
        tools: &Vec<ToolDefinition>,
    ) -> Result<StreamResponseStream, ProviderError> {
        // If the caller already named a backend, bypass tier selection and forward the
        // model directly to the underlying virtual provider.
        if !model.provider.is_empty() {
            return self.inner.stream(model, messages, tools).await;
        }
        if self.tiers.is_empty() {
            return Err(ProviderError::RequestFailed(format!(
                "no tiers configured for tier provider '{}'",
                self.name
            )));
        }
        // "0" -> tiers[0] (best), "1" -> tiers[1], ...; non-numeric ids are unknown tiers.
        let requested = model
            .id
            .parse::<usize>()
            .map_err(|_| ProviderError::ModelNotFound(model.id.clone()))?;
        // Clamp to the last tier when the requested index runs past the end.
        let index = requested.min(self.tiers.len() - 1);
        let target = &self.tiers[index];
        self.inner.stream(target, messages, tools).await
    }

    fn name(&self) -> &str {
        &self.name
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::HashMap;
    use std::sync::Mutex;

    use crate::core::providers::StreamResponse;
    use crate::core::types::{ContentBlock, TextContent, TokenUsage};
    use futures::{StreamExt, stream};

    /// A backend that records the model id it was asked to stream and returns a
    /// single deterministic chunk. Shares its log via `Arc` so the test can read it
    /// after the call, even though the provider is registered into the tier.
    struct RecordingProvider {
        name: &'static str,
        seen: Arc<Mutex<Vec<String>>>,
    }

    #[async_trait]
    impl Provider for RecordingProvider {
        async fn stream<'a>(
            &self,
            model: &Model,
            _messages: Box<dyn Iterator<Item = &'a Message> + Send + 'a>,
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

    /// A request addressed to the tier provider by index (what the agent would hold).
    /// `provider` is empty so tier selection kicks in.
    fn tier_request(index: &str) -> Model {
        Model {
            id: index.to_string(),
            provider: String::new(),
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
    async fn routes_by_index_with_first_tier_best() {
        let seen = Arc::new(Mutex::new(Vec::new()));
        let tp = TierProvider::new(
            "tier",
            vec![target_model("back", "real-high"), target_model("back", "real-mid")],
            vec![("back", backend(seen.clone()))],
        );

        drain(tp.stream(&tier_request("0"), Box::new(std::iter::empty::<&Message>()), &vec![]).await.unwrap()).await;
        drain(tp.stream(&tier_request("1"), Box::new(std::iter::empty::<&Message>()), &vec![]).await.unwrap()).await;

        // "0" -> best (real-high), "1" -> real-mid; the backend saw target ids, never the indices.
        let observed = seen.lock().unwrap().clone();
        assert_eq!(observed, vec!["real-high".to_string(), "real-mid".to_string()]);
    }

    #[tokio::test]
    async fn non_empty_provider_bypasses_tier_selection() {
        let seen = Arc::new(Mutex::new(Vec::new()));
        let tp = TierProvider::new(
            "tier",
            vec![target_model("back", "tier-model")],
            vec![("back", backend(seen.clone()))],
        );

        // The caller addresses the backend directly: model.id is the real model id and
        // model.provider names the backend. Tier selection is skipped and the model is
        // forwarded verbatim (the backend never sees the tier's "tier-model").
        let direct = Model {
            id: "direct-model".to_string(),
            provider: "back".to_string(),
            context_window: 0,
            base_url: String::new(),
            headers: HashMap::new(),
        };
        drain(tp.stream(&direct, Box::new(std::iter::empty::<&Message>()), &vec![]).await.unwrap()).await;

        let observed = seen.lock().unwrap().clone();
        assert_eq!(observed, vec!["direct-model".to_string()]);
    }

    #[tokio::test]
    async fn bypass_with_unknown_provider_does_not_fall_back_to_tiers() {
        // No backends registered, but a tier exists. A bypass request names a provider
        // nobody knows: it must surface the virtual provider's error rather than be
        // reinterpreted as tier selection.
        let tp = TierProvider::new("tier", vec![target_model("back", "tier-model")], Vec::<(&str, Arc<dyn Provider>)>::new());

        let direct = Model {
            id: "direct-model".to_string(),
            provider: "ghost".to_string(),
            context_window: 0,
            base_url: String::new(),
            headers: HashMap::new(),
        };
        let result = tp.stream(&direct, Box::new(std::iter::empty::<&Message>()), &vec![]).await;
        assert!(matches!(result, Err(ProviderError::RequestFailed(_))));
    }


    #[tokio::test]
    async fn out_of_range_index_clamps_to_last_tier() {
        let seen = Arc::new(Mutex::new(Vec::new()));
        let tp = TierProvider::new(
            "tier",
            vec![target_model("back", "real-high"), target_model("back", "real-mid")],
            vec![("back", backend(seen.clone()))],
        );

        // 5 far exceeds the two tiers -> resolves to the last one.
        drain(tp.stream(&tier_request("5"), Box::new(std::iter::empty::<&Message>()), &vec![]).await.unwrap()).await;

        let observed = seen.lock().unwrap().clone();
        assert_eq!(observed, vec!["real-mid".to_string()]);
    }

    #[tokio::test]
    async fn non_numeric_index_is_model_not_found() {
        let seen = Arc::new(Mutex::new(Vec::new()));
        let tp = TierProvider::new("tier", vec![target_model("back", "real-high")], vec![("back", backend(seen))]);

        let result = tp.stream(&tier_request("nope"), Box::new(std::iter::empty::<&Message>()), &vec![]).await;
        assert!(matches!(result, Err(ProviderError::ModelNotFound(ref s)) if s == "nope"));
    }

    #[tokio::test]
    async fn missing_target_provider_is_request_failed() {
        // The only tier points at a backend nobody registered.
        let tp = TierProvider::new(
            "tier",
            vec![target_model("no-such-provider", "real-high")],
            Vec::<(&str, Arc<dyn Provider>)>::new(),
        );

        let result = tp.stream(&tier_request("0"), Box::new(std::iter::empty::<&Message>()), &vec![]).await;
        assert!(matches!(result, Err(ProviderError::RequestFailed(_))));
    }

    #[tokio::test]
    async fn empty_tiers_is_request_failed() {
        let tp = TierProvider::new("tier", vec![], Vec::<(&str, Arc<dyn Provider>)>::new());

        let result = tp.stream(&tier_request("0"), Box::new(std::iter::empty::<&Message>()), &vec![]).await;
        assert!(matches!(result, Err(ProviderError::RequestFailed(_))));
    }

    #[test]
    fn name_returns_configured() {
        let tp = TierProvider::new("tier", vec![], Vec::<(&str, Arc<dyn Provider>)>::new());
        assert_eq!(tp.name(), "tier");
    }

    #[tokio::test]
    async fn two_distinct_backends_no_crosstalk() {
        let seen_a = Arc::new(Mutex::new(Vec::new()));
        let seen_b = Arc::new(Mutex::new(Vec::new()));
        let tp = TierProvider::new(
            "tier",
            vec![target_model("a", "real-high"), target_model("b", "real-mid")],
            vec![
                ("a", Arc::new(RecordingProvider { name: "a", seen: seen_a.clone() }) as Arc<dyn Provider>),
                ("b", Arc::new(RecordingProvider { name: "b", seen: seen_b.clone() }) as Arc<dyn Provider>),
            ],
        );

        drain(tp.stream(&tier_request("0"), Box::new(std::iter::empty::<&Message>()), &vec![]).await.unwrap()).await;
        drain(tp.stream(&tier_request("1"), Box::new(std::iter::empty::<&Message>()), &vec![]).await.unwrap()).await;

        assert_eq!(seen_a.lock().unwrap().as_slice(), &["real-high".to_string()]);
        assert_eq!(seen_b.lock().unwrap().as_slice(), &["real-mid".to_string()]);
    }
}

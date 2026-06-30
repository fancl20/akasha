//! Default base agent template.
//!
//! [`AgentBuilder::base`] is [`AgentBuilder::new`] pre-wired with the
//! cross-cutting extensions every real agent uses — [`SchemaVerification`] then
//! [`CircuitBreaker`] — so a caller gets schema validation and output bounding
//! without folding the chain by hand. It returns a partially-configured
//! [`AgentBuilder`]: keep chaining tools/extensions and call `.build()`.
//!
//! [`AgentBuilder`]: super::builder::AgentBuilder
//! [`SchemaVerification`]: crate::extra::extensions::schema::SchemaVerification
//! [`CircuitBreaker`]: crate::extra::extensions::circuit_breaker::CircuitBreaker

use std::sync::{Arc, Mutex};

use crate::core::providers::{Model, Provider};
use crate::core::session::Session;
use crate::extra::agents::builder::AgentBuilder;
use crate::extra::extensions::circuit_breaker::CircuitBreaker;
use crate::extra::extensions::schema::SchemaVerification;

impl AgentBuilder {
    /// A default base template: [`new`](AgentBuilder::new) with the cross-cutting
    /// extension combo every real agent uses —
    /// [`SchemaVerification`](crate::extra::extensions::schema::SchemaVerification)
    /// then
    /// [`CircuitBreaker`](crate::extra::extensions::circuit_breaker::CircuitBreaker)
    /// — already chained. Returns a partially-configured builder; add
    /// tools/extensions and finish with [`.build()`](AgentBuilder::build).
    ///
    /// The standard pair runs first, so anything appended via
    /// [`.extension()`](AgentBuilder::extension) lands after them — e.g. a mux
    /// fallback ends up innermost, matching `And(Schema, And(Circuit, ext))`.
    pub fn base(model: Model, provider: Arc<dyn Provider>, session: Arc<Mutex<dyn Session>>) -> Self {
        AgentBuilder::new(model, provider, session)
            .extension(SchemaVerification::new())
            .extension(CircuitBreaker::new())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    use std::collections::HashMap;

    use async_trait::async_trait;

    use crate::core::providers::{ProviderError, StreamResponseStream};
    use crate::core::session::InMemorySession;
    use crate::core::types::{Message, ToolDefinition};

    /// A provider that is never actually streamed from — `base` only needs a
    /// type implementing [`Provider`] to construct the builder; these tests
    /// build and inspect, never prompt.
    struct NullProvider;

    #[async_trait]
    impl Provider for NullProvider {
        async fn stream<'a>(
            &self,
            _: &Model,
            _: Box<dyn Iterator<Item = &'a Message> + Send + 'a>,
            _: &Vec<ToolDefinition>,
        ) -> Result<StreamResponseStream, ProviderError> {
            unreachable!("base template tests never prompt the model")
        }

        fn name(&self) -> &str {
            "null"
        }
    }

    fn model() -> Model {
        Model {
            id: "m".to_string(),
            provider: "p".to_string(),
            context_window: 0,
            base_url: String::new(),
            headers: HashMap::new(),
        }
    }

    fn provider() -> Arc<dyn Provider> {
        Arc::new(NullProvider)
    }

    fn session() -> Arc<Mutex<dyn Session>> {
        InMemorySession::new().arc()
    }

    /// `base` wires Schema + Circuit (two extensions → `And`); no tools are
    /// enabled (deny-all), and the model threads through untouched.
    #[tokio::test]
    async fn base_wires_schema_then_circuit() {
        let agent = AgentBuilder::base(model(), provider(), session()).build().await.unwrap();
        assert_eq!(agent.extension.name(), "and");
        assert!(agent.state.tools.definitions().is_empty(), "deny-all: nothing enabled");
        assert_eq!(agent.state.model.id, "m");
    }
}

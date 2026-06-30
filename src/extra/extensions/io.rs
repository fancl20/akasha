//! Channel-backed I/O bridge between an [`Agent`] and any transport.
//!
//! [`IOExtension`] wires an agent's streaming output and turn-by-turn input to a
//! pair of unbounded channels: content blocks and tool-call notifications flow
//! **out** as [`OutputEvent`]s, and the next user message flows **in** at each
//! turn boundary.
//!
//! [`IOExtension::new`] returns the extension alongside the inbound message
//! sender and the outbound event receiver; add it to an agent's extension chain
//! **last**, so its turn-end input gating runs after every other extension. The
//! caller spawns `agent.prompt(first)` itself — the bridge owns no task of its
//! own.
//!
//! [`Agent`]: crate::core::agent::Agent
//! [`Session`]: crate::core::session::Session
//! [`Extension`]: crate::core::extensions::Extension

use async_trait::async_trait;
use tokio::sync::{mpsc, oneshot};

use crate::core::agent::AgentState;
use crate::core::extensions::{Extension, ExtensionError, ToolCallDecision};
use crate::core::providers::StreamResponse;
use crate::core::types::{ContentBlock, Message};

/// One unit of agent output a transport renders.
///
/// `Finish` carries the ack sender so the transport signals once it has flushed
/// the turn's output; the agent then waits for the next inbound message before
/// continuing, gating the conversation to the transport's pace.
pub enum OutputEvent {
    /// A streamed content block from the assistant message in progress.
    Append(ContentBlock),
    /// A tool was called; a transport may surface it as an activity notice.
    Notification(String),
    /// The agent finished a turn. Ack the sender once the output is flushed.
    Finish(oneshot::Sender<Result<(), ExtensionError>>),
}

impl IOExtension {
    /// Construct the extension and its transport channels: `tx` feeds inbound
    /// [`Message`]s (one per turn), `rx` drains outbound [`OutputEvent`]s. Add
    /// the returned extension to an agent's chain **last** so its turn-end input
    /// gating runs after every other extension.
    pub fn new() -> (IOExtension, mpsc::UnboundedSender<Message>, mpsc::UnboundedReceiver<OutputEvent>) {
        let (event_tx, event_rx) = mpsc::unbounded_channel::<OutputEvent>();
        let (msg_tx, msg_rx) = mpsc::unbounded_channel::<Message>();
        (IOExtension { tx: event_tx, rx: msg_rx }, msg_tx, event_rx)
    }
}

/// [`Extension`] that bridges an agent to a transport over two channels.
///
/// Streaming output (content blocks, tool-call notices) is sent out as
/// [`OutputEvent`]s; at each turn end it signals [`OutputEvent::Finish`], waits
/// for the transport's ack, then takes the next inbound message and appends it
/// to the session — driving a multi-turn conversation from outside the agent.
///
/// Constructed by [`IOExtension::new`]; a transport never builds one directly.
pub struct IOExtension {
    tx: mpsc::UnboundedSender<OutputEvent>,
    rx: mpsc::UnboundedReceiver<Message>,
}

fn dropped() -> ExtensionError {
    ExtensionError::ExtensionFailed { name: "io".to_string(), message: "transport channel dropped".to_string() }
}

#[async_trait]
impl Extension for IOExtension {
    fn name(&self) -> &str {
        "io"
    }

    async fn on_message_update(&mut self, chunk: &StreamResponse) -> Result<(), ExtensionError> {
        for block in &chunk.message.content {
            self.tx.send(OutputEvent::Append(block.clone())).map_err(|_| dropped())?;
        }
        Ok(())
    }

    async fn on_tool_execution_start(
        &mut self,
        _tool_call_id: &str,
        name: &str,
        args: &serde_json::Value,
    ) -> Result<ToolCallDecision, ExtensionError> {
        let notification = format!("{name}: {args}");
        self.tx.send(OutputEvent::Notification(notification)).map_err(|_| dropped())?;
        Ok(ToolCallDecision::Allow)
    }

    async fn on_turn_end(&mut self, state: AgentState) -> Result<AgentState, ExtensionError> {
        let (tx, rx) = oneshot::channel();
        self.tx.send(OutputEvent::Finish(tx)).map_err(|_| dropped())?;
        // The ack carries its own Result (transport flush status); `let _` discards
        // the inner value once the await itself has succeeded.
        let _ = rx.await.map_err(|_| dropped())?;

        let msg = self.rx.recv().await.ok_or_else(dropped)?;
        state
            .session
            .lock()
            .unwrap()
            .append(msg)
            .map_err(|e| ExtensionError::ExtensionFailed { name: "io".to_string(), message: e.to_string() })?;
        Ok(state)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::Arc;

    use crate::core::agent::Agent;
    use crate::core::providers::{Model, Provider, ProviderError, StreamResponse, StreamResponseStream};
    use crate::core::session::{InMemorySession, Session};
    use crate::core::tools::ToolRegistry;
    use crate::core::types::{TextContent, TokenUsage, ToolDefinition};
    use async_trait::async_trait;
    use futures::stream;
    use std::collections::HashMap;
    use std::time::Duration;
    use tokio::time::timeout;

    /// Streams one fixed assistant text message per `stream()` call — enough to
    /// drive the agent loop one turn (text-only, no tool calls).
    struct TextProvider(Message);

    #[async_trait]
    impl Provider for TextProvider {
        async fn stream<'a>(
            &self,
            _model: &Model,
            _messages: Box<dyn Iterator<Item = &'a Message> + Send + 'a>,
            _tools: &Vec<ToolDefinition>,
        ) -> Result<StreamResponseStream, ProviderError> {
            let resp = StreamResponse {
                message: self.0.clone(),
                usage: TokenUsage {
                    input_tokens: 0,
                    output_tokens: 0,
                    cache_read_tokens: None,
                    cache_write_tokens: None,
                },
                stop_reason: Some("stop".to_string()),
            };
            Ok(Box::pin(stream::iter(vec![resp])))
        }

        fn name(&self) -> &str {
            "text"
        }
    }

    fn assistant(text: &str) -> Message {
        Message {
            role: "assistant".to_string(),
            content: vec![ContentBlock::Text(TextContent { content: text.to_string() })],
        }
    }

    fn user(text: &str) -> Message {
        Message {
            role: "user".to_string(),
            content: vec![ContentBlock::Text(TextContent { content: text.to_string() })],
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

    /// Receive the next event, failing (with a timeout) if the bridge stalls.
    async fn next(rx: &mut mpsc::UnboundedReceiver<OutputEvent>) -> OutputEvent {
        timeout(Duration::from_secs(2), rx.recv()).await.expect("event within timeout").expect("event channel open")
    }

    fn expect_append_text(ev: OutputEvent, expect: &str) {
        match ev {
            OutputEvent::Append(ContentBlock::Text(t)) => assert_eq!(t.content, expect),
            _ => panic!("expected Append(Text)"),
        }
    }

    /// An `IOExtension` (from [`IOExtension::new`]) wired into an agent + a
    /// caller-spawned `agent.prompt` task streams the agent's output, gates on
    /// `Finish` between turns, and advances to a fresh turn when a message
    /// arrives on the inbound channel — the transport-side contract, independent
    /// of any concrete transport.
    #[tokio::test]
    async fn streams_output_gates_on_finish_and_feeds_next_turn() {
        let (io, tx, mut rx) = IOExtension::new();
        let mut agent = Agent {
            state: AgentState { model: model(), tools: ToolRegistry::new(), session: InMemorySession::new().arc() },
            provider: Arc::new(TextProvider(assistant("reply"))),
            extension: Box::new(io),
        };

        // The caller starts the agent task itself; the bridge carries the first prompt.
        let task = tokio::spawn(async move { agent.prompt(user("hello")).await });

        // Turn 1: the streamed reply, then a finish handshake.
        expect_append_text(next(&mut rx).await, "reply");
        let ack = match next(&mut rx).await {
            OutputEvent::Finish(tx) => tx,
            _ => panic!("expected Finish after the turn"),
        };
        ack.send(Ok(())).unwrap();
        tx.send(user("again")).unwrap();

        // Turn 2: the feed drove a fresh turn — same contract repeats.
        expect_append_text(next(&mut rx).await, "reply");
        assert!(matches!(next(&mut rx).await, OutputEvent::Finish(_)), "turn 2 also finishes");

        task.abort();
    }
}

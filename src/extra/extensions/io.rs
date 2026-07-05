//! Channel-backed I/O bridge between an [`Agent`] and any transport.
//!
//! [`IoExtension`] wires an agent's streaming output and turn-by-turn input to a
//! pair of unbounded channels: content blocks and tool-call results flow **out**
//! as [`OutputEvent`]s, and the next user message flows **in** at each turn
//! boundary.
//!
//! [`IoExtension::new`] returns the extension alongside the inbound message
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
use crate::core::extensions::{Extension, ExtensionError};
use crate::core::providers::StreamResponse;
use crate::core::tools::ToolError;
use crate::core::types::{ContentBlock, Message, TextContent, ToolResult, ToolResultContent};

/// One unit of agent output a transport renders.
///
/// `Finish` carries the ack sender so the transport signals once it has flushed
/// the turn's output; the agent then waits for the next inbound message before
/// continuing, gating the conversation to the transport's pace.
pub enum OutputEvent {
    /// A streamed content block from the assistant message in progress. A
    /// `ContentBlock::ToolCall` here opens a tool call; its matching [`ToolEnd`]
    /// carries the result.
    ///
    /// [`ToolEnd`]: OutputEvent::ToolEnd
    Append(ContentBlock),
    /// A tool call resolved — its result is ready to render. Paired with an
    /// earlier `Append(ContentBlock::ToolCall)` carrying the same `id`. Emitted
    /// for both successful and failed executions (`is_error` set on failure).
    ToolEnd { id: String, result: ToolResult },
    /// The agent finished a turn. Ack the sender once the output is flushed.
    Finish(oneshot::Sender<Result<(), ExtensionError>>),
}

impl IoExtension {
    /// Construct the extension and its transport channels: `tx` feeds inbound
    /// [`Message`]s (one per turn), `rx` drains outbound [`OutputEvent`]s. Add
    /// the returned extension to an agent's chain **last** so its turn-end input
    /// gating runs after every other extension.
    pub fn new() -> (IoExtension, mpsc::UnboundedSender<Message>, mpsc::UnboundedReceiver<OutputEvent>) {
        let (event_tx, event_rx) = mpsc::unbounded_channel::<OutputEvent>();
        let (msg_tx, msg_rx) = mpsc::unbounded_channel::<Message>();
        (IoExtension { tx: event_tx, rx: msg_rx }, msg_tx, event_rx)
    }
}

/// [`Extension`] that bridges an agent to a transport over two channels.
///
/// Streaming output (content blocks, tool-call notices) is sent out as
/// [`OutputEvent`]s; at each turn end it signals [`OutputEvent::Finish`], waits
/// for the transport's ack, then takes the next inbound message and appends it
/// to the session — driving a multi-turn conversation from outside the agent.
///
/// Constructed by [`IoExtension::new`]; a transport never builds one directly.
pub struct IoExtension {
    tx: mpsc::UnboundedSender<OutputEvent>,
    rx: mpsc::UnboundedReceiver<Message>,
}

fn dropped() -> ExtensionError {
    ExtensionError::ExtensionFailed { name: "io".to_string(), message: "transport channel dropped".to_string() }
}

#[async_trait]
impl Extension for IoExtension {
    fn name(&self) -> &str {
        "io"
    }

    async fn on_message_update(&mut self, chunk: &StreamResponse) -> Result<(), ExtensionError> {
        for block in &chunk.message.content {
            self.tx.send(OutputEvent::Append(block.clone())).map_err(|_| dropped())?;
        }
        Ok(())
    }

    async fn tool_execution_end(
        &mut self,
        tool_call_id: &str,
        result: Result<ToolResult, ToolError>,
    ) -> Result<Result<ToolResult, ToolError>, ExtensionError> {
        // Surface the resolved tool call to the transport — synthesizing an
        // error `ToolResult` for failures, since the agent only constructs that
        // shape after this hook returns (see `agent_loop`). The reducer marks
        // status from `is_error`. The original `result` is passed through
        // unchanged so Schema/Circuit (run earlier in the chain) see the raw
        // outcome; here we only observe it for rendering.
        let view = match &result {
            Ok(r) => r.clone(),
            Err(e) => ToolResult {
                tool_call_id: Some(tool_call_id.to_string()),
                content: vec![ToolResultContent::Text(TextContent { content: e.to_string() })],
                is_error: true,
            },
        };
        self.tx.send(OutputEvent::ToolEnd { id: tool_call_id.to_string(), result: view }).map_err(|_| dropped())?;
        Ok(result)
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

    /// An `IoExtension` (from [`IoExtension::new`]) wired into an agent + a
    /// caller-spawned `agent.prompt` task streams the agent's output, gates on
    /// `Finish` between turns, and advances to a fresh turn when a message
    /// arrives on the inbound channel — the transport-side contract, independent
    /// of any concrete transport.
    #[tokio::test]
    async fn streams_output_gates_on_finish_and_feeds_next_turn() {
        let (io, tx, mut rx) = IoExtension::new();
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

    /// `tool_execution_end` emits a `ToolEnd` carrying the result for both
    /// success and failure, and passes the original result through unchanged.
    #[tokio::test]
    async fn tool_execution_end_emits_tool_end_and_passes_through() {
        use crate::core::tools::ToolError;
        use crate::core::types::{ToolResult, ToolResultContent};

        let (io, _tx, mut rx) = IoExtension::new();
        let mut ext: Box<dyn Extension> = Box::new(io);

        // Success: emits ToolEnd with the result; passthrough preserves the Ok.
        let ok = ToolResult {
            tool_call_id: None,
            content: vec![ToolResultContent::Text(TextContent { content: "ok".to_string() })],
            is_error: false,
        };
        let out = ext.tool_execution_end("c1", Ok(ok.clone())).await.unwrap().unwrap();
        assert_eq!(out.content, ok.content);
        match rx.try_recv() {
            Ok(OutputEvent::ToolEnd { id, result }) => {
                assert_eq!(id, "c1");
                assert!(!result.is_error);
            }
            _ => panic!("expected ToolEnd for the ok result"),
        }

        // Failure: synthesizes an error ToolEnd; passthrough keeps the Err.
        let err = ext.tool_execution_end("c2", Err(ToolError::Execution("boom".to_string()))).await;
        assert!(err.is_ok(), "passthrough wraps the inner Err in Ok");
        assert!(matches!(err.unwrap(), Err(ToolError::Execution(_))));
        match rx.try_recv() {
            Ok(OutputEvent::ToolEnd { id, result }) => {
                assert_eq!(id, "c2");
                assert!(result.is_error, "failure surfaces as an error result");
            }
            _ => panic!("expected ToolEnd for the err result"),
        }

        assert!(rx.try_recv().is_err(), "no further events emitted");
    }
}

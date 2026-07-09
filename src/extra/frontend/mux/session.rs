//! Multi-topic session orchestration over [`SessionManager`] + [`ViewModel`].
//!
//! [`Mux`] is a thin, frontend-agnostic orchestrator that gives one user a
//! multi-topic conversation: each topic is a forked sub-session driven as its own
//! [`ViewModel`] run, and an ephemeral router agent decides which topic a message
//! belongs to. It replaces the old bespoke `MuxSession` meta-session by composing
//! existing primitives instead of reimplementing them:
//!
//! - **Topic lifecycle** — [`SessionManager::fork`]/[get](SessionManager::get), the
//!   same primitive the subagent engine uses (no hand-rolled loader).
//! - **User-interaction binding** — [`ViewModel`] attaches input focus + output to
//!   the active topic run; the user talks to it directly (no `pending`/deferred-switch
//!   state machine, no private `Custom`-block control channel).
//! - **Routing** — a one-turn router agent over a throwaway session, so the main
//!   session stays a *lean* topic registry (id + summary per topic).
//!
//! The main [`SessionManager`] **is** the router session: its message history holds
//! the topic registry. Topics are forks of it. A topic that detects a subject change
//! calls the `handoff` tool; `Mux` records a summary, parks the topic, runs the
//! router, and starts/resumes the next topic.
//!
//! [`ViewModel`]: crate::extra::frontend::viewmodel::ViewModel
//! [`SessionManager`]: crate::extra::agents::builder::SessionManager

use std::collections::HashMap;
use std::sync::{Arc, Mutex};

use async_trait::async_trait;

use crate::core::agent::Agent;
use crate::core::extensions::{Extension, ExtensionError};
use crate::core::session::{InMemorySession, Session};
use crate::core::tools::{ToolError, ToolHandler, ToolRegistry};
use crate::core::types::{ContentBlock, Message, TextContent, ToolDefinition, ToolResult, ToolResultContent};
use crate::extra::agents::builder::{AgentBuilder, SessionManager};
use crate::extra::extensions::control::{ControlExtension, fold_message, stop_message};
use crate::extra::frontend::viewmodel::{AgentId, InputRouter, Step, Transcript, ViewModel};

/// Bound the router's context: only the most recent this many registry records are
/// seeded into the ephemeral router session. The main session keeps the full durable
/// log; this just caps per-route token cost (replacing the old daily-id rollover).
const MAX_REGISTRY_RECORDS: usize = 64;

/// Builds a fresh base agent over a given session — the caller's customization point
/// (model, provider, tools, extensions, `tools_enable`). Mux layers the mux machinery
/// on top (`ControlExtension` + `handoff` for topics, `route` for the router), so the
/// factory should build a plain base agent and not add those itself.
type AgentFactory = Arc<dyn Fn(Arc<Mutex<dyn Session>>) -> AgentBuilder + Send + Sync>;

/// Frontend-agnostic multi-topic orchestrator. Owns a [`ViewModel`] of topic runs
/// and the [`SessionManager`] that is both the router registry and the fork source.
///
/// Not a [`Session`] itself — drive it from a render loop with [`Mux::step`], feed
/// messages via the sender from [`Mux::input`], and render with
/// [`Mux::transcript`] / [`Mux::ack`] exactly as a bare `ViewModel`.
pub struct Mux {
    vm: ViewModel,
    mgr: Arc<Mutex<dyn SessionManager>>,
    /// Internal delivery to the focused run (kept private; the frontend never uses it).
    router: Arc<InputRouter>,
    /// Single message hub: the frontend always sends here, and `step` forwards to the
    /// focused topic (a channel hop, no model call) or routes when none is focused.
    inbox_tx: tokio::sync::mpsc::UnboundedSender<Message>,
    inbox: tokio::sync::mpsc::UnboundedReceiver<Message>,
    /// The caller's per-run customization point; see [`AgentFactory`].
    factory: AgentFactory,
    sysprompt: Message,
    /// fork id -> the run driving it (live or ended-but-retained).
    topics: HashMap<String, AgentId>,
    /// per-run handoff capture: a `Some(summary)` means the topic has handed off.
    handoff_slots: HashMap<AgentId, Arc<Mutex<Option<String>>>>,
}

impl Mux {
    /// Build a `Mux` over `mgr` (the main session = router registry + fork source).
    /// `factory` constructs each topic/router agent over the session Mux hands it
    /// (a topic fork, or the router's throwaway session) — the caller's customization
    /// point. `sysprompt` is appended to each fresh topic fork after its `stop`.
    pub fn new(
        mgr: Arc<Mutex<dyn SessionManager>>,
        factory: impl Fn(Arc<Mutex<dyn Session>>) -> AgentBuilder + Send + Sync + 'static,
        sysprompt: Message,
    ) -> Mux {
        let vm = ViewModel::new();
        let router = vm.input();
        let (inbox_tx, inbox) = tokio::sync::mpsc::unbounded_channel::<Message>();
        Mux {
            vm,
            mgr,
            router,
            inbox_tx,
            inbox,
            factory: Arc::new(factory),
            sysprompt,
            topics: HashMap::new(),
            handoff_slots: HashMap::new(),
        }
    }

    /// Cloneable sender for the task that collects user messages. Every message goes
    /// through this single hub; `step` forwards it to the focused topic or routes it.
    pub fn input(&self) -> tokio::sync::mpsc::UnboundedSender<Message> {
        self.inbox_tx.clone()
    }

    /// Borrowed transcript for `id` (delegates to the [`ViewModel`]).
    pub fn transcript(&self, id: AgentId) -> Option<&Transcript> {
        self.vm.transcript(id)
    }

    /// Ack a pending finished turn for `id` (delegates to the [`ViewModel`]).
    pub fn ack(&mut self, id: AgentId, result: Result<(), ExtensionError>) {
        self.vm.ack(id, result);
    }

    /// Pump one event: route an inbound message or reduce a run event. Returns the
    /// next renderable [`Step`] for the focused topic, or `None` at shutdown.
    /// Handoffs are absorbed here (the old topic is finalized, the next is bound);
    /// the caller only renders what this returns.
    pub async fn step(&mut self) -> Option<Step> {
        loop {
            // Both branches borrow disjoint fields (inbox / vm), so the select! can
            // poll them concurrently without starving one for the other.
            let event = tokio::select! {
                msg = self.inbox.recv() => Event::Inbox(msg),
                step = self.vm.step() => Event::Vm(step),
            };
            match event {
                Event::Inbox(None) | Event::Vm(None) => return None,
                Event::Inbox(Some(msg)) => {
                    // Forward to the focused topic (a channel hop, no model call), or
                    // route when nothing is focused (bootstrap / mid-handoff).
                    if self.vm.focus().is_some() {
                        let _ = self.router.send(msg);
                    } else {
                        self.route_to_topic(msg).await;
                    }
                }
                Event::Vm(Some(step)) => {
                    let id = step_id(&step);
                    if let Step::AgentEnded { id } = step {
                        // A handed-off topic ends via StopAfterHandoff; a filled slot
                        // distinguishes a handoff from a crash.
                        let handoff = self.handoff_slots.get(&id).and_then(|s| s.lock().unwrap().take());
                        if let Some(summary) = handoff {
                            self.on_handoff(id, summary).await;
                            // Finalize the old topic's render slot so the frontend
                            // resets before the next topic streams.
                            return Some(Step::TurnFinished { id });
                        }
                        // Real crash: drop our bookkeeping and surface the end.
                        self.handoff_slots.remove(&id);
                        self.topics.retain(|_, run| *run != id);
                        if self.vm.focus() == Some(id) {
                            return Some(Step::AgentEnded { id });
                        }
                        continue;
                    }
                    // Streaming / turn-finish: surface only the focused topic. Events
                    // from detached (parked) topics are silently consumed.
                    if self.vm.focus() == Some(id) {
                        return Some(step);
                    }
                }
            }
        }
    }

    /// Record the summary, find the message that triggered the handoff, park the
    /// topic, and route that message to the next topic.
    async fn on_handoff(&mut self, id: AgentId, summary: String) {
        let topic_id = self.topic_id_of(id);
        // Read the triggering message before mutating the fork (the hidden fold below
        // drops it from the provider view, but it stays in raw storage until then).
        let msg = topic_id.as_deref().and_then(|t| self.last_user_message(t));
        if let Some(tid) = &topic_id {
            let fork = {
                let mut mgr = self.mgr.lock().unwrap();
                let _ = mgr.append(registry_record(tid, &summary));
                mgr.get(tid).ok()
            };
            if let Some(fork) = fork {
                hide_handoff_exchange(&fork);
            }
        }
        // Park the topic: stop rendering + unfocus. Its transcript is retained for
        // scrollback; re-routing to it later restarts a run from the persisted fork.
        self.vm.detach(id, true);
        self.handoff_slots.remove(&id);
        if let Some(msg) = msg {
            self.route_to_topic(msg).await;
        }
    }

    /// Run the ephemeral router over `msg` and bind the topic it selects — starting
    /// a new fork, resuming an ended one, or re-focusing a live one.
    async fn route_to_topic(&mut self, msg: Message) {
        let decision = self.route_decision(msg.clone()).await;
        match decision.topic_id {
            Some(id) if self.topices_has_fork(&id) => self.bind_existing(id, msg).await,
            _ => self.bind_new(msg).await,
        }
    }

    /// True if `id` is a known topic with a persistable fork in the manager.
    fn topices_has_fork(&self, id: &str) -> bool {
        self.mgr.lock().unwrap().get(id).is_ok()
    }

    /// Re-focus a live run, or stop the ended one and restart from its fork.
    async fn bind_existing(&mut self, id: String, msg: Message) {
        let old = self.topics.get(&id).copied();
        if let Some(old) = old && self.vm.is_running(old) {
            self.vm.attach(old, true);
            let _ = self.router.send(msg);
            return;
        }
        if let Some(old) = old {
            self.vm.stop(old);
            self.handoff_slots.remove(&old);
        }
        let Ok(fork) = self.mgr.lock().unwrap().get(&id) else { return };
        self.start_topic(id, fork, msg).await;
    }

    /// Fork a fresh topic session, isolate it from the registry, and start a run.
    async fn bind_new(&mut self, msg: Message) {
        let Ok((id, fork)) = self.mgr.lock().unwrap().fork() else { return };
        {
            let mut f = fork.lock().unwrap();
            // stop_message drops the inherited registry prefix from the provider view;
            // the sysprompt is re-appended after it so the topic still has its instructions.
            let _ = f.append(stop_message());
            let _ = f.append(self.sysprompt.clone());
        }
        self.start_topic(id, fork, msg).await;
    }

    /// Build a topic run over `fork` with a fresh handoff slot and start it.
    async fn start_topic(&mut self, id: String, fork: Arc<Mutex<dyn Session>>, msg: Message) {
        // Only the focused topic renders; detach whatever was focused first.
        if let Some(old) = self.vm.focus() {
            self.vm.detach(old, true);
        }
        let slot = Arc::new(Mutex::new(None::<String>));
        let builder = (self.factory)(fork)
            .extension(ControlExtension::new())
            .tool(HandoffTool { slot: slot.clone() })
            .extension(StopAfterHandoff { slot: slot.clone() })
            .tools_enable(["handoff"]);
        match self.vm.start(builder, msg).await {
            Ok(run) => {
                self.topics.insert(id, run);
                self.handoff_slots.insert(run, slot);
            }
            Err(e) => eprintln!("mux: failed to start topic run: {e}"),
        }
    }

    /// Drive the ephemeral router one turn over `msg` + the recent registry, then
    /// read its `route` tool call. The temp session is discarded, leaving the main
    /// session a lean registry.
    async fn route_decision(&self, msg: Message) -> RouteDecision {
        let temp: Arc<Mutex<dyn Session>> = InMemorySession::new().arc();
        {
            let mut t = temp.lock().unwrap();
            let _ = t.append(Message {
                role: "user".into(),
                content: vec![ContentBlock::Text(TextContent { content: include_str!("prompt_router.md").to_string() })],
            });
            let records: Vec<Message> = self.mgr.lock().unwrap().messages().cloned().collect();
            for record in records.iter().rev().take(MAX_REGISTRY_RECORDS).rev() {
                let _ = t.append(record.clone());
            }
        }

        // Reuse the caller's factory for the router, but narrow its tool pool to just
        // `route` (the factory's own enables can't widen it back — the pool is replaced).
        let mut route_tools = ToolRegistry::new();
        route_tools.register(Box::new(RouteTool));
        let agent = (self.factory)(temp.clone())
            .tools(route_tools)
            .tools_enable(["route"])
            .build()
            .await;
        let mut agent = match agent {
            Ok(a) => a,
            Err(e) => {
                eprintln!("mux: failed to build router: {e}");
                return RouteDecision::default();
            }
        };
        // The router calls `route`, gets a result, emits a final assistant turn, and
        // returns Ok; we read the decision from the (discarded) temp session.
        let _ = Agent::prompt(&mut agent, msg).await;
        let (id, summary) = extract_route(&*temp.lock().unwrap());
        RouteDecision { topic_id: id, summary }
    }

    fn topic_id_of(&self, run: AgentId) -> Option<String> {
        self.topics.iter().find_map(|(t, r)| (*r == run).then(|| t.clone()))
    }

    /// The last user message in a topic's fork — the one that triggered a handoff.
    fn last_user_message(&self, topic_id: &str) -> Option<Message> {
        let fork = self.mgr.lock().unwrap().get(topic_id).ok()?;
        let fork = fork.lock().unwrap();
        fork.messages().rev().find(|m| m.role == "user").cloned()
    }
}

/// Which the router selected: resume `topic_id`, or `None` to start a new topic.
#[derive(Default)]
struct RouteDecision {
    topic_id: Option<String>,
    #[allow(dead_code)]
    summary: String,
}

enum Event {
    Inbox(Option<Message>),
    Vm(Option<Step>),
}

fn step_id(step: &Step) -> AgentId {
    match step {
        Step::Updated { id } | Step::TurnFinished { id } | Step::AgentEnded { id } => *id,
    }
}

/// Build a registry record (id + summary) appended to the main session on handoff.
fn registry_record(topic_id: &str, summary: &str) -> Message {
    Message {
        role: "user".into(),
        content: vec![
            ContentBlock::Text(TextContent { content: format!("topic_id: {topic_id}\n") }),
            ContentBlock::Text(TextContent { content: format!("summary: {summary}") }),
        ],
    }
}

/// Append a hidden fold over the handoff exchange — the triggering user message, the
/// `handoff` tool call, and its result — so re-entering the topic does not surface its
/// own handoff in the provider view. The range is `[last_user_idx, len)`: the last user
/// message (the one that triggered the handoff) and every message after it. Empty text
/// ⇒ the range is dropped with no placeholder (the originals stay in raw storage).
fn hide_handoff_exchange(fork: &Arc<Mutex<dyn Session>>) {
    let mut f = fork.lock().unwrap();
    let Some(start) = (|| {
        let msgs: Vec<&Message> = f.messages().collect();
        let start = msgs.iter().rposition(|m| m.role == "user")?;
        (start < msgs.len()).then_some(start)
    })() else {
        return;
    };
    let n = f.messages().count();
    let (_, fold) = fold_message(start, n, "");
    let _ = f.append(fold);
}

/// Scan the router's temp session for its `route` tool call.
fn extract_route(temp: &dyn Session) -> (Option<String>, String) {
    for m in temp.messages().rev() {
        if m.role != "assistant" {
            continue;
        }
        for block in &m.content {
            if let ContentBlock::ToolCall(tc) = block
                && tc.name == "route"
            {
                let id = tc
                    .arguments
                    .get("topic_id")
                    .and_then(|v| v.as_str())
                    .filter(|s| !s.is_empty())
                    .map(str::to_string);
                let summary = tc
                    .arguments
                    .get("summary")
                    .and_then(|v| v.as_str())
                    .unwrap_or("topic")
                    .to_string();
                return (id, summary);
            }
        }
    }
    (None, "topic".to_string())
}

// ─── router tool ──────────────────────────────────────────────────────────────

/// The router's only tool. It records nothing — `Mux` reads the call from the
/// discarded temp session — and simply confirms so the router's turn ends cleanly.
struct RouteTool;

#[async_trait]
impl ToolHandler for RouteTool {
    fn definition(&self) -> ToolDefinition {
        ToolDefinition {
            name: "route".to_string(),
            description: "Select the topic to handle the latest user message. Pass `topic_id` to \
                resume an existing topic (matched against the listed summaries); omit it to start a \
                new topic. Always include a short `summary` of the topic."
                .to_string(),
            parameters: serde_json::json!({
                "type": "object",
                "properties": {
                    "topic_id": {
                        "type": "string",
                        "description": "topic_id of an existing topic to resume; omit or empty for a new topic."
                    },
                    "summary": {
                        "type": "string",
                        "description": "Short summary of the topic, used for future routing."
                    }
                },
                "required": ["summary"]
            }),
        }
    }

    async fn execute(
        &self,
        _cancel: futures::channel::oneshot::Receiver<bool>,
        _params: serde_json::Value,
    ) -> Result<ToolResult, ToolError> {
        Ok(ToolResult {
            tool_call_id: None,
            content: vec![ToolResultContent::Text(TextContent { content: "routed".to_string() })],
            is_error: false,
        })
    }
}

// ─── handoff tool + stop extension ────────────────────────────────────────────

/// Called by a topic LLM that detects a subject change. Captures the `summary` into
/// `slot` (first call wins) and confirms; [`StopAfterHandoff`] then ends the run so
/// there is no extra model call or stray user-facing output after the handoff.
struct HandoffTool {
    slot: Arc<Mutex<Option<String>>>,
}

#[async_trait]
impl ToolHandler for HandoffTool {
    fn definition(&self) -> ToolDefinition {
        ToolDefinition {
            name: "handoff".to_string(),
            description: "Signal that the latest user message belongs to a different topic, with a \
                short `summary` of the current one. Call this instead of answering when the subject \
                has changed."
                .to_string(),
            parameters: serde_json::json!({
                "type": "object",
                "properties": {
                    "summary": {
                        "type": "string",
                        "description": "Short summary of the current topic, used for future routing."
                    }
                },
                "required": ["summary"]
            }),
        }
    }

    async fn execute(
        &self,
        _cancel: futures::channel::oneshot::Receiver<bool>,
        params: serde_json::Value,
    ) -> Result<ToolResult, ToolError> {
        let summary = params
            .get("summary")
            .and_then(|v| v.as_str())
            .unwrap_or("topic")
            .to_string();
        let mut slot = self.slot.lock().unwrap();
        if slot.is_none() {
            *slot = Some(summary);
        }
        Ok(ToolResult {
            tool_call_id: None,
            content: vec![ToolResultContent::Text(TextContent { content: "handed off".to_string() })],
            is_error: false,
        })
    }
}

/// Stops the topic's agent loop once [`HandoffTool`] has captured a summary. Mirrors
/// the subagent engine's `AbortOnYield`: the next `on_message_start` after a capture
/// returns `ExtensionError::Stopped`, ending the run so `Mux` sees `AgentEnded` with a
/// filled slot.
struct StopAfterHandoff {
    slot: Arc<Mutex<Option<String>>>,
}

#[async_trait]
impl Extension for StopAfterHandoff {
    fn name(&self) -> &str {
        "mux/stop-after-handoff"
    }

    async fn on_message_start(&mut self, messages: Vec<Message>) -> Result<Vec<Message>, ExtensionError> {
        if self.slot.lock().unwrap().is_some() {
            return Err(ExtensionError::Stopped {
                name: "mux/stop-after-handoff".to_string(),
                message: "topic handed off; ending run".to_string(),
            });
        }
        Ok(messages)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::core::providers::{Model, Provider, ProviderError, StreamResponse, StreamResponseStream};
    use crate::core::session::InMemorySession;
    use crate::core::types::{TokenUsage, ToolCall};
    use crate::extra::agents::builder::SessionAdapter;
    use async_trait::async_trait;
    use futures::stream;
    use std::collections::{HashMap, VecDeque};
    use std::sync::atomic::{AtomicUsize, Ordering};
    use std::time::{Duration, Instant};
    use tokio::time::timeout;

    /// Replays a queue of scripted assistant messages, one per `stream()` call. The
    /// last call index is exposed so a test can assert how many calls a run made.
    struct ScriptedProvider {
        turns: Arc<Mutex<VecDeque<Message>>>,
        calls: Arc<AtomicUsize>,
        seen: Arc<Mutex<Vec<Vec<ContentBlock>>>>,
    }

    impl ScriptedProvider {
        fn new(turns: Vec<Message>) -> Self {
            Self {
                turns: Arc::new(Mutex::new(turns.into())),
                calls: Arc::new(AtomicUsize::new(0)),
                seen: Arc::new(Mutex::new(Vec::new())),
            }
        }
    }

    #[async_trait]
    impl Provider for ScriptedProvider {
        async fn stream<'a>(
            &self,
            _model: &Model,
            messages: Box<dyn Iterator<Item = &'a Message> + Send + 'a>,
            _tools: &Vec<ToolDefinition>,
        ) -> Result<StreamResponseStream, ProviderError> {
            let blocks: Vec<ContentBlock> = messages.flat_map(|m| m.content.iter().cloned()).collect();
            self.seen.lock().unwrap().push(blocks.clone());
            self.calls.fetch_add(1, Ordering::SeqCst);
            let msg = self.turns.lock().unwrap().pop_front().unwrap_or_else(|| Message {
                role: "assistant".to_string(),
                content: vec![ContentBlock::Text(TextContent { content: String::new() })],
            });
            let stop = if msg.content.iter().any(|b| matches!(b, ContentBlock::ToolCall(_))) {
                "tool_calls"
            } else {
                "stop"
            };
            Ok(Box::pin(stream::iter(vec![StreamResponse {
                message: msg,
                usage: TokenUsage {
                    input_tokens: 0,
                    output_tokens: 0,
                    cache_read_tokens: None,
                    cache_write_tokens: None,
                },
                stop_reason: Some(stop.to_string()),
            }])))
        }
        fn name(&self) -> &str {
            "scripted"
        }
    }

    /// A provider that dispatches by the offered tools: calls where `route` is
    /// offered (the router) draw from one queue; calls where `handoff` is offered
    /// (a topic) draw from another. Lets a test script router and topic turns
    /// independently against a single shared provider.
    struct DualProvider {
        router: Arc<Mutex<VecDeque<Message>>>,
        topic: Arc<Mutex<VecDeque<Message>>>,
        calls: Arc<AtomicUsize>,
    }

    impl DualProvider {
        fn new(router: Vec<Message>, topic: Vec<Message>) -> Self {
            Self {
                router: Arc::new(Mutex::new(router.into())),
                topic: Arc::new(Mutex::new(topic.into())),
                calls: Arc::new(AtomicUsize::new(0)),
            }
        }
    }

    #[async_trait]
    impl Provider for DualProvider {
        async fn stream<'a>(
            &self,
            _model: &Model,
            _messages: Box<dyn Iterator<Item = &'a Message> + Send + 'a>,
            tools: &Vec<ToolDefinition>,
        ) -> Result<StreamResponseStream, ProviderError> {
            self.calls.fetch_add(1, Ordering::SeqCst);
            let is_router = tools.iter().any(|t| t.name == "route");
            let queue = if is_router { self.router.clone() } else { self.topic.clone() };
            let msg = queue.lock().unwrap().pop_front().unwrap_or_else(|| Message {
                role: "assistant".to_string(),
                content: vec![ContentBlock::Text(TextContent { content: String::new() })],
            });
            let stop = if msg.content.iter().any(|b| matches!(b, ContentBlock::ToolCall(_))) {
                "tool_calls"
            } else {
                "stop"
            };
            Ok(Box::pin(stream::iter(vec![StreamResponse {
                message: msg,
                usage: TokenUsage {
                    input_tokens: 0,
                    output_tokens: 0,
                    cache_read_tokens: None,
                    cache_write_tokens: None,
                },
                stop_reason: Some(stop.to_string()),
            }])))
        }
        fn name(&self) -> &str {
            "dual"
        }
    }

    fn model() -> Model {
        Model {
            id: "m".into(),
            provider: "p".into(),
            context_window: 0,
            base_url: String::new(),
            headers: HashMap::new(),
        }
    }

    fn manager() -> Arc<Mutex<dyn SessionManager>> {
        Arc::new(Mutex::new(SessionAdapter::new(InMemorySession::new(), || InMemorySession::new().arc())))
    }

    fn sysprompt() -> Message {
        Message {
            role: "user".into(),
            content: vec![ContentBlock::Text(TextContent { content: "SYSTEM".to_string() })],
        }
    }

    /// A minimal factory: a base agent over the given session. Production callers add
    /// their own tools/extensions/enables here; Mux layers handoff/control on top.
    fn factory(
        model: Model,
        provider: Arc<dyn Provider>,
    ) -> impl Fn(Arc<Mutex<dyn Session>>) -> AgentBuilder + Send + Sync + 'static {
        move |fork| AgentBuilder::base(model.clone(), provider.clone(), fork)
    }

    fn route_call(topic_id: Option<&str>) -> Message {
        let args = match topic_id {
            Some(id) => serde_json::json!({ "topic_id": id, "summary": "s" }),
            None => serde_json::json!({ "summary": "s" }),
        };
        Message {
            role: "assistant".into(),
            content: vec![ContentBlock::ToolCall(ToolCall { id: "r".into(), name: "route".into(), arguments: args })],
        }
    }

    fn text_reply(t: &str) -> Message {
        Message {
            role: "assistant".into(),
            content: vec![ContentBlock::Text(TextContent { content: t.into() })],
        }
    }

    fn handoff_call() -> Message {
        Message {
            role: "assistant".into(),
            content: vec![ContentBlock::ToolCall(ToolCall {
                id: "h".into(),
                name: "handoff".into(),
                arguments: serde_json::json!({ "summary": "leaving" }),
            })],
        }
    }

    fn user(t: &str) -> Message {
        Message {
            role: "user".into(),
            content: vec![ContentBlock::Text(TextContent { content: t.into() })],
        }
    }

    async fn step(mux: &mut Mux) -> Option<Step> {
        // Generous bound: scripted providers finish in milliseconds, but the E2E
        // test drives real model calls (router + topic turns) inside one step.
        timeout(Duration::from_secs(60), mux.step()).await.expect("step within timeout")
    }

    /// Step (acking along the way) until `mux`'s focused run finishes a turn.
    async fn finish_turn(mux: &mut Mux) -> AgentId {
        let deadline = Instant::now() + Duration::from_secs(120);
        loop {
            assert!(deadline > Instant::now(), "timed out waiting for a finished turn");
            match step(mux).await {
                Some(Step::TurnFinished { id }) => {
                    mux.ack(id, Ok(()));
                    return id;
                }
                Some(Step::AgentEnded { id }) => return id,
                Some(_) => {}
                None => panic!("mux stepped to None"),
            }
        }
    }

    /// Step until a run other than `exclude` finishes a turn, acking along the way.
    async fn finish_other_turn(mux: &mut Mux, exclude: AgentId) -> AgentId {
        let deadline = Instant::now() + Duration::from_secs(120);
        loop {
            assert!(deadline > Instant::now(), "timed out waiting for another turn");
            match step(mux).await {
                Some(Step::TurnFinished { id }) => {
                    mux.ack(id, Ok(()));
                    if id != exclude {
                        return id;
                    }
                }
                Some(_) => {}
                None => panic!("mux stepped to None"),
            }
        }
    }

    /// First message with no existing topics: the router starts a new topic, the run
    /// is focused, and the registry prefix is dropped from the topic's provider view.
    #[tokio::test]
    async fn first_message_routes_to_a_new_topic() {
        // router → route(new); topic → reply.
        let provider = Arc::new(ScriptedProvider::new(vec![route_call(None), text_reply("about rust")]));
        let seen = provider.seen.clone();
        let mut mux = Mux::new(manager(), factory(model(), provider), sysprompt());
        let input = mux.input();
        input.send(user("tell me about rust")).unwrap();

        let id = finish_turn(&mut mux).await;
        assert_eq!(mux.transcript(id).unwrap().turns.len(), 1, "one topic turn");
        assert!(mux.vm.focus() == Some(id), "topic is focused");

        // The topic saw SYSTEM + the user message — not the router prompt or any
        // registry record (ControlExtension dropped the inherited fork prefix).
        let last_view = seen.lock().unwrap().last().unwrap().clone();
        let texts: String = last_view
            .iter()
            .filter_map(|b| match b { ContentBlock::Text(t) => Some(t.content.as_str()), _ => None })
            .collect::<Vec<_>>()
            .join("|");
        assert!(texts.contains("SYSTEM"), "topic sees its sysprompt: {texts}");
        assert!(texts.contains("tell me about rust"), "topic sees the user message: {texts}");
        assert!(!texts.contains("route") && !texts.contains("topic_id"), "router/registry noise is dropped: {texts}");
    }

    /// A topic that calls `handoff` ends its run (no second provider call after the
    /// tool result), records a summary, and routes the triggering message to a new
    /// topic — focus moves off the old run.
    #[tokio::test]
    async fn handoff_ends_run_and_routes_to_next_topic() {
        // router: initial route + handoff route (both new).
        // topic: A.turn1 reply, A.turn2 handoff, B.turn1 reply.
        let provider = Arc::new(DualProvider::new(
            vec![route_call(None), route_call(None)],
            vec![text_reply("a"), handoff_call(), text_reply("b")],
        ));
        let calls = provider.calls.clone();
        let mut mux = Mux::new(manager(), factory(model(), provider), sysprompt());
        let input = mux.input();

        input.send(user("rust")).unwrap();
        let a = finish_turn(&mut mux).await; // A turn 1 ("a")
        let calls_after_a = calls.load(Ordering::SeqCst);

        input.send(user("now food")).unwrap();
        // A's handoff surfaces TurnFinished{A}; then B streams and finishes.
        let b = finish_other_turn(&mut mux, a).await;

        assert_ne!(b, a, "a different topic run handled the handoff");
        assert_eq!(mux.vm.focus(), Some(b), "focus moved to the new topic");
        // A.handoff (1) + router re-route (1) + B.turn1 (1) = 3 calls since A.turn1.
        // If StopAfterHandoff failed, A would have made an extra call after the handoff.
        assert_eq!(calls.load(Ordering::SeqCst) - calls_after_a, 3, "no extra provider call after the handoff");
    }

    /// A handoff appends an id+summary registry record to the main session, visible
    /// to the next router turn.
    #[tokio::test]
    async fn handoff_appends_registry_record_visible_to_router() {
        let mgr = manager();
        let provider = Arc::new(DualProvider::new(
            vec![route_call(None), route_call(None)],
            vec![handoff_call()], // A.turn1 hands off immediately
        ));
        let mut mux = Mux::new(mgr.clone(), factory(model(), provider), sysprompt());
        let input = mux.input();

        input.send(user("a")).unwrap();
        let _ = finish_turn(&mut mux).await; // A.turn1 hands off → routed onward; surfaces TurnFinished{A}

        let records: Vec<Message> = mgr.lock().unwrap().messages().cloned().collect();
        let has_summary = records.iter().any(|m| {
            m.content
                .iter()
                .any(|b| matches!(b, ContentBlock::Text(t) if t.content.contains("summary: leaving")))
        });
        assert!(has_summary, "registry should hold A's handoff summary: {records:?}");
    }

    /// A handoff hides its own exchange (triggering message + handoff tool call + result)
    /// in the topic's fork, so resuming the topic later shows a clean history. The
    /// sysprompt is retained; the handoff/trigger do not surface through ControlExtension.
    #[tokio::test]
    async fn handoff_hides_exchange_from_future_topic_view() {
        let mgr = manager();
        let provider = Arc::new(DualProvider::new(
            vec![route_call(None), route_call(None)],
            vec![handoff_call()], // A.turn1 hands off immediately
        ));
        let mut mux = Mux::new(mgr.clone(), factory(model(), provider), sysprompt());
        let input = mux.input();
        input.send(user("trigger msg")).unwrap();
        let a = finish_turn(&mut mux).await;
        let tid = mux.topic_id_of(a).expect("topic has a fork id");

        // Render the fork exactly as the topic's provider would on resume — its run
        // already carries ControlExtension. The exchange must be gone, sysprompt kept.
        let fork = mgr.lock().unwrap().get(&tid).unwrap();
        let raw: Vec<Message> = fork.lock().unwrap().messages().cloned().collect();
        let mut ctrl = ControlExtension::new();
        let rendered = ctrl.on_message_start(raw).await.unwrap();
        let view: String = rendered
            .iter()
            .flat_map(|m| {
                m.content.iter().filter_map(|b| match b {
                    ContentBlock::Text(t) => Some(t.content.as_str()),
                    _ => None,
                })
            })
            .collect::<Vec<_>>()
            .join("|");
        assert!(view.contains("SYSTEM"), "sysprompt retained on resume: {view}");
        assert!(!view.contains("handoff"), "handoff tool call hidden: {view}");
        assert!(!view.contains("trigger msg"), "triggering user message hidden: {view}");
    }

    /// `extract_route` reads the router's decision from the temp session, picking an
    /// existing id when supplied and `None` when omitted.
    #[test]
    fn extract_route_parses_decision() {
        let mut temp = InMemorySession::new();
        temp.append(Message {
            role: "assistant".into(),
            content: vec![ContentBlock::ToolCall(ToolCall {
                id: "r".into(),
                name: "route".into(),
                arguments: serde_json::json!({ "topic_id": "abc", "summary": "x" }),
            })],
        })
        .unwrap();
        let (id, _) = extract_route(&temp);
        assert_eq!(id.as_deref(), Some("abc"));

        let mut temp = InMemorySession::new();
        temp.append(Message {
            role: "assistant".into(),
            content: vec![ContentBlock::ToolCall(ToolCall {
                id: "r".into(),
                name: "route".into(),
                arguments: serde_json::json!({ "summary": "x" }),
            })],
        })
        .unwrap();
        let (id, _) = extract_route(&temp);
        assert!(id.is_none(), "omitted topic_id ⇒ new topic");
    }

    // --- E2E (gated on DEEPSEEK_API_KEY) ---

    fn api_key() -> Option<String> {
        std::env::var("DEEPSEEK_API_KEY").ok()
    }

    fn real_model() -> Model {
        Model {
            id: "deepseek-v4-flash".to_string(),
            provider: "deepseek".to_string(),
            context_window: 64000,
            base_url: String::new(),
            headers: HashMap::new(),
        }
    }

    /// Three prompts across distinct topics. The primary regression guard: every
    /// prompt cycle completes cleanly (turn ends on an assistant message). With a
    /// real model it also confirms routing happened (≥1 handoff ⇒ a registry record).
    #[tokio::test]
    async fn e2e_multi_topic_routes_across_turns() {
        let key = match api_key() {
            Some(k) => k,
            None => return,
        };
        use crate::extra::providers::deepseek::DeepSeekProvider;
        let provider: Arc<dyn Provider> = Arc::new(DeepSeekProvider::new(key));

        let mgr = manager();
        let mgr_for_inspect = mgr.clone();
        let sysprompt = Message {
            role: "user".into(),
            content: vec![ContentBlock::Text(TextContent {
                content: "Answer concisely. If the user's latest message changes the subject, call the \
                    `handoff` tool with a short summary of the current topic instead of answering."
                    .to_string(),
            })],
        };
        let mut mux = Mux::new(mgr, factory(real_model(), provider), sysprompt);
        let input = mux.input();

        for prompt in ["One sentence on the Rust programming language.", "One sentence on Italian cuisine.", "Compare Rust with C++ in two sentences."] {
            input.send(user(prompt)).unwrap();
            let _ = finish_turn(&mut mux).await;
        }

        // Each cycle ended on an assistant turn (finish_turn guarantees it); and the
        // model drove at least one handoff, leaving a registry record in the main session.
        let registry_len = mgr_for_inspect.lock().unwrap().messages().count();
        assert!(registry_len > 0, "mux should have routed across topics (registry empty)");
    }
}

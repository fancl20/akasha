//! Frontend-agnostic projection of one or more agent runs: a redux-style
//! [`Transcript`] reduced from each agent's [`OutputEvent`] stream, driven by a
//! [`ViewModel`] that owns the runs.
//!
//! [`OutputEvent`]: crate::extra::extensions::io::OutputEvent

use std::collections::{HashMap, HashSet};
use std::sync::{Arc, Mutex};

use tokio::sync::{mpsc, oneshot};
use tokio::task::JoinHandle;

use crate::core::extensions::ExtensionError;
use crate::core::types::{ContentBlock, Message, TextContent, ToolCall, ToolResult};
use crate::extra::agents::builder::AgentBuilder;
use crate::extra::extensions::io::OutputEvent;

// ─── redux state ──────────────────────────────────────────────────────────────

/// Stable identifier for a turn within a transcript.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct TurnId(pub u64);

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TurnStatus {
    /// Assistant output is still streaming / tools are resolving.
    Streaming,
    /// The turn finished (a `Finish` event was reduced).
    Finished,
}

/// One renderable block within a turn. Streaming text/reasoning accumulates into
/// a single trailing block; a tool call is a live block whose status and result
/// fill in as the turn progresses.
#[derive(Debug, Clone)]
pub enum Block {
    Text(TextContent),
    Reasoning(TextContent),
    ToolCall(ToolCallView),
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ToolStatus {
    /// No result received yet this turn.
    Running,
    /// `tool_execution_end` returned `Ok`.
    Done,
    /// `tool_execution_end` returned `Err`, or an error result.
    Error,
}

#[derive(Debug, Clone)]
pub struct ToolCallView {
    pub call: ToolCall,
    pub status: ToolStatus,
    pub result: Option<ToolResult>,
}

#[derive(Debug, Clone)]
pub struct Turn {
    pub id: TurnId,
    pub status: TurnStatus,
    pub blocks: Vec<Block>,
}

impl Turn {
    fn new(id: TurnId) -> Self {
        Self { id, status: TurnStatus::Streaming, blocks: Vec::new() }
    }
}

/// The redux-style renderable state: the agent's output, grouped by turn.
///
/// Turns are opened automatically by the reducer — on the first event that
/// follows a finished turn. The transcript holds the agent's output per turn,
/// not the user prompts that drove it; each frontend renders its own prompts
/// (Telegram echoes user messages natively, a web UI has the text it submitted).
#[derive(Debug, Clone, Default)]
pub struct Transcript {
    pub turns: Vec<Turn>,
    next_id: u64,
}

impl Transcript {
    pub fn new() -> Self {
        Self::default()
    }

    /// Borrowed view of the live (last, streaming) turn, if any.
    pub fn live(&self) -> Option<&Turn> {
        self.turns.last().filter(|t| t.status == TurnStatus::Streaming)
    }

    /// Ensure a streaming turn exists — opening a fresh one when the last turn is
    /// finished (or there are none) — then return it mutably.
    fn ensure_live(&mut self) -> &mut Turn {
        let need_new = self.turns.last().is_none_or(|t| t.status == TurnStatus::Finished);
        if need_new {
            let id = TurnId(self.next_id);
            self.next_id += 1;
            self.turns.push(Turn::new(id));
        }
        self.turns.last_mut().expect("just pushed or already had a live turn")
    }

    /// Reduce a streamed content block: accumulate text/reasoning deltas into the
    /// trailing matching block (mirroring `StreamResponse::merge`), or open a tool
    /// call. Other blocks (image / tool-result / custom) aren't streamed as
    /// assistant content and are ignored.
    pub fn append(&mut self, block: ContentBlock) {
        let turn = self.ensure_live();
        match block {
            ContentBlock::Text(t) => match turn.blocks.last_mut() {
                Some(Block::Text(existing)) => existing.content.push_str(&t.content),
                _ => turn.blocks.push(Block::Text(t)),
            },
            ContentBlock::Reasoning(t) => match turn.blocks.last_mut() {
                Some(Block::Reasoning(existing)) => existing.content.push_str(&t.content),
                _ => turn.blocks.push(Block::Reasoning(t)),
            },
            ContentBlock::ToolCall(call) => {
                turn.blocks.push(Block::ToolCall(ToolCallView { call, status: ToolStatus::Running, result: None }))
            }
            _ => {}
        }
    }

    /// Reduce a resolved tool call: fill the matching `ToolCall` block's result
    /// and status, matched by `id` within the live turn.
    pub fn tool_end(&mut self, id: &str, result: ToolResult) {
        let turn = self.ensure_live();
        for block in &mut turn.blocks {
            if let Block::ToolCall(view) = block
                && view.call.id == id
            {
                view.status = if result.is_error { ToolStatus::Error } else { ToolStatus::Done };
                view.result = Some(result);
                return;
            }
        }
    }

    /// Mark the live turn finished. Any tool still `Running` (e.g. a tool denied
    /// at `on_tool_execution_start`, where `tool_execution_end` is skipped) is
    /// marked `Done` so a frontend never shows a hung tool.
    pub fn finish_turn(&mut self) {
        if let Some(turn) = self.turns.last_mut() {
            turn.status = TurnStatus::Finished;
            for block in &mut turn.blocks {
                if let Block::ToolCall(view) = block
                    && view.status == ToolStatus::Running
                {
                    view.status = ToolStatus::Done;
                }
            }
        }
    }
}

// ─── input router ─────────────────────────────────────────────────────────────

/// Errors routing a message to a run's input.
#[derive(Debug, thiserror::Error)]
pub enum RouteError {
    /// No run is input-focused.
    #[error("no agent is input-focused")]
    NoFocus,
    /// The focused run's input channel closed (it ended or was stopped).
    #[error("agent channel closed")]
    Closed,
}

#[derive(Default)]
struct RouterInner {
    focus: Option<AgentId>,
    senders: HashMap<AgentId, mpsc::UnboundedSender<Message>>,
}

/// Cloneable, thread-safe input hub: routes typed messages to the single focused
/// run. Shared between the ViewModel and the input handler (e.g. Telegram's
/// message handler) via [`ViewModel::input`], so routing a message never needs
/// `&mut` on the ViewModel — keeping input off the render loop's critical section.
///
/// Mirrors the `Arc<Mutex<…>>`-hub precedent of the session mux.
pub struct InputRouter {
    inner: Mutex<RouterInner>,
}

impl InputRouter {
    fn new() -> Self {
        Self { inner: Mutex::new(RouterInner::default()) }
    }

    /// Route `msg` to the focused run's inbound channel.
    pub fn send(&self, msg: Message) -> Result<(), RouteError> {
        let inner = self.inner.lock().unwrap();
        let Some(focus) = inner.focus else { return Err(RouteError::NoFocus) };
        let Some(tx) = inner.senders.get(&focus) else { return Err(RouteError::NoFocus) };
        tx.send(msg).map_err(|_| RouteError::Closed)
    }

    /// The currently focused run, if any.
    pub fn focus(&self) -> Option<AgentId> {
        self.inner.lock().unwrap().focus
    }

    fn register(&self, id: AgentId, tx: mpsc::UnboundedSender<Message>) {
        self.inner.lock().unwrap().senders.insert(id, tx);
    }

    fn set_focus(&self, focus: Option<AgentId>) {
        self.inner.lock().unwrap().focus = focus;
    }

    /// Drop a run's sender and clear focus if it matched. Idempotent.
    fn remove(&self, id: AgentId) {
        let mut inner = self.inner.lock().unwrap();
        inner.senders.remove(&id);
        if inner.focus == Some(id) {
            inner.focus = None;
        }
    }
}

// ─── view model ───────────────────────────────────────────────────────────────

/// Identifies one run within a [`ViewModel`].
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct AgentId(pub u64);

/// What happened when the ViewModel pumped one fan-in event.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Step {
    /// The run's transcript changed — re-render it.
    Updated { id: AgentId },
    /// The run finished a turn and is awaiting an ack before continuing. Render
    /// the final state, then call [`ViewModel::ack`] once it is flushed.
    TurnFinished { id: AgentId },
    /// The run's task ended (its prompt loop returned). Its transcript is
    /// retained for scrollback; call [`ViewModel::stop`] to drop it.
    AgentEnded { id: AgentId },
}

/// One attached run: its reduced transcript, turn pacing, and the spawned agent
/// + forwarder tasks.
///
/// NOTE: `JoinHandle::drop` *detaches* a task — it does not abort it. The
/// [`ViewModel`] destructor calls `.abort()` on both handles explicitly; do not
/// rely on field-drop order here to tear tasks down.
struct Run {
    transcript: Transcript,
    pending_ack: Option<oneshot::Sender<Result<(), ExtensionError>>>,
    agent_task: JoinHandle<()>,
    forwarder_task: JoinHandle<()>,
    ended: bool,
}

/// Internal fan-in event: a run's output, or a signal that its stream closed.
enum FanEvent {
    Output(OutputEvent),
    Ended,
}

/// Owns N agent runs and reduces each one's [`OutputEvent`] stream into its own
/// [`Transcript`], multiplexed through a single fan-in channel.
///
/// Construct with [`ViewModel::new`], then spawn runs with [`ViewModel::start`].
/// Drive rendering by looping [`ViewModel::step`]; pace each run by calling
/// [`ViewModel::ack`] after flushing a finished turn. The inbound-message router
/// from [`ViewModel::input`] feeds the focused run's next turn — clone it for
/// whichever task collects user input. It is separate from the ViewModel so input
/// never contends with the render loop for `&mut self`.
///
/// Each run carries two independent attachment flags: **input** (which run
/// receives typed messages — at most one, the [`ViewModel::focus`]) and **output**
/// (which transcripts the frontend renders — an advisory set, since reduction is
/// unconditional). Toggle them with [`ViewModel::attach`] / [`ViewModel::detach`].
///
/// [`OutputEvent`]: crate::extra::extensions::io::OutputEvent
pub struct ViewModel {
    runs: HashMap<AgentId, Run>,
    fan_in: mpsc::UnboundedReceiver<(AgentId, FanEvent)>,
    /// Clone-source only: each [`ViewModel::start`] clones this into a forwarder
    /// task. Never sent on directly. Kept as a field so `start(&mut self, …)` can
    /// mint new forwarders without re-creating the channel.
    fan_tx: mpsc::UnboundedSender<(AgentId, FanEvent)>,
    router: Arc<InputRouter>,
    output: HashSet<AgentId>,
    next_id: u64,
}

impl Default for ViewModel {
    fn default() -> Self {
        let (fan_tx, fan_in) = mpsc::unbounded_channel::<(AgentId, FanEvent)>();
        Self {
            runs: HashMap::new(),
            fan_in,
            fan_tx,
            router: Arc::new(InputRouter::new()),
            output: HashSet::new(),
            next_id: 0,
        }
    }
}

impl ViewModel {
    /// An empty ViewModel owning no runs, alongside its shared input router.
    ///
    /// The router is ViewModel-scoped (not per-run), so it is handed out here
    /// rather than from [`Self::start`] — clone it for whichever task collects
    /// user input. Spawn runs with [`Self::start`]; dropping the ViewModel aborts
    /// them. ([`Self::input`] re-obtains the handle from a borrowed ViewModel.)
    pub fn new() -> (Self, Arc<InputRouter>) {
        let vm = Self::default();
        let input = vm.router.clone();
        (vm, input)
    }

    /// Build the agent from `builder`, spawn `agent.prompt(first)` plus a forwarder
    /// that fans its output into the ViewModel's stream, register its input
    /// channel, and auto-attach it (input + output). Returns the new run's id.
    /// Dropping the ViewModel aborts the run.
    pub async fn start(&mut self, builder: AgentBuilder, first: Message) -> anyhow::Result<AgentId> {
        let id = AgentId(self.next_id);
        self.next_id += 1;

        let (builder, mut rx, msg_tx) = builder.bind_io();
        let mut agent = builder.build().await?;

        let agent_task = tokio::spawn(async move {
            if let Err(e) = agent.prompt(first).await {
                eprintln!("agent prompt error: {e}");
            }
        });

        let fan_tx = self.fan_tx.clone();
        let forwarder_task = tokio::spawn(async move {
            while let Some(event) = rx.recv().await {
                if fan_tx.send((id, FanEvent::Output(event))).is_err() {
                    break;
                }
            }
            let _ = fan_tx.send((id, FanEvent::Ended));
        });

        self.runs.insert(
            id,
            Run { transcript: Transcript::new(), pending_ack: None, agent_task, forwarder_task, ended: false },
        );
        self.router.register(id, msg_tx);
        self.attach(id, true);
        Ok(id)
    }

    /// Attach a run. Always renders it; `input=true` also makes it the input focus
    /// (displacing any prior focus). `input=false` = render-only.
    pub fn attach(&mut self, id: AgentId, input: bool) {
        self.output.insert(id);
        if input {
            self.router.set_focus(Some(id));
        }
    }

    /// Detach a run. Always clears the input focus if it is `id`; `output=true`
    /// also stops rendering it. `output=false` = input-only (unfocus, keep
    /// rendering).
    pub fn detach(&mut self, id: AgentId, output: bool) {
        if self.router.focus() == Some(id) {
            self.router.set_focus(None);
        }
        if output {
            self.output.remove(&id);
        }
    }

    /// Forcefully drop a run: abort its tasks and remove its transcript, input
    /// sender, and render membership. Use to clean up a finished run (after
    /// [`Step::AgentEnded`]) once its transcript is no longer needed.
    pub fn stop(&mut self, id: AgentId) {
        if let Some(run) = self.runs.remove(&id) {
            run.agent_task.abort();
            run.forwarder_task.abort();
        }
        self.router.remove(id);
        self.output.remove(&id);
    }

    /// The current renderable state for `id`, if the run is held (including one
    /// that has ended but not yet been [`Self::stop`]ped).
    pub fn transcript(&self, id: AgentId) -> Option<&Transcript> {
        self.runs.get(&id).map(|r| &r.transcript)
    }

    /// All run ids the ViewModel still holds (running or ended-but-not-stopped).
    pub fn ids(&self) -> Vec<AgentId> {
        self.runs.keys().copied().collect()
    }

    /// The run currently receiving typed input, if any.
    pub fn focus(&self) -> Option<AgentId> {
        self.router.focus()
    }

    /// Whether `id`'s transcript is in the render set.
    pub fn is_output_attached(&self, id: AgentId) -> bool {
        self.output.contains(&id)
    }

    /// Whether `id`'s task is still running (false once [`Step::AgentEnded`] fired).
    pub fn is_running(&self, id: AgentId) -> bool {
        self.runs.get(&id).is_some_and(|r| !r.ended)
    }

    /// A cloneable handle to the input router, for the task that collects user
    /// input. Sending a message routes it to the focused run.
    pub fn input(&self) -> Arc<InputRouter> {
        self.router.clone()
    }

    /// Pump one fan-in event through the reducer. Returns `None` when the stream
    /// closed (no runs remain). [`Step::TurnFinished`] means a run's turn is done
    /// and it is blocked on [`Self::ack`]; [`Step::AgentEnded`] means a run's task
    /// ended (its transcript is retained until [`Self::stop`]).
    pub async fn step(&mut self) -> Option<Step> {
        loop {
            let (id, fan_event) = self.fan_in.recv().await?;
            return Some(match fan_event {
                FanEvent::Ended => {
                    let Some(run) = self.runs.get_mut(&id) else {
                        continue;
                    };
                    run.agent_task.abort();
                    run.forwarder_task.abort();
                    run.ended = true;
                    self.router.remove(id);
                    Step::AgentEnded { id }
                }
                FanEvent::Output(event) => {
                    let Some(run) = self.runs.get_mut(&id) else {
                        continue;
                    };
                    match event {
                        OutputEvent::Append(block) => {
                            run.transcript.append(block);
                            Step::Updated { id }
                        }
                        OutputEvent::ToolEnd { id: call_id, result } => {
                            run.transcript.tool_end(&call_id, result);
                            Step::Updated { id }
                        }
                        OutputEvent::Finish(ack) => {
                            run.transcript.finish_turn();
                            if self.output.contains(&id) {
                                run.pending_ack = Some(ack);
                                Step::TurnFinished { id }
                            } else {
                                // Nothing renders this run, so there is nothing to
                                // flush-pace against — ack immediately and let the
                                // agent block on the input gate (it would not get
                                // input unless focused anyway).
                                let _ = ack.send(Ok(()));
                                Step::Updated { id }
                            }
                        }
                    }
                }
            });
        }
    }

    /// Ack the pending finish for `id`, unblocking the run for its next turn.
    /// Call once the finished turn's output has been flushed to the frontend; a
    /// frontend that does not pace rendering may ack immediately.
    pub fn ack(&mut self, id: AgentId, result: Result<(), ExtensionError>) {
        if let Some(run) = self.runs.get_mut(&id)
            && let Some(ack) = run.pending_ack.take()
        {
            let _ = ack.send(result);
        }
    }
}

impl Drop for ViewModel {
    fn drop(&mut self) {
        for run in self.runs.values_mut() {
            run.agent_task.abort();
            run.forwarder_task.abort();
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    use std::collections::{HashMap, HashSet};
    use std::sync::Arc;
    use std::sync::atomic::{AtomicU64, Ordering};
    use std::time::{Duration, Instant};

    use async_trait::async_trait;
    use futures::stream;
    use tokio::time::timeout;

    use crate::core::providers::{Model, Provider, ProviderError, StreamResponse, StreamResponseStream};
    use crate::core::session::{InMemorySession, Session};
    use crate::core::tools::{ToolError, ToolHandler};
    use crate::core::types::{TokenUsage, ToolDefinition, ToolResultContent};

    // ── Transcript reducer (pure) ──────────────────────────────────────────────

    #[test]
    fn append_accumulates_text_and_reasoning_separately() {
        let mut t = Transcript::new();
        t.append(ContentBlock::Text(TextContent { content: "Hello".into() }));
        t.append(ContentBlock::Text(TextContent { content: ", world".into() }));
        t.append(ContentBlock::Reasoning(TextContent { content: "thinking".into() }));
        t.append(ContentBlock::Text(TextContent { content: "after".into() }));

        let turn = &t.turns[0];
        assert_eq!(turn.blocks.len(), 3);
        assert!(matches!(&turn.blocks[0], Block::Text(t) if t.content == "Hello, world"));
        assert!(matches!(&turn.blocks[1], Block::Reasoning(r) if r.content == "thinking"));
        assert!(matches!(&turn.blocks[2], Block::Text(t) if t.content == "after"));
    }

    #[test]
    fn finish_then_append_opens_a_new_turn() {
        let mut t = Transcript::new();
        t.append(ContentBlock::Text(TextContent { content: "first".into() }));
        t.finish_turn();
        t.append(ContentBlock::Text(TextContent { content: "second".into() }));

        assert_eq!(t.turns.len(), 2);
        assert_eq!(t.turns[0].id, TurnId(0));
        assert_eq!(t.turns[0].status, TurnStatus::Finished);
        assert_eq!(t.turns[1].id, TurnId(1));
        assert_eq!(t.turns[1].status, TurnStatus::Streaming);
    }

    #[test]
    fn tool_end_marks_error_and_success_results() {
        let mut t = Transcript::new();
        t.append(ContentBlock::ToolCall(ToolCall {
            id: "x".into(),
            name: "n".into(),
            arguments: serde_json::json!({}),
        }));
        assert!(matches!(
            &t.turns[0].blocks[0],
            Block::ToolCall(ToolCallView { status: ToolStatus::Running, result: None, .. })
        ));

        t.tool_end("x", ToolResult { tool_call_id: None, content: vec![], is_error: true });
        assert!(matches!(
            &t.turns[0].blocks[0],
            Block::ToolCall(ToolCallView { status: ToolStatus::Error, result: Some(r), .. }) if r.is_error
        ));
    }

    #[test]
    fn finish_turn_closes_running_tools() {
        let mut t = Transcript::new();
        t.append(ContentBlock::ToolCall(ToolCall {
            id: "y".into(),
            name: "n".into(),
            arguments: serde_json::json!({}),
        }));
        t.finish_turn();
        assert!(matches!(&t.turns[0].blocks[0], Block::ToolCall(ToolCallView { status: ToolStatus::Done, .. })));
    }

    // ── ViewModel (drives real agents) ─────────────────────────────────────────

    /// A provider that streams a scripted sequence of messages, one per `stream()`
    /// call, repeating the last once exhausted — enough to drive a turn that emits
    /// a tool call then final text.
    struct ScriptedProvider {
        messages: Vec<Message>,
        call: AtomicU64,
    }

    impl ScriptedProvider {
        fn new(messages: Vec<Message>) -> Self {
            Self { messages, call: AtomicU64::new(0) }
        }
    }

    #[async_trait]
    impl Provider for ScriptedProvider {
        async fn stream<'a>(
            &self,
            _model: &Model,
            _messages: Box<dyn Iterator<Item = &'a Message> + Send + 'a>,
            _tools: &Vec<ToolDefinition>,
        ) -> Result<StreamResponseStream, ProviderError> {
            let i = self.call.fetch_add(1, Ordering::SeqCst) as usize;
            let message = self.messages.get(i).cloned().unwrap_or_else(|| self.messages.last().unwrap().clone());
            let stop_reason = if message.content.iter().any(|b| matches!(b, ContentBlock::ToolCall(_))) {
                "tool_calls"
            } else {
                "stop"
            };
            let resp = StreamResponse {
                message,
                usage: TokenUsage {
                    input_tokens: 0,
                    output_tokens: 0,
                    cache_read_tokens: None,
                    cache_write_tokens: None,
                },
                stop_reason: Some(stop_reason.to_string()),
            };
            Ok(Box::pin(stream::iter(vec![resp])))
        }

        fn name(&self) -> &str {
            "scripted"
        }
    }

    /// A provider that always errors — drives a run to a natural exit so the
    /// forwarder emits `Ended`.
    struct ErrorProvider;

    #[async_trait]
    impl Provider for ErrorProvider {
        async fn stream<'a>(
            &self,
            _model: &Model,
            _messages: Box<dyn Iterator<Item = &'a Message> + Send + 'a>,
            _tools: &Vec<ToolDefinition>,
        ) -> Result<StreamResponseStream, ProviderError> {
            Err(ProviderError::RequestFailed("boom".into()))
        }

        fn name(&self) -> &str {
            "error"
        }
    }

    struct StubTool;

    #[async_trait]
    impl ToolHandler for StubTool {
        fn definition(&self) -> ToolDefinition {
            ToolDefinition {
                name: "search".into(),
                description: "stub".into(),
                parameters: serde_json::json!({"type":"object"}),
            }
        }
        async fn execute(
            &self,
            _cancel: futures::channel::oneshot::Receiver<bool>,
            _params: serde_json::Value,
        ) -> Result<ToolResult, ToolError> {
            Ok(ToolResult {
                tool_call_id: None,
                content: vec![ToolResultContent::Text(TextContent { content: "result".into() })],
                is_error: false,
            })
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

    fn session() -> Arc<Mutex<dyn Session>> {
        InMemorySession::new().arc()
    }

    fn toolcall_msg(id: &str, name: &str) -> Message {
        Message {
            role: "assistant".into(),
            content: vec![ContentBlock::ToolCall(ToolCall {
                id: id.into(),
                name: name.into(),
                arguments: serde_json::json!({"q":"x"}),
            })],
        }
    }

    fn text_msg(text: &str) -> Message {
        Message { role: "assistant".into(), content: vec![ContentBlock::Text(TextContent { content: text.into() })] }
    }

    fn user_msg(text: &str) -> Message {
        Message { role: "user".into(), content: vec![ContentBlock::Text(TextContent { content: text.into() })] }
    }

    async fn step(vm: &mut ViewModel) -> Option<Step> {
        timeout(Duration::from_secs(2), vm.step()).await.expect("step within timeout")
    }

    fn last_tool(vm: &ViewModel, id: AgentId) -> &ToolCallView {
        vm.transcript(id)
            .and_then(|t| t.turns.last())
            .and_then(|t| {
                t.blocks.iter().find_map(|b| match b {
                    Block::ToolCall(tc) => Some(tc),
                    _ => None,
                })
            })
            .expect("a tool block")
    }

    /// Step until `id`'s turn finishes (acking any other run that finishes along
    /// the way), then ack `id`. Asserts the run did not end first.
    async fn finish_turn(vm: &mut ViewModel, id: AgentId) {
        let deadline = Instant::now() + Duration::from_secs(2);
        loop {
            assert!(deadline > Instant::now(), "timed out waiting for turn finish on {id:?}");
            match step(vm).await {
                Some(Step::TurnFinished { id: f }) => {
                    vm.ack(f, Ok(()));
                    if f == id {
                        return;
                    }
                }
                Some(Step::AgentEnded { id: e }) if e == id => panic!("run {id:?} ended before its turn finished"),
                Some(_) => {}
                None => panic!("fan-in closed waiting for turn finish on {id:?}"),
            }
        }
    }

    fn turn_text(vm: &ViewModel, id: AgentId, turn: usize) -> String {
        vm.transcript(id).unwrap().turns[turn]
            .blocks
            .iter()
            .find_map(|b| match b {
                Block::Text(t) => Some(t.content.clone()),
                _ => None,
            })
            .unwrap()
    }

    /// A turn with one tool call then final text reduces, in order, into:
    /// `Append(ToolCall)` → `ToolEnd` → `Append(Text)` → `Finish`, leaving a
    /// single finished turn whose tool block carries its result.
    #[tokio::test]
    async fn reduces_a_turn_with_a_tool_call() {
        let provider = Arc::new(ScriptedProvider::new(vec![toolcall_msg("t1", "search"), text_msg("all done")]));
        let builder = AgentBuilder::new(model(), provider, session()).tool(StubTool).tools_enable(["search"]);
        let (mut vm, _) = ViewModel::new();
        let id = vm.start(builder, user_msg("go")).await.unwrap();

        // Append(ToolCall): opens the turn and a Running tool block.
        assert_eq!(step(&mut vm).await, Some(Step::Updated { id }));
        assert_eq!(last_tool(&vm, id).call.name, "search");
        assert_eq!(last_tool(&vm, id).status, ToolStatus::Running);
        assert_eq!(vm.transcript(id).unwrap().turns.len(), 1);

        // ToolEnd: fills the result, marks Done.
        assert_eq!(step(&mut vm).await, Some(Step::Updated { id }));
        assert_eq!(last_tool(&vm, id).status, ToolStatus::Done);
        assert!(last_tool(&vm, id).result.as_ref().is_some_and(|r| !r.is_error));

        // Append(Text): adds the final text block.
        assert_eq!(step(&mut vm).await, Some(Step::Updated { id }));
        assert_eq!(turn_text(&vm, id, 0), "all done");

        // Finish: the turn is closed; ack unblocks the run for the next turn.
        assert_eq!(step(&mut vm).await, Some(Step::TurnFinished { id }));
        assert_eq!(vm.transcript(id).unwrap().turns[0].status, TurnStatus::Finished);
        vm.ack(id, Ok(()));
        // Drop aborts the (now input-waiting) run task.
    }

    /// Two concurrently-running runs reduce into separate transcripts; the fan-in
    /// interleaves their events but never crosses their content.
    #[tokio::test]
    async fn two_agents_reduce_independently() {
        let provider_a = Arc::new(ScriptedProvider::new(vec![text_msg("hello from A")]));
        let provider_b = Arc::new(ScriptedProvider::new(vec![text_msg("hello from B")]));

        let (mut vm, _) = ViewModel::new();
        let a = vm.start(AgentBuilder::new(model(), provider_a, session()), user_msg("to a")).await.unwrap();
        let b = vm.start(AgentBuilder::new(model(), provider_b, session()), user_msg("to b")).await.unwrap();
        assert_ne!(a, b);

        let mut finished = HashSet::new();
        let deadline = Instant::now() + Duration::from_secs(2);
        while finished.len() < 2 && deadline > Instant::now() {
            match step(&mut vm).await {
                Some(Step::TurnFinished { id }) => {
                    finished.insert(id);
                    vm.ack(id, Ok(()));
                }
                Some(_) => {}
                None => break,
            }
        }
        assert!(finished.contains(&a) && finished.contains(&b), "both runs should finish a turn");

        assert_eq!(turn_text(&vm, a, 0), "hello from A");
        assert_eq!(turn_text(&vm, b, 0), "hello from B");
    }

    /// Starting a run attaches input (focus); starting a second displaces it. A
    /// message routed afterwards reaches only the focused run.
    #[tokio::test]
    async fn single_focus_input_displacement() {
        let provider_a = Arc::new(ScriptedProvider::new(vec![text_msg("A reply")]));
        let provider_b = Arc::new(ScriptedProvider::new(vec![text_msg("B reply")]));

        let (mut vm, input) = ViewModel::new();
        let a = vm.start(AgentBuilder::new(model(), provider_a, session()), user_msg("first to a")).await.unwrap();
        assert_eq!(vm.focus(), Some(a));
        finish_turn(&mut vm, a).await; // A's turn 1, then A blocks on input.

        let b = vm.start(AgentBuilder::new(model(), provider_b, session()), user_msg("first to b")).await.unwrap();
        assert_eq!(vm.focus(), Some(b), "starting B should displace focus from A");
        finish_turn(&mut vm, b).await; // B's turn 1.

        // Route a second message — only B (the focus) receives it.
        input.send(user_msg("second")).unwrap();
        finish_turn(&mut vm, b).await; // B's turn 2.

        assert_eq!(vm.transcript(a).unwrap().turns.len(), 1, "A never received the second message");
        assert_eq!(vm.transcript(b).unwrap().turns.len(), 2, "B received both turns");
    }

    /// With output detached, a turn still reduces but its `Finish` auto-acks and
    /// surfaces as `Updated` (not `TurnFinished`) — nothing is rendering it to
    /// flush-pace against. The transcript still accumulates.
    #[tokio::test(flavor = "current_thread")]
    async fn output_detach_keeps_reducing_and_auto_acks() {
        let provider = Arc::new(ScriptedProvider::new(vec![text_msg("turn one")]));
        let (mut vm, _) = ViewModel::new();
        let id = vm.start(AgentBuilder::new(model(), provider, session()), user_msg("go")).await.unwrap();
        // Turn 1 is in flight (first prompt delivered directly). Detach output
        // before it finishes: current_thread keeps the spawned task unpolled
        // until we await, so this ordering is deterministic.
        vm.detach(id, true);
        assert!(!vm.is_output_attached(id));
        assert_eq!(vm.focus(), None);

        // Append → Updated.
        assert_eq!(step(&mut vm).await, Some(Step::Updated { id }));
        // Finish → auto-acked → Updated (not TurnFinished).
        assert_eq!(step(&mut vm).await, Some(Step::Updated { id }));
        assert_eq!(vm.transcript(id).unwrap().turns.len(), 1, "transcript still accumulates while detached");
        assert_eq!(vm.transcript(id).unwrap().turns[0].status, TurnStatus::Finished);
    }

    /// Detaching output never touches the transcript: re-attaching shows the full
    /// history, not a frozen or truncated view.
    #[tokio::test]
    async fn detach_output_keeps_transcript_intact_for_reattach() {
        let provider = Arc::new(ScriptedProvider::new(vec![text_msg("turn one")]));
        let (mut vm, _) = ViewModel::new();
        let id = vm.start(AgentBuilder::new(model(), provider, session()), user_msg("go")).await.unwrap();
        finish_turn(&mut vm, id).await;
        assert_eq!(vm.transcript(id).unwrap().turns.len(), 1);

        vm.detach(id, true); // stop rendering + unfocus
        assert!(!vm.is_output_attached(id));
        assert_eq!(vm.transcript(id).unwrap().turns.len(), 1, "transcript preserved while detached");

        vm.attach(id, false); // re-render only (no focus)
        assert!(vm.is_output_attached(id));
        assert_eq!(vm.transcript(id).unwrap().turns.len(), 1, "full history visible on re-attach");
        assert_eq!(vm.transcript(id).unwrap().turns[0].status, TurnStatus::Finished);
    }

    /// A run whose task ends naturally emits `AgentEnded`; its transcript is
    /// retained for scrollback, and `stop` drops it.
    #[tokio::test(flavor = "current_thread")]
    async fn agent_ended_retains_transcript_then_stop_cleans_up() {
        let provider = Arc::new(ErrorProvider);
        let (mut vm, _) = ViewModel::new();
        let id = vm.start(AgentBuilder::new(model(), provider, session()), user_msg("go")).await.unwrap();
        assert!(vm.is_running(id));

        // The provider errors on the first stream → prompt returns Err → task
        // ends → forwarder emits Ended.
        assert_eq!(step(&mut vm).await, Some(Step::AgentEnded { id }));
        assert!(!vm.is_running(id), "marked ended");
        assert_eq!(vm.focus(), None, "focus cleared on exit");
        assert!(vm.transcript(id).is_some(), "transcript retained for scrollback");

        vm.stop(id);
        assert!(vm.transcript(id).is_none(), "stop drops the transcript");
        assert!(vm.ids().is_empty(), "stop removes the run");
    }
}

use std::iter::once;
use std::sync::{Arc, Mutex};

use async_trait::async_trait;
use uuid::Uuid;

use crate::core::agent::AgentState;
use crate::core::extensions::{Extension, ExtensionError};
use crate::core::session::{Session, SessionError};
use crate::core::tools::{ToolError, ToolHandler};
use crate::core::types::{ContentBlock, Message, TextContent, ToolDefinition, ToolResult, ToolResultContent};

type SessionLoader = Arc<dyn Fn(&str) -> anyhow::Result<Box<dyn Session>> + Send + Sync + 'static>;
pub struct MuxSession {
    loader: SessionLoader,
    active: Option<(String, Box<dyn Session>)>,
    router: Box<dyn Session>,
    pending: Vec<Message>,
    switching: bool,

    prompt_router: Message,
}

impl MuxSession {
    pub fn new(id: &str, loader: SessionLoader) -> anyhow::Result<(Arc<Mutex<Self>>, MuxExtension, MuxSwitchTool)> {
        let mux = Arc::new(Mutex::new(MuxSession {
            active: None,
            router: loader(id)?,
            loader,
            pending: Vec::new(),
            switching: false,
            prompt_router: Message {
                role: "user".to_string(),
                content: vec![ContentBlock::Text(TextContent {
                    content: include_str!("prompt_router.md").to_string(),
                })],
            },
        }));
        mux.lock().unwrap().switch(&Uuid::now_v7().to_string())?;
        anyhow::Ok((mux.clone(), MuxExtension { session: mux.clone() }, MuxSwitchTool { session: mux.clone() }))
    }

    fn flush(&mut self) -> Result<(), SessionError> {
        for msg in std::mem::take(&mut self.pending) {
            self.active.as_mut().unwrap().1.append(msg)?;
        }
        Ok(())
    }

    fn switch(&mut self, id: &str) -> Result<(), SessionError> {
        let session = (self.loader)(id).map_err(|e| SessionError::Failed { message: e.to_string() })?;
        self.active = Some((id.to_string(), session));
        self.switching = true;
        Ok(())
    }

    fn truncate_pending(&mut self) {
        if let Some(idx) = self.pending.iter().position(|m| m.role != "user") {
            self.pending.truncate(idx);
        }
    }
}

impl Session for MuxSession {
    fn append(&mut self, message: Message) -> Result<(), SessionError> {
        self.pending.push(message);
        Ok(())
    }

    fn messages(&self) -> Box<dyn Iterator<Item = &Message> + '_> {
        match self.active.as_ref() {
            Some(session) => Box::new(session.1.messages().chain(self.pending.iter())),
            None => Box::new(
                self.router
                    .messages()
                    .chain(
                        self.pending
                            .iter()
                            .take(self.pending.iter().position(|m| m.role != "user").unwrap_or(self.pending.len())),
                    )
                    .chain(once(&self.prompt_router)),
            ),
        }
    }
}

pub struct MuxSwitchTool {
    session: Arc<Mutex<MuxSession>>,
}

#[async_trait]
impl ToolHandler for MuxSwitchTool {
    fn definition(&self) -> ToolDefinition {
        ToolDefinition {
            name: "session-mux-switch".to_string(),
            description: "Switch to an existing conversation or create a new one. \
                - For regular session must only provide summary to switch to router session. \
                - For router sesion optionally omit next_id to auto-generate new session."
                .to_string(),
            parameters: serde_json::json!({
                "type": "object",
                "properties": {
                    "next_id": {
                        "type": "string",
                        "description": "Target conversation ID."
                    },
                    "summary": {
                        "type": "string",
                        "description": "Brief summary of the current conversation topic EXCLUDING the message triggers the switch, used to identify this session for future routing."
                    }
                }
            }),
        }
    }

    async fn execute(
        &self,
        _cancel: futures::channel::oneshot::Receiver<bool>,
        params: serde_json::Value,
    ) -> Result<ToolResult, ToolError> {
        let mut mux = self.session.lock().unwrap();
        match mux.active {
            Some(_) => match params.get("summary").and_then(|v| v.as_str()).unwrap_or("") {
                "" => Ok(ToolResult {
                    tool_call_id: None,
                    content: vec![ToolResultContent::Text(TextContent {
                        content: format!("summary is required when switching to router session."),
                    })],
                    is_error: true,
                }),
                summary => {
                    let id = mux.active.as_ref().unwrap().0.clone();
                    mux.router
                        .append(Message {
                            role: "user".to_string(),
                            content: vec![
                                ContentBlock::Text(TextContent { content: format!("id: {}", id) }),
                                ContentBlock::Text(TextContent { content: format!("summary: {}", summary) }),
                            ],
                        })
                        .map_err(|e| ToolError::Execution(e.to_string()))?;
                    mux.active = None;
                    Ok(ToolResult {
                        tool_call_id: None,
                        content: vec![ToolResultContent::Text(TextContent { content: format!("id: {}", id) })],
                        is_error: false,
                    })
                }
            },
            None => {
                let id = params
                    .get("next_id")
                    .and_then(|v| v.as_str())
                    .map_or_else(|| Uuid::now_v7().to_string(), |v| v.to_string());
                mux.switch(&id).map_err(|e| ToolError::Execution(e.to_string()))?;

                Ok(ToolResult {
                    tool_call_id: None,
                    content: vec![ToolResultContent::Text(TextContent {
                        content: format!("switched to session '{id}'."),
                    })],
                    is_error: false,
                })
            }
        }
    }
}

pub struct MuxExtension {
    session: Arc<Mutex<MuxSession>>,
}

#[async_trait]
impl Extension for MuxExtension {
    fn name(&self) -> &str {
        "session/mux"
    }

    async fn on_message_start(&mut self, messages: Vec<Message>) -> Result<Vec<Message>, ExtensionError> {
        let mut mux = self.session.lock().unwrap();
        if mux.switching {
            mux.switching = false;
            mux.truncate_pending();
            return Ok(mux.messages().cloned().collect());
        }
        Ok(messages)
    }

    async fn on_turn_end(&mut self, state: AgentState) -> Result<AgentState, ExtensionError> {
        let mut mux = self.session.lock().unwrap();

        // fallback if router session failed to switch.
        if mux.active.is_none() {
            mux.switch(&Uuid::now_v7().to_string()).map_err(|e| ExtensionError::ExtensionFailed {
                name: "session/mux".to_string(),
                message: e.to_string(),
            })?;
            mux.truncate_pending();
        }

        mux.flush()
            .map_err(|e| ExtensionError::ExtensionFailed { name: "session/mux".to_string(), message: e.to_string() })?;
        Ok(state)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::core::agent::Agent;
    use crate::core::providers::{Model, Provider};
    use crate::core::session::InMemorySession;
    use crate::core::tools::ToolRegistry;
    use crate::extra::providers::deepseek::DeepSeekProvider;
    use std::collections::HashMap;

    // --- Helpers ---

    fn text_msg(role: &str, content: &str) -> Message {
        Message {
            role: role.to_string(),
            content: vec![ContentBlock::Text(TextContent { content: content.to_string() })],
        }
    }

    fn make_mux() -> (Arc<Mutex<MuxSession>>, MuxExtension, MuxSwitchTool) {
        // Unit tests exercise the session mechanics, so each loaded session starts empty.
        MuxSession::new("router", Arc::new(|_id| Ok(Box::new(InMemorySession::new())))).unwrap()
    }

    fn tool_result(call_id: &str, content: &str) -> Message {
        Message {
            role: "tool".to_string(),
            content: vec![ContentBlock::ToolResult(ToolResult {
                tool_call_id: Some(call_id.to_string()),
                content: vec![ToolResultContent::Text(TextContent { content: content.to_string() })],
                is_error: false,
            })],
        }
    }

    fn make_mux_with_tracker() -> (Arc<Mutex<MuxSession>>, MuxExtension, MuxSwitchTool, Arc<Mutex<Vec<String>>>) {
        let loaded_ids = Arc::new(Mutex::new(Vec::new()));
        let ids = loaded_ids.clone();
        let mux = MuxSession::new(
            "router",
            Arc::new(move |id| {
                ids.lock().unwrap().push(id.to_string());
                Ok(Box::new(InMemorySession::new()))
            }),
        )
        .unwrap();
        (mux.0.clone(), mux.1, mux.2, loaded_ids)
    }

    // --- Unit tests ---

    #[test]
    fn new_starts_with_active_session() {
        let (mux, _, _) = make_mux();
        let m = mux.lock().unwrap();
        assert!(m.active.is_some(), "new() should create an initial active session via switch");
        assert_eq!(m.messages().count(), 0, "active session should be empty initially");
    }

    #[test]
    fn append_adds_to_pending() {
        let (mux, _, _) = make_mux();
        let mut m = mux.lock().unwrap();
        m.append(text_msg("user", "hello")).unwrap();

        // The appended message lives in pending and is visible via messages()
        assert_eq!(m.pending.len(), 1, "message should be in pending");
        assert_eq!(m.pending[0].role, "user");
        let msgs: Vec<_> = m.messages().collect();
        assert_eq!(msgs.len(), 1);
    }

    #[test]
    fn messages_chains_active_and_pending() {
        let (mux, _, _) = make_mux();
        let mut m = mux.lock().unwrap();

        // Set up active session with one message
        m.switch("s1").unwrap();
        m.active.as_mut().unwrap().1.append(text_msg("user", "existing")).unwrap();

        // Add pending message
        m.append(text_msg("user", "pending")).unwrap();

        let msgs: Vec<_> = m.messages().collect();
        assert_eq!(msgs.len(), 2);
        assert_eq!(msgs[0].role, "user");
        assert_eq!(msgs[1].role, "user");
    }

    #[test]
    fn flush_moves_pending_to_active() {
        let (mux, _, _) = make_mux();
        let mut m = mux.lock().unwrap();

        m.switch("s1").unwrap();
        m.append(text_msg("user", "hello")).unwrap();
        m.append(text_msg("assistant", "hi")).unwrap();
        assert_eq!(m.pending.len(), 2);

        m.flush().unwrap();

        assert_eq!(m.pending.len(), 0, "pending should be empty after flush");
        let active = &m.active.as_ref().unwrap().1;
        let msgs: Vec<_> = active.messages().collect();
        assert_eq!(msgs.len(), 2, "active session should have the flushed messages");
    }

    #[test]
    fn switch_creates_active_session() {
        let (mux, _, _, loaded_ids) = make_mux_with_tracker();
        let mut m = mux.lock().unwrap();

        m.switch("conv-1").unwrap();

        assert!(m.active.is_some());
        assert_eq!(m.active.as_ref().unwrap().0, "conv-1");
        assert_eq!(loaded_ids.lock().unwrap().last().unwrap(), "conv-1");
    }

    #[test]
    fn switch_binds_and_arms_without_routing_or_truncating() {
        let (mux, _, _) = make_mux();
        let mut m = mux.lock().unwrap();

        m.switch("s1").unwrap();
        let router_before = m.router.messages().count();

        // Routing context comes from the user message appended in MuxSwitchTool::execute,
        // so switch() must not forward pending to the router; and truncation is deferred to
        // on_message_start / on_turn_end, so switch() leaves pending untouched.
        m.pending.push(text_msg("user", "switch topic"));
        m.pending.push(tool_result("c1", "session-mux-switch result"));

        m.switch("s2").unwrap();

        assert_eq!(m.router.messages().count(), router_before, "switch must not append anything to the router");
        assert_eq!(m.active.as_ref().unwrap().0, "s2");
        assert!(m.switching, "switch arms the switching flag");
        assert_eq!(m.pending.len(), 2, "switch leaves pending for on_message_start / on_turn_end to truncate");
    }

    #[test]
    fn retain_pending_users_drops_assistant_with_tool_calls() {
        use crate::core::types::ToolCall;

        let (mux, _, _) = make_mux();
        let mut m = mux.lock().unwrap();
        m.switch("s1").unwrap();

        // pending: [user, assistant with switch + tavily tool calls]
        m.pending.push(text_msg("user", "question"));
        m.pending.push(Message {
            role: "assistant".to_string(),
            content: vec![
                ContentBlock::Text(TextContent { content: "reasoning".to_string() }),
                ContentBlock::ToolCall(ToolCall {
                    id: "call_1".to_string(),
                    name: "session-mux-switch".to_string(),
                    arguments: serde_json::json!({}),
                }),
                ContentBlock::ToolCall(ToolCall {
                    id: "call_2".to_string(),
                    name: "tavily_extract".to_string(),
                    arguments: serde_json::json!({}),
                }),
            ],
        });

        m.truncate_pending();

        // The whole non-user tail is dropped — the new session does not inherit the
        // router/switch tool use.
        assert_eq!(m.pending.len(), 1, "only the leading user message is kept");
        assert_eq!(m.pending[0].role, "user");
    }

    #[tokio::test]
    async fn on_turn_end_fallback_flushes_only_users() {
        let (mux, mut ext, _) = make_mux();
        {
            let mut m = mux.lock().unwrap();
            m.active = None; // router failed to switch back → fallback will fire
            m.pending.push(text_msg("user", "question"));
            m.pending.push(text_msg("assistant", "router reasoning"));
            m.pending.push(tool_result("c1", "switch result"));
        }

        let state = AgentState {
            model: Model {
                id: "m".to_string(),
                provider: "p".to_string(),
                context_window: 0,
                base_url: String::new(),
                headers: HashMap::new(),
            },
            tools: ToolRegistry::new(),
            session: mux.clone(),
        };
        ext.on_turn_end(state).await.unwrap();

        // switch() + retain_pending_users() + flush(): the new session must start from the
        // user message only, not the router's reasoning or the switch result.
        let m = mux.lock().unwrap();
        assert!(m.active.is_some(), "fallback should bind a new active session");
        let active: Vec<_> = m.active.as_ref().unwrap().1.messages().cloned().collect();
        assert_eq!(active.len(), 1, "new session should contain only the user message");
        assert!(active.iter().all(|x| x.role == "user"), "router reasoning must not leak into the new session");
    }

    #[tokio::test]
    async fn on_message_start_drops_switch_result() {
        let (mux, mut ext, tool) = make_mux();
        {
            let mut m = mux.lock().unwrap();
            m.active = None; // start in routing session
        }

        // router→active: execute binds the active session and arms the switching flag.
        let (_, rx) = futures::channel::oneshot::channel();
        tool.execute(rx, serde_json::json!({"next_id": "s2"})).await.unwrap();

        // The agent loop appends the switch tool result to pending after execute() returns.
        {
            let mut m = mux.lock().unwrap();
            m.pending.push(text_msg("user", "question"));
            m.pending.push(tool_result("c_switch", "switched to session 's2'."));
        }

        // The view read upstream of the hook is dirty — a dangling tool result with no
        // preceding tool call, which providers reject.
        let dirty: Vec<_> = mux.lock().unwrap().messages().cloned().collect();
        assert_eq!(dirty.iter().filter(|m| m.role == "tool").count(), 1);

        // on_message_start finalizes the switch: truncates the result and returns a clean view.
        let clean = ext.on_message_start(dirty).await.unwrap();
        assert_eq!(clean.iter().filter(|m| m.role == "tool").count(), 0, "switch result must be dropped");

        let m = mux.lock().unwrap();
        assert!(m.pending.iter().all(|m| m.role == "user"), "pending holds only user messages after finalize");
        assert!(!m.switching, "switching flag is cleared");
    }

    #[test]
    fn retain_pending_users_keeps_only_leading_users() {
        let (mux, _, _) = make_mux();
        let mut m = mux.lock().unwrap();

        m.switch("s1").unwrap();

        // Pending: user, assistant, user
        m.pending.push(text_msg("user", "q1"));
        m.pending.push(text_msg("assistant", "a1"));
        m.pending.push(text_msg("user", "q2"));

        m.truncate_pending();

        // Truncated at the first non-user message, so the trailing user message is dropped too.
        assert_eq!(m.pending.len(), 1, "pending is truncated at the first non-user message");
        assert_eq!(m.pending[0].content[0], ContentBlock::Text(TextContent { content: "q1".to_string() }));
    }

    #[tokio::test]
    async fn active_without_summary_returns_error() {
        let (mux, _, tool) = make_mux();
        {
            let mut m = mux.lock().unwrap();
            m.switch("s1").unwrap();
        }

        let (_, rx) = futures::channel::oneshot::channel();
        let result = tool.execute(rx, serde_json::json!({})).await.unwrap();

        assert!(result.is_error, "should return error when no summary provided with active session");
        let text = match &result.content[0] {
            ToolResultContent::Text(t) => &t.content,
            _ => panic!("expected text content"),
        };
        assert!(text.contains("summary is required"), "error message should mention summary, got: {text}");
    }

    #[tokio::test]
    async fn execute_to_router_appends_routing_message() {
        let (mux, _, tool) = make_mux();
        {
            let mut m = mux.lock().unwrap();
            m.switch("s1").unwrap();
        }

        let (_, rx) = futures::channel::oneshot::channel();
        let result = tool.execute(rx, serde_json::json!({"summary": "talked about rust"})).await.unwrap();

        assert!(!result.is_error, "switching to router with a summary should succeed");

        let m = mux.lock().unwrap();
        assert!(m.active.is_none(), "active should be cleared after switching to router session");
        let router_msgs: Vec<_> = m.router.messages().cloned().collect();
        assert_eq!(router_msgs.len(), 1, "router should receive one routing message");
        assert_eq!(router_msgs[0].role, "user");
        let text: String = router_msgs[0]
            .content
            .iter()
            .filter_map(|b| match b {
                ContentBlock::Text(t) => Some(t.content.as_str()),
                _ => None,
            })
            .collect::<Vec<_>>()
            .join(" ");
        assert!(text.contains("id: s1"), "routing message should carry the source session id, got: {text}");
        assert!(text.contains("summary: talked about rust"), "routing message should carry the summary, got: {text}");
    }

    #[tokio::test]
    async fn execute_to_active_switches_session() {
        let (mux, _, tool) = make_mux();
        {
            let mut m = mux.lock().unwrap();
            m.active = None; // start in routing session
        }

        let (_, rx) = futures::channel::oneshot::channel();
        let result = tool.execute(rx, serde_json::json!({"next_id": "s2"})).await.unwrap();

        assert!(!result.is_error);
        let m = mux.lock().unwrap();
        assert!(m.active.is_some(), "should switch into an active session");
        assert_eq!(m.active.as_ref().unwrap().0, "s2");
        assert!(m.switching, "router→active switch should arm the switching flag for on_message_start");
    }

    // --- E2E test ---

    fn api_key() -> Option<String> {
        std::env::var("DEEPSEEK_API_KEY").ok()
    }

    fn test_model() -> Model {
        Model {
            id: "deepseek-v4-flash".to_string(),
            provider: "deepseek".to_string(),
            context_window: 64000,
            base_url: String::new(),
            headers: HashMap::new(),
        }
    }

    fn user_text(content: &str) -> Message {
        Message {
            role: "user".to_string(),
            content: vec![ContentBlock::Text(TextContent { content: content.to_string() })],
        }
    }

    #[tokio::test]
    async fn test_e2e_mux_session_multi_turn() {
        let key = match api_key() {
            Some(k) => k,
            None => return,
        };

        let provider: Arc<dyn Provider> = Arc::new(DeepSeekProvider::new(key));

        // Prime every session with the topic-change instruction and track how many sessions
        // the loader creates, so the test can confirm the mux actually routed.
        let loaded_ids = Arc::new(Mutex::new(Vec::new()));
        let tracked = loaded_ids.clone();
        let (mux, mux_ext, mux_tool) = MuxSession::new(
            "router",
            Arc::new(move |id| {
                tracked.lock().unwrap().push(id.to_string());
                let mut session = Box::new(InMemorySession::new());
                session.append(Message {
                    role: "user".to_string(),
                    content: vec![ContentBlock::Text(TextContent {
                        content: "MUST detect whether topic changed before respond and if so, use the tool to switch session.".to_string(),
                    })],
                })?;
                Ok(session)
            }),
        )
        .unwrap();

        let mut tools = ToolRegistry::new();
        tools.register(Box::new(mux_tool));

        let agent_state = AgentState { model: test_model(), tools, session: mux };
        let mut agent = Agent { state: agent_state, provider, extension: Box::new(mux_ext) };

        // Three turns across distinct topics. The primary regression this guards is a
        // provider rejecting the message chain because of a dangling tool result after a
        // switch — every prompt must complete cleanly. Exact routing is model-driven, so we
        // only assert that the mux routed beyond the initial router + active sessions.
        agent
            .prompt(user_text("Tell me briefly about the Rust programming language in one sentence."))
            .await
            .expect("first prompt should succeed");
        agent
            .prompt(user_text("Tell me about Italian cuisine in one sentence."))
            .await
            .expect("second prompt should succeed");
        agent
            .prompt(user_text("Now compare Rust with C++ in two sentences."))
            .await
            .expect("third prompt should succeed");

        // Initial construction loads the router + one active session (2). More than that
        // means a session-mux-switch created or re-entered a session.
        let session_count = loaded_ids.lock().unwrap().len();
        assert!(
            session_count > 2,
            "mux should have routed to a session beyond the initial two, loaded {session_count} sessions"
        );

        // prompt() only returns once the last message is an assistant response.
        let last_role = agent.state.session.lock().unwrap().messages().last().map(|m| m.role.clone());
        assert_eq!(last_role.as_deref(), Some("assistant"), "final message should be an assistant response");
    }
}

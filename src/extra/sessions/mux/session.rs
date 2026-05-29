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
    skip_next: bool,

    prompt_router: Message,
}

impl MuxSession {
    pub fn new(id: &str, loader: SessionLoader) -> anyhow::Result<(Arc<Mutex<Self>>, MuxExtension, MuxSwitchTool)> {
        let mux = Arc::new(Mutex::new(MuxSession {
            active: None,
            router: loader(id)?,
            loader,
            pending: Vec::new(),
            skip_next: false,
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

        for msg in self.pending.iter().filter(|m| match m.content.last() {
            Some(ContentBlock::ToolResult(ToolResult { is_error: false, .. })) => true,
            Some(ContentBlock::ToolCall(..)) => true,
            _ => false,
        }) {
            self.router.append(msg.clone())?;
        }

        self.active = Some((id.to_string(), session));
        if let Some(idx) = self.pending.iter().position(|m| m.role != "user") {
            self.pending.truncate(idx);
        }
        Ok(())
    }
}

impl Session for MuxSession {
    fn append(&mut self, message: Message) -> Result<(), SessionError> {
        if self.skip_next {
            self.skip_next = false
        } else {
            self.pending.push(message);
        }
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
                        "description": "Brief summary of the current conversation topic, used to identify this session for future routing."
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
                _ => Ok(ToolResult {
                    tool_call_id: None,
                    content: vec![ToolResultContent::Text(TextContent {
                        content: format!("id: {}", mux.active.take().unwrap().0),
                    })],
                    is_error: false,
                }),
            },
            None => {
                let id = params
                    .get("next_id")
                    .and_then(|v| v.as_str())
                    .map_or_else(|| Uuid::now_v7().to_string(), |v| v.to_string());
                mux.switch(&id).map_err(|e| ToolError::Execution(e.to_string()))?;
                mux.skip_next = true;

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

    async fn on_turn_end(&mut self, state: AgentState) -> Result<AgentState, ExtensionError> {
        let mut mux = self.session.lock().unwrap();

        // fallback if router session failed to switch.
        if mux.active.is_none() {
            mux.switch(&Uuid::now_v7().to_string()).map_err(|e| ExtensionError::ExtensionFailed {
                name: "session/mux".to_string(),
                message: e.to_string(),
            })?;
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
    use crate::core::providers::{Model, Registry};
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

    fn tool_result_msg(content: &str) -> Message {
        Message {
            role: "tool".to_string(),
            content: vec![ContentBlock::ToolResult(ToolResult {
                tool_call_id: None,
                content: vec![ToolResultContent::Text(TextContent { content: content.to_string() })],
                is_error: false,
            })],
        }
    }

    fn make_mux() -> (Arc<Mutex<MuxSession>>, MuxExtension, MuxSwitchTool) {
        MuxSession::new("router", Arc::new(|_id| Ok(Box::new(InMemorySession::new())))).unwrap()
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
    fn switch_routes_switch_tool_result_to_router() {
        let (mux, _, _) = make_mux();
        let mut m = mux.lock().unwrap();

        // First switch to create an active session
        m.switch("s1").unwrap();

        // Simulate a realistic flow: user asks, assistant calls tool, tool returns switch result
        m.append(text_msg("user", "switch topic")).unwrap();
        let switch_msg = tool_result_msg("session-mux-switch\nid: s1\ntalked about rust");
        m.append(switch_msg).unwrap();

        // Switch triggers routing of switch messages to router
        m.switch("s2").unwrap();

        // Router should have received the switch message
        let router_msgs: Vec<_> = m.router.messages().collect();
        assert_eq!(router_msgs.len(), 1, "router should have the switch tool result");
    }

    #[test]
    fn switch_truncates_pending_at_first_non_user() {
        let (mux, _, _) = make_mux();
        let mut m = mux.lock().unwrap();

        m.switch("s1").unwrap();

        // Pending: user, assistant, user
        m.append(text_msg("user", "q1")).unwrap();
        m.append(text_msg("assistant", "a1")).unwrap();
        m.append(text_msg("user", "q2")).unwrap();

        m.switch("s2").unwrap();

        // switch truncates pending at the first non-user role message
        assert_eq!(m.pending.len(), 1, "pending should be truncated at first non-user message");
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

    fn extract_text(msg: &Message) -> String {
        msg.content
            .iter()
            .filter_map(|b| match b {
                ContentBlock::Text(t) => Some(t.content.as_str()),
                _ => None,
            })
            .collect::<Vec<_>>()
            .join(" ")
            .to_lowercase()
    }

    #[tokio::test]
    async fn test_e2e_mux_session_multi_turn() {
        let key = match api_key() {
            Some(k) => k,
            None => return,
        };

        let mut registry = Registry::new();
        registry.register("deepseek", Box::new(DeepSeekProvider::new(key)));

        let (mux, mux_ext, mux_tool) = make_mux();
        let mut tools = ToolRegistry::new();
        tools.register(Box::new(mux_tool));

        let agent_state = AgentState { model: test_model(), tools, session: mux };
        let mut agent = Agent { state: agent_state, models: Arc::new(registry), extension: Box::new(mux_ext) };

        // Turn 1: topic A — Rust
        let msg_a = Message {
            role: "user".to_string(),
            content: vec![ContentBlock::Text(TextContent {
                content: "Tell me briefly about the Rust programming language in one sentence.".to_string(),
            })],
        };
        agent.prompt(msg_a).await.expect("first prompt should succeed");

        // Turn 2: topic B — Italian cuisine (topic switch)
        let msg_b = Message {
            role: "user".to_string(),
            content: vec![ContentBlock::Text(TextContent {
                content: "Tell me about Italian cuisine in one sentence.".to_string(),
            })],
        };
        agent.prompt(msg_b).await.expect("second prompt should succeed");

        // Turn 3: back to topic A — compare Rust with C++
        let msg_c = Message {
            role: "user".to_string(),
            content: vec![ContentBlock::Text(TextContent {
                content: "Now compare Rust with C++ in two sentences.".to_string(),
            })],
        };
        agent.prompt(msg_c).await.expect("third prompt should succeed");

        let output: Vec<_> = agent.state.session.lock().unwrap().messages().cloned().collect();

        // Verify message structure: three user-assistant pairs in alternation
        assert_eq!(output.len(), 6, "should have 6 messages (3 user + 3 assistant), got {}", output.len());
        for (i, expected_role) in ["user", "assistant", "user", "assistant", "user", "assistant"].iter().enumerate() {
            assert_eq!(output[i].role, *expected_role, "message {i} should be {expected_role}, got {}", output[i].role);
        }

        // Verify the three user messages are exactly what we sent
        let user_msgs: Vec<_> = output.iter().filter(|m| m.role == "user").map(|m| extract_text(m)).collect();
        assert!(user_msgs[0].contains("rust"), "first user message should be about Rust, got: {}", user_msgs[0]);
        assert!(
            user_msgs[1].contains("italian cuisine"),
            "second user message should be about Italian cuisine, got: {}",
            user_msgs[1]
        );
        assert!(
            user_msgs[2].contains("rust") && user_msgs[2].contains("c++"),
            "third user message should compare Rust and C++, got: {}",
            user_msgs[2]
        );
    }
}

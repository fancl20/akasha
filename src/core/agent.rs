use std::sync::{Arc, Mutex};

use futures::StreamExt;

use crate::core::extensions::{Extension, ToolCallDecision};
use crate::core::providers::{Model, Provider, StreamResponse};
use crate::core::session::{Session, SessionError};
use crate::core::tools::ToolRegistry;
use crate::core::types::{ContentBlock, Message, TextContent, ToolResult, ToolResultContent};

/// Errors from agent operations.
#[derive(Debug, thiserror::Error)]
pub enum AgentError {
    #[error("provider error: {0}")]
    Provider(#[from] crate::core::providers::ProviderError),
    #[error("tool error: {0}")]
    Tool(String),
    #[error("extension error: {0}")]
    Extension(#[from] crate::core::extensions::ExtensionError),
    #[error("session error: {0}")]
    Session(#[from] SessionError),
    #[error("agent aborted")]
    Aborted,
    #[error("{0}")]
    Other(String),
}

#[derive(Clone)]
pub struct AgentState {
    pub model: Model,
    pub tools: ToolRegistry,
    pub session: Arc<Mutex<dyn Session>>,
}

pub struct Agent {
    pub state: AgentState,
    pub provider: Arc<dyn Provider>,
    pub extension: Box<dyn Extension>,
}

impl Agent {
    pub async fn prompt(&mut self, message: Message) -> Result<(), AgentError> {
        self.state = self.extension.on_agent_start(self.state.clone()).await?;
        self.state.session.lock().unwrap().append(message)?;
        loop {
            let mut state = self.extension.on_turn_start(self.state.clone()).await?;

            state = agent_loop(state, self.provider.as_ref(), self.extension.as_mut()).await?;

            self.state = self.extension.on_turn_end(state).await?;
            match self.state.session.lock().unwrap().messages().last() {
                Some(Message { role, .. }) if role == "assistant" => break,
                _ => {}
            }
        }
        self.state = self.extension.on_agent_end(self.state.clone()).await?;
        return Ok(());
    }
}

pub async fn agent_loop(
    state: AgentState,
    provider: &dyn Provider,
    extension: &mut dyn Extension,
) -> Result<AgentState, AgentError> {
    loop {
        let messages = state.session.lock().unwrap().messages().cloned().collect();
        let messages = extension.on_message_start(messages).await?;

        let mut response = StreamResponse::new();
        let mut stream = provider.stream(&state.model, Box::new(messages.iter()), &state.tools.definitions()).await?;
        while let Some(chunk) = stream.next().await {
            extension.on_message_update(&chunk).await?;
            response.merge(chunk);
        }
        extension.on_message_end(&response).await?;

        let resp = response.message;
        state.session.lock().unwrap().append(resp.clone())?;

        let tool_calls: Vec<(String, String, serde_json::Value)> = resp
            .content
            .iter()
            .filter_map(|b| match b {
                ContentBlock::ToolCall(tc) => Some((tc.id.clone(), tc.name.clone(), tc.arguments.clone())),
                _ => None,
            })
            .collect();

        if tool_calls.len() == 0 {
            return Ok(state);
        }

        for (id, name, arguments) in tool_calls {
            let decision = extension.on_tool_execution_start(&id, &name, &arguments).await?;
            if let ToolCallDecision::Deny(reason) = decision {
                state.session.lock().unwrap().append(Message {
                    role: "tool".to_string(),
                    content: vec![ContentBlock::ToolResult(ToolResult {
                        tool_call_id: Some(id),
                        content: vec![ToolResultContent::Text(TextContent { content: reason })],
                        is_error: true,
                    })],
                })?;
                continue;
            }

            let Some(handler) = state.tools.get(&name) else {
                state.session.lock().unwrap().append(Message {
                    role: "tool".to_string(),
                    content: vec![ContentBlock::ToolResult(ToolResult {
                        tool_call_id: Some(id),
                        content: vec![ToolResultContent::Text(TextContent {
                            content: format!("error: no handler registered for tool '{name}'"),
                        })],
                        is_error: true,
                    })],
                })?;
                continue;
            };

            let (_, rx) = futures::channel::oneshot::channel();
            let raw_result = handler.execute(rx, arguments).await;

            let result = extension.tool_execution_end(&id, raw_result).await?;
            let result = match result {
                Ok(mut r) => {
                    r.tool_call_id = Some(id);
                    r
                }
                Err(e) => ToolResult {
                    tool_call_id: Some(id),
                    content: vec![ToolResultContent::Text(TextContent {
                        content: format!("error: tool '{name}' execution failed: {e}"),
                    })],
                    is_error: true,
                },
            };

            let tool_msg = Message { role: "tool".to_string(), content: vec![ContentBlock::ToolResult(result)] };
            state.session.lock().unwrap().append(tool_msg)?;
        }
    }
}

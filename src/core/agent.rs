use futures::StreamExt;

use crate::core::extensions::Extension;
use crate::core::providers::{Model, Registry, StreamResponse};
use crate::core::tools::ToolRegistry;
use crate::core::types::{ContentBlock, Message, Request};

/// Errors from agent operations.
#[derive(Debug, thiserror::Error)]
pub enum AgentError {
    #[error("provider error: {0}")]
    Provider(#[from] crate::core::providers::ProviderError),
    #[error("tool error: {0}")]
    Tool(String),
    #[error("extension error: {0}")]
    Extension(#[from] crate::core::extensions::ExtensionError),
    #[error("agent aborted")]
    Aborted,
    #[error("{0}")]
    Other(String),
}

pub async fn run(
    request: &Request,
    model: &Model,
    models: &Registry,
    tools: &ToolRegistry,
    extension: &Box<dyn Extension>,
) -> Result<Vec<Message>, AgentError> {
    let provider = models.get(&model.provider).ok_or_else(|| {
        AgentError::Other(format!("no provider registered for '{}'", model.provider))
    })?;

    let mut request = Request {
        messages: request.messages.clone().clone(),
        tools: request.tools.clone(),
    };
    let mut output: Vec<Message> = Vec::new();

    loop {
        request = extension.on_request(request).await?;

        let mut response = StreamResponse::new();
        let mut stream = std::pin::pin!(provider.stream(model, &request).await?);
        while let Some(chunk) = stream.next().await {
            extension.on_response_chunk(&chunk).await?;
            response.merge(chunk);
        }
        extension.on_response(&response).await?;

        let resp = response.message;
        request.messages.push(resp.clone());
        output.push(resp.clone());

        let tool_calls: Vec<(String, String, serde_json::Value)> = resp
            .content
            .iter()
            .filter_map(|b| match b {
                ContentBlock::ToolCall {
                    id,
                    name,
                    arguments,
                } => Some((id.clone(), name.clone(), arguments.clone())),
                _ => None,
            })
            .collect();

        if tool_calls.len() == 0 {
            return Ok(output);
        }

        for (id, name, arguments) in tool_calls {
            let handler = tools.get(&name).ok_or_else(|| {
                AgentError::Tool(format!("no handler registered for tool '{name}'"))
            })?;

            let cancel = tokio::sync::watch::channel(false).1;
            let result = handler
                .execute(cancel, arguments)
                .await
                .map_err(|e| AgentError::Tool(format!("tool '{name}' execution failed: {e}")))?;

            let tool_msg = Message {
                role: "tool".to_string(),
                content: vec![ContentBlock::ToolResult {
                    tool_call_id: id,
                    content: result,
                }],
            };

            request.messages.push(tool_msg.clone());
            output.push(tool_msg);
        }
    }
}

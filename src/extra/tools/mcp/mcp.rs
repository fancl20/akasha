use std::sync::Arc;

use async_trait::async_trait;
use rmcp::RoleClient;
use rmcp::model::{CallToolRequestParams, CallToolResult, RawContent};
use rmcp::service::{RunningService, ServiceExt};
use rmcp::transport::StreamableHttpClientTransport;
use rmcp::transport::streamable_http_client::StreamableHttpClientTransportConfig;

use crate::core::tools::{ToolError, ToolHandler, ToolRegistry};
use crate::core::types::{TextContent, ToolDefinition, ToolResult, ToolResultContent};
use crate::extra::tools::mcp::config;

/// Error type for MCP connection and tool discovery.
#[derive(Debug, thiserror::Error)]
pub enum McpToolError {
    #[error("connection error: {0}")]
    Connection(String),
    #[error("tool discovery error: {0}")]
    Discovery(String),
}

/// Wraps a single MCP server tool as a [`ToolHandler`].
///
/// Each instance holds a clone of the [`Peer`] for making calls and an
/// `Arc` reference to the [`RunningService`] that keeps the underlying
/// transport alive for as long as any tool from the same server is in use.
pub struct McpTool {
    definition: ToolDefinition,
    peer: rmcp::Peer<RoleClient>,
    // Prevents the MCP connection from being dropped while tools are alive.
    _connection: Arc<RunningService<RoleClient, ()>>,
}

#[async_trait]
impl ToolHandler for McpTool {
    fn definition(&self) -> ToolDefinition {
        self.definition.clone()
    }

    async fn execute(
        &self,
        _cancel: futures::channel::oneshot::Receiver<bool>,
        params: serde_json::Value,
    ) -> Result<ToolResult, ToolError> {
        let arguments = params.as_object().cloned();

        let mut req = CallToolRequestParams::new(self.definition.name.clone());
        if let Some(args) = arguments {
            req = req.with_arguments(args);
        }

        let result = self.peer.call_tool(req).await.map_err(|e| ToolError::Execution(e.to_string()))?;

        let content = to_result_content(&result);

        if result.is_error.unwrap_or(false) {
            let text = text_from_content(&content);
            Err(ToolError::Execution(text))
        } else {
            Ok(ToolResult { tool_call_id: None, content, is_error: false })
        }
    }
}

impl From<&RawContent> for ToolResultContent {
    fn from(raw: &RawContent) -> Self {
        match raw {
            RawContent::Text(t) => ToolResultContent::Text(TextContent { content: t.text.clone() }),
            RawContent::Resource(r) => match &r.resource {
                rmcp::model::ResourceContents::TextResourceContents { text, .. } => {
                    ToolResultContent::Text(TextContent { content: text.clone() })
                }
                _ => ToolResultContent::Text(TextContent { content: format!("{raw:?}") }),
            },
            _ => ToolResultContent::Text(TextContent { content: format!("{raw:?}") }),
        }
    }
}

/// Converts MCP tool result content into [`ToolResultContent`].
fn to_result_content(result: &CallToolResult) -> Vec<ToolResultContent> {
    result.content.iter().map(|c| (&c.raw).into()).collect()
}

/// Extracts readable text from [`ToolResultContent`] for error messages.
fn text_from_content(content: &[ToolResultContent]) -> String {
    let texts: Vec<&str> = content
        .iter()
        .filter_map(|c| match c {
            ToolResultContent::Text(t) => Some(t.content.as_str()),
            _ => None,
        })
        .collect();

    if texts.is_empty() { serde_json::to_string(content).unwrap_or_default() } else { texts.join("\n") }
}

/// Connects to an MCP server via streamable HTTP, discovers its tools, and
/// registers each one in the supplied [`ToolRegistry`].
///
/// Returns an `Arc` to the [`RunningService`] so the caller can shut down the
/// connection when it is no longer needed (dropping all clones closes the
/// transport).
///
/// # Example
///
/// ```ignore
/// use akasha::core::tools::ToolRegistry;
/// use akasha::extra::tools::mcp;
/// use akasha::extra::tools::mcp::config::StreamableHttpConfig;
///
/// let mut tools = ToolRegistry::new();
/// let service = mcp::register(
///     &mut tools,
///     &StreamableHttpConfig {
///         url: "http://localhost:8000/mcp".to_string(),
///         headers: Default::default(),
///     },
/// )
/// .await
/// .unwrap();
/// ```
pub async fn register(
    registry: &mut ToolRegistry,
    server: &config::StreamableHttpConfig,
) -> Result<Arc<RunningService<RoleClient, ()>>, McpToolError> {
    let mut header_map = reqwest::header::HeaderMap::new();
    for (key, value) in &server.headers {
        let name = reqwest::header::HeaderName::from_bytes(key.as_bytes())
            .map_err(|e| McpToolError::Connection(format!("invalid header name '{key}': {e}")))?;
        let val = reqwest::header::HeaderValue::from_str(value)
            .map_err(|e| McpToolError::Connection(format!("invalid header value for '{key}': {e}")))?;
        header_map.insert(name, val);
    }

    let client = reqwest::Client::builder()
        .default_headers(header_map)
        .build()
        .map_err(|e| McpToolError::Connection(e.to_string()))?;

    let transport =
        StreamableHttpClientTransport::with_client(client, StreamableHttpClientTransportConfig::with_uri(&*server.url));
    let service = ().serve(transport).await.map_err(|e| McpToolError::Connection(e.to_string()))?;
    let service = Arc::new(service);
    let tools = service.list_all_tools().await.map_err(|e| McpToolError::Discovery(e.to_string()))?;

    for tool in tools {
        let name = tool.name.into_owned();
        if !server.is_tool_allowed(&name) {
            continue;
        }

        let definition = ToolDefinition {
            name,
            description: tool.description.map(|d| d.into_owned()).unwrap_or_default(),
            parameters: serde_json::Value::Object(tool.input_schema.as_ref().clone()),
        };

        registry.register(McpTool { definition, peer: service.peer().clone(), _connection: service.clone() }.into());
    }

    Ok(service)
}

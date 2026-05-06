use std::sync::Arc;

use async_trait::async_trait;
use rmcp::RoleClient;
use rmcp::model::{CallToolRequestParams, CallToolResult, RawContent};
use rmcp::service::{RunningService, ServiceExt};
use rmcp::transport::StreamableHttpClientTransport;
use rmcp::transport::streamable_http_client::StreamableHttpClientTransportConfig;

use crate::core::tools::{ToolError, ToolHandler, ToolRegistry};
use crate::core::types::ToolDefinition;

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
        _cancel: tokio::sync::watch::Receiver<bool>,
        params: serde_json::Value,
    ) -> Result<String, ToolError> {
        let arguments = params.as_object().cloned();

        let mut req = CallToolRequestParams::new(self.definition.name.clone());
        if let Some(args) = arguments {
            req = req.with_arguments(args);
        }

        let result = self
            .peer
            .call_tool(req)
            .await
            .map_err(|e| ToolError::Execution(e.to_string()))?;

        let text = format_content(&result);

        if result.is_error.unwrap_or(false) {
            Err(ToolError::Execution(text))
        } else {
            Ok(text)
        }
    }
}

/// Extracts readable text from a [`CallToolResult`].
///
/// Joins all text and text-resource content items with newlines. Falls back to
/// JSON-serialising the full content list when no textual items are present
/// (e.g. image-only results).
fn format_content(result: &CallToolResult) -> String {
    let texts: Vec<String> = result
        .content
        .iter()
        .filter_map(|c| match &c.raw {
            RawContent::Text(t) => Some(t.text.clone()),
            RawContent::Resource(r) => match &r.resource {
                rmcp::model::ResourceContents::TextResourceContents { text, .. } => {
                    Some(text.clone())
                }
                _ => None,
            },
            _ => None,
        })
        .collect();

    if texts.is_empty() {
        serde_json::to_string(&result.content).unwrap_or_default()
    } else {
        texts.join("\n")
    }
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
///
/// let mut tools = ToolRegistry::new();
/// let service = mcp::register(
///     &mut tools,
///     "http://localhost:8000/mcp",
///     HeaderMap::new(),
/// )
/// .await
/// .unwrap();
/// ```
pub async fn register(
    registry: &mut ToolRegistry,
    uri: &str,
    headers: reqwest::header::HeaderMap,
) -> Result<Arc<RunningService<RoleClient, ()>>, McpToolError> {
    let client = reqwest::Client::builder()
        .default_headers(headers)
        .build()
        .map_err(|e| McpToolError::Connection(e.to_string()))?;

    let transport = StreamableHttpClientTransport::with_client(
        client,
        StreamableHttpClientTransportConfig::with_uri(uri),
    );
    let service = ().serve(transport).await.map_err(|e| McpToolError::Connection(e.to_string()))?;
    let service = Arc::new(service);
    let tools = service
        .list_all_tools()
        .await
        .map_err(|e| McpToolError::Discovery(e.to_string()))?;

    for tool in tools {
        let definition = ToolDefinition {
            name: tool.name.into_owned(),
            description: tool.description.map(|d| d.into_owned()).unwrap_or_default(),
            parameters: serde_json::Value::Object(tool.input_schema.as_ref().clone()),
        };

        registry.register(Box::new(McpTool {
            definition,
            peer: service.peer().clone(),
            _connection: service.clone(),
        }));
    }

    Ok(service)
}

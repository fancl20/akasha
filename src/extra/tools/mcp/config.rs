use std::collections::HashMap;

use serde::Deserialize;

/// Top-level MCP configuration, compatible with the standard format used by
/// Claude Desktop, VS Code, Cline, and others.
///
/// ```json
/// {
///   "mcpServers": {
///     "my-server": {
///       "url": "http://localhost:3000/mcp",
///       "headers": { "Authorization": "Bearer token" }
///     }
///   }
/// }
/// ```
#[derive(Debug, Clone, Deserialize)]
pub struct McpConfig {
    #[serde(rename = "mcpServers", alias = "servers")]
    pub mcp_servers: HashMap<String, ServerEntry>,
}

/// A single server entry in the config.
///
/// Uses `#[serde(untagged)]` to discriminate by structure:
/// - entries with `url` → `StreamableHttp`
/// - entries with `command` → `Stdio`
/// - anything else → `Unknown`
#[derive(Debug, Clone, Deserialize)]
#[serde(untagged)]
pub enum ServerEntry {
    /// Streamable HTTP transport (also covers legacy SSE configs).
    StreamableHttp(StreamableHttpConfig),
    /// Unrecognised config blob.
    Unknown(serde_json::Value),
}

/// Configuration for Streamable HTTP transport.
#[derive(Debug, Clone, Deserialize)]
pub struct StreamableHttpConfig {
    /// MCP server endpoint URL.
    pub url: String,
    /// Optional HTTP headers (e.g. `Authorization`).
    #[serde(default)]
    pub headers: HashMap<String, String>,
}

impl ServerEntry {
    /// Returns the [`StreamableHttpConfig`] if this is a supported entry,
    /// or an error explaining why it cannot be used.
    pub fn into_config(self) -> Result<StreamableHttpConfig, McpConfigError> {
        match self {
            Self::StreamableHttp(cfg) => Ok(cfg),
            Self::Unknown(v) => Err(McpConfigError::InvalidConfig {
                detail: format!("unrecognised server config: {v}"),
            }),
        }
    }
}

#[derive(Debug, thiserror::Error)]
pub enum McpConfigError {
    #[error("unsupported transport '{transport}'; only StreamableHttp (url-based) is supported")]
    UnsupportedTransport { transport: String },
    #[error("invalid config: {detail}")]
    InvalidConfig { detail: String },
    #[error(transparent)]
    Parse(#[from] serde_json::Error),
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn minimal_streamable_http() {
        let cfg: McpConfig =
            serde_json::from_str(r#"{"mcpServers":{"s":{"url":"http://localhost/mcp"}}}"#).unwrap();
        let entry = cfg.mcp_servers.get("s").unwrap().clone();
        let http = entry.into_config().unwrap();
        assert_eq!(http.url, "http://localhost/mcp");
        assert!(http.headers.is_empty());
    }

    #[test]
    fn with_headers() {
        let cfg: McpConfig = serde_json::from_str(
            r#"{"mcpServers":{"s":{"url":"http://localhost/mcp","headers":{"Authorization":"Bearer tok"}}}}"#,
        )
        .unwrap();
        let http = cfg
            .mcp_servers
            .get("s")
            .unwrap()
            .clone()
            .into_config()
            .unwrap();
        assert_eq!(http.headers["Authorization"], "Bearer tok");
    }

    #[test]
    fn servers_alias() {
        let cfg: McpConfig =
            serde_json::from_str(r#"{"servers":{"s":{"url":"http://localhost/mcp"}}}"#).unwrap();
        assert!(cfg.mcp_servers.contains_key("s"));
    }

    #[test]
    fn unknown_rejected() {
        let cfg: McpConfig = serde_json::from_str(r#"{"mcpServers":{"s":{"foo":"bar"}}}"#).unwrap();
        let err = cfg.mcp_servers["s"].clone().into_config().unwrap_err();
        assert!(matches!(err, McpConfigError::InvalidConfig { .. }));
    }

    #[test]
    fn empty_config() {
        let cfg: McpConfig = serde_json::from_str(r#"{"mcpServers":{}}"#).unwrap();
        assert!(cfg.mcp_servers.is_empty());
    }
}

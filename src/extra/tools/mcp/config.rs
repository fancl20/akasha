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
    /// Tool allowlist. When non-empty, only tools whose name appears here
    /// are registered. Empty means "allow everything" (the default).
    #[serde(default)]
    pub allow: Vec<String>,
    /// Tool denylist. A tool whose name appears here is never registered,
    /// and a deny match overrides an allow match.
    #[serde(default)]
    pub deny: Vec<String>,
}

impl StreamableHttpConfig {
    /// Returns `true` if a tool with the given name should be registered.
    ///
    /// A tool is allowed when it is not in `deny` and, when `allow` is
    /// non-empty, is present in `allow`. An empty `allow` list means
    /// "allow everything".
    pub fn is_tool_allowed(&self, name: &str) -> bool {
        if self.deny.iter().any(|p| p == name) {
            return false;
        }
        self.allow.is_empty() || self.allow.iter().any(|p| p == name)
    }
}

impl ServerEntry {
    /// Returns the [`StreamableHttpConfig`] if this is a supported entry,
    /// or an error explaining why it cannot be used.
    pub fn into_config(self) -> Result<StreamableHttpConfig, McpConfigError> {
        match self {
            Self::StreamableHttp(cfg) => Ok(cfg),
            Self::Unknown(v) => {
                Err(McpConfigError::InvalidConfig { detail: format!("unrecognised server config: {v}") })
            }
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
        let cfg: McpConfig = serde_json::from_str(r#"{"mcpServers":{"s":{"url":"http://localhost/mcp"}}}"#).unwrap();
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
        let http = cfg.mcp_servers.get("s").unwrap().clone().into_config().unwrap();
        assert_eq!(http.headers["Authorization"], "Bearer tok");
    }

    #[test]
    fn servers_alias() {
        let cfg: McpConfig = serde_json::from_str(r#"{"servers":{"s":{"url":"http://localhost/mcp"}}}"#).unwrap();
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

    fn http(allow: &[&str], deny: &[&str]) -> StreamableHttpConfig {
        StreamableHttpConfig {
            url: "http://localhost/mcp".into(),
            headers: HashMap::new(),
            allow: allow.iter().map(|s| (*s).into()).collect(),
            deny: deny.iter().map(|s| (*s).into()).collect(),
        }
    }

    #[test]
    fn empty_lists_allow_everything() {
        let cfg = http(&[], &[]);
        assert!(cfg.is_tool_allowed("anything"));
        assert!(cfg.is_tool_allowed(""));
    }

    #[test]
    fn allow_exact_name() {
        let cfg = http(&["filesystem_read", "git_status"], &[]);
        assert!(cfg.is_tool_allowed("filesystem_read"));
        assert!(cfg.is_tool_allowed("git_status"));
        assert!(!cfg.is_tool_allowed("filesystem_write"));
        assert!(!cfg.is_tool_allowed("git"));
    }

    #[test]
    fn deny_overrides_allow() {
        let cfg = http(&["git_status"], &["git_status"]);
        assert!(!cfg.is_tool_allowed("git_status"));
    }

    #[test]
    fn deny_without_allow() {
        let cfg = http(&[], &["dangerous", "exact"]);
        assert!(!cfg.is_tool_allowed("dangerous"));
        assert!(!cfg.is_tool_allowed("exact"));
        assert!(cfg.is_tool_allowed("safe"));
    }

    #[test]
    fn parses_allow_and_deny() {
        let cfg: McpConfig = serde_json::from_str(
            r#"{"mcpServers":{"s":{"url":"http://localhost/mcp","allow":["git_status"],"deny":["git_push"]}}}"#,
        )
        .unwrap();
        let http = cfg.mcp_servers["s"].clone().into_config().unwrap();
        assert_eq!(http.allow, vec!["git_status"]);
        assert_eq!(http.deny, vec!["git_push"]);
        assert!(!http.is_tool_allowed("git_push"));
        assert!(http.is_tool_allowed("git_status"));
    }
}

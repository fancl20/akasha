//! Circuit-breaker extension that bounds tool output size.
//!
//! [`CircuitBreaker`] is a cross-cutting [`Extension`] that inspects every tool
//! result in [`tool_execution_end`](Extension::tool_execution_end). When the
//! concatenated text of a result exceeds [`MAX_OUTPUT_LENGTH`] characters, it
//! clips the output to the first [`MAX_OUTPUT_LENGTH`] characters and converts
//! the result into an error flagged `MAX_OUTPUT_LENGTH`.
//!
//! This stops a single over-long tool return — e.g. a full scraped web page —
//! from being persisted into the session and blowing past the model's context
//! window on every subsequent turn.

use async_trait::async_trait;

use crate::core::extensions::{Extension, ExtensionError};
use crate::core::tools::ToolError;
use crate::core::types::{TextContent, ToolResult, ToolResultContent};

/// Maximum number of characters a tool result may contain before the circuit
/// breaker clips it. Doubles as the error marker surfaced to the model.
const MAX_OUTPUT_LENGTH: usize = 30_000;

/// Bounds tool output size. See the module docs.
///
/// Compose it with other extensions via [`And`](super::combinator::And). It is
/// stateless, so place it last in the chain so it clips the final result the
/// other extensions (e.g. schema verification) produce.
pub struct CircuitBreaker;

impl CircuitBreaker {
    pub fn new() -> Self {
        Self
    }
}

impl Default for CircuitBreaker {
    fn default() -> Self {
        Self::new()
    }
}

/// Concatenates the `Text` blocks of a tool result — the same shape the provider
/// serializes when sending the result to the model.
fn result_text(result: &ToolResult) -> String {
    result
        .content
        .iter()
        .filter_map(|c| match c {
            ToolResultContent::Text(t) => Some(t.content.as_str()),
            _ => None,
        })
        .collect::<Vec<_>>()
        .join("")
}

#[async_trait]
impl Extension for CircuitBreaker {
    fn name(&self) -> &str {
        "circuit_breaker"
    }

    async fn tool_execution_end(
        &mut self,
        _tool_call_id: &str,
        result: Result<ToolResult, ToolError>,
    ) -> Result<Result<ToolResult, ToolError>, ExtensionError> {
        // A failed execution has no result to measure; pass it through untouched.
        let mut result = match result {
            Ok(r) => r,
            err @ Err(_) => return Ok(err),
        };

        let text = result_text(&result);
        if text.chars().count() <= MAX_OUTPUT_LENGTH {
            return Ok(Ok(result));
        }

        // Keep the leading slice so the model still gets *some* output, then flag
        // the result as an error so it knows the return was truncated.
        let clipped: String = text.chars().take(MAX_OUTPUT_LENGTH).collect();
        result.content = vec![
            ToolResultContent::Text(TextContent { content: clipped }),
            ToolResultContent::Text(TextContent {
                content: "\n[MAX_OUTPUT_LENGTH: output exceeded and was clipped]".to_string(),
            }),
        ];
        result.is_error = true;
        Ok(Ok(result))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn ok_result(text: &str) -> Result<ToolResult, ToolError> {
        Ok(ToolResult {
            tool_call_id: Some("call_1".to_string()),
            content: vec![ToolResultContent::Text(TextContent { content: text.to_string() })],
            is_error: false,
        })
    }

    fn result_text(out: &ToolResult) -> String {
        out.content
            .iter()
            .filter_map(|c| match c {
                ToolResultContent::Text(t) => Some(t.content.as_str()),
                _ => None,
            })
            .collect::<Vec<_>>()
            .join("")
    }

    #[tokio::test]
    async fn passes_short_result_through() {
        let mut ext = CircuitBreaker::new();
        let out = ext.tool_execution_end("c1", ok_result("small output")).await.unwrap().unwrap();
        assert!(!out.is_error);
        assert_eq!(result_text(&out), "small output");
        assert_eq!(out.tool_call_id.as_deref(), Some("call_1"));
    }

    #[tokio::test]
    async fn boundary_at_limit_passes() {
        let mut ext = CircuitBreaker::new();
        let exactly = "a".repeat(MAX_OUTPUT_LENGTH);
        let out = ext.tool_execution_end("c1", ok_result(&exactly)).await.unwrap().unwrap();
        assert!(!out.is_error, "exactly at the limit is not over it");
        assert_eq!(result_text(&out).chars().count(), MAX_OUTPUT_LENGTH);
    }

    #[tokio::test]
    async fn clips_over_limit_to_first_slice_and_flags_error() {
        let mut ext = CircuitBreaker::new();
        let original = "x".repeat(MAX_OUTPUT_LENGTH + 5);
        let out = ext.tool_execution_end("c1", ok_result(&original)).await.unwrap().unwrap();

        assert!(out.is_error, "over-limit result must be flagged as an error");
        let text = result_text(&out);
        assert!(text.contains("MAX_OUTPUT_LENGTH"), "error marker must reach the model: {text}");
        // The clipped body is exactly the first MAX_OUTPUT_LENGTH chars, followed by the marker.
        assert!(text.starts_with(&"x".repeat(MAX_OUTPUT_LENGTH)));
        assert_eq!(out.tool_call_id.as_deref(), Some("call_1"), "tool_call_id is preserved");
    }

    #[tokio::test]
    async fn clips_on_char_boundary_for_multibyte() {
        let mut ext = CircuitBreaker::new();
        // 'é' is two bytes; a char-boundary-safe clip must not panic and must keep ≤ limit chars.
        let original = "é".repeat(MAX_OUTPUT_LENGTH + 10);
        let out = ext.tool_execution_end("c1", ok_result(&original)).await.unwrap().unwrap();
        assert!(out.is_error);
        let text = result_text(&out);
        let body = text.split("\n[").next().unwrap();
        assert_eq!(body.chars().count(), MAX_OUTPUT_LENGTH);
    }

    #[tokio::test]
    async fn passes_tool_error_through() {
        let mut ext = CircuitBreaker::new();
        let err = Err(ToolError::Execution("boom".to_string()));
        let out = ext.tool_execution_end("c1", err).await.unwrap();
        assert!(out.is_err(), "a ToolError must pass through unchanged");
    }
}

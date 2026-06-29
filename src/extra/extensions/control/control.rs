//! Conversation control extension: **stop** (drop history) and **fold** (summarize a run).
//!
//! [`ControlExtension`] is a cross-cutting [`Extension`] that rewrites the message
//! stream in [`on_message_start`](Extension::on_message_start) — the last point before
//! the provider is called — so compaction applies only to what the model sees. Tools and
//! other extensions still observe the full, unaltered conversation through
//! [`Session::messages`], which is what they need to *decide* when and what to compact.
//!
//! Two control markers are recognised. Both ride on a `custom`-role message as a
//! [`ContentBlock::Custom`] block, so the provider never sees them raw (its
//! `role != "custom"` filter strips them):
//!
//! * **stop** — "ignore everything before this": the stop marker and every preceding
//!   message are dropped from the rendered view. When multiple stops appear, the most
//!   recent one wins.
//! * **fold** — "collapse a range and replace with an id + summary": the raw messages at
//!   indices `start..end` (a half-open range into [`Session::messages`]) are collapsed into
//!   a single placeholder carrying the fold `id` and `text`. The range must precede the
//!   marker; the originals stay in the raw session, so the [`ControlUnfoldTool`] can
//!   surface them again on demand.
//!
//! Because the rendering is a stateless pure function of the raw messages, there is no
//! cached view to keep in sync — it is recomputed each turn at the single place it is
//! needed. Build the markers with [`stop_message`] / [`fold_message`], wire
//! [`ControlExtension`] into the agent's extension chain last (so its output is the
//! final provider view), and register [`ControlUnfoldTool`] against the same session
//! handle so the model can expand a fold.
//!
//! [`Extension`]: crate::core::extensions::Extension
//! [`Session::messages`]: crate::core::session::Session::messages

use std::ops::Range;
use std::sync::{Arc, Mutex};

use async_trait::async_trait;
use uuid::Uuid;

use crate::core::extensions::Extension;
use crate::core::session::Session;
use crate::core::tools::{ToolError, ToolHandler};
use crate::core::types::{ContentBlock, Message, TextContent, ToolDefinition, ToolResult, ToolResultContent};

/// Custom-content type marking a **stop** control message.
const STOP_TYPE: &str = "session-control-stop";
/// Custom-content type marking a **fold** control message.
const FOLD_TYPE: &str = "session-control-fold";

/// Stateless [`Extension`] that applies stop/fold compaction to the message stream at
/// [`on_message_start`](Extension::on_message_start) — so only the provider sees the
/// compacted view; everything else reads the raw session.
///
/// It holds no state: the transform is a pure function of the messages it receives.
/// Compose it last in an [`And`](crate::extra::extensions::combinator::And) chain so its
/// output is the final provider view.
pub struct ControlExtension;

impl ControlExtension {
    pub fn new() -> Self {
        Self
    }
}

impl Default for ControlExtension {
    fn default() -> Self {
        Self::new()
    }
}

#[async_trait]
impl Extension for ControlExtension {
    fn name(&self) -> &str {
        "session/control"
    }

    async fn on_message_start(
        &mut self,
        messages: Vec<Message>,
    ) -> Result<Vec<Message>, crate::core::extensions::ExtensionError> {
        Ok(render(messages))
    }
}

/// Tool that expands a folded section back into its original messages.
///
/// Given the `id` shown in a folded summary placeholder, it returns the full content of
/// the messages that were collapsed there — read straight from the raw session, so it
/// works even though [`ControlExtension`] never stores the originals. A later `stop`
/// removes the placeholder from the rendered view but does not affect this tool: the
/// originals are still returned by `id`.
///
/// [`ControlExtension`]: ControlExtension
pub struct ControlUnfoldTool {
    session: Arc<Mutex<dyn Session>>,
}

impl ControlUnfoldTool {
    /// Build the tool over `session` — the same handle the agent drives and
    /// [`ControlExtension`] renders — so the raw messages it reads are ground truth.
    ///
    /// [`ControlExtension`]: ControlExtension
    pub fn new(session: Arc<Mutex<dyn Session>>) -> Self {
        Self { session }
    }
}

#[async_trait]
impl ToolHandler for ControlUnfoldTool {
    fn definition(&self) -> ToolDefinition {
        ToolDefinition {
            name: "session-control-unfold".to_string(),
            description: "Expand a previously folded section of the conversation back into its \
                original messages. Pass the fold `id` found in a folded summary placeholder; \
                returns the full content of the messages that were collapsed there."
                .to_string(),
            parameters: serde_json::json!({
                "type": "object",
                "properties": {
                    "id": {
                        "type": "string",
                        "description": "The fold id shown in a folded summary placeholder."
                    }
                },
                "required": ["id"]
            }),
        }
    }

    async fn execute(
        &self,
        _cancel: futures::channel::oneshot::Receiver<bool>,
        params: serde_json::Value,
    ) -> Result<ToolResult, ToolError> {
        let id = params.get("id").and_then(|v| v.as_str()).unwrap_or("");
        let raw: Vec<Message> = self.session.lock().unwrap().messages().cloned().collect();

        let folded = raw.iter().enumerate().find_map(|(idx, m)| {
            let fold = m.custom(FOLD_TYPE)?;
            if fold.get("id").and_then(|v| v.as_str()) != Some(id) {
                return None;
            }
            let start = fold.get("start").and_then(|v| v.as_u64()).unwrap_or(0) as usize;
            let end = fold.get("end").and_then(|v| v.as_u64()).unwrap_or(0) as usize;
            let hi = end.min(idx).min(raw.len());
            let lo = start.min(hi);
            (lo < hi).then(|| raw[lo..hi].to_vec())
        });

        match folded {
            Some(msgs) => {
                let body = serde_json::to_string_pretty(&msgs).unwrap_or_default();
                Ok(ToolResult {
                    tool_call_id: None,
                    content: vec![ToolResultContent::Text(TextContent { content: body })],
                    is_error: false,
                })
            }
            None => Ok(ToolResult {
                tool_call_id: None,
                content: vec![ToolResultContent::Text(TextContent {
                    content: format!("no folded content found for id '{id}'"),
                })],
                is_error: true,
            }),
        }
    }
}

// ---- control message constructors ----

/// Build a **stop** control message: append it to a session to drop every message at or
/// before this point from the provider's view.
pub fn stop_message() -> Message {
    Message {
        role: "custom".to_string(),
        content: vec![ContentBlock::Custom { r#type: STOP_TYPE.to_string(), content: serde_json::json!({}) }],
    }
}

/// Build a **fold** control message collapsing raw messages at indices `start..end` (a
/// half-open range into [`Session::messages`]) into a summary placeholder. Returns the
/// generated fold `id` (hand it to the unfold tool) alongside the message to append.
///
/// [`Session::messages`]: crate::core::session::Session::messages
pub fn fold_message(start: usize, end: usize, text: &str) -> (String, Message) {
    let id = Uuid::now_v7().to_string();
    (id.clone(), fold_message_with_id(start, end, &id, text))
}

/// Build a **fold** control message with a caller-supplied `id`.
///
/// [`Session::messages`]: crate::core::session::Session::messages
pub fn fold_message_with_id(start: usize, end: usize, id: &str, text: &str) -> Message {
    Message {
        role: "custom".to_string(),
        content: vec![ContentBlock::Custom {
            r#type: FOLD_TYPE.to_string(),
            content: serde_json::json!({ "id": id, "start": start, "end": end, "text": text }),
        }],
    }
}

/// The single rendered message that stands in for a folded run. It exposes the fold `id`
/// (so the model can call the unfold tool) alongside the caller-supplied summary `text`.
fn fold_ref(id: &str, count: usize, text: &str) -> Message {
    Message {
        role: "user".to_string(),
        content: vec![ContentBlock::Text(TextContent {
            content: format!(
                "[folded summary of {count} earlier message(s); call the unfold tool with id=\"{id}\" to retrieve them]\n\n{text}"
            ),
        })],
    }
}

/// Render the raw message stream into the provider view: apply stops (most recent wins)
/// and folds (collapse a raw-index range into a placeholder).
///
/// Each rendered entry remembers the raw-index span it represents, so a fold keyed by raw
/// indices locates its targets even across earlier folds — collapsing every entry whose
/// span falls inside `start..end` and replacing the run with one placeholder at that
/// position. Non-overlapping folds are exact; a range that only partially overlaps an
/// earlier fold's placeholder cannot split it and is left alone.
fn render(raw: Vec<Message>) -> Vec<Message> {
    let mut rendered: Vec<(Range<usize>, Message)> = Vec::with_capacity(raw.len());

    for (idx, msg) in raw.into_iter().enumerate() {
        if msg.custom(STOP_TYPE).is_some() {
            rendered.clear();
        } else if let Some(fold) = msg.custom(FOLD_TYPE) {
            let id = fold.get("id").and_then(|v| v.as_str()).unwrap_or("").to_string();
            let start = fold.get("start").and_then(|v| v.as_u64()).unwrap_or(0) as usize;
            let end = fold.get("end").and_then(|v| v.as_u64()).unwrap_or(0) as usize;
            let text = fold.get("text").and_then(|v| v.as_str()).unwrap_or("").to_string();
            // The marker can only fold what precedes it; clamp the range to what's been seen.
            let hi = end.min(idx);
            let lo = start.min(hi);
            // Raw order is preserved, so the entries whose spans fall inside lo..hi form one
            // contiguous run; collapse it and drop a placeholder where the first one sat.
            if let Some(first) = rendered.iter().position(|(s, _)| s.start >= lo && s.end <= hi) {
                let last = first + rendered[first..].iter().take_while(|(s, _)| s.start >= lo && s.end <= hi).count();
                let drained: Vec<(Range<usize>, Message)> = rendered.drain(first..last).collect();
                let count = drained.iter().map(|(s, _)| s.end - s.start).sum::<usize>();
                rendered.insert(first, (lo..hi, fold_ref(&id, count, &text)));
            }
            // else: nothing in range is currently live (dropped by a stop, or already
            // subsumed by an earlier fold) — leave no placeholder.
        } else {
            rendered.push((idx..idx + 1, msg));
        }
    }

    rendered.into_iter().map(|(_, m)| m).collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::core::session::InMemorySession;

    // --- helpers ---

    fn text_msg(role: &str, content: &str) -> Message {
        Message {
            role: role.to_string(),
            content: vec![ContentBlock::Text(TextContent { content: content.to_string() })],
        }
    }

    fn msg_text(msg: &Message) -> String {
        msg.content
            .iter()
            .filter_map(|b| match b {
                ContentBlock::Text(t) => Some(t.content.as_str()),
                _ => None,
            })
            .collect::<Vec<_>>()
            .join("")
    }

    fn roles(rendered: &[Message]) -> Vec<&str> {
        rendered.iter().map(|m| m.role.as_str()).collect()
    }

    fn is_control(msg: &Message) -> bool {
        msg.custom(STOP_TYPE).is_some() || msg.custom(FOLD_TYPE).is_some()
    }

    // --- pass-through ---

    #[test]
    fn render_passes_regular_messages_through_unchanged() {
        let raw = vec![text_msg("user", "hello"), text_msg("assistant", "hi"), text_msg("user", "again")];
        let rendered = render(raw);
        assert_eq!(rendered.len(), 3);
        assert_eq!(roles(&rendered), ["user", "assistant", "user"]);
        assert!(rendered.iter().all(|m| !is_control(m)));
    }

    #[test]
    fn unrelated_custom_blocks_pass_through() {
        // A Custom block that is not one of our control types must surface untouched.
        let raw = vec![Message {
            role: "custom".to_string(),
            content: vec![ContentBlock::Custom {
                r#type: "session-mux-switch".to_string(),
                content: serde_json::json!({ "id": "x" }),
            }],
        }];
        let rendered = render(raw);
        assert_eq!(rendered.len(), 1, "non-control custom messages are not consumed");
    }

    // --- stop ---

    #[test]
    fn stop_drops_everything_before_it() {
        let raw =
            vec![text_msg("user", "old1"), text_msg("assistant", "old2"), stop_message(), text_msg("user", "fresh")];
        let rendered = render(raw);
        assert_eq!(rendered.len(), 1, "stop must drop itself and everything before it");
        assert_eq!(msg_text(&rendered[0]), "fresh");
    }

    #[test]
    fn most_recent_stop_wins() {
        let raw =
            vec![text_msg("user", "a"), stop_message(), text_msg("user", "b"), stop_message(), text_msg("user", "c")];
        let rendered = render(raw);
        assert_eq!(rendered.len(), 1);
        assert_eq!(msg_text(&rendered[0]), "c");
    }

    // --- fold ---

    #[test]
    fn fold_replaces_preceding_messages_with_placeholder() {
        let raw = vec![
            text_msg("user", "keep"),
            text_msg("user", "fold1"),
            text_msg("assistant", "fold2"),
            fold_message_with_id(1, 3, "fold-id", "summary of fold1/fold2"),
        ];
        let rendered = render(raw);
        assert_eq!(rendered.len(), 2, "the two folded messages collapse into one placeholder");
        assert_eq!(msg_text(&rendered[0]), "keep");
        let placeholder = msg_text(&rendered[1]);
        assert!(placeholder.contains("fold-id"), "placeholder must expose the fold id, got: {placeholder}");
        assert!(placeholder.contains("summary of fold1/fold2"), "placeholder must carry the summary text");
    }

    #[test]
    fn fold_clamps_end_to_preceding_messages() {
        // `end` past the marker's position is clamped: only raw index 0 ("only") exists.
        let raw = vec![text_msg("user", "only"), fold_message_with_id(0, 5, "id", "everything")];
        let rendered = render(raw);
        assert_eq!(rendered.len(), 1, "end past the available range folds only what exists");
        assert!(msg_text(&rendered[0]).contains("everything"));
    }

    #[test]
    fn fold_preserves_placeholders_across_subsequent_folds() {
        let raw = vec![
            text_msg("user", "m1"),
            text_msg("user", "m2"),
            text_msg("user", "m3"),
            fold_message_with_id(1, 3, "f1", "first"), // folds raw 1..3 (m2, m3) -> [m1, P1]
            text_msg("user", "m4"),                    // [m1, P1, m4]
            fold_message_with_id(4, 5, "f2", "second"), // folds raw 4..5 (m4) -> [m1, P1, P2]
        ];
        let rendered = render(raw);
        assert_eq!(rendered.len(), 3);
        assert_eq!(msg_text(&rendered[0]), "m1");
        assert!(msg_text(&rendered[1]).contains("first"));
        assert!(msg_text(&rendered[2]).contains("second"));
    }

    #[test]
    fn fold_can_collapse_an_interior_range() {
        // Folding a range that is not at the tail leaves the messages after it visible —
        // something the trailing-count form could not express.
        let raw = vec![
            text_msg("user", "a"),                    // 0
            text_msg("user", "b"),                    // 1
            text_msg("user", "c"),                    // 2
            text_msg("user", "d"),                    // 3
            fold_message_with_id(1, 3, "mid", "b+c"), // folds raw 1..3 (b, c)
        ];
        let rendered = render(raw);
        assert_eq!(rendered.len(), 3, "b and c collapse into one placeholder; a and d remain");
        assert_eq!(msg_text(&rendered[0]), "a");
        assert!(msg_text(&rendered[1]).contains("b+c"));
        assert_eq!(msg_text(&rendered[2]), "d", "message after the folded range stays visible");
    }

    #[test]
    fn fold_with_empty_range_is_a_noop() {
        let raw = vec![text_msg("user", "a"), fold_message_with_id(0, 0, "empty", "nothing")];
        let rendered = render(raw);
        assert_eq!(rendered.len(), 1, "an empty range folds nothing and leaves no placeholder");
        assert_eq!(msg_text(&rendered[0]), "a");
    }

    #[test]
    fn outer_fold_absorbs_an_inner_placeholder() {
        // f1 folds b..c into a placeholder; f2 then folds a range that contains it.
        let raw = vec![
            text_msg("user", "a"),
            text_msg("user", "b"),
            text_msg("user", "c"),
            fold_message_with_id(1, 3, "f1", "bc"), // -> placeholder over raw 1..3
            text_msg("user", "d"),
            fold_message_with_id(0, 5, "f2", "abcd"), // folds raw 0..5, swallowing f1's placeholder
        ];
        // Everything from a through d collapses into the outer placeholder.
        let rendered = render(raw);
        assert_eq!(rendered.len(), 1);
        assert!(msg_text(&rendered[0]).contains("abcd"));
    }

    #[test]
    fn stop_after_fold_empties_the_render() {
        let raw = vec![
            text_msg("user", "m1"),
            text_msg("user", "m2"),
            fold_message_with_id(0, 2, "fid", "sum"),
            stop_message(),
        ];
        // The stop drops everything before it — including the fold's placeholder.
        assert!(render(raw).is_empty());
    }

    // --- unfold tool ---

    #[tokio::test]
    async fn unfold_tool_returns_folded_content() {
        let session = InMemorySession::new().arc();
        let (id, fold) = fold_message(0, 2, "qa pair");
        {
            let mut s = session.lock().unwrap();
            s.append(text_msg("user", "question")).unwrap();
            s.append(text_msg("assistant", "answer")).unwrap();
            s.append(fold).unwrap();
        }

        let tool = ControlUnfoldTool::new(session);
        let (_, rx) = futures::channel::oneshot::channel();
        let result = tool.execute(rx, serde_json::json!({ "id": id })).await.unwrap();
        assert!(!result.is_error);
        let body = match &result.content[0] {
            ToolResultContent::Text(t) => &t.content,
            _ => panic!("expected text content"),
        };
        assert!(body.contains("question"), "folded content must include originals: {body}");
        assert!(body.contains("answer"), "folded content must include originals: {body}");
    }

    #[tokio::test]
    async fn unfold_tool_still_works_when_messages_follow_the_fold() {
        let session = InMemorySession::new().arc();
        let (id, fold) = fold_message(1, 2, "sum");
        {
            let mut s = session.lock().unwrap();
            s.append(text_msg("user", "m1")).unwrap();
            s.append(text_msg("user", "m2")).unwrap();
            s.append(fold).unwrap();
            s.append(text_msg("user", "m3")).unwrap();
            s.append(text_msg("assistant", "m4")).unwrap();
        }

        let tool = ControlUnfoldTool::new(session);
        let (_, rx) = futures::channel::oneshot::channel();
        let result = tool.execute(rx, serde_json::json!({ "id": id })).await.unwrap();
        assert!(!result.is_error);
        let body = match &result.content[0] {
            ToolResultContent::Text(t) => &t.content,
            _ => panic!("expected text content"),
        };
        assert!(body.contains("m2"), "folded content must include the original even with later messages: {body}");
    }

    #[tokio::test]
    async fn unfold_tool_errors_on_unknown_id() {
        let session = InMemorySession::new().arc();
        let tool = ControlUnfoldTool::new(session);

        let (_, rx) = futures::channel::oneshot::channel();
        let result = tool.execute(rx, serde_json::json!({ "id": "nope" })).await.unwrap();
        assert!(result.is_error);
        let body = match &result.content[0] {
            ToolResultContent::Text(t) => &t.content,
            _ => panic!("expected text content"),
        };
        assert!(body.contains("no folded content"), "error should explain the missing id: {body}");
    }

    #[tokio::test]
    async fn unfold_tool_ignores_stop() {
        let session = InMemorySession::new().arc();
        let (id, fold) = fold_message(0, 2, "sum");
        {
            let mut s = session.lock().unwrap();
            s.append(text_msg("user", "m1")).unwrap();
            s.append(text_msg("user", "m2")).unwrap();
            s.append(fold).unwrap();
            s.append(stop_message()).unwrap();
        }

        let tool = ControlUnfoldTool::new(session);
        let (_, rx) = futures::channel::oneshot::channel();
        let result = tool.execute(rx, serde_json::json!({ "id": id })).await.unwrap();
        assert!(!result.is_error, "unfold returns the folded content even when a stop follows");
        let body = match &result.content[0] {
            ToolResultContent::Text(t) => &t.content,
            _ => panic!("expected text content"),
        };
        assert!(body.contains("m1"), "folded content must still be returned despite the stop: {body}");
        assert!(body.contains("m2"), "folded content must still be returned despite the stop: {body}");
    }
}

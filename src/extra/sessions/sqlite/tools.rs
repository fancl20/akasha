use async_trait::async_trait;
use rusqlite::Connection;
use schemars::JsonSchema;
use serde::Deserialize;
use std::sync::{Arc, Mutex};

use crate::core::tools::{ToolError, ToolHandler, ToolRegistry};
use crate::core::types::{Message, TextContent, ToolDefinition, ToolResult, ToolResultContent};
use crate::extra::sessions::sqlite::SqliteSession;

pub struct RangeIter<'a> {
    db: &'a Connection,
    curr_id: Option<i64>,
    head_id: i64,
}

impl Iterator for RangeIter<'_> {
    type Item = (i64, Message);

    fn next(&mut self) -> Option<Self::Item> {
        let id = self.curr_id.take()?;
        let (content_hash, prev_id): (String, Option<i64>) = self
            .db
            .query_row("SELECT content, prev FROM cards WHERE id = ?1", [id], |row| Ok((row.get(0)?, row.get(1)?)))
            .ok()?;
        let body: String =
            self.db.query_row("SELECT body FROM contents WHERE hash = ?1", [&content_hash], |row| row.get(0)).ok()?;

        if id != self.head_id {
            self.curr_id = prev_id;
        }
        Some((id, serde_json::from_str(&body).ok()?))
    }
}

#[derive(Deserialize, JsonSchema)]
pub struct MessageRange {
    /// The newer card ID (start of range, walked from)
    pub tail_id: i64,
    /// The older card ID (end of range, walked towards)
    pub head_id: i64,
}

impl MessageRange {
    pub fn iter<'a>(&self, db: &'a Connection) -> Result<RangeIter<'a>, ToolError> {
        db.query_row("SELECT 1 FROM cards WHERE id = ?1", [self.tail_id], |_| Ok(()))
            .map_err(|e| ToolError::Execution(format!("card {} not found: {e}", self.tail_id)))?;
        Ok(RangeIter { db, curr_id: Some(self.tail_id), head_id: self.head_id })
    }
}

pub struct ExpandRangeTool {
    definition: ToolDefinition,
    db: Arc<Mutex<Connection>>,
}

#[async_trait]
impl ToolHandler for ExpandRangeTool {
    fn definition(&self) -> ToolDefinition {
        self.definition.clone()
    }

    async fn execute(
        &self,
        _cancel: tokio::sync::watch::Receiver<bool>,
        params: serde_json::Value,
    ) -> Result<ToolResult, ToolError> {
        let range: MessageRange = serde_json::from_value(params).map_err(|e| ToolError::Validation(e.to_string()))?;

        let db = self.db.lock().map_err(|_| ToolError::Execution("lock poisoned".into()))?;

        let messages: Vec<Message> = range.iter(&db)?.map(|(_, msg)| msg).collect();
        let content = vec![ToolResultContent::Text(TextContent {
            content: serde_json::to_string(&messages)
                .map_err(|e| ToolError::Execution(format!("failed to serialize messages: {e}")))?,
        })];

        Ok(ToolResult { tool_call_id: None, content, is_error: false })
    }
}

impl SqliteSession {
    pub fn register_tools(&self, registry: &mut ToolRegistry) {
        registry.register(ExpandRangeTool{
            definition: ToolDefinition {
                name: "message_range".to_string(),
                description: "Retrieve messages in a range between two IDs. returning all messages from new to old in the range inclusive of both endpoints.".to_string(),
                parameters: serde_json::to_value(schemars::schema_for!(MessageRange))
                    .expect("MessageRangeParams schema is always valid"),
            },
            db: self.db(),
        }.into());
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::core::session::Session;
    use crate::core::types::{ContentBlock, TextContent};

    fn text_msg(role: &str, content: &str) -> Message {
        Message {
            role: role.to_string(),
            content: vec![ContentBlock::Text(TextContent { content: content.to_string() })],
        }
    }

    fn cancel_rx() -> tokio::sync::watch::Receiver<bool> {
        let (_, rx) = tokio::sync::watch::channel(false);
        rx
    }

    fn make_tool() -> (SqliteSession, ExpandRangeTool) {
        let session = SqliteSession::new(":memory:", "test").unwrap();
        let tool = ExpandRangeTool {
            definition: ToolDefinition {
                name: "message_range".into(),
                description: String::new(),
                parameters: serde_json::Value::Null,
            },
            db: session.db(),
        };
        (session, tool)
    }

    /// Helper: append N messages and return the card IDs in order.
    fn append_messages(session: &mut SqliteSession, n: usize) -> Vec<i64> {
        let mut ids = Vec::new();
        for i in 0..n {
            session.append(text_msg("user", &format!("msg_{i}"))).unwrap();
            ids.push(session.head_id().parse::<i64>().unwrap());
        }
        ids
    }

    /// Extract the returned messages from a ToolResult.
    fn extract_messages(result: &ToolResult) -> Vec<Message> {
        assert!(!result.is_error, "tool returned an error");
        let text = match &result.content[0] {
            ToolResultContent::Text(t) => &t.content,
            _ => panic!("expected text content"),
        };
        serde_json::from_str(text).unwrap()
    }

    #[tokio::test]
    async fn full_range() {
        let (mut session, tool) = make_tool();
        let ids = append_messages(&mut session, 4);

        let result =
            tool.execute(cancel_rx(), serde_json::json!({ "tail_id": ids[3], "head_id": ids[0] })).await.unwrap();

        let msgs = extract_messages(&result);
        // Walked tail→head so order is newest-first: msg_3, msg_2, msg_1, msg_0
        assert_eq!(msgs.len(), 4);
        assert_eq!(msgs[0].role, "user");
        for (i, msg) in msgs.iter().enumerate() {
            let expected = format!("msg_{}", 3 - i);
            assert_eq!(msg.content[0], ContentBlock::Text(TextContent { content: expected }),);
        }
    }

    #[tokio::test]
    async fn single_card_range() {
        let (mut session, tool) = make_tool();
        let ids = append_messages(&mut session, 3);

        // tail == head → exactly one card
        let result =
            tool.execute(cancel_rx(), serde_json::json!({ "tail_id": ids[1], "head_id": ids[1] })).await.unwrap();

        let msgs = extract_messages(&result);
        assert_eq!(msgs.len(), 1);
        assert_eq!(msgs[0].content[0], ContentBlock::Text(TextContent { content: "msg_1".into() }),);
    }

    #[tokio::test]
    async fn subrange() {
        let (mut session, tool) = make_tool();
        let ids = append_messages(&mut session, 5);

        // Range from ids[4] back to ids[2] → 3 messages: msg_4, msg_3, msg_2
        let result =
            tool.execute(cancel_rx(), serde_json::json!({ "tail_id": ids[4], "head_id": ids[2] })).await.unwrap();

        let msgs = extract_messages(&result);
        assert_eq!(msgs.len(), 3);
        for (i, msg) in msgs.iter().enumerate() {
            let expected = format!("msg_{}", 4 - i);
            assert_eq!(msg.content[0], ContentBlock::Text(TextContent { content: expected }),);
        }
    }

    #[tokio::test]
    async fn nonexistent_tail_returns_error() {
        let (_session, tool) = make_tool();

        let err = tool.execute(cancel_rx(), serde_json::json!({ "tail_id": 9999, "head_id": 1 })).await.unwrap_err();

        match err {
            ToolError::Execution(msg) => assert!(msg.contains("card 9999 not found")),
            other => panic!("expected Execution error, got: {other}"),
        }
    }

    #[tokio::test]
    async fn invalid_params_returns_validation_error() {
        let (_session, tool) = make_tool();

        let err = tool.execute(cancel_rx(), serde_json::json!({ "bogus": true })).await.unwrap_err();

        match err {
            ToolError::Validation(msg) => assert!(msg.contains("tail_id")),
            other => panic!("expected Validation error, got: {other}"),
        }
    }
}

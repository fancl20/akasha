use rusqlite::{Connection, params};
use std::sync::{Arc, Mutex};
use std::time::{SystemTime, UNIX_EPOCH};

use crate::core::session::{Session, SessionError};
use crate::core::types::Message;

const SCHEMA: &str = include_str!("db.sql");

pub struct SqliteSession {
    db: Arc<Mutex<Connection>>,
    ref_name: Option<String>,
    tail_id: Option<i64>,
    messages: Vec<Message>,
}

fn session_err(e: impl std::fmt::Display) -> SessionError {
    SessionError::Failed { message: e.to_string() }
}

impl SqliteSession {
    pub fn new(db_path: &str, ref_name: &str) -> Result<Self, SessionError> {
        let conn = Connection::open(db_path).map_err(session_err)?;
        conn.execute_batch(SCHEMA).map_err(session_err)?;

        let tail_id =
            conn.query_row("SELECT tail FROM refs WHERE ref = ?1", [&ref_name], |row| row.get::<_, i64>(0)).ok();

        let mut messages = Vec::new();
        let mut iter_id = tail_id;
        while let Some(id) = iter_id {
            let (content, prev_id): (String, Option<i64>) = conn
                .query_row("SELECT content, prev FROM messages WHERE id = ?1", [id], |row| {
                    Ok((row.get(0)?, row.get(1)?))
                })
                .map_err(session_err)?;
            messages.push(serde_json::from_str(&content).map_err(session_err)?);
            iter_id = prev_id;
        }
        messages.reverse();

        Ok(Self { db: Arc::new(Mutex::new(conn)), ref_name: Some(ref_name.to_string()), tail_id, messages })
    }

    pub fn db(&self) -> Arc<Mutex<Connection>> {
        self.db.clone()
    }
}

impl Session for SqliteSession {
    fn messages(&self) -> &Vec<Message> {
        &self.messages
    }

    fn append(&mut self, message: Message) -> Result<(), SessionError> {
        let mut db = self.db.lock().map_err(|_| SessionError::Failed { message: "lock poisoned".into() })?;
        let tx = db.transaction().map_err(session_err)?;

        let content = serde_json::to_string(&message).map_err(session_err)?;
        let now = SystemTime::now().duration_since(UNIX_EPOCH).unwrap_or_default().as_secs() as i64;

        let id: i64 = tx
            .query_row(
                "INSERT INTO messages (content, created_at, prev) VALUES (?1, ?2, ?3) RETURNING id",
                params![&content, now, self.tail_id],
                |row| row.get(0),
            )
            .map_err(session_err)?;

        if let Some(ref_name) = &self.ref_name {
            tx.execute("INSERT OR REPLACE INTO refs (ref, tail) VALUES (?1, ?2)", params![ref_name, id])
                .map_err(session_err)?;
        }
        tx.commit().map_err(session_err)?;

        self.tail_id = Some(id);
        self.messages.push(message);
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::core::types::{ContentBlock, TextContent};

    fn text_msg(role: &str, content: &str) -> Message {
        Message {
            role: role.to_string(),
            content: vec![ContentBlock::Text(TextContent { content: content.to_string() })],
        }
    }

    fn make_session(ref_name: &str) -> SqliteSession {
        SqliteSession::new(":memory:", ref_name).unwrap()
    }

    #[test]
    fn empty_session() {
        let s = make_session("test");
        assert!(s.messages().is_empty());
    }

    #[test]
    fn append_and_messages() {
        let mut s = make_session("test");
        s.append(text_msg("user", "hello")).unwrap();
        s.append(text_msg("assistant", "hi")).unwrap();
        s.append(text_msg("user", "how are you")).unwrap();
        let ctx = s.messages();
        assert_eq!(ctx.len(), 3);
        assert_eq!(ctx[0].role, "user");
        assert_eq!(ctx[1].role, "assistant");
        assert_eq!(ctx[2].role, "user");
    }

    #[test]
    fn open_reloads_chain() {
        let db_path = std::env::temp_dir().join(format!("zk_test_{}", std::process::id()));
        let path = db_path.to_str().unwrap();

        let mut s = SqliteSession::new(path, "conv1").unwrap();
        s.append(text_msg("user", "hello")).unwrap();
        s.append(text_msg("assistant", "hi")).unwrap();
        drop(s);

        let loaded = SqliteSession::new(path, "conv1").unwrap();
        let ctx = loaded.messages();
        assert_eq!(ctx.len(), 2);
        assert_eq!(ctx[0].role, "user");
        assert_eq!(ctx[1].role, "assistant");

        let _ = std::fs::remove_file(db_path);
    }
}

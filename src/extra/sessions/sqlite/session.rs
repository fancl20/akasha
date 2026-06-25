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

fn load(conn: &Connection, tail_id: Option<i64>) -> Result<Vec<Message>, SessionError> {
    let mut messages = Vec::new();
    let mut iter_id = tail_id;
    while let Some(id) = iter_id {
        let (content, prev_id): (String, Option<i64>) = conn
            .query_row("SELECT content, prev FROM messages WHERE id = ?1", [id], |row| Ok((row.get(0)?, row.get(1)?)))
            .map_err(session_err)?;
        messages.push(serde_json::from_str(&content).map_err(session_err)?);
        iter_id = prev_id;
    }
    messages.reverse();
    Ok(messages)
}

impl SqliteSession {
    pub fn new(db_path: &str, ref_name: &str) -> Result<Self, SessionError> {
        let db = Connection::open(db_path).map_err(session_err)?;
        db.execute_batch(SCHEMA).map_err(session_err)?;

        let tail_id =
            db.query_row("SELECT tail FROM refs WHERE ref = ?1", [&ref_name], |row| row.get::<_, i64>(0)).ok();
        let messages = load(&db, tail_id)?;

        Ok(Self { db: Arc::new(Mutex::new(db)), ref_name: Some(ref_name.to_string()), tail_id, messages })
    }

    pub fn db(&self) -> Arc<Mutex<Connection>> {
        self.db.clone()
    }

    pub fn fork(&self, ref_name: &str) -> Result<Self, SessionError> {
        // Pin the new ref at the current tail so it inherits this session's history.
        if let Some(tail) = self.tail_id {
            self.db
                .lock()
                .unwrap()
                .execute("INSERT OR REPLACE INTO refs (ref, tail) VALUES (?1, ?2)", params![ref_name, tail])
                .map_err(session_err)?;
        }

        Ok(Self {
            db: self.db.clone(),
            ref_name: Some(ref_name.to_string()),
            tail_id: self.tail_id,
            messages: self.messages.clone(),
        })
    }

    pub fn get(&self, ref_name: &str) -> Result<Self, SessionError> {
        let db = self.db.lock().unwrap();
        let tail_id = self
            .db
            .lock()
            .unwrap()
            .query_row("SELECT tail FROM refs WHERE ref = ?1", [&ref_name], |row| row.get::<_, i64>(0))
            .map_err(|e| match e {
                rusqlite::Error::QueryReturnedNoRows => {
                    SessionError::Failed { message: format!("ref not found: {ref_name}") }
                }
                e => session_err(e),
            })?;
        let messages = load(&db, Some(tail_id))?;
        Ok(Self { db: self.db.clone(), ref_name: Some(ref_name.to_string()), tail_id: Some(tail_id), messages })
    }
}

impl Session for SqliteSession {
    fn messages(&self) -> Box<dyn Iterator<Item = &Message> + '_> {
        Box::new(self.messages.iter())
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
        assert!(s.messages().next().is_none());
    }

    #[test]
    fn append_and_messages() {
        let mut s = make_session("test");
        s.append(text_msg("user", "hello")).unwrap();
        s.append(text_msg("assistant", "hi")).unwrap();
        s.append(text_msg("user", "how are you")).unwrap();
        let ctx: Vec<_> = s.messages().collect();
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
        let ctx: Vec<_> = loaded.messages().collect();
        assert_eq!(ctx.len(), 2);
        assert_eq!(ctx[0].role, "user");
        assert_eq!(ctx[1].role, "assistant");

        let _ = std::fs::remove_file(db_path);
    }

    #[test]
    fn fork_shares_parent_then_diverges() {
        let mut a = make_session("main");
        a.append(text_msg("user", "hello")).unwrap();
        a.append(text_msg("assistant", "hi")).unwrap();

        // Fork shares the parent chain at the current tail.
        let mut b = a.fork("topic").unwrap();
        assert_eq!(a.messages().count(), 2);
        assert_eq!(b.messages().count(), 2);

        // Each session appends independently from the fork point.
        a.append(text_msg("user", "a-only")).unwrap();
        b.append(text_msg("user", "b-only")).unwrap();

        let a_ctx: Vec<_> = a.messages().collect();
        let b_ctx: Vec<_> = b.messages().collect();

        // Shared prefix is identical, tails diverge.
        assert_eq!(a_ctx.len(), 3);
        assert_eq!(b_ctx.len(), 3);
        assert_eq!(a_ctx[0].role, b_ctx[0].role);
        assert_eq!(a_ctx[1].role, b_ctx[1].role);
        assert_ne!(a_ctx[2].content, b_ctx[2].content);
    }

    #[test]
    fn fork_persists_ref_for_reload() {
        let db_path = std::env::temp_dir().join(format!("zk_fork_{}", std::process::id()));
        let path = db_path.to_str().unwrap();

        {
            let mut a = SqliteSession::new(path, "main").unwrap();
            a.append(text_msg("user", "hello")).unwrap();
            a.append(text_msg("assistant", "hi")).unwrap();
            // Fork pins "topic" at the current tail and is then dropped.
            let _ = a.fork("topic").unwrap();
        }

        // Reopening the forked ref reconstructs the shared parent chain.
        let loaded = SqliteSession::new(path, "topic").unwrap();
        let ctx: Vec<_> = loaded.messages().collect();
        assert_eq!(ctx.len(), 2);
        assert_eq!(ctx[0].role, "user");
        assert_eq!(ctx[1].role, "assistant");

        let _ = std::fs::remove_file(db_path);
    }

    #[test]
    fn fork_empty_session_diverges() {
        let a = make_session("main");
        // Forking an empty session yields an empty session; the ref row is
        // deferred until the first append.
        let mut b = a.fork("topic").unwrap();
        assert_eq!(b.messages().count(), 0);

        b.append(text_msg("user", "b-only")).unwrap();
        assert_eq!(a.messages().count(), 0);
        assert_eq!(b.messages().count(), 1);
    }

    #[test]
    fn get_returns_existing_ref() {
        let mut s = make_session("main");
        s.append(text_msg("user", "hello")).unwrap();
        s.append(text_msg("assistant", "hi")).unwrap();

        // get reopens the ref over the shared connection.
        let loaded = s.get("main").unwrap();
        let ctx: Vec<_> = loaded.messages().collect();
        assert_eq!(ctx.len(), 2);
        assert_eq!(ctx[0].role, "user");
        assert_eq!(ctx[1].role, "assistant");
    }

    #[test]
    fn get_errors_on_missing_ref() {
        let s = make_session("main");
        let err = s.get("nope").err().expect("get should error for a missing ref");
        assert!(matches!(err, SessionError::Failed { message } if message.contains("nope")));
    }

    #[test]
    fn get_sees_latest_state_and_shares_connection() {
        let mut a = make_session("main");
        a.append(text_msg("user", "hello")).unwrap();
        a.fork("topic").unwrap();

        // get reopens the forked ref over the shared connection.
        let mut b = a.get("topic").unwrap();
        assert_eq!(b.messages().count(), 1);

        // Appending through the reopened session advances that ref only.
        b.append(text_msg("user", "b-only")).unwrap();
        assert_eq!(b.messages().count(), 2);

        // main is unaffected by topic's append...
        assert_eq!(a.messages().count(), 1);

        // ...and reopening topic again reflects the append (shared connection).
        let c = a.get("topic").unwrap();
        assert_eq!(c.messages().count(), 2);
    }
}

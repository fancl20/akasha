use rusqlite::{Connection, params};
use sha2::{Digest, Sha256};
use std::sync::{Arc, Mutex};
use std::time::{SystemTime, UNIX_EPOCH};

use crate::core::session::{Session, SessionError};
use crate::core::types::Message;

const SCHEMA: &str = include_str!("db.sql");

pub struct SqliteSession {
    db: Arc<Mutex<Connection>>,
    ref_name: String,
    head_id: Option<i64>,
    messages: Vec<Message>,
}

fn session_err(e: impl std::fmt::Display) -> SessionError {
    SessionError::Failed { message: e.to_string() }
}

impl SqliteSession {
    pub fn new(db_path: &str, ref_name: &str) -> Result<Self, SessionError> {
        let conn = Connection::open(db_path).map_err(session_err)?;
        conn.execute_batch(SCHEMA).map_err(session_err)?;

        let head_id =
            conn.query_row("SELECT tail FROM refs WHERE ref = ?1", [&ref_name], |row| row.get::<_, i64>(0)).ok();

        let mut messages = Vec::new();
        let mut iter_id = head_id;
        while let Some(id) = iter_id {
            let (content_hash, prev_id): (String, Option<i64>) = conn
                .query_row("SELECT content, prev FROM cards WHERE id = ?1", [id], |row| Ok((row.get(0)?, row.get(1)?)))
                .map_err(session_err)?;
            let body: String = conn
                .query_row("SELECT body FROM contents WHERE hash = ?1", [&content_hash], |row| row.get(0))
                .map_err(session_err)?;
            messages.push(serde_json::from_str(&body).map_err(session_err)?);
            iter_id = prev_id;
        }
        messages.reverse();

        Ok(Self { db: Arc::new(Mutex::new(conn)), ref_name: ref_name.to_string(), head_id, messages })
    }

    pub fn db(&self) -> Arc<Mutex<Connection>> {
        self.db.clone()
    }
}

impl Session for SqliteSession {
    fn head_id(&self) -> String {
        self.head_id.map_or("null".to_string(), |id| id.to_string())
    }

    fn context(&self) -> &Vec<Message> {
        &self.messages
    }

    fn append(&mut self, message: Message) -> Result<(), SessionError> {
        let mut db = self.db.lock().map_err(|_| SessionError::Failed { message: "lock poisoned".into() })?;
        let tx = db.transaction().map_err(session_err)?;

        let body = serde_json::to_string(&message).map_err(session_err)?;
        let hash = format!("{:x}", Sha256::digest(body.as_bytes()));
        let now = SystemTime::now().duration_since(UNIX_EPOCH).unwrap_or_default().as_secs() as i64;

        tx.execute("INSERT OR IGNORE INTO contents (hash, body) VALUES (?1, ?2)", params![&hash, &body])
            .map_err(session_err)?;
        let id: i64 = tx
            .query_row(
                "INSERT INTO cards (content, created_at, prev) VALUES (?1, ?2, ?3) RETURNING id",
                params![&hash, now, self.head_id],
                |row| row.get(0),
            )
            .map_err(session_err)?;
        tx.execute("INSERT OR REPLACE INTO refs (ref, tail) VALUES (?1, ?2)", rusqlite::params![&self.ref_name, id])
            .map_err(session_err)?;
        tx.commit().map_err(session_err)?;

        self.head_id = Some(id);
        self.messages.push(message);
        Ok(())
    }

    fn create(&self) -> Box<dyn Session> {
        let new_ref =
            format!("session:{}", SystemTime::now().duration_since(UNIX_EPOCH).unwrap_or_default().as_nanos());
        Box::new(Self { db: self.db.clone(), ref_name: new_ref, head_id: None, messages: Vec::new() })
    }

    fn fork(&self) -> Box<dyn Session> {
        let forked_ref = format!("{}:fork:{}", self.ref_name, self.head_id.map_or("null".into(), |id| id.to_string()));
        Box::new(Self {
            db: self.db.clone(),
            ref_name: forked_ref,
            head_id: self.head_id,
            messages: self.messages.clone(),
        })
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
        assert_eq!(s.head_id(), "null");
        assert!(s.context().is_empty());
    }

    #[test]
    fn append_and_context() {
        let mut s = make_session("test");
        s.append(text_msg("user", "hello")).unwrap();
        s.append(text_msg("assistant", "hi")).unwrap();
        s.append(text_msg("user", "how are you")).unwrap();
        let ctx = s.context();
        assert_eq!(ctx.len(), 3);
        assert_eq!(ctx[0].role, "user");
        assert_eq!(ctx[1].role, "assistant");
        assert_eq!(ctx[2].role, "user");
    }

    #[test]
    fn head_id_changes() {
        let mut s = make_session("test");
        let id0 = s.head_id();
        s.append(text_msg("user", "a")).unwrap();
        let id1 = s.head_id();
        s.append(text_msg("user", "b")).unwrap();
        let id2 = s.head_id();
        assert_ne!(id0, id1);
        assert_ne!(id1, id2);
    }

    #[test]
    fn fork_shares_prefix() {
        let mut s = make_session("test");
        s.append(text_msg("user", "1")).unwrap();
        s.append(text_msg("user", "2")).unwrap();
        let forked = s.fork();
        assert_eq!(s.context(), forked.context());
        assert_eq!(s.head_id(), forked.head_id());
    }

    #[test]
    fn fork_independence() {
        let mut s = make_session("test");
        s.append(text_msg("user", "shared")).unwrap();
        s.append(text_msg("assistant", "response")).unwrap();
        let mut forked = s.fork();
        s.append(text_msg("user", "branch A")).unwrap();
        forked.append(text_msg("user", "branch B")).unwrap();
        assert_eq!(s.context().len(), 3);
        assert_eq!(forked.context().len(), 3);
        assert_ne!(s.head_id(), forked.head_id());
    }

    #[test]
    fn create_is_empty() {
        let mut s = make_session("test");
        s.append(text_msg("user", "hello")).unwrap();
        let created = s.create();
        assert_eq!(created.head_id(), "null");
        assert!(created.context().is_empty());
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
        let ctx = loaded.context();
        assert_eq!(ctx.len(), 2);
        assert_eq!(ctx[0].role, "user");
        assert_eq!(ctx[1].role, "assistant");

        let _ = std::fs::remove_file(db_path);
    }
}

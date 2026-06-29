//! Session fork/get management for the opinionated sub-agent framework the
//! [`AgentBuilder`] produces.
//!
//! [`SessionManager`] extends [`Session`]: a manager *is* its main session, so
//! the main conversation is driven directly through `append`/`messages`.
//! [`fork`](SessionManager::fork) and [`get`](SessionManager::get) return forks
//! as owned `Box<dyn SessionManager>` — each itself a session (and forkable
//! again). It has two implementations:
//!
//! - [`SqliteSession`] implements it natively: a fork persists a database ref
//!   named after the UUID, so it survives restarts and `get` reopens it from the
//!   DB (cache-free).
//! - [`SessionAdapter`] adapts any [`Session`]: forks are live sessions stored as
//!   a shared `Arc<Mutex<dyn Session>>`; each handle owns a read-cache of the
//!   messages (mirrored on `append`) so `messages()` returns borrowed refs from
//!   owned data without holding the lock.
//!
//! [`AgentBuilder`]: super::AgentBuilder
//! [`SqliteSession`]: crate::extra::sessions::sqlite::SqliteSession
//! [`Session`]: crate::core::session::Session

use std::collections::HashMap;
use std::sync::{Arc, Mutex};

use uuid::Uuid;

use crate::core::session::{Session, SessionError};
use crate::core::types::Message;
use crate::extra::sessions::sqlite::SqliteSession;

/// A [`Session`] that can also fork itself and reopen forks by UUID.
///
/// Because [`Session`] is a supertrait, a session manager *is* its main session:
/// interact with the main conversation through `append`/`messages` directly.
/// [`fork`](Self::fork) and [`get`](Self::get) return owned
/// `Box<dyn SessionManager>` values, which are themselves usable as sessions
/// (the [`Session`] methods are available on them via the supertrait).
///
/// [`Session`]: crate::core::session::Session
pub trait SessionManager: Session {
    fn fork(&mut self) -> Result<(String, Box<dyn SessionManager>), SessionError>;
    fn get(&self, id: &str) -> Result<Box<dyn SessionManager>, SessionError>;
}

impl SessionManager for SqliteSession {
    fn fork(&mut self) -> Result<(String, Box<dyn SessionManager>), SessionError> {
        let id = Uuid::now_v7().to_string();
        Ok((id.clone(), Box::new(SqliteSession::fork(self, &id)?)))
    }

    fn get(&self, id: &str) -> Result<Box<dyn SessionManager>, SessionError> {
        Ok(Box::new(SqliteSession::get(self, id)?))
    }
}

/// A shared, live session: the backing [`Session`] behind a [`SessionAdapter`].
///
/// Forks share one of these, so an `append` through any handle reaches the live
/// session and is visible to a later [`get`]. Matches the `Arc<Mutex<dyn Session>>`
/// shape used for sessions throughout the crate.
///
/// [`get`]: SessionAdapter::get
/// [`Session`]: crate::core::session::Session
type Shared = Arc<Mutex<dyn Session>>;

/// A shared registry of fork UUID → live fork session.
type Registry = Arc<Mutex<HashMap<String, Shared>>>;

/// Adapts any [`Session`] into a [`SessionManager`] with live, shared forks.
///
/// The adapter *is* its session: `append`/`messages` operate on it directly. It
/// holds the backing session as a shared [`Shared`] (`Arc<Mutex<dyn Session>>`)
/// and an owned read-cache of its messages. `append` writes to **both** — the
/// shared backing (so other handles and a later [`get`](Self::get) see it) and
/// the cache — while `messages` returns borrowed refs from the owned cache, with
/// no lock held across the return.
///
/// [`fork`](Self::fork) snapshots the current messages into a fresh shared
/// session registered under a UUID; [`get`](Self::get) opens a handle to that
/// shared session, rebuilding its cache so it reflects appends made through any
/// other handle. Forks are in-memory only; for persistence use [`SqliteSession`]
/// directly.
///
/// [`Session`]: crate::core::session::Session
pub struct SessionAdapter {
    backing: Shared,
    messages: Vec<Message>,
    forks: Registry,
    fresh: Arc<dyn Fn() -> Shared + Send + Sync>,
}

fn poison() -> SessionError {
    SessionError::Failed { message: "session lock poisoned".into() }
}

impl SessionAdapter {
    /// Wrap `session` as the main session to fork from. `fresh` constructs each
    /// empty backing session a fork is replayed into (e.g.
    /// `|| InMemorySession::new().arc()`).
    pub fn new(session: impl Session + 'static, fresh: impl Fn() -> Shared + Send + Sync + 'static) -> Self {
        // `Arc<Mutex<S>>` coerces to `Arc<Mutex<dyn Session>>` the same way
        // [`Session::arc`] does — `Mutex<S>: Unsize<Mutex<dyn Session>>`.
        Self::from_shared(Arc::new(Mutex::new(session)), Arc::new(Mutex::new(HashMap::new())), Arc::new(fresh))
    }

    /// Build a handle over `backing`, seeding its read-cache from the live
    /// session and sharing `forks`/`fresh` so the handle is itself forkable.
    fn from_shared(backing: Shared, forks: Registry, fresh: Arc<dyn Fn() -> Shared + Send + Sync>) -> Self {
        let messages = backing.lock().unwrap_or_else(|e| e.into_inner()).messages().cloned().collect();
        Self { backing, messages, forks, fresh }
    }
}

impl Session for SessionAdapter {
    fn append(&mut self, message: Message) -> Result<(), SessionError> {
        // Live write: forward to the shared backing so other handles / a later
        // `get` observe it, then mirror into the owned read-cache.
        self.backing.lock().map_err(|_| poison())?.append(message.clone())?;
        self.messages.push(message);
        Ok(())
    }

    fn messages(&self) -> Box<dyn DoubleEndedIterator<Item = &Message> + '_> {
        Box::new(self.messages.iter())
    }
}

impl SessionManager for SessionAdapter {
    fn fork(&mut self) -> Result<(String, Box<dyn SessionManager>), SessionError> {
        let id = Uuid::now_v7().to_string();
        let backing: Shared = (self.fresh)();
        for message in &self.messages {
            backing.lock().map_err(|_| poison())?.append(message.clone())?;
        }
        self.forks.lock().map_err(|_| poison())?.insert(id.clone(), backing.clone());
        Ok((id, Box::new(Self::from_shared(backing, self.forks.clone(), self.fresh.clone()))))
    }

    fn get(&self, id: &str) -> Result<Box<dyn SessionManager>, SessionError> {
        let backing = self
            .forks
            .lock()
            .map_err(|_| poison())?
            .get(id)
            .cloned()
            .ok_or_else(|| SessionError::Failed { message: format!("fork not found: {id}") })?;
        Ok(Box::new(Self::from_shared(backing, self.forks.clone(), self.fresh.clone())))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    use crate::core::session::InMemorySession;
    use crate::core::types::{ContentBlock, Message, TextContent};

    fn text_msg(role: &str, content: &str) -> Message {
        Message {
            role: role.to_string(),
            content: vec![ContentBlock::Text(TextContent { content: content.to_string() })],
        }
    }

    /// Roles of a manager's session — works on any `SessionManager` since
    /// `Session` is a supertrait (no upcast needed).
    fn roles(mgr: &dyn SessionManager) -> Vec<String> {
        mgr.messages().map(|m| m.role.clone()).collect()
    }

    fn last_text(mgr: &dyn SessionManager) -> Option<String> {
        mgr.messages().last().and_then(|m| {
            m.content.iter().find_map(|b| match b {
                ContentBlock::Text(t) => Some(t.content.clone()),
                _ => None,
            })
        })
    }

    fn inmem_factory() -> impl Fn() -> Shared + Send + Sync + 'static {
        || InMemorySession::new().arc()
    }

    // --- SessionAdapter (generic, live shared forks) ---

    #[test]
    fn adapter_is_its_main_session() {
        let mut mgr = SessionAdapter::new(InMemorySession::new(), inmem_factory());
        mgr.append(text_msg("user", "hello")).unwrap();
        mgr.append(text_msg("assistant", "hi")).unwrap();
        assert_eq!(roles(&mgr), vec!["user".to_string(), "assistant".to_string()]);
    }

    #[test]
    fn adapter_fork_returns_a_session_with_messages() {
        let mut main = InMemorySession::new();
        main.append(text_msg("user", "hello")).unwrap();
        main.append(text_msg("assistant", "hi")).unwrap();

        let mut mgr = SessionAdapter::new(main, inmem_factory());
        let (_id, forked) = mgr.fork().unwrap();
        assert_eq!(roles(&*forked), vec!["user".to_string(), "assistant".to_string()]);
    }

    #[test]
    fn adapter_fork_is_independent_from_main() {
        let mut main = InMemorySession::new();
        main.append(text_msg("user", "hello")).unwrap();

        let mut mgr = SessionAdapter::new(main, inmem_factory());
        let (_id, mut forked) = mgr.fork().unwrap();

        mgr.append(text_msg("user", "main-only")).unwrap();
        forked.append(text_msg("user", "fork-only")).unwrap();

        assert_eq!(roles(&mgr), vec!["user".to_string(), "user".to_string()]);
        assert_eq!(roles(&*forked), vec!["user".to_string(), "user".to_string()]);
        assert_ne!(last_text(&mgr).as_deref(), Some("fork-only"));
        assert_ne!(last_text(&*forked).as_deref(), Some("main-only"));
    }

    #[test]
    fn adapter_fork_is_itself_a_session_manager() {
        let mut main = InMemorySession::new();
        main.append(text_msg("user", "hello")).unwrap();

        let mut mgr = SessionAdapter::new(main, inmem_factory());
        let (_id, mut forked) = mgr.fork().unwrap();
        let (_gid, grandchild) = forked.fork().unwrap();
        assert_eq!(roles(&*grandchild), vec!["user".to_string()]);
    }

    #[test]
    fn adapter_get_retrieves_a_fork() {
        let mut main = InMemorySession::new();
        main.append(text_msg("user", "hello")).unwrap();
        main.append(text_msg("assistant", "hi")).unwrap();

        let mut mgr = SessionAdapter::new(main, inmem_factory());
        let (id, _forked) = mgr.fork().unwrap();
        let got = mgr.get(&id).unwrap();
        assert_eq!(roles(&*got), vec!["user".to_string(), "assistant".to_string()]);
    }

    #[test]
    fn adapter_fork_appends_are_live_across_gets() {
        let mut main = InMemorySession::new();
        main.append(text_msg("user", "hello")).unwrap();

        let mut mgr = SessionAdapter::new(main, inmem_factory());
        let (id, mut forked) = mgr.fork().unwrap();
        // Append through one handle; a later get() of the same fork observes it,
        // because both share the fork's backing session.
        forked.append(text_msg("assistant", "fork-only")).unwrap();

        let got = mgr.get(&id).unwrap();
        assert_eq!(roles(&*got), vec!["user".to_string(), "assistant".to_string()]);
    }

    #[test]
    fn adapter_get_unknown_errors() {
        let mgr = SessionAdapter::new(InMemorySession::new(), inmem_factory());
        let err = mgr.get("nope").err().expect("unknown fork should error");
        assert!(matches!(err, SessionError::Failed { .. }));
    }

    // --- SqliteSession (native, DB-backed SessionManager) ---

    fn sqlite(ref_name: &str) -> SqliteSession {
        SqliteSession::new(":memory:", ref_name).unwrap()
    }

    #[test]
    fn sqlite_is_its_main_session() {
        let mut s = sqlite("main");
        s.append(text_msg("user", "hello")).unwrap();
        assert_eq!(roles(&s), vec!["user".to_string()]);
    }

    #[test]
    fn sqlite_fork_returns_a_session_with_messages() {
        let mut s = sqlite("main");
        s.append(text_msg("user", "hello")).unwrap();
        s.append(text_msg("assistant", "hi")).unwrap();

        let (_id, forked) = SessionManager::fork(&mut s).unwrap();
        assert_eq!(roles(&*forked), vec!["user".to_string(), "assistant".to_string()]);
    }

    #[test]
    fn sqlite_fork_is_independent_from_main() {
        let mut s = sqlite("main");
        s.append(text_msg("user", "hello")).unwrap();

        let (_id, forked) = SessionManager::fork(&mut s).unwrap();
        s.append(text_msg("user", "main-only")).unwrap();

        assert_eq!(roles(&s), vec!["user".to_string(), "user".to_string()]);
        assert_eq!(roles(&*forked), vec!["user".to_string()]);
    }

    #[test]
    fn sqlite_get_reopens_a_fork() {
        let mut s = sqlite("main");
        s.append(text_msg("user", "hello")).unwrap();
        s.append(text_msg("assistant", "hi")).unwrap();

        let (id, _forked) = SessionManager::fork(&mut s).unwrap();
        let got = SessionManager::get(&s, &id).unwrap();
        assert_eq!(roles(&*got), vec!["user".to_string(), "assistant".to_string()]);
    }

    #[test]
    fn sqlite_get_unknown_errors() {
        let s = sqlite("main");
        let err = SessionManager::get(&s, "nope").err().expect("unknown fork should error");
        assert!(matches!(err, SessionError::Failed { .. }));
    }
}

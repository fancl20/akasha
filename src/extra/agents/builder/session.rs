//! Session fork/get management for the opinionated sub-agent framework the
//! [`AgentBuilder`] produces.
//!
//! [`SessionManager`] extends [`Session`]: a manager *is* its main session, so
//! the main conversation is driven directly through `append`/`messages`.
//! [`fork`](SessionManager::fork) and [`get`](SessionManager::get) return forks
//! as `Arc<Mutex<dyn Session>>` — drivable sessions the caller can hand
//! straight to an [`Agent`]. Forks are plain sessions, not managers; only the
//! root manager creates and reopens them. It has two implementations:
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
//! [`Agent`]: crate::core::agent::Agent
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
/// [`fork`](Self::fork) and [`get`](Self::get) return `Arc<Mutex<dyn Session>>`
/// — the forked branch as a drivable session, ready to hand to an
/// [`Agent`](crate::core::agent::Agent). Forks are plain sessions, not managers;
/// only the root manager creates and reopens them.
///
/// [`Session`]: crate::core::session::Session
pub trait SessionManager: Session {
    fn fork(&mut self) -> Result<(String, Arc<Mutex<dyn Session>>), SessionError>;
    fn get(&self, id: &str) -> Result<Arc<Mutex<dyn Session>>, SessionError>;
}

impl SessionManager for SqliteSession {
    fn fork(&mut self) -> Result<(String, Arc<Mutex<dyn Session>>), SessionError> {
        let id = Uuid::now_v7().to_string();
        Ok((id.clone(), SqliteSession::fork(self, &id)?.arc()))
    }

    fn get(&self, id: &str) -> Result<Arc<Mutex<dyn Session>>, SessionError> {
        Ok(SqliteSession::get(self, id)?.arc())
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
/// session registered under a UUID and returns it; [`get`](Self::get) returns
/// that shared session, so it reflects appends made through any other handle.
/// Forks are in-memory only; for persistence use [`SqliteSession`] directly.
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
    /// session and sharing `forks`/`fresh`.
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
    fn fork(&mut self) -> Result<(String, Arc<Mutex<dyn Session>>), SessionError> {
        let id = Uuid::now_v7().to_string();
        let backing: Shared = (self.fresh)();
        for message in &self.messages {
            backing.lock().map_err(|_| poison())?.append(message.clone())?;
        }
        self.forks.lock().map_err(|_| poison())?.insert(id.clone(), backing.clone());
        Ok((id, backing))
    }

    fn get(&self, id: &str) -> Result<Arc<Mutex<dyn Session>>, SessionError> {
        self.forks
            .lock()
            .map_err(|_| poison())?
            .get(id)
            .cloned()
            .ok_or_else(|| SessionError::Failed { message: format!("fork not found: {id}") })
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

    /// Roles of any session — works on a manager or a forked `Arc<Mutex<dyn
    /// Session>>` (via a lock), since `Session` exposes `messages` directly.
    fn roles(s: &dyn Session) -> Vec<String> {
        s.messages().map(|m| m.role.clone()).collect()
    }

    fn last_text(s: &dyn Session) -> Option<String> {
        s.messages().last().and_then(|m| {
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
    fn dyn_session_manager_coerces_to_dyn_session() {
        // Probe: can an Arc<Mutex<dyn SessionManager>> be handed to the agent as
        // an Arc<Mutex<dyn Session>>? (trait upcasting through Mutex). If this
        // compiles, the builder can hold the manager trait object and serve both
        // the agent and the subagent engine from one Arc.
        fn as_session(m: Arc<Mutex<dyn SessionManager>>) -> Arc<Mutex<dyn crate::core::session::Session>> {
            m
        }
        let mgr: Arc<Mutex<dyn SessionManager>> =
            Arc::new(Mutex::new(SessionAdapter::new(InMemorySession::new(), inmem_factory())));
        let session: Arc<Mutex<dyn crate::core::session::Session>> = as_session(mgr);
        session.lock().unwrap().append(text_msg("user", "ok")).unwrap();
    }

    #[test]
    fn adapter_fork_returns_a_session_with_messages() {
        let mut main = InMemorySession::new();
        main.append(text_msg("user", "hello")).unwrap();
        main.append(text_msg("assistant", "hi")).unwrap();

        let mut mgr = SessionAdapter::new(main, inmem_factory());
        let (_id, forked) = mgr.fork().unwrap();
        assert_eq!(roles(&*forked.lock().unwrap()), vec!["user".to_string(), "assistant".to_string()]);
    }

    #[test]
    fn adapter_fork_is_independent_from_main() {
        let mut main = InMemorySession::new();
        main.append(text_msg("user", "hello")).unwrap();

        let mut mgr = SessionAdapter::new(main, inmem_factory());
        let (_id, forked) = mgr.fork().unwrap();

        mgr.append(text_msg("user", "main-only")).unwrap();
        forked.lock().unwrap().append(text_msg("user", "fork-only")).unwrap();

        assert_eq!(roles(&mgr), vec!["user".to_string(), "user".to_string()]);
        assert_eq!(roles(&*forked.lock().unwrap()), vec!["user".to_string(), "user".to_string()]);
        assert_ne!(last_text(&mgr).as_deref(), Some("fork-only"));
        assert_ne!(last_text(&*forked.lock().unwrap()).as_deref(), Some("main-only"));
    }

    #[test]
    fn adapter_get_retrieves_a_fork() {
        let mut main = InMemorySession::new();
        main.append(text_msg("user", "hello")).unwrap();
        main.append(text_msg("assistant", "hi")).unwrap();

        let mut mgr = SessionAdapter::new(main, inmem_factory());
        let (id, _forked) = mgr.fork().unwrap();
        let got = mgr.get(&id).unwrap();
        assert_eq!(roles(&*got.lock().unwrap()), vec!["user".to_string(), "assistant".to_string()]);
    }

    #[test]
    fn adapter_fork_appends_are_live_across_gets() {
        let mut main = InMemorySession::new();
        main.append(text_msg("user", "hello")).unwrap();

        let mut mgr = SessionAdapter::new(main, inmem_factory());
        let (id, forked) = mgr.fork().unwrap();
        // Append through one handle; a later get() of the same fork observes it,
        // because both share the fork's backing session.
        forked.lock().unwrap().append(text_msg("assistant", "fork-only")).unwrap();

        let got = mgr.get(&id).unwrap();
        assert_eq!(roles(&*got.lock().unwrap()), vec!["user".to_string(), "assistant".to_string()]);
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
        assert_eq!(roles(&*forked.lock().unwrap()), vec!["user".to_string(), "assistant".to_string()]);
    }

    #[test]
    fn sqlite_fork_is_independent_from_main() {
        let mut s = sqlite("main");
        s.append(text_msg("user", "hello")).unwrap();

        let (_id, forked) = SessionManager::fork(&mut s).unwrap();
        s.append(text_msg("user", "main-only")).unwrap();

        assert_eq!(roles(&s), vec!["user".to_string(), "user".to_string()]);
        assert_eq!(roles(&*forked.lock().unwrap()), vec!["user".to_string()]);
    }

    #[test]
    fn sqlite_get_reopens_a_fork() {
        let mut s = sqlite("main");
        s.append(text_msg("user", "hello")).unwrap();
        s.append(text_msg("assistant", "hi")).unwrap();

        let (id, _forked) = SessionManager::fork(&mut s).unwrap();
        let got = SessionManager::get(&s, &id).unwrap();
        assert_eq!(roles(&*got.lock().unwrap()), vec!["user".to_string(), "assistant".to_string()]);
    }

    #[test]
    fn sqlite_get_unknown_errors() {
        let s = sqlite("main");
        let err = SessionManager::get(&s, "nope").err().expect("unknown fork should error");
        assert!(matches!(err, SessionError::Failed { .. }));
    }
}

use sha2::{Digest, Sha256};

use crate::core::types::Message;

#[derive(Debug, thiserror::Error)]
pub enum SessionError {
    #[error("session error: {message}")]
    Failed { message: String },
}

pub trait Session: Send + Sync {
    fn head_id(&self) -> String;
    fn context(&self) -> &Vec<Message>;
    fn append(&mut self, message: Message) -> Result<(), SessionError>;
    fn create(&self) -> Box<dyn Session>;
    fn fork(&self) -> Box<dyn Session>;
}

impl Clone for Box<dyn Session> {
    fn clone(&self) -> Self {
        self.fork()
    }
}

impl<T: Session + 'static> From<T> for Box<dyn Session> {
    fn from(session: T) -> Self {
        Box::new(session)
    }
}

#[derive(Clone)]
pub struct InMemorySession {
    hash: String,
    messages: Vec<Message>,
}

impl InMemorySession {
    pub fn new() -> Self {
        Self { hash: "null".to_string(), messages: Vec::new() }
    }
}

impl Session for InMemorySession {
    fn head_id(&self) -> String {
        self.hash.clone()
    }

    fn context(&self) -> &Vec<Message> {
        &self.messages
    }

    fn append(&mut self, message: Message) -> Result<(), SessionError> {
        let mut hasher = Sha256::new();
        hasher.update(self.hash.as_bytes());
        hasher.update(
            serde_json::to_string(&message).map_err(|e| SessionError::Failed { message: e.to_string() })?.as_bytes(),
        );

        self.hash = format!("{:x}", hasher.finalize());
        self.messages.push(message);
        Ok(())
    }

    fn create(&self) -> Box<dyn Session> {
        Self::new().into()
    }

    fn fork(&self) -> Box<dyn Session> {
        self.clone().into()
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

    #[test]
    fn empty_session() {
        let session = InMemorySession::new();
        assert_eq!(session.head_id(), "null");
        assert!(session.context().is_empty());
    }

    #[test]
    fn append_and_context() {
        let mut session = InMemorySession::new();
        session.append(text_msg("user", "hello")).unwrap();
        session.append(text_msg("assistant", "hi")).unwrap();
        session.append(text_msg("user", "how are you")).unwrap();

        let ctx = session.context();
        assert_eq!(ctx.len(), 3);
        assert_eq!(ctx[0].role, "user");
        assert_eq!(ctx[1].role, "assistant");
        assert_eq!(ctx[2].role, "user");
    }

    #[test]
    fn head_id_changes() {
        let mut session = InMemorySession::new();
        let id0 = session.head_id();
        session.append(text_msg("user", "a")).unwrap();
        let id1 = session.head_id();
        session.append(text_msg("user", "b")).unwrap();
        let id2 = session.head_id();

        assert_ne!(id0, id1);
        assert_ne!(id1, id2);
    }

    #[test]
    fn hash_is_deterministic() {
        let mut s1 = InMemorySession::new();
        let mut s2 = InMemorySession::new();

        s1.append(text_msg("user", "hello")).unwrap();
        s2.append(text_msg("user", "hello")).unwrap();

        assert_eq!(s1.head_id(), s2.head_id());

        s1.append(text_msg("assistant", "world")).unwrap();
        s2.append(text_msg("assistant", "world")).unwrap();

        assert_eq!(s1.head_id(), s2.head_id());
    }

    #[test]
    fn fork_independence() {
        let mut session = InMemorySession::new();
        session.append(text_msg("user", "shared")).unwrap();
        session.append(text_msg("assistant", "response")).unwrap();

        let mut forked = session.fork();
        session.append(text_msg("user", "branch A")).unwrap();
        forked.append(text_msg("user", "branch B")).unwrap();

        let ctx_a = session.context();
        let ctx_b = forked.context();

        assert_eq!(ctx_a.len(), 3);
        assert_eq!(ctx_b.len(), 3);
        assert_eq!(ctx_a[0].role, "user");
        assert_eq!(ctx_a[1].role, "assistant");
        assert_ne!(ctx_a[2].content, ctx_b[2].content);
        // Different messages → different hashes
        assert_ne!(session.head_id(), forked.head_id());
    }

    #[test]
    fn fork_shares_prefix() {
        let mut session = InMemorySession::new();
        session.append(text_msg("user", "1")).unwrap();
        session.append(text_msg("user", "2")).unwrap();

        let forked = session.fork();
        assert_eq!(session.context(), forked.context());
        assert_eq!(session.head_id(), forked.head_id());
    }
}

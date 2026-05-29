use std::sync::{Arc, Mutex};

use crate::core::types::Message;

#[derive(Debug, thiserror::Error)]
pub enum SessionError {
    #[error("session error: {message}")]
    Failed { message: String },
}

pub trait Session: Send + Sync {
    fn append(&mut self, message: Message) -> Result<(), SessionError>;
    fn messages(&self) -> &Vec<Message>;
    fn arc(self) -> Arc<Mutex<dyn Session>>
    where
        Self: Sized + 'static,
    {
        Arc::new(Mutex::new(self))
    }
}

pub struct InMemorySession {
    messages: Vec<Message>,
}

impl InMemorySession {
    pub fn new() -> Self {
        Self { messages: Vec::new() }
    }
}

impl Session for InMemorySession {
    fn messages(&self) -> &Vec<Message> {
        &self.messages
    }

    fn append(&mut self, message: Message) -> Result<(), SessionError> {
        self.messages.push(message);
        Ok(())
    }
}

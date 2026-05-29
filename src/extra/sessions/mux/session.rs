use serde_json::{Map, Value};
use uuid::Uuid;

use crate::core::session::{Session, SessionError};
use crate::core::types::{ContentBlock, Message, ToolCall};

pub struct MuxSession {
    active: Option<String>,
    session: Box<dyn Session>,
    messages: Vec<Message>,
    switching: bool,
}

impl MuxSession {
    pub fn new(session: Box<dyn Session>) -> Result<Self, SessionError> {
        let mut session = Self { active: None, session, messages: Vec::new(), switching: false };
        session.build_messages();
        if session.active.is_none() {
            session.switch(&Uuid::now_v7().to_string())?;
        }
        Ok(session)
    }

    fn switch(&mut self, id: &str) -> Result<(), SessionError> {
        self.active = Some(id.to_string());
        let mut args = Map::new();
        args.insert("next_id".to_string(), Value::String(id.to_string()));
        self.append(Message {
            role: "session/mux".to_string(),
            content: vec![ContentBlock::ToolCall(ToolCall {
                id: "".to_string(),
                name: "session/mux/switch".to_string(),
                arguments: Value::Object(args),
            })],
        })
    }

    fn build_messages(&mut self) {
        let all = self.session.messages();
        self.messages.clear();
        if let Some(msg) = all.get(1) {
            self.messages.push(msg.clone());
        } else {
            return;
        }

        let mut id: Option<&str> = None;
        for msg in all.iter() {
            if msg.role == "session/mux" {
                if let Some(ContentBlock::ToolCall(tc)) = msg.content.last() {
                    id = tc.arguments.get("next_id").and_then(|v| v.as_str());
                }
                continue;
            }

            if id == self.active.as_deref() {
                self.messages.push(msg.clone());
            } else if msg.role == "user" {
                *self.messages.last_mut().unwrap() = msg.clone();
            }
        }
    }
}

impl Session for MuxSession {
    fn messages(&self) -> &Vec<Message> {
        &self.messages
    }

    fn append(&mut self, mut message: Message) -> Result<(), SessionError> {
        if let Some(ContentBlock::ToolCall(ToolCall { id: _, name, arguments })) = message.content.last_mut() {
            match name.as_str() {
                "session/mux/route" => self.switching = true,
                "session/mux/switch" => {
                    message.role = "session/mux".to_string();
                    if arguments.get("next_id").is_none() {
                        arguments["next_id"] = serde_json::Value::String(Uuid::now_v7().to_string());
                    }
                    self.active = arguments["next_id"].as_str().map(|s| s.to_string());
                    self.switching = false;
                }
                _ => (),
            };
        }

        if self.switching {
            return Ok(());
        }
        self.session.append(message)?;
        self.build_messages();
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::core::session::InMemorySession;
    use crate::core::types::TextContent;

    fn text_msg(role: &str, text: &str) -> Message {
        Message { role: role.to_string(), content: vec![ContentBlock::Text(TextContent { content: text.to_string() })] }
    }

    fn switch_call(id: &str) -> Message {
        let mut args = Map::new();
        args.insert("next_id".to_string(), Value::String(id.to_string()));
        Message {
            role: "assistant".to_string(),
            content: vec![ContentBlock::ToolCall(ToolCall {
                id: "".to_string(),
                name: "session/mux/switch".to_string(),
                arguments: Value::Object(args),
            })],
        }
    }

    fn route_call() -> Message {
        Message {
            role: "assistant".to_string(),
            content: vec![ContentBlock::ToolCall(ToolCall {
                id: "".to_string(),
                name: "session/mux/route".to_string(),
                arguments: Value::Object(Map::new()),
            })],
        }
    }

    fn text_of(msg: &Message) -> Option<String> {
        msg.content.iter().find_map(|c| match c {
            ContentBlock::Text(t) => Some(t.content.clone()),
            _ => None,
        })
    }

    fn all_texts(session: &MuxSession) -> Vec<String> {
        session.messages().iter().filter_map(text_of).collect()
    }

    #[test]
    fn new_session_has_active_id_and_empty_messages() {
        let mux = MuxSession::new(Box::new(InMemorySession::new())).unwrap();
        assert!(mux.active.is_some());
        assert!(mux.messages().is_empty());
    }

    #[test]
    fn append_and_read_messages() {
        let mut mux = MuxSession::new(Box::new(InMemorySession::new())).unwrap();
        mux.append(text_msg("user", "hello")).unwrap();
        mux.append(text_msg("assistant", "hi")).unwrap();

        let texts = all_texts(&mux);
        assert!(texts.contains(&"hello".to_string()));
        assert!(texts.contains(&"hi".to_string()));
    }

    #[test]
    fn switch_changes_active_conversation() {
        let mut mux = MuxSession::new(Box::new(InMemorySession::new())).unwrap();
        let initial = mux.active.clone();

        mux.append(switch_call("conv-b")).unwrap();

        assert_eq!(mux.active.as_deref(), Some("conv-b"));
        assert_ne!(mux.active, initial);
    }

    #[test]
    fn messages_filtered_by_active_conversation() {
        let mut mux = MuxSession::new(Box::new(InMemorySession::new())).unwrap();

        mux.append(text_msg("user", "A1")).unwrap();
        mux.append(text_msg("assistant", "A-reply")).unwrap();
        mux.append(switch_call("conv-b")).unwrap();
        mux.append(text_msg("user", "B1")).unwrap();
        mux.append(text_msg("assistant", "B-reply")).unwrap();

        let texts = all_texts(&mux);
        assert!(texts.iter().any(|t| t == "B-reply"));
    }

    #[test]
    fn switch_back_restores_previous_messages() {
        let mut mux = MuxSession::new(Box::new(InMemorySession::new())).unwrap();
        let conv_a = mux.active.clone().unwrap();

        mux.append(text_msg("user", "A1")).unwrap();
        mux.append(text_msg("assistant", "A-reply")).unwrap();
        // Switch away and back without adding messages in the other conversation
        mux.append(switch_call("conv-b")).unwrap();
        mux.append(switch_call(&conv_a)).unwrap();

        let texts = all_texts(&mux);
        assert!(texts.iter().any(|t| t == "A-reply"));
    }

    #[test]
    fn inactive_user_message_replaces_last() {
        let mut mux = MuxSession::new(Box::new(InMemorySession::new())).unwrap();
        let conv_a = mux.active.clone().unwrap();

        mux.append(text_msg("user", "A1")).unwrap();
        mux.append(text_msg("assistant", "A-reply")).unwrap();
        mux.append(switch_call("conv-b")).unwrap();
        mux.append(text_msg("user", "B1")).unwrap();
        mux.append(switch_call(&conv_a)).unwrap();

        // When conv-b's user messages precede the switch back to conv-a,
        // they replace the last visible message (the assistant reply).
        let msgs = mux.messages();
        let last_user = msgs.iter().rev().find(|m| m.role == "user");
        assert!(last_user.is_some());
        assert_eq!(text_of(last_user.unwrap()).as_deref(), Some("B1"));
    }

    #[test]
    fn mux_control_messages_hidden() {
        let mut mux = MuxSession::new(Box::new(InMemorySession::new())).unwrap();

        mux.append(text_msg("user", "hello")).unwrap();
        mux.append(switch_call("conv-b")).unwrap();
        mux.append(text_msg("user", "world")).unwrap();

        assert!(!mux.messages().iter().any(|m| m.role == "session/mux"));
    }

    #[test]
    fn route_drops_all_messages_until_switch() {
        let mut mux = MuxSession::new(Box::new(InMemorySession::new())).unwrap();

        mux.append(text_msg("user", "before")).unwrap();
        mux.append(route_call()).unwrap();
        mux.append(text_msg("user", "dropped")).unwrap();
        mux.append(text_msg("assistant", "also dropped")).unwrap();

        let texts = all_texts(&mux);
        assert!(texts.iter().any(|t| t == "before"));
        assert!(!texts.iter().any(|t| t.contains("dropped")));
    }

    #[test]
    fn switch_after_route_restores_appending() {
        let mut mux = MuxSession::new(Box::new(InMemorySession::new())).unwrap();

        mux.append(route_call()).unwrap();
        mux.append(text_msg("user", "dropped")).unwrap();
        mux.append(switch_call("new-conv")).unwrap();
        mux.append(text_msg("user", "visible")).unwrap();

        let texts = all_texts(&mux);
        assert!(texts.iter().any(|t| t == "visible"));
        assert!(!texts.iter().any(|t| t.contains("dropped")));
    }

    #[test]
    fn switch_without_id_auto_generates_uuid() {
        let mut mux = MuxSession::new(Box::new(InMemorySession::new())).unwrap();
        let initial = mux.active.clone();

        mux.append(Message {
            role: "assistant".to_string(),
            content: vec![ContentBlock::ToolCall(ToolCall {
                id: "".to_string(),
                name: "session/mux/switch".to_string(),
                arguments: Value::Object(Map::new()),
            })],
        })
        .unwrap();

        assert!(mux.active.is_some());
        assert_ne!(mux.active, initial);
    }
}

use async_trait::async_trait;
use std::collections::HashSet;
use std::sync::Arc;
use teloxide::dispatching::{Dispatcher, HandlerExt, UpdateFilterExt};
use teloxide::prelude::Requester;
use teloxide::types::{ChatId, Message as TgMessage, Update};
use teloxide::utils::command::BotCommands;
use teloxide::{Bot, RequestError, dptree};

use crate::core::agent;
use crate::core::extensions::{Extension, ExtensionError};
use crate::core::providers::{Model, Registry, StreamResponse};
use crate::core::tools::ToolRegistry;
use crate::core::types::{ContentBlock, Message, Request};

#[derive(BotCommands, Clone)]
#[command(rename_rule = "lowercase")]
enum Command {
    Start,
}

/// Extension that forwards completed agent responses to a Telegram chat.
///
/// Created per-message with the originating `chat_id`, so it always
/// replies to the correct conversation.
pub struct TelegramExtension {
    bot: Bot,
    chat_id: ChatId,
}

impl TelegramExtension {
    pub fn new(bot: Bot, chat_id: ChatId) -> Self {
        Self { bot, chat_id }
    }
}

#[async_trait]
impl Extension for TelegramExtension {
    fn name(&self) -> &str {
        "telegram"
    }

    /// Called by the agent loop once a full (non-streaming) response is
    /// assembled.  Extracts text blocks and sends them back to the chat.
    async fn on_response(&self, resp: &StreamResponse) -> Result<(), ExtensionError> {
        let text: String = resp
            .message
            .content
            .iter()
            .filter_map(|block| match block {
                ContentBlock::Text { content } => Some(content.as_str()),
                _ => None,
            })
            .collect();

        if !text.is_empty() {
            self.bot
                .send_message(self.chat_id, &text)
                .await
                .map_err(|e| ExtensionError::ExtensionFailed {
                    name: self.name().to_string(),
                    message: e.to_string(),
                })?;
        }

        Ok(())
    }
}

async fn handle_start(bot: Bot, msg: TgMessage) -> Result<(), RequestError> {
    bot.send_message(
        msg.chat.id,
        "Hello! I'm Akasha, your AI assistant. How can I help you?",
    )
    .await?;
    Ok(())
}

async fn handle_message(
    bot: Bot,
    msg: TgMessage,
    model: Arc<Model>,
    models: Arc<Registry>,
    tools: Arc<ToolRegistry>,
    allowed_users: Arc<HashSet<u64>>,
) -> Result<(), RequestError> {
    let chat_id = msg.chat.id;

    let user_id = match msg.from {
        Some(ref user) => user.id.0,
        None => return Ok(()),
    };
    if !allowed_users.is_empty() && !allowed_users.contains(&user_id) {
        let _ = bot
            .send_message(chat_id, "You are not authorized to use this bot.")
            .await;
        return Ok(());
    }

    let text = match msg.text() {
        Some(t) => t,
        None => return Ok(()),
    };

    let ext: Box<dyn Extension> = Box::new(TelegramExtension::new(bot.clone(), chat_id));

    let request = Request {
        messages: vec![Message {
            role: "user".into(),
            content: vec![ContentBlock::Text {
                content: text.to_owned(),
            }],
        }],
        tools: tools.definitions(),
    };

    if let Err(e) = agent::run(&request, &model, &*models, &*tools, &ext).await {
        let _ = bot.send_message(chat_id, format!("agent error: {e}")).await;
    }

    Ok(())
}

/// Start the Telegram bot: long-poll for messages, run the agent for each
/// one, and forward responses back via [`TelegramExtension::on_response`].
pub async fn run(
    token: impl Into<String>,
    model: Model,
    models: Registry,
    tools: ToolRegistry,
    allowed_users: HashSet<u64>,
) {
    let bot = Bot::new(token.into());
    let model = Arc::new(model);
    let models = Arc::new(models);
    let tools = Arc::new(tools);
    let allowed_users = Arc::new(allowed_users);

    let handler = dptree::entry()
        .branch(
            Update::filter_message()
                .filter_command::<Command>()
                .endpoint(handle_start),
        )
        .branch(Update::filter_message().endpoint(handle_message));

    Dispatcher::builder(bot, handler)
        .dependencies(dptree::deps![model, models, tools, allowed_users])
        .build()
        .dispatch()
        .await;
}

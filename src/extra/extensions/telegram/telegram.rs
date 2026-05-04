use async_trait::async_trait;
use std::collections::HashSet;
use std::sync::Arc;
use std::time::{Duration, SystemTime};
use teloxide::dispatching::dialogue::{Dialogue, InMemStorage};
use teloxide::dispatching::{Dispatcher, HandlerExt, UpdateFilterExt, dialogue};
use teloxide::prelude::Requester;
use teloxide::types::{ChatAction, ChatId, Message as TgMessage, Update};
use teloxide::utils::command::BotCommands;
use teloxide::{ApiError, Bot, RequestError, dptree};
use tokio::sync::{mpsc, oneshot};

use crate::core::agent;
use crate::core::extensions::{Extension, ExtensionError};
use crate::core::providers::{Model, Registry, StreamResponse};
use crate::core::tools::ToolRegistry;
use crate::core::types::{ContentBlock, Message, Request};

enum StreamEvent {
    Append(ContentBlock),
    Finish(oneshot::Sender<Result<(), ExtensionError>>),
}

/// Extension that streams agent responses into Telegram messages via
/// edit-as-you-go, splitting at line boundaries when the 4096-char
/// limit is reached.
///
/// Chunk text is sent through an unbounded channel to a background task
/// so that `on_response_chunk` never blocks on Telegram API calls.
pub struct TelegramExtension {
    tx: mpsc::UnboundedSender<StreamEvent>,
}

impl Clone for TelegramExtension {
    fn clone(&self) -> Self {
        Self {
            tx: self.tx.clone(),
        }
    }
}

impl TelegramExtension {
    pub fn new(bot: Bot, chat_id: ChatId) -> Self {
        let (tx, rx) = mpsc::unbounded_channel();
        tokio::spawn(edit_loop(bot, chat_id, rx));
        Self { tx }
    }
}

/// Background task: receives chunks from the channel, batches them,
/// and flushes to Telegram with throttling.
async fn edit_loop(bot: Bot, chat_id: ChatId, mut rx: mpsc::UnboundedReceiver<StreamEvent>) {
    let mut pending = String::new();
    let mut message_id = None;
    let mut wait_until = SystemTime::UNIX_EPOCH;

    loop {
        let event = match rx.try_recv() {
            Ok(msg) => Some(msg),
            Err(mpsc::error::TryRecvError::Empty) => rx.recv().await,
            Err(mpsc::error::TryRecvError::Disconnected) => None,
        };

        let mut finish_tx = None;
        match event {
            Some(StreamEvent::Append(ContentBlock::Text { content })) => pending.push_str(&content),
            Some(StreamEvent::Append(ContentBlock::Reasoning { .. })) => (), // Allow reasoning event to trigger typing action.
            Some(StreamEvent::Append(..)) => continue,
            Some(StreamEvent::Finish(done)) => {
                finish_tx = Some(done);
                pending.push('\n');
                wait_until = SystemTime::UNIX_EPOCH;
            }
            None if pending.trim().is_empty() => return,
            _ => wait_until = SystemTime::UNIX_EPOCH,
        }

        if SystemTime::now() < wait_until {
            continue;
        }

        while !pending.trim().is_empty() {
            let end = pending
                .char_indices()
                .nth(4095)
                .map(|(i, c)| i + c.len_utf8());
            let boundary = pending[..end.unwrap_or(pending.len())]
                .rfind("\n")
                .map(|i| i + 1)
                .or(end)
                .unwrap_or(pending.len());

            match (message_id, pending[..boundary].trim()) {
                (_, "") => (),
                (Some(id), sending) => match bot.edit_message_text(chat_id, id, sending).await {
                    Ok(_) | Err(RequestError::Api(ApiError::MessageNotModified)) => (),
                    Err(e) => eprintln!("telegram edit error: {e}"),
                },
                (None, sending) => match bot.send_message(chat_id, sending).await {
                    Ok(msg) => message_id = Some(msg.id),
                    Err(e) => eprintln!("telegram send error: {e}"),
                },
            }
            wait_until = SystemTime::now() + Duration::from_secs(4);

            if end.is_some() {
                message_id = None;
                pending = pending[boundary..].to_owned();
            }

            if finish_tx.is_none() || end.is_none() {
                break;
            }
        }

        // If no message sent because pending is empty, set the chat action to typing.
        if SystemTime::now() >= wait_until {
            let _ = bot.send_chat_action(chat_id, ChatAction::Typing).await;
            wait_until = SystemTime::now() + Duration::from_secs(4);
        }

        // Prepare the next turn.
        if let Some(tx) = finish_tx {
            message_id = None;
            pending.clear();
            let _ = tx.send(Ok(()));
        }
    }
}

#[async_trait]
impl Extension for TelegramExtension {
    fn name(&self) -> &str {
        "telegram"
    }

    async fn on_response_chunk(&self, chunk: &StreamResponse) -> Result<(), ExtensionError> {
        for block in &chunk.message.content {
            self.tx
                .send(StreamEvent::Append(block.clone()))
                .map_err(|_| ExtensionError::ExtensionFailed {
                    name: "telegram".to_string(),
                    message: "editor task dropped".to_string(),
                })?;
        }

        Ok(())
    }

    async fn on_response(&self, _resp: &StreamResponse) -> Result<(), ExtensionError> {
        let (done_tx, done_rx) = oneshot::channel();
        self.tx.send(StreamEvent::Finish(done_tx)).map_err(|_| {
            ExtensionError::ExtensionFailed {
                name: "telegram".to_string(),
                message: "editor task dropped".to_string(),
            }
        })?;
        done_rx.await.map_err(|_| ExtensionError::ExtensionFailed {
            name: "telegram".to_string(),
            message: "editor task crashed".to_string(),
        })?
    }
}

#[derive(Clone, Default)]
struct ChatState {
    messages: Vec<Message>,
    extension: Option<TelegramExtension>,
}

type AgentDialogue = Dialogue<ChatState, InMemStorage<ChatState>>;

#[derive(BotCommands, Clone)]
#[command(rename_rule = "lowercase", description = "Bot commands")]
enum Command {
    #[command(description = "clear conversation history")]
    Clear,
    #[command(description = "show available commands")]
    Help,
}

type HandlerResult = Result<(), Box<dyn std::error::Error + Send + Sync>>;

async fn command_handler(
    bot: Bot,
    dialogue: AgentDialogue,
    msg: TgMessage,
    cmd: Command,
) -> HandlerResult {
    match cmd {
        Command::Clear => {
            dialogue.exit().await?;
            bot.send_message(msg.chat.id, "Conversation history cleared.")
                .await?;
        }
        Command::Help => {
            bot.send_message(msg.chat.id, Command::descriptions().to_string())
                .await?;
        }
    }
    Ok(())
}

async fn handle_message(
    bot: Bot,
    dialogue: AgentDialogue,
    msg: TgMessage,
    model: Arc<Model>,
    models: Arc<Registry>,
    tools: Arc<ToolRegistry>,
) -> HandlerResult {
    let chat_id = msg.chat.id;

    let text = match msg.text() {
        Some(t) => t,
        None => return Ok(()),
    };

    let mut state = dialogue.get_or_default().await?;

    state.messages.push(Message {
        role: "user".into(),
        content: vec![ContentBlock::Text {
            content: text.to_owned(),
        }],
    });

    let request = Request {
        messages: state.messages.clone(),
        tools: tools.definitions(),
    };

    let ext: Box<dyn Extension> = Box::new(
        state
            .extension
            .get_or_insert_with(|| TelegramExtension::new(bot.clone(), chat_id))
            .clone(),
    );

    match agent::run(&request, &model, &*models, &*tools, &ext).await {
        Ok(new_messages) => {
            state.messages.extend(new_messages);
            dialogue.update(state).await?;
        }
        Err(e) => {
            let _ = bot.send_message(chat_id, format!("agent error: {e}")).await;
        }
    }

    Ok(())
}

pub async fn run(
    token: impl Into<String>,
    model: Model,
    models: Registry,
    tools: ToolRegistry,
    allowed_ids: HashSet<u64>,
) {
    let bot = Bot::new(token.into());
    let model = Arc::new(model);
    let models = Arc::new(models);
    let tools = Arc::new(tools);
    let allowed_ids = Arc::new(allowed_ids);

    let handler = dialogue::enter::<Update, InMemStorage<ChatState>, ChatState, _>().branch(
        Update::filter_message()
            .filter(|msg: TgMessage, allowed_ids: Arc<HashSet<u64>>| {
                msg.from
                    .map(|user| allowed_ids.contains(&user.id.0))
                    .unwrap_or_default()
            })
            .branch(
                dptree::entry()
                    .filter_command::<Command>()
                    .endpoint(command_handler),
            )
            .branch(dptree::entry().endpoint(handle_message)),
    );

    Dispatcher::builder(bot, handler)
        .dependencies(dptree::deps![
            InMemStorage::<ChatState>::new(),
            model,
            models,
            tools,
            allowed_ids
        ])
        .build()
        .dispatch()
        .await;
}

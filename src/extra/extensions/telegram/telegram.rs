use async_trait::async_trait;
use std::collections::HashSet;
use std::sync::{Arc, Mutex};
use std::time::{Duration, SystemTime};
use teloxide::dispatching::dialogue::{Dialogue, InMemStorage};
use teloxide::dispatching::{Dispatcher, UpdateFilterExt, UpdateHandler, dialogue};
use teloxide::prelude::Requester;
use teloxide::sugar::request::RequestLinkPreviewExt;
use teloxide::types::{ChatAction, ChatId, Message as TgMessage, MessageId, Update};
use teloxide::utils::command::BotCommands;
use teloxide::{ApiError, Bot, RequestError, dptree};
use tokio::sync::{mpsc, oneshot};
use tokio::task::JoinSet;

use crate::core::agent::{Agent, AgentState};
use crate::core::extensions::{Extension, ExtensionError, ToolCallDecision};
use crate::core::providers::StreamResponse;
use crate::core::types::{ContentBlock, Message, TextContent};
use crate::extra::extensions::combinator::And;

enum StreamEvent {
    Append(ContentBlock),
    Notification(String),
    Finish(oneshot::Sender<Result<(), ExtensionError>>),
}

pub struct TelegramExtension {
    tx: mpsc::UnboundedSender<StreamEvent>,
    rx: mpsc::UnboundedReceiver<Message>,
}

#[async_trait]
impl Extension for TelegramExtension {
    fn name(&self) -> &str {
        "telegram"
    }

    async fn on_message_update(&mut self, chunk: &StreamResponse) -> Result<(), ExtensionError> {
        for block in &chunk.message.content {
            self.tx.send(StreamEvent::Append(block.clone())).map_err(|_| {
                ExtensionError::ExtensionFailed {
                    name: "telegram".to_string(),
                    message: "updater task dropped".to_string(),
                }
            })?;
        }
        Ok(())
    }

    async fn on_tool_execution_start(
        &mut self,
        _tool_call_id: &str,
        name: &str,
        args: &serde_json::Value,
    ) -> Result<ToolCallDecision, ExtensionError> {
        let notification = format!("Tool: {}: {}", name, args);
        self.tx.send(StreamEvent::Notification(notification)).map_err(|_| {
            ExtensionError::ExtensionFailed {
                name: "telegram".to_string(),
                message: "updater task dropped".to_string(),
            }
        })?;
        Ok(ToolCallDecision::Allow)
    }

    async fn on_turn_end(&mut self, mut state: AgentState) -> Result<AgentState, ExtensionError> {
        let (tx, rx) = oneshot::channel();
        self.tx.send(StreamEvent::Finish(tx)).map_err(|_| ExtensionError::ExtensionFailed {
            name: "telegram".to_string(),
            message: "updater task dropped".to_string(),
        })?;
        let _ = rx.await.map_err(|_| ExtensionError::ExtensionFailed {
            name: "telegram".to_string(),
            message: "updater task dropped".to_string(),
        })?;

        state.messages.push(self.rx.recv().await.ok_or(ExtensionError::ExtensionFailed {
            name: "telegram".to_string(),
            message: "input channel dropped".to_string(),
        })?);
        Ok(state)
    }
}

/// Background task: receives chunks from the channel, batches them,
/// and flushes to Telegram with throttling.
async fn update_chat(bot: Bot, chat_id: ChatId, mut rx: mpsc::UnboundedReceiver<StreamEvent>) {
    let mut pending = String::new();
    let mut message_id = None;
    let mut wait_until = SystemTime::UNIX_EPOCH;

    let update_message = async |id: Option<MessageId>, sending: &str| match (id, sending) {
        (_, "") => id,
        (Some(id), sending) => {
            match bot.edit_message_text(chat_id, id, sending).disable_link_preview(true).await {
                Ok(_) | Err(RequestError::Api(ApiError::MessageNotModified)) => (),
                Err(e) => eprintln!("telegram edit error: {e}"),
            };
            Some(id)
        }
        (None, sending) => {
            match bot.send_message(chat_id, sending).disable_link_preview(true).await {
                Ok(msg) => Some(msg.id),
                Err(e) => {
                    eprintln!("telegram send error: {e}");
                    None
                }
            }
        }
    };

    loop {
        let event = match rx.try_recv() {
            Ok(msg) => Some(msg),
            Err(mpsc::error::TryRecvError::Empty) => rx.recv().await,
            Err(mpsc::error::TryRecvError::Disconnected) => None,
        };

        let mut finish_tx = None;
        match event {
            Some(StreamEvent::Append(ContentBlock::Text(t))) => pending.push_str(&t.content),
            Some(StreamEvent::Append(ContentBlock::Reasoning { .. })) => (), // Allow reasoning event to trigger typing action.
            Some(StreamEvent::Append(..)) => continue,
            Some(StreamEvent::Notification(t)) if pending.is_empty() => {
                message_id = update_message(message_id, &t[..t.ceil_char_boundary(512)]).await
            }
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
            let end = pending.char_indices().nth(4095).map(|(i, c)| i + c.len_utf8());
            let boundary = pending[..end.unwrap_or(pending.len())]
                .rfind("\n")
                .map(|i| i + 1)
                .or(end)
                .unwrap_or(pending.len());

            message_id = update_message(message_id, pending[..boundary].trim()).await;
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

#[derive(Clone, Default)]
enum State {
    #[default]
    Idle,
    Running {
        tasks: Arc<Mutex<JoinSet<()>>>,
        tx: mpsc::UnboundedSender<Message>,
    },
}

type AgentDialogue = Dialogue<State, InMemStorage<State>>;
type HandlerResult = Result<(), Box<dyn std::error::Error + Send + Sync>>;

#[derive(BotCommands, Clone)]
#[command(rename_rule = "lowercase", description = "Bot commands")]
enum Command {
    #[command(description = "reset agent status")]
    Reset,
    #[command(description = "show available commands")]
    Help,
}

async fn command_handler(
    bot: Bot,
    dialogue: AgentDialogue,
    msg: TgMessage,
    cmd: Command,
) -> HandlerResult {
    match cmd {
        Command::Reset => {
            if let Ok(State::Running { tasks, .. }) = dialogue.get_or_default().await {
                tasks.lock().expect("tasks lock poisoned").abort_all();
            }
            dialogue.exit().await?;
            bot.send_message(msg.chat.id, "agent aborted and reset.").await?;
        }
        Command::Help => {
            bot.send_message(msg.chat.id, Command::descriptions().to_string()).await?;
        }
    }
    Ok(())
}

async fn handle_message(
    bot: Bot,
    dialogue: AgentDialogue,
    msg: TgMessage,
    agent_factory: Arc<dyn Fn() -> Agent + Send + Sync + 'static>,
) -> HandlerResult {
    let prompt = if let Some(text) = msg.text() {
        Message {
            role: "user".into(),
            content: vec![ContentBlock::Text(TextContent { content: text.to_owned() })],
        }
    } else {
        return Ok(());
    };

    match dialogue.get_or_default().await? {
        State::Idle => {
            let (event_tx, event_rx) = mpsc::unbounded_channel();
            let (msg_tx, msg_rx) = mpsc::unbounded_channel();

            let mut agent = agent_factory();
            agent.extension =
                And::new(agent.extension, TelegramExtension { tx: event_tx, rx: msg_rx }).into();

            let mut tasks = JoinSet::new();
            tasks.spawn(async move {
                if let Err(e) = agent.prompt(prompt).await {
                    eprintln!("agent prompt error: {}", e);
                }
            });
            tasks.spawn(update_chat(bot, msg.chat.id, event_rx));

            dialogue
                .update(State::Running { tasks: Arc::new(Mutex::new(tasks)), tx: msg_tx })
                .await?;
        }
        State::Running { tasks: _, tx } => {
            tx.send(prompt)?;
        }
    }

    Ok(())
}

fn schema(
    ids: Arc<HashSet<u64>>,
) -> UpdateHandler<Box<dyn std::error::Error + Send + Sync + 'static>> {
    let allowed_filter =
        move |msg: TgMessage| msg.from.map(|user| ids.contains(&user.id.0)).unwrap_or_default();

    let command_handler = teloxide::filter_command::<Command, _>()
        .branch(dptree::case![Command::Help].endpoint(command_handler))
        .branch(dptree::case![Command::Reset].endpoint(command_handler));

    let message_handler = Update::filter_message()
        .filter(allowed_filter)
        .branch(command_handler)
        .branch(dptree::entry().endpoint(handle_message));

    dialogue::enter::<Update, InMemStorage<State>, State, _>().branch(message_handler)
}

pub async fn dispatch(
    token: impl Into<String>,
    allowed_ids: HashSet<u64>,
    agent_factory: Arc<dyn Fn() -> Agent + Send + Sync + 'static>,
) {
    Dispatcher::builder(Bot::new(token.into()), schema(Arc::new(allowed_ids)))
        .dependencies(dptree::deps![InMemStorage::<State>::new(), agent_factory])
        .build()
        .dispatch()
        .await;
}

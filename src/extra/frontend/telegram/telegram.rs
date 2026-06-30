use std::collections::HashSet;
use std::sync::Arc;
use std::time::{Duration, SystemTime};
use teloxide::dispatching::dialogue::{Dialogue, InMemStorage};
use teloxide::dispatching::{Dispatcher, UpdateFilterExt, UpdateHandler, dialogue};
use teloxide::prelude::Requester;
use teloxide::sugar::request::RequestLinkPreviewExt;
use teloxide::types::{ChatAction, ChatId, Message as TgMessage, MessageId, Update, User};
use teloxide::utils::command::BotCommands;
use teloxide::{ApiError, Bot, RequestError, dptree};
use tokio::sync::{Mutex, mpsc};
use tokio::task::JoinSet;

use crate::core::agent::Agent;
use crate::core::types::{ContentBlock, Message, TextContent};
use crate::extra::extensions::io::{IOExtension, OutputEvent};

/// Background task: receives events from the channel, batches them,
/// and flushes to Telegram with throttling.
async fn update_chat(bot: Bot, chat_id: ChatId, mut rx: mpsc::UnboundedReceiver<OutputEvent>) {
    let mut pending = String::new();
    let mut message_id = None;
    let mut wait_until = SystemTime::UNIX_EPOCH;

    let update_message = async |id: Option<MessageId>, sending: &str| match (id, sending) {
        (_, "") => id,
        (Some(id), sending) => {
            match bot.edit_message_text(chat_id, id, sending).disable_link_preview(true).await {
                Ok(_) | Err(RequestError::Api(ApiError::MessageNotModified)) => {}
                Err(e) => eprintln!("telegram edit error: {e}"),
            };
            Some(id)
        }
        (None, sending) => match bot.send_message(chat_id, sending).disable_link_preview(true).await {
            Ok(msg) => Some(msg.id),
            Err(e) => {
                eprintln!("telegram send error: {e}");
                None
            }
        },
    };

    loop {
        let event = match rx.try_recv() {
            Ok(msg) => Some(msg),
            Err(mpsc::error::TryRecvError::Empty) => rx.recv().await,
            Err(mpsc::error::TryRecvError::Disconnected) => None,
        };

        let mut finish_tx = None;
        match event {
            Some(OutputEvent::Append(ContentBlock::Text(t))) => pending.push_str(&t.content),
            Some(OutputEvent::Append(ContentBlock::Reasoning { .. })) => {} // Allow reasoning event to trigger typing action.
            Some(OutputEvent::Append(..)) => continue,
            Some(OutputEvent::Notification(t)) if pending.is_empty() => {
                message_id = update_message(message_id, &t[..t.ceil_char_boundary(512)]).await
            }
            Some(OutputEvent::Finish(done)) => {
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
            let boundary =
                pending[..end.unwrap_or(pending.len())].rfind("\n").map(|i| i + 1).or(end).unwrap_or(pending.len());

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
type AgentFactory = Arc<dyn Fn(Option<User>) -> anyhow::Result<Agent> + Send + Sync + 'static>;

#[derive(BotCommands, Clone)]
#[command(rename_rule = "lowercase", description = "Bot commands")]
enum Command {
    #[command(description = "abort and reset agent")]
    Reset,
    #[command(description = "show available commands")]
    Help,
    #[command(description = "start talk with the agent")]
    Start,
    #[command(description = "switch session for the prompt that follows")]
    Switch(String),
}

async fn command_handler(
    bot: Bot,
    dialogue: AgentDialogue,
    msg: TgMessage,
    cmd: Command,
    factory: AgentFactory,
) -> HandlerResult {
    match cmd {
        Command::Start => {}
        Command::Reset => {
            if let Ok(State::Running { tasks, .. }) = dialogue.get_or_default().await {
                tasks.lock().await.abort_all();
            }
            dialogue.exit().await?;
            bot.send_message(msg.chat.id, "agent aborted and reset.").await?;
        }
        Command::Help => {
            bot.send_message(msg.chat.id, Command::descriptions().to_string()).await?;
        }
        Command::Switch(text) => {
            let prompt = Message {
                role: "user".into(),
                content: vec![ContentBlock::Text(TextContent { content: format!("switch session: {text}") })],
            };
            handle_prompt(bot, dialogue, msg, prompt, factory).await?;
        }
    }
    Ok(())
}

async fn handle_message(bot: Bot, dialogue: AgentDialogue, msg: TgMessage, factory: AgentFactory) -> HandlerResult {
    let text = match msg.text() {
        Some(text) => text.to_owned(),
        None => return Ok(()),
    };
    let prompt = Message { role: "user".into(), content: vec![ContentBlock::Text(TextContent { content: text })] };
    handle_prompt(bot, dialogue, msg, prompt, factory).await
}

/// Drives the agent dialogue with `prompt`, starting a new run when idle or
/// queuing the prompt onto an in-flight run.
async fn handle_prompt(
    bot: Bot,
    dialogue: AgentDialogue,
    msg: TgMessage,
    prompt: Message,
    factory: AgentFactory,
) -> HandlerResult {
    match dialogue.get_or_default().await? {
        State::Idle => {
            let agent = factory(msg.from)?;
            let (mut agent, tx, rx) = IOExtension::bind(agent);
            let mut tasks = JoinSet::new();
            tasks.spawn(async move {
                if let Err(e) = agent.prompt(prompt).await {
                    eprintln!("agent prompt error: {e}");
                }
            });
            tasks.spawn(update_chat(bot, msg.chat.id, rx));

            dialogue.update(State::Running { tasks: Arc::new(Mutex::new(tasks)), tx: tx }).await?;
        }
        State::Running { tx, .. } => {
            tx.send(prompt)?;
        }
    }

    Ok(())
}

fn schema(ids: Arc<HashSet<u64>>) -> UpdateHandler<Box<dyn std::error::Error + Send + Sync + 'static>> {
    let allowed_filter = move |msg: TgMessage| msg.from.map(|user| ids.contains(&user.id.0)).unwrap_or_default();

    let command_handler = teloxide::filter_command::<Command, _>()
        .branch(dptree::case![Command::Start].endpoint(command_handler))
        .branch(dptree::case![Command::Help].endpoint(command_handler))
        .branch(dptree::case![Command::Reset].endpoint(command_handler))
        .branch(dptree::filter(|x| matches!(x, Command::Switch(..))).endpoint(command_handler));

    let message_handler = Update::filter_message()
        .filter(allowed_filter)
        .branch(command_handler)
        .branch(dptree::entry().endpoint(handle_message));

    dialogue::enter::<Update, InMemStorage<State>, State, _>().branch(message_handler)
}

pub async fn dispatch(
    token: impl Into<String>,
    allowed_ids: HashSet<u64>,
    factory: AgentFactory,
) -> Result<(), RequestError> {
    let bot = Bot::new(token.into());
    bot.set_my_commands(Command::bot_commands()).await?;
    Dispatcher::builder(bot, schema(Arc::new(allowed_ids)))
        .dependencies(dptree::deps![InMemStorage::<State>::new(), factory])
        .build()
        .dispatch()
        .await;
    Ok(())
}

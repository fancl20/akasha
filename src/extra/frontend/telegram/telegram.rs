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

use crate::core::types::{ContentBlock, Message, TextContent};
use crate::extra::frontend::mux::Mux;
use crate::extra::frontend::viewmodel::{Block, Step, Turn};

/// A Telegram-flavored flattening of a turn's blocks into one message: assistant
/// text verbatim, tool calls as `name: args` (rendered when the call is triggered,
/// matching the prior tool-notification). Reasoning is skipped.
fn render_turn(turn: &Turn) -> String {
    turn.blocks
        .last()
        .map(|block| match block {
            Block::Text(t) => t.content.clone(),
            Block::Reasoning(_) => "".to_string(),
            Block::ToolCall(tc) => {
                let note = format!("{}: {}", tc.call.name, tc.call.arguments);
                note[..note.ceil_char_boundary(512)].to_string()
            }
        })
        .unwrap_or_default()
}

/// Edit `id` in place when `Some`, else send a new message. Empty `sending` is a
/// no-op. Returns the message id to keep editing (which may be `None` on a send
/// failure, so the next call retries as a fresh send).
async fn update_message(bot: &Bot, chat_id: ChatId, id: Option<MessageId>, sending: &str) -> Option<MessageId> {
    match (id, sending) {
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
    }
}

/// Background task: steps the view model and renders the live turn to Telegram
/// with throttled edits, flushing and acking each finished turn.
///
/// `sealed` is the byte offset into the rendered turn already committed to prior
/// messages; the live message edits only the tail past it. When that tail would
/// exceed Telegram's 4095-char cap it is sealed (the offset advances and
/// `message_id` resets) and the remainder continues in a fresh message — one
/// chunk per throttle window while streaming, the whole tail on finish. The
/// offset resets each turn.
async fn update_chat(bot: Bot, chat_id: ChatId, mut mux: Mux) {
    let mut message_id = None;
    let mut sealed = 0usize;
    let mut wait_until = SystemTime::UNIX_EPOCH;

    loop {
        let (id, finish) = match mux.step().await {
            None => return,
            Some(Step::TurnFinished { id }) => (id, true),
            Some(Step::Updated { id }) if SystemTime::now() >= wait_until => (id, false),
            Some(Step::AgentEnded { .. }) => return,
            Some(_) => continue,
        };

        let text = mux.transcript(id).and_then(|t| t.turns.last().map(render_turn)).unwrap_or_default();
        while !text[sealed..].trim().is_empty() {
            let rest = &text[sealed..];
            let end = rest.char_indices().nth(4095).map(|(i, c)| i + c.len_utf8());
            let boundary = rest[..end.unwrap_or(rest.len())].rfind('\n').map(|i| i + 1).or(end).unwrap_or(rest.len());
            message_id = update_message(&bot, chat_id, message_id, rest[..boundary].trim()).await;
            wait_until = SystemTime::now() + Duration::from_secs(4);
            if end.is_some() {
                message_id = None;
                sealed += boundary;
            }
            if !finish || end.is_none() {
                break;
            }
        }

        // If no message sent because pending is empty, set the chat action to typing.
        if SystemTime::now() >= wait_until {
            let _ = bot.send_chat_action(chat_id, ChatAction::Typing).await;
            wait_until = SystemTime::now() + Duration::from_secs(4);
        }

        if finish {
            message_id = None;
            sealed = 0;
            mux.ack(id, Ok(()));
        }
    }
}

#[derive(Clone, Default)]
enum State {
    #[default]
    Idle,
    Running {
        tasks: Arc<Mutex<JoinSet<()>>>,
        input: mpsc::UnboundedSender<Message>,
    },
}

type AgentDialogue = Dialogue<State, InMemStorage<State>>;
type HandlerResult = Result<(), Box<dyn std::error::Error + Send + Sync>>;
type AgentFactory = Arc<dyn Fn(Option<User>) -> anyhow::Result<Mux> + Send + Sync + 'static>;

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
            let mux = factory(msg.from)?;
            let input = mux.input();
            input.send(prompt)?;

            let mut tasks = JoinSet::new();
            tasks.spawn(update_chat(bot, msg.chat.id, mux));
            dialogue.update(State::Running { tasks: Arc::new(Mutex::new(tasks)), input }).await?;
        }
        State::Running { input, .. } => {
            input.send(prompt)?;
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

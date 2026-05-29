use std::collections::{HashMap, HashSet};
use std::path::PathBuf;
use std::sync::Arc;

use akasha::core::agent::{Agent, AgentState};
use akasha::core::extensions::NoopExtension;
use akasha::core::providers::{Model, Registry};
use akasha::core::session::{InMemorySession, Session};
use akasha::core::tools::ToolRegistry;
use akasha::core::types::{ContentBlock, Message, TextContent};
use akasha::extra::extensions::telegram;
use akasha::extra::providers::deepseek::DeepSeekProvider;
use akasha::extra::tools::mcp;
use clap::Parser;

#[derive(Parser)]
#[command(name = "akasha", version, about = "a paw from cat")]
struct Cli {
    #[arg(long, env = "TELEGRAM_BOT_TOKEN")]
    telegram_token: String,
    #[arg(long)]
    provider: Option<String>,
    #[arg(long)]
    api_key: Option<String>,
    #[arg(long)]
    base_url: Option<String>,
    #[arg(long, value_delimiter = ',')]
    allowed_ids: Vec<u64>,
    #[arg(long)]
    mcps: Option<String>,

    #[arg(long, env = "DEEPSEEK_API_KEY", hide_env_values = true)]
    deepseek: Option<String>,
}

impl Cli {
    fn resolve(mut self) -> Self {
        if self.provider.is_none() && self.api_key.is_none() {
            if let Some(key) = self.deepseek.take() {
                self.provider = Some("deepseek".to_string());
                self.api_key = Some(key);
            }
        }
        return self;
    }
}

#[tokio::main]
async fn main() {
    let cli = Cli::parse().resolve();

    let provider = cli.provider.expect("provider is required");
    let api_key = cli.api_key.expect("api_key is required");
    let base_url = cli.base_url;

    let mut models = Registry::new();
    models.register("deepseek", Box::new(DeepSeekProvider::new(api_key)));

    let model = Model {
        id: "deepseek-v4-flash".into(),
        provider: provider,
        context_window: 384_000,
        base_url: base_url.unwrap_or_default(),
        headers: HashMap::from([("reasoning_effort".into(), "max".into())]),
    };

    let mut tools = ToolRegistry::new();
    if let Some(mcps_path) = &cli.mcps {
        let raw = std::fs::read_to_string(mcps_path).expect("failed to read mcps config");
        let cfg: mcp::config::McpConfig = serde_json::from_str(&raw).expect("failed to parse mcps config");
        for (name, entry) in cfg.mcp_servers {
            let server = entry.into_config().unwrap_or_else(|e| {
                panic!("invalid config for mcp server '{name}': {e}");
            });
            mcp::register(&mut tools, &server)
                .await
                .unwrap_or_else(|e| panic!("failed to connect to mcp server '{name}': {e}"));
        }
    }

    let allowed_ids: HashSet<u64> = cli.allowed_ids.into_iter().collect();
    let models = Arc::new(models);

    let prompt = Message {
        role: "user".into(),
        content: vec![ContentBlock::Text(TextContent {
            content: "1. use tools to assist user. 2. respond concisely without format, table, title for readability on phone. 3. keep the tone dry and neutral.".into(),
        })],
    };

    if let Err(e) = telegram::dispatch(
        cli.telegram_token,
        allowed_ids,
        Arc::new(move |user| {
            let mut tools = tools.clone();
            let session = match user {
                Some(user) => {
                    // let dir = data_dir().join("db");
                    // let _ = std::fs::create_dir_all(&dir);
                    // SqliteSession::new(
                    //     dir.join(format!("{}.db", user.id.0)).to_str().ok_or(anyhow::anyhow!("invalid db path"))?,
                    //     &uuid::Uuid::now_v7().to_string(),
                    // )
                    // .map_err(|e| anyhow::anyhow!(e))
                    // .inspect(|session| session.register_tools(&mut tools))?
                    InMemorySession::new().arc()
                }
                None => InMemorySession::new().arc(),
            };
            session.lock().unwrap().append(prompt.clone()).map_err(|e| anyhow::anyhow!(e))?;

            Ok(Agent {
                state: AgentState { model: model.clone(), tools: tools.clone(), session },
                models: models.clone(),
                extension: NoopExtension.into(),
            })
        }),
    )
    .await
    {
        panic!("dispatch error: {e}")
    };
}

fn data_dir() -> PathBuf {
    let base = std::env::var("XDG_DATA_HOME").ok().filter(|s| !s.is_empty()).map(PathBuf::from).unwrap_or_else(|| {
        let home = std::env::var("HOME").unwrap_or_else(|_| ".".to_string());
        PathBuf::from(home).join(".local/share")
    });
    base.join("akasha")
}

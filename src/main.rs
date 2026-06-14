use std::collections::{HashMap, HashSet};
use std::ops::Sub;
use std::path::PathBuf;
use std::sync::{Arc, Mutex};

use akasha::core::agent::{Agent, AgentState};
use akasha::core::extensions::NoopExtension;
use akasha::core::providers::{Model, Provider};
use akasha::core::session::{InMemorySession, Session};
use akasha::core::tools::ToolRegistry;
use akasha::core::types::{ContentBlock, Message, TextContent};
use akasha::extra::extensions::telegram;
use akasha::extra::providers::deepseek::DeepSeekProvider;
use akasha::extra::sessions::mux::session::MuxSession;
use akasha::extra::sessions::sqlite::SqliteSession;
use akasha::extra::tools::mcp;
use chrono::{Duration, Local};
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

    let model = Model {
        id: "deepseek-v4-pro".into(),
        provider: provider,
        context_window: 384_000,
        base_url: base_url.unwrap_or_default(),
        headers: HashMap::from([("reasoning_effort".into(), "max".into())]),
    };
    let provider: Arc<dyn Provider> = Arc::new(DeepSeekProvider::new(api_key));

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

    let prompt = Message {
        role: "user".into(),
        content: vec![ContentBlock::Text(TextContent { content: include_str!("prompt.md").to_string() })],
    };

    if let Err(e) = telegram::dispatch(
        cli.telegram_token,
        allowed_ids,
        Arc::new(move |user| {
            let (session, extension, tools) = match user {
                Some(user) => {
                    let dir = data_dir().join("db");
                    let _ = std::fs::create_dir_all(&dir);
                    let db_path_str = dir
                        .join(format!("{}.db", user.id.0))
                        .to_str()
                        .ok_or(anyhow::anyhow!("invalid db path"))?
                        .to_string();

                    let prompt = prompt.clone();
                    let loader = Arc::new(move |ref_name: &str| -> anyhow::Result<Box<dyn Session>> {
                        let mut session = Box::new(SqliteSession::new(&db_path_str, ref_name)?);
                        session.append(prompt.clone()).map_err(|e| anyhow::anyhow!("{e}"))?;
                        Ok(session)
                    });

                    let id = Local::now().sub(Duration::hours(4)).format("%Y-%m-%d").to_string();
                    let (mux, ext, tool) = MuxSession::new(&id, loader)?;
                    let mut tools = tools.clone();
                    tools.register(Box::new(tool));
                    let mux: Arc<Mutex<dyn Session>> = mux;

                    (mux, ext.into(), tools)
                }
                None => (InMemorySession::new().arc(), NoopExtension.into(), tools.clone()),
            };

            Ok(Agent {
                state: AgentState { model: model.clone(), tools: tools, session },
                provider: provider.clone(),
                extension,
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

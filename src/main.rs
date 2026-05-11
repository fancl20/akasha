use std::collections::{HashMap, HashSet};
use std::sync::Arc;

use akasha::core::agent::{Agent, AgentState};
use akasha::core::extensions::NoopExtension;
use akasha::core::providers::{Model, Registry};
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
    #[arg(long, env = "OPENAI_API_KEY", hide_env_values = true)]
    openai: Option<String>,
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
        let cfg: mcp::config::McpConfig =
            serde_json::from_str(&raw).expect("failed to parse mcps config");
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

    telegram::dispatch(
        cli.telegram_token,
        allowed_ids,
        Arc::new(move || Agent {
            state: AgentState {
                model: model.clone(),
                tools: tools.clone(),
                messages: vec![prompt.clone()],
            },
            models: models.clone(),
            extension: Box::new(NoopExtension),
        }),
    )
    .await;
}

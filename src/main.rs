use clap::Parser;
use figment::providers::{Env, Serialized};
use figment::Figment;

use openai_trtllm::config::Config;
use openai_trtllm::startup;
use openai_trtllm::telemetry;

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    let config: Config = Figment::new()
        .merge(Env::prefixed("OPENAI_TRTLLM_"))
        .merge(Serialized::defaults(Config::parse()))
        .extract()
        .unwrap();

    telemetry::init_subscriber("openai_trtllm", "info", config.otlp_endpoint.clone())?;

    startup::run_server(config).await
}

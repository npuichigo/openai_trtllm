use clap::Parser;
use serde::{Deserialize, Serialize};

#[derive(Parser, Debug, Serialize, Deserialize)]
pub struct Config {
    /// Host to bind to
    #[arg(long, short = 'H', default_value_t = String::from("0.0.0.0"))]
    pub host: String,
    /// Port to bind to
    #[arg(long, short, default_value_t = 3000)]
    pub port: usize,

    /// Endpoint of OpenTelemetry collector
    #[arg(long, short)]
    #[serde(skip_serializing_if = "Option::is_none")]
    pub otlp_endpoint: Option<String>,
}

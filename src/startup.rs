use anyhow::Context;
use axum::routing::{get, post};
use axum::Router;
use axum_tracing_opentelemetry::middleware::OtelAxumLayer;

use crate::config::Config;
use crate::routes;
use crate::triton::grpc_inference_service_client::GrpcInferenceServiceClient;

pub async fn run_server(config: Config) -> anyhow::Result<()> {
    tracing::info!("Connecting to triton endpoint: {}", config.triton_endpoint);
    let grpc_client = GrpcInferenceServiceClient::connect(config.triton_endpoint)
        .await
        .context("failed to connect triton endpoint")?;

    let app = Router::new()
        .route("/v1/completions", post(routes::compat_completions))
        .route(
            "/v1/chat/completions",
            post(routes::compat_chat_completions),
        )
        .with_state(grpc_client)
        .layer(OtelAxumLayer::default())
        .route("/health_check", get(routes::health_check));

    let address = format!("{}:{}", config.host, config.port);
    tracing::info!("Starting server at {}", address);

    let listener = tokio::net::TcpListener::bind(address).await.unwrap();
    axum::serve(listener, app)
        .with_graceful_shutdown(shutdown_signal())
        .await?;

    Ok(())
}

async fn shutdown_signal() {
    tokio::signal::ctrl_c()
        .await
        .expect("failed to install CTRL+C signal handler");

    opentelemetry::global::shutdown_tracer_provider();
}

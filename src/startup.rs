use anyhow::Context;
use axum::routing::{get, post};
use axum::Router;
use axum::middleware::{self, Next};
use axum::http::{Request, StatusCode};
use axum::response::Response;
use axum::body::Body;
use axum_tracing_opentelemetry::middleware::OtelAxumLayer;

use crate::config::Config;
use crate::history::HistoryBuilder;
use crate::routes;
use crate::state::AppState;
use crate::triton::grpc_inference_service_client::GrpcInferenceServiceClient;

async fn auth_middleware(
    req: Request<Body>,
    next: Next,
    api_key: Option<String>,
) -> Result<Response, StatusCode> {
    if let Some(ref key) = api_key {
        if let Some(auth_header) = req.headers().get("Authorization") {
            if let Ok(auth_str) = auth_header.to_str() {
                if auth_str == format!("Bearer {}", key) {
                    return Ok(next.run(req).await);
                }
            }
        }
        Err(StatusCode::UNAUTHORIZED)
    } else {
        Ok(next.run(req).await)
    }
}

pub async fn run_server(config: Config) -> anyhow::Result<()> {
    tracing::info!("Connecting to triton endpoint: {}", config.triton_endpoint);
    let grpc_client = GrpcInferenceServiceClient::connect(config.triton_endpoint)
        .await
        .context("failed to connect triton endpoint")?;

    let history_builder =
        HistoryBuilder::new(&config.history_template, &config.history_template_file)?;
    let state = AppState {
        grpc_client,
        history_builder,
    };

    let api_key = config.api_key.clone();

    let app = Router::new()
        .route("/v1/completions", post(routes::compat_completions))
        .route(
            "/v1/chat/completions",
            post(routes::compat_chat_completions),
        )
        .route("/health_check", get(routes::health_check))
        .with_state(state)
        .layer(OtelAxumLayer::default())
        .layer(middleware::from_fn(move |req, next| {
            auth_middleware(req, next, api_key.clone())
        }));

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

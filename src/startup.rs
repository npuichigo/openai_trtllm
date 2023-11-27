use anyhow::Context;
use axum::routing::{get, post};
use axum::Router;
use axum_prometheus::PrometheusMetricLayer;
use axum_tracing_opentelemetry::middleware::{OtelAxumLayer, OtelInResponseLayer};
use tower::ServiceBuilder;
use tower_http::request_id::MakeRequestUuid;
use tower_http::{
    trace::TraceLayer,
    trace::{DefaultMakeSpan, DefaultOnResponse},
    ServiceBuilderExt,
};
use tracing::Level;

use crate::config::Config;
use crate::routes;
use crate::triton::grpc_inference_service_client::GrpcInferenceServiceClient;

pub async fn run_server(config: Config) -> anyhow::Result<()> {
    let (prometheus_layer, metric_handle) = PrometheusMetricLayer::pair();

    let grpc_client = GrpcInferenceServiceClient::connect(config.triton_endpoint)
        .await
        .context("failed to connect triton endpoint")?;

    let app = Router::new()
        .route("/v1/completions", post(routes::compat_completions))
        .route("/health_check", get(routes::health_check))
        .route("/metrics", get(|| async move { metric_handle.render() }))
        .with_state(grpc_client)
        .layer(prometheus_layer)
        .layer(OtelInResponseLayer)
        .layer(OtelAxumLayer::default())
        .layer(
            ServiceBuilder::new()
                .set_x_request_id(MakeRequestUuid)
                .layer(
                    TraceLayer::new_for_http()
                        .make_span_with(
                            DefaultMakeSpan::new()
                                .include_headers(true)
                                .level(Level::DEBUG),
                        )
                        .on_response(
                            DefaultOnResponse::new()
                                .include_headers(true)
                                .level(Level::DEBUG),
                        ),
                )
                .propagate_x_request_id(),
        );

    let address = format!("{}:{}", config.host, config.port);
    tracing::info!("Starting server at {}", address);

    axum::Server::bind(&address.parse()?)
        .serve(app.into_make_service())
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

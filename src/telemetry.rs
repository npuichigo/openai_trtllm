use opentelemetry::trace::TraceError;
use opentelemetry::{global, KeyValue};
use opentelemetry_otlp::WithExportConfig;
use opentelemetry_sdk::propagation::TraceContextPropagator;
use opentelemetry_sdk::trace as sdktrace;
use opentelemetry_sdk::{runtime, Resource};
use tracing_subscriber::util::SubscriberInitExt;
use tracing_subscriber::{layer::SubscriberExt, EnvFilter, Layer};

fn init_tracer(name: &str, otlp_endpoint: &str) -> Result<sdktrace::Tracer, TraceError> {
    opentelemetry_otlp::new_pipeline()
        .tracing()
        .with_exporter(
            opentelemetry_otlp::new_exporter()
                .tonic()
                .with_endpoint(otlp_endpoint),
        )
        .with_trace_config(
            sdktrace::config()
                .with_resource(Resource::new(vec![KeyValue::new(
                    "service.name",
                    name.to_owned(),
                )]))
                .with_sampler(sdktrace::Sampler::AlwaysOn),
        )
        .install_batch(runtime::Tokio)
}

/// Compose multiple layers into a `tracing`'s subscriber.
///
/// # Implementation Notes
///
/// We are using `impl Subscriber` as return type to avoid having to spell out the actual
/// type of the returned subscriber, which is indeed quite complex.
pub fn init_subscriber(
    name: &str,
    env_filter: &str,
    otlp_endpoint: Option<String>,
) -> anyhow::Result<()> {
    global::set_text_map_propagator(TraceContextPropagator::new());

    let env_filter = EnvFilter::try_from_default_env()
        .unwrap_or_else(|_| EnvFilter::new(env_filter))
        .add_directive("otel::tracing=trace".parse()?)
        .add_directive("otel=debug".parse()?);

    let telemetry_layer = if let Some(otlp_endpoint) = otlp_endpoint {
        let tracer = init_tracer(name, &otlp_endpoint)?;

        Some(
            tracing_opentelemetry::layer()
                .with_error_records_to_exceptions(true)
                .with_tracer(tracer),
        )
    } else {
        None
    };

    let fmt_layer = if cfg!(debug_assertions) {
        tracing_subscriber::fmt::layer()
            .pretty()
            .with_line_number(true)
            .with_thread_names(true)
            .boxed()
    } else {
        tracing_subscriber::fmt::layer()
            .json()
            .flatten_event(true)
            .boxed()
    };

    tracing_subscriber::registry()
        .with(env_filter)
        .with(telemetry_layer)
        .with(fmt_layer)
        .init();

    Ok(())
}

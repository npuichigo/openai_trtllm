use axum::http::HeaderMap;
use opentelemetry::global;
use opentelemetry::propagation::Injector;
use tonic::Request;
use tracing::Span;
use tracing_opentelemetry::OpenTelemetrySpanExt;

pub struct MetadataMap<'a>(&'a mut tonic::metadata::MetadataMap);

impl<'a> Injector for MetadataMap<'a> {
    /// Set a key and value in the MetadataMap.  Does nothing if the key or value are not valid inputs
    fn set(&mut self, key: &str, value: String) {
        if let Ok(key) = tonic::metadata::MetadataKey::from_bytes(key.as_bytes()) {
            if let Ok(val) = tonic::metadata::MetadataValue::try_from(&value) {
                self.0.insert(key, val);
            }
        }
    }
}

pub(crate) fn propagate_context<T>(request: &mut Request<T>, header: &HeaderMap) {
    let mut metadata_map = MetadataMap(request.metadata_mut());

    // Propagate the current opentelemetry context
    let cx = Span::current().context();
    global::get_text_map_propagator(|propagator| propagator.inject_context(&cx, &mut metadata_map));

    // Propagate x-request-id header to the request if it exists
    if let Some(x_request_id) = header.get("x-request-id") {
        if let Ok(x_request_id) = x_request_id.to_str() {
            metadata_map.set("x-request-id", x_request_id.to_string());
        }
    }
}

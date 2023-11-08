pub mod config;
pub mod routes;
pub mod startup;
pub mod telemetry;
mod utils;

pub(crate) mod triton {
    tonic::include_proto!("inference");
}

use tonic::transport::Channel;
use crate::history::HistoryBuilder;
use crate::triton::grpc_inference_service_client::GrpcInferenceServiceClient;

#[derive(Clone)]
pub struct AppState {
    pub grpc_client: GrpcInferenceServiceClient<Channel>,
    pub history_builder: HistoryBuilder
}
use crate::history::HistoryBuilder;
use crate::triton::grpc_inference_service_client::GrpcInferenceServiceClient;
use tonic::transport::Channel;

#[derive(Clone)]
pub struct AppState {
    pub grpc_client: GrpcInferenceServiceClient<Channel>,
    pub history_builder: HistoryBuilder,
}

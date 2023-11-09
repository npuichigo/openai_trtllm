//! https://platform.openai.com/docs/api-reference/completions/create
use crate::error::AppError;
use crate::triton::grpc_inference_service_client::GrpcInferenceServiceClient;
use crate::triton::ServerLiveRequest;
use crate::utils::string_or_seq_string;
use anyhow::Context;
use axum::extract::State;
use axum::response::{IntoResponse, Response};
use axum::Json;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use tonic::transport::Channel;
use tracing::instrument;

#[instrument(name = "completions", skip(client, request))]
pub async fn compat_completions(
    client: State<GrpcInferenceServiceClient<Channel>>,
    request: Json<CompletionRequest>,
) -> Response {
    tracing::debug!(
        "Received request with streaming set to: {}",
        &request.stream
    );

    if request.stream {
        todo!()
    } else {
        completions(client, request).await.into_response()
    }
}

#[instrument(name = "non-streaming completions", skip(client, request), err(Debug))]
pub async fn completions(
    State(mut client): State<GrpcInferenceServiceClient<Channel>>,
    Json(request): Json<CompletionRequest>,
) -> Result<Json<CompletionResponse>, AppError> {
    let response = client
        .server_live(ServerLiveRequest {})
        .await
        .context("failed to call triton grpc method")?;
    tracing::info!("Server live response: {:?}", response.into_inner());

    Ok(Json(CompletionResponse {
        id: "fake-id".to_string(),
        object: "text_completion".to_string(),
        created: 0,
        model: "fake-model".to_string(),
        choices: vec![CompletionResponseChoices {
            text: "fake-text".to_string(),
            index: 0,
            logprobs: None,
            finish_reason: None,
        }],
        usage: None,
    }))
}

#[derive(Deserialize, Debug)]
pub struct CompletionRequest {
    /// ID of the model to use.
    pub model: String,
    /// The prompt(s) to generate completions for, encoded as a string, array of strings, array of
    /// tokens, or array of token arrays.
    #[serde(deserialize_with = "string_or_seq_string")]
    pub prompt: Vec<String>,
    /// Generates best_of completions server-side and returns the "best" (the one with the highest
    /// log probability per token). Results cannot be streamed.
    #[serde(default = "default_best_of")]
    pub best_of: usize,
    /// Echo back the prompt in addition to the completion
    #[serde(default = "default_echo")]
    pub echo: bool,
    /// Number between -2.0 and 2.0. Positive values penalize new tokens based on their existing
    // frequency in the text so far, decreasing the model's likelihood to repeat the same line
    // verbatim.
    #[serde(default = "default_frequency_penalty")]
    pub frequency_penalty: f32,
    /// Modify the likelihood of specified tokens appearing in the completion.
    pub logit_bias: Option<HashMap<String, f32>>,
    /// Include the log probabilities on the logprobs most likely tokens, as well the chosen tokens.
    pub logprobs: Option<usize>,
    /// The maximum number of tokens to generate in the completion.
    #[serde(default = "default_max_tokens")]
    pub max_tokens: usize,
    // How many completions to generate for each prompt.
    #[serde(default = "default_n")]
    pub n: usize,
    /// Number between -2.0 and 2.0. Positive values penalize new tokens based on whether they
    /// appear in the text so far, increasing the model's likelihood to talk about new topics.
    #[serde(default = "default_presence_penalty")]
    pub presence_penalty: f32,
    /// Up to 4 sequences where the API will stop generating further tokens. The returned text will
    /// not contain the stop sequence.
    pub stop: Option<Vec<String>>,
    /// Whether to stream back partial progress.
    #[serde(default = "default_stream")]
    pub stream: bool,
    /// The suffix that comes after a completion of inserted text.
    pub suffix: Option<String>,
    /// What sampling temperature to use, between 0 and 2. Higher values like 0.8 will make the
    /// output more random, while lower values like 0.2 will make it more focused and deterministic.
    #[serde(default = "default_temperature")]
    pub temperature: f32,
    /// An alternative to sampling with temperature, called nucleus sampling, where the model
    /// considers the results of the tokens with top_p probability mass. So 0.1 means only the
    /// tokens comprising the top 10% probability mass are considered.
    #[serde(default = "default_top_p")]
    pub top_p: f32,
    /// A unique identifier representing your end-user, which can help OpenAI to monitor and detect
    /// abuse.
    pub user: Option<String>,
}

#[derive(Serialize, Debug)]
pub struct CompletionResponse {
    /// A unique identifier for the completion.
    id: String,
    /// The object type, which is always "text_completion"
    object: String,
    /// The Unix timestamp (in seconds) of when the completion was created.
    created: u64,
    /// The model used for completion.
    model: String,
    /// The list of completion choices the model generated for the input prompt.
    choices: Vec<CompletionResponseChoices>,
    /// Usage statistics for the completion request.
    usage: Option<Usage>,
}

#[derive(Serialize, Debug)]
struct CompletionResponseChoices {
    text: String,
    index: usize,
    logprobs: Option<()>,
    finish_reason: Option<FinishReason>,
}

#[derive(Serialize, Debug)]
pub enum FinishReason {
    /// The model hit a natural stop point or a provided stop sequence.
    Stop,
    /// The maximum number of tokens specified in the request was reached.
    Length,
    /// Content was omitted due to a flag from our content filters.
    ContentFilter,
}

#[derive(Serialize, Debug, Default)]
pub struct Usage {
    /// Number of tokens in the prompt.
    pub prompt_tokens: usize,
    /// Number of tokens in the generated completion.
    pub completion_tokens: usize,
    /// Total number of tokens used in the request (prompt + completion).
    pub total_tokens: usize,
}

fn default_best_of() -> usize {
    1
}

fn default_echo() -> bool {
    false
}

fn default_frequency_penalty() -> f32 {
    0.0
}

fn default_max_tokens() -> usize {
    16
}

fn default_n() -> usize {
    1
}

fn default_presence_penalty() -> f32 {
    0.0
}

fn default_stream() -> bool {
    false
}

fn default_temperature() -> f32 {
    1.0
}

fn default_top_p() -> f32 {
    1.0
}

//! https://platform.openai.com/docs/api-reference/completions/create
use std::collections::HashMap;
use std::iter::IntoIterator;
use std::time::{SystemTime, UNIX_EPOCH};

use anyhow::Context;
use async_stream::{stream, try_stream};
use axum::extract::State;
use axum::response::sse::{Event, KeepAlive, Sse};
use axum::response::{IntoResponse, Response};
use axum::Json;
use serde::{Deserialize, Serialize};
use serde_json::json;
use tonic::codegen::tokio_stream::Stream;
use tonic::transport::Channel;
use tracing;
use tracing::instrument;
use uuid::Uuid;

use crate::error::AppError;
use crate::triton::grpc_inference_service_client::GrpcInferenceServiceClient;
use crate::triton::request::{Builder, InferTensorData};
use crate::triton::ModelInferRequest;
use crate::utils::{deserialize_bytes_tensor, string_or_seq_string};

#[instrument(name = "completions", skip(client, request))]
pub(crate) async fn compat_completions(
    client: State<GrpcInferenceServiceClient<Channel>>,
    request: Json<CompletionCreateParams>,
) -> Response {
    tracing::debug!("request: {:?}", request);

    if request.stream {
        completions_stream(client, request).await.into_response()
    } else {
        completions(client, request).await.into_response()
    }
}

#[instrument(name = "streaming completions", skip(client, request))]
async fn completions_stream(
    State(mut client): State<GrpcInferenceServiceClient<Channel>>,
    Json(request): Json<CompletionCreateParams>,
) -> Result<Sse<impl Stream<Item = anyhow::Result<Event>>>, AppError> {
    let id = format!("cmpl-{}", Uuid::new_v4());
    let created = SystemTime::now().duration_since(UNIX_EPOCH)?.as_secs();

    let model_name = request.model.clone();
    let request = build_triton_request(request)?;

    let response_stream = try_stream! {
        let request_stream = stream! { yield request };

        let mut stream = client
            .model_stream_infer(tonic::Request::new(request_stream))
            .await
            .context("failed to call triton grpc method model_stream_infer")?
            .into_inner();

        while let Some(response) = stream.message().await? {
            if !response.error_message.is_empty() {
                tracing::error!("received error message from triton: {}", response.error_message);

                // Corresponds to https://github.com/openai/openai-python/blob/17ac6779958b2b74999c634c4ea4c7b74906027a/src/openai/_streaming.py#L113
                yield Event::default().event("error").json_data(json!({
                    "error": {
                        "status_code": 500,
                        "message": "Internal Server Error"
                    }
                })).unwrap();
                return;
            }
            let infer_response = response
                .infer_response
                .context("empty infer response received")?;
            tracing::debug!("triton infer response: {:?}", infer_response);

            let raw_content = infer_response
                .raw_output_contents
                .last()
                .context("empty raw output contents")?;
            let content = deserialize_bytes_tensor(raw_content.clone())?
                .into_iter()
                .map(|s| s.replace("</s>", ""))
                .collect::<String>();

            if !content.is_empty() {
                let response = Completion {
                    id: id.clone(),
                    object: "text_completion".to_string(),
                    created,
                    model: model_name.clone(),
                    choices: vec![CompletionChoice {
                        text: content,
                        index: 0,
                        logprobs: None,
                        finish_reason: None,
                    }],
                    usage: None,
                };
                yield Event::default().json_data(response).unwrap();
            }
        }
        let response = Completion {
            id,
            object: "text_completion".to_string(),
            created,
            model: model_name,
            choices: vec![CompletionChoice {
                text: String::new(),
                index: 0,
                logprobs: None,
                finish_reason: Some(FinishReason::Stop),
            }],
            usage: None,
        };
        yield Event::default().json_data(response).unwrap();

        // OpenAI stream response terminated by a data: [DONE] message.
        yield Event::default().data("[DONE]");
    };

    Ok(Sse::new(response_stream).keep_alive(KeepAlive::default()))
}

#[instrument(name = "non-streaming completions", skip(client, request), err(Debug))]
async fn completions(
    State(mut client): State<GrpcInferenceServiceClient<Channel>>,
    Json(request): Json<CompletionCreateParams>,
) -> Result<Json<Completion>, AppError> {
    let model_name = request.model.clone();
    let request = build_triton_request(request)?;
    let request_stream = stream! { yield request };
    let mut stream = client
        .model_stream_infer(tonic::Request::new(request_stream))
        .await
        .context("failed to call triton grpc method model_stream_infer")?
        .into_inner();

    let mut contents: Vec<String> = Vec::new();
    while let Some(response) = stream.message().await? {
        if !response.error_message.is_empty() {
            return Err(anyhow::anyhow!(
                "error message received from triton: {}",
                response.error_message
            )
            .into());
        }
        let infer_response = response
            .infer_response
            .context("empty infer response received")?;
        tracing::debug!("triton infer response: {:?}", infer_response);

        let raw_content = infer_response
            .raw_output_contents
            .last()
            .context("empty raw output contents")?;
        let content = deserialize_bytes_tensor(raw_content.clone())?
            .into_iter()
            .map(|s| s.trim().replace("</s>", ""))
            .collect();
        contents.push(content);
    }

    Ok(Json(Completion {
        id: format!("cmpl-{}", Uuid::new_v4()),
        object: "text_completion".to_string(),
        created: SystemTime::now().duration_since(UNIX_EPOCH)?.as_secs(),
        model: model_name,
        choices: vec![CompletionChoice {
            text: contents.into_iter().collect(),
            index: 0,
            logprobs: None,
            finish_reason: Some(FinishReason::Stop),
        }],
        usage: None,
    }))
}

fn build_triton_request(request: CompletionCreateParams) -> anyhow::Result<ModelInferRequest> {
    Builder::new()
        .model_name(request.model)
        .input(
            "text_input",
            [1, 1],
            InferTensorData::Bytes(
                request
                    .prompt
                    .into_iter()
                    .map(|s| s.as_bytes().to_vec())
                    .collect(),
            ),
        )
        .input(
            "max_tokens",
            [1, 1],
            InferTensorData::Int32(vec![request.max_tokens as i32]),
        )
        .input(
            "bad_words",
            [1, 1],
            InferTensorData::Bytes(vec!["".as_bytes().to_vec()]),
        )
        .input(
            "stop_words",
            [1, 1],
            InferTensorData::Bytes(vec!["</s>".as_bytes().to_vec()]),
        )
        .input(
            "stream",
            [1, 1],
            InferTensorData::Bool(vec![request.stream]),
        )
        .build()
        .context("failed to build triton request")
}

#[allow(dead_code)]
#[derive(Deserialize, Debug)]
pub(crate) struct CompletionCreateParams {
    /// ID of the model to use.
    model: String,
    /// The prompt(s) to generate completions for, encoded as a string, array of strings, array of
    /// tokens, or array of token arrays.
    #[serde(deserialize_with = "string_or_seq_string")]
    prompt: Vec<String>,
    /// Generates best_of completions server-side and returns the "best" (the one with the highest
    /// log probability per token). Results cannot be streamed.
    #[serde(default = "default_best_of")]
    best_of: usize,
    /// Echo back the prompt in addition to the completion
    #[serde(default = "default_echo")]
    echo: bool,
    /// Number between -2.0 and 2.0. Positive values penalize new tokens based on their existing
    /// frequency in the text so far, decreasing the model's likelihood to repeat the same line
    /// verbatim.
    #[serde(default = "default_frequency_penalty")]
    frequency_penalty: f32,
    /// Modify the likelihood of specified tokens appearing in the completion.
    logit_bias: Option<HashMap<String, f32>>,
    /// Include the log probabilities on the logprobs most likely tokens, as well the chosen tokens.
    logprobs: Option<usize>,
    /// The maximum number of tokens to generate in the completion.
    #[serde(default = "default_max_tokens")]
    max_tokens: usize,
    /// How many completions to generate for each prompt.
    #[serde(default = "default_n")]
    n: usize,
    /// Number between -2.0 and 2.0. Positive values penalize new tokens based on whether they
    /// appear in the text so far, increasing the model's likelihood to talk about new topics.
    #[serde(default = "default_presence_penalty")]
    presence_penalty: f32,
    /// If specified, our system will make a best effort to sample deterministically, such that
    /// repeated requests with the same seed and parameters should return the same result.
    seed: Option<usize>,
    /// Up to 4 sequences where the API will stop generating further tokens. The returned text will
    /// not contain the stop sequence.
    stop: Option<Vec<String>>,
    /// Whether to stream back partial progress.
    #[serde(default = "default_stream")]
    stream: bool,
    /// The suffix that comes after a completion of inserted text.
    suffix: Option<String>,
    /// What sampling temperature to use, between 0 and 2. Higher values like 0.8 will make the
    /// output more random, while lower values like 0.2 will make it more focused and deterministic.
    #[serde(default = "default_temperature")]
    temperature: f32,
    /// An alternative to sampling with temperature, called nucleus sampling, where the model
    /// considers the results of the tokens with top_p probability mass. So 0.1 means only the
    /// tokens comprising the top 10% probability mass are considered.
    #[serde(default = "default_top_p")]
    top_p: f32,
    /// A unique identifier representing your end-user, which can help OpenAI to monitor and detect
    /// abuse.
    user: Option<String>,
}

#[derive(Serialize, Debug)]
struct Completion {
    /// A unique identifier for the completion.
    id: String,
    /// The object type, which is always "text_completion"
    object: String,
    /// The Unix timestamp (in seconds) of when the completion was created.
    created: u64,
    /// The model used for completion.
    model: String,
    /// The list of completion choices the model generated for the input prompt.
    choices: Vec<CompletionChoice>,
    /// Usage statistics for the completion request.
    usage: Option<Usage>,
}

#[derive(Serialize, Debug)]
struct CompletionChoice {
    text: String,
    index: usize,
    logprobs: Option<()>,
    finish_reason: Option<FinishReason>,
}

#[allow(dead_code)]
#[derive(Serialize, Debug)]
#[serde(rename_all = "snake_case")]
enum FinishReason {
    /// The model hit a natural stop point or a provided stop sequence.
    Stop,
    /// The maximum number of tokens specified in the request was reached.
    Length,
    /// Content was omitted due to a flag from our content filters.
    ContentFilter,
}

#[derive(Serialize, Debug, Default)]
struct Usage {
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

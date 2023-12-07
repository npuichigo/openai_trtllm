# openai_trtllm - OpenAI-compatible API for TensorRT-LLM

Provide [TensorRT-LLM](https://github.com/NVIDIA/TensorRT-LLM) and [NVIDIA Triton Inference Server](https://github.com/triton-inference-server/tensorrtllm_backend)
with an OpenAI-compatible API. This allows you to integrate with [langchain](https://github.com/langchain-ai/langchain)

## Get started
Follow the [tensorrtllm_backend tutorial](https://github.com/triton-inference-server/tensorrtllm_backend#using-the-tensorrt-llm-backend)
to build your TensorRT engine, and launch a triton server. We provide an `Baichuan` example below to follow.

You need to clone the repository with dependencies to build the project.
```bash
# Make sure you have git-lfs installed (https://git-lfs.com)
git lfs install
git submodule update --init --recursive

# If lfs files are not downloaded, run the following command
git lfs pull
```

### Build with Docker
Make sure you have [Docker](https://docs.docker.com/get-docker/) and [Docker Compose](https://docs.docker.com/compose/)
installed.
```bash
docker compose up --build
```
### Build locally
```bash
cargo run --release
```

The parameters can be set with environment variables or command line arguments:
```bash
./target/release/openai_trtllm --help
Usage: openai_trtllm [OPTIONS]

Options:
  -H, --host <HOST>                        Host to bind to [default: 0.0.0.0]
  -p, --port <PORT>                        Port to bind to [default: 3000]
  -t, --triton-endpoint <TRITON_ENDPOINT>  [default: http://localhost:16001]
  -o, --otlp-endpoint <OTLP_ENDPOINT>      Endpoint of OpenTelemetry collector
  -h, --help                               Print help
```

## Example
We provide a model template in `models/Baichuan` to let you follow. Since we're unknown of your hardware, we don't
provide a pre-built TensorRT engine. You need to follow the steps below to build your own engine.

1. Download the [Baichuan](https://huggingface.co/baichuan-inc/Baichuan2-13B-Chat) model from HuggingFace.
   ```bash
   # Make sure you have git-lfs installed (https://git-lfs.com)
   git lfs install
   git clone https://huggingface.co/baichuan-inc/Baichuan2-13B-Chat models/download/Baichuan2-13B-Chat
   ```
2. We provide a pre-built docker which is slightly newer than [v0.6.1](https://github.com/NVIDIA/TensorRT-LLM/tree/v0.6.1).
You are free to test on other versions.
   ```bash
   docker run --rm -it --shm-size=2g --ulimit memlock=-1 --ulimit stack=67108864 --gpus=all -v /models:/models npuichigo/tritonserver-trtllm:711a28d bash
   ```
3. Follow the [tutorial](https://github.com/NVIDIA/TensorRT-LLM/tree/v0.6.1/examples/baichuan) here to build your engine.
   ```bash
   # int8 for example [with inflight batching]
   python /app/tensorrt_llm/examples/baichuan/build.py \
     --model_version v2_13b \
     --model_dir /models/download/Baichuan2-13B-Chat \
     --output_dir /models/baichuan/tensorrt_llm/1 \
     --max_input_len 4096 \
     --max_output_len 1024 \
     --dtype float16 \
     --use_gpt_attention_plugin float16 \
     --use_gemm_plugin float16 \
     --enable_context_fmha \
     --use_weight_only \
     --use_inflight_batching
   ```
   After the build, the engine will be saved to `/models/baichuan/tensorrt_llm/1/` to be used by Triton.
4. Make sure the `models/baichuan/preprocessing/config.pbtxt` and `models/baichuan/postprocessing/config.pbtxt` refer
to the correct tokenizer directory. For example:
   ```bash
   parameters {
     key: "tokenizer_dir"
     value: {
       string_value: "/models/download/Baichuan2-13B-Chat"
     }
   }
   ```
5. Go ahead to launch the server, better with docker-compose.
 
## Tracing
We are tracing performance metrics using tracing, tracing-opentelemetry and opentelemetry-otlp crates.

Let's say you are running a Jaeger instance locally, you can run the following command to start it:
```bash
docker run --rm --name jaeger \
  -p 6831:6831/udp \
  -p 6832:6832/udp \
  -p 5778:5778 \
  -p 16686:16686 \
  -p 4317:4317 \
  -p 4318:4318 \
  -p 14250:14250 \
  -p 14268:14268 \
  -p 14269:14269 \
  -p 9411:9411 \
  jaegertracing/all-in-one:1.51
  
```

To enable tracing, set the `OTLP_ENDPOINT` environment variable or `--otlp-endpoint` command line
argument to the endpoint of your OpenTelemetry collector.
```bash
OTLP_ENDPOINT=http://localhost:4317 cargo run --release
```

## References
- [cria](https://github.com/AmineDiro/cria)

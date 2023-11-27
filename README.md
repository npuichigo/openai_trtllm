# openai_trtllm - OpenAI-compatible API for TensorRT-LLM

Provide [TensorRT-LLM](https://github.com/NVIDIA/TensorRT-LLM) and [NVIDIA Triton Inference Server](https://github.com/triton-inference-server/tensorrtllm_backend)
with an OpenAI-compatible API. This allows you to integrate with [langchain](https://github.com/langchain-ai/langchain)

## Get started
Follow the [tensorrtllm_backend tutorial](https://github.com/triton-inference-server/tensorrtllm_backend#using-the-tensorrt-llm-backend)
to build your TensorRT engine, and launch a triton server.

Build with Cargo
```bash
git submodule update --init --recursive
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

FROM rust:1.74.0-bookworm as chef

WORKDIR /app

RUN apt-get update && apt-get install lld clang protobuf-compiler -y

RUN cargo install cargo-chef --locked

FROM chef as planner

COPY . .

# Compute a lock-like file for our project
RUN cargo chef prepare --recipe-path recipe.json

FROM chef as builder

COPY --from=planner /app/recipe.json recipe.json

# Build our project dependencies, not our application!
RUN cargo chef cook --release --recipe-path recipe.json

# Up to this point, if our dependency tree stays the same,
# all layers should be cached.
COPY . .

# Build our project
RUN cargo build --release --bin openai_trtllm

FROM debian:bookworm-slim AS runtime

WORKDIR /app

RUN apt-get update -y \
    && apt-get install -y --no-install-recommends openssl ca-certificates \
    # Clean up
    && apt-get autoremove -y \
    && apt-get clean -y \
    && rm -rf /var/lib/apt/lists/*

COPY --from=builder /app/target/release/openai_trtllm openai_trtllm

ENTRYPOINT ["./openai_trtllm"]

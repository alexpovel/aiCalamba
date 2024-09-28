FROM rust:1.81 as builder

WORKDIR /app
COPY Cargo.toml Cargo.lock ./
COPY src ./src

RUN cargo build --release

FROM debian:bookworm-slim
RUN apt-get update && apt-get install -y libssl-dev ca-certificates
COPY --from=builder /app/target/release/aicalamba /usr/local/bin/aicalamba
ENTRYPOINT ["aicalamba"]

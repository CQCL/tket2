# Use the official Rust image as the base image
FROM rust:1.73-alpine3.18 AS builder

# Install musl-dev package
RUN apk add --no-cache musl-dev

# Set the working directory to /app
WORKDIR /app

# Copy the Rust project files to the container
COPY . .

# Build the Rust binary in release mode
RUN cargo build --release

# Use a smaller base image for the final image
FROM debian:latest

# Set the working directory to /app
WORKDIR /app

# Copy the Rust binary from the builder image to the final image
COPY --from=builder /app/target/release/taso-optimiser .

# Start the Rust binary
CMD ["/app/taso-optimiser"]
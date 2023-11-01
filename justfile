test:
    cargo test --all-features

fix:
    cargo clippy --all-targets --all-features --workspace --fix --allow-staged

ptest:
    (cd pyrs && maturin develop && pytest)
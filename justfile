test:
    cargo test

fix:
    cargo clippy --fix --allow-staged

ptest:
    (cd pyrs && maturin develop && pytest)
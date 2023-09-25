test:
    cargo test

fix:
    cargo flippy --fix --allow-staged

ptest:
    (cd pyrs && maturin develop && pytest)
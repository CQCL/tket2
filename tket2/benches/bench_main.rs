//! Benchmarks for the tket2 crate.

#[allow(dead_code)]
mod benchmarks;

use criterion::criterion_main;

criterion_main! {
    benchmarks::hash::benches,
}

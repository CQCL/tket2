//! Benchmarks for the tket crate.

#[allow(dead_code)]
mod benchmarks;

use criterion::criterion_main;

criterion_main! {
    benchmarks::hash::benches,
}

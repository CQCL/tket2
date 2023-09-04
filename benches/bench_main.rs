#[allow(dead_code)]
mod benchmarks;

use criterion::criterion_main;

criterion_main! {
    benchmarks::hash::benches,
    benchmarks::pattern_match::benches,
}

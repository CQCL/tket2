#[allow(dead_code)]
mod benchmarks;

use criterion::criterion_main;

#[cfg(feature = "portmatching")]
criterion_main! {
    benchmarks::hash::benches,
    benchmarks::pattern_match::benches,
}

#[cfg(not(feature = "portmatching"))]
criterion_main! {
    benchmarks::hash::benches,
}

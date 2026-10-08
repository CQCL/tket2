//! Benchmark the unexpanded Guppy handoff suite through HUGR resynthesis.

use std::collections::BTreeMap;
use std::fs;
use std::panic::{AssertUnwindSafe, catch_unwind};
use std::path::Path;
use std::time::Instant;

use hugr::envelope::EnvelopeConfig;
use hugr::{Hugr, HugrView};
use serde_json::{Value, json};
use tket::Circuit;
use tket::passes::{ComposablePass, InlineFunctionsPass, Normalize, PauliGraphResynthesis};

fn counts(hugr: &Hugr) -> Value {
    let mut quantum = BTreeMap::<String, usize>::new();
    let mut structure = BTreeMap::<String, usize>::new();
    for node in hugr.nodes() {
        let op = hugr.get_optype(node);
        if let Some(ext) = op.as_extension_op() {
            if ext.extension_id().to_string() == "tket.quantum" {
                *quantum.entry(ext.unqualified_id().to_string()).or_default() += 1;
            }
        }
        for (name, present) in [
            ("TailLoop", op.is_tail_loop()),
            ("CFG", op.is_cfg()),
            ("Conditional", op.is_conditional()),
            ("Call", op.is_call()),
            ("CallIndirect", op.is_call_indirect()),
        ] {
            if present {
                *structure.entry(name.to_owned()).or_default() += 1;
            }
        }
    }
    let explicit_t = quantum.get("T").unwrap_or(&0) + quantum.get("Tdg").unwrap_or(&0);
    let toffolis = *quantum.get("Toffoli").unwrap_or(&0);
    json!({
        "explicit_t_tdg_nodes": explicit_t,
        "toffoli_nodes": toffolis,
        "t_with_7t_toffoli_lowering": explicit_t + 7 * toffolis,
        "quantum_ops": quantum,
        "structure": structure,
        "nodes": hugr.num_nodes(),
    })
}

fn run(input: &Circuit, enable_t: bool, output: &Path) -> Value {
    let mut circuit = input.clone();
    let start = Instant::now();
    let outcome = catch_unwind(AssertUnwindSafe(|| {
        PauliGraphResynthesis::default()
            .with_t_optimization(enable_t)
            .run(circuit.hugr_mut())
    }));
    let seconds = start.elapsed().as_secs_f64();
    match outcome {
        Ok(Ok(())) => match circuit.hugr().validate() {
            Ok(()) => {
                fs::write(output, circuit.store_str(EnvelopeConfig::text()).unwrap()).unwrap();
                json!({"status": "ok", "seconds": seconds, "counts": counts(circuit.hugr()), "output": output})
            }
            Err(err) => {
                json!({"status": "validation_error", "seconds": seconds, "error": err.to_string()})
            }
        },
        Ok(Err(err)) => {
            json!({"status": "error", "seconds": seconds, "error": err.to_string(), "debug_error": format!("{err:?}")})
        }
        Err(err) => {
            let message = err
                .downcast_ref::<String>()
                .cloned()
                .or_else(|| err.downcast_ref::<&str>().map(|s| s.to_string()))
                .unwrap_or_else(|| "non-string panic".to_owned());
            json!({"status": "panic", "seconds": seconds, "error": message})
        }
    }
}

fn main() {
    let directory = std::env::args()
        .nth(1)
        .expect("usage: handoff_benchmark DIRECTORY");
    let directory = Path::new(&directory);
    let manifest: Value =
        serde_json::from_str(&fs::read_to_string(directory.join("manifest.json")).unwrap())
            .unwrap();
    fs::create_dir_all(directory.join("outputs")).unwrap();
    let mut results = Vec::new();
    for artifact in manifest["artifacts"].as_array().unwrap() {
        let file = artifact["file"].as_str().unwrap();
        let name = Path::new(file).file_stem().unwrap().to_str().unwrap();
        eprintln!("Processing {name}");
        let loaded = Circuit::load_str(fs::read_to_string(directory.join(file)).unwrap(), None);
        let input = match loaded {
            Ok(input) => input,
            Err(error) => {
                results.push(
                    json!({"circuit": name, "status": "load_error", "error": error.to_string()}),
                );
                continue;
            }
        };
        // Record the pass's own structural preparation separately from optimisation.
        let mut prepared = input.clone();
        let preparation = InlineFunctionsPass::default()
            .run(prepared.hugr_mut())
            .map_err(|e| e.to_string())
            .and_then(|_| {
                Normalize::default()
                    .run(prepared.hugr_mut())
                    .map_err(|e| e.to_string())
            });
        let prepared_counts = match preparation {
            Ok(_) => counts(prepared.hugr()),
            Err(error) => json!({"error": error}),
        };
        let off = run(
            &input,
            false,
            &directory.join(format!("outputs/{name}_resynthesis.hugr")),
        );
        let on = run(
            &input,
            true,
            &directory.join(format!("outputs/{name}_t_optimization.hugr")),
        );
        println!(
            "{name}: resynthesis={} T optimisation={}",
            off["status"], on["status"]
        );
        results.push(json!({"circuit": name, "raw_static_counts": counts(input.hugr()), "prepared_static_counts": prepared_counts, "resynthesis_only": off, "resynthesis_t_optimization": on}));
        fs::write(directory.join("results.json"), serde_json::to_string_pretty(&json!({
            "settings": {"t_optimization": true, "ancilla_budget": null, "seed": 0, "scope": "default", "comptime_expansion_added": false},
            "count_kind": "static_gate_nodes_not_execution_weighted",
            "artifacts": results,
        })).unwrap()).unwrap();
    }
}

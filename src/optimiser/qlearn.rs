use std::collections::HashMap;
use std::time::Instant;

use fxhash::FxHashSet;
use hugr::Hugr;
use rurel::mdp::{Agent, State};
use rurel::strategy::explore::RandomExploration;
use rurel::strategy::learn::QLearning;
use rurel::strategy::terminate::FixedIterations;
use rurel::AgentTrainer;

use crate::circuit::CircuitHash;
use crate::rewrite::strategy::RewriteStrategy;
use crate::rewrite::Rewriter;
use crate::Circuit;

use super::{TasoLogger, TasoOptimiser};

#[derive(PartialEq, Eq, Clone)]
struct MyState<T> {
    circuit: T,
}

impl<T: Circuit> std::hash::Hash for MyState<T> {
    fn hash<H: std::hash::Hasher>(&self, state: &mut H) {
        self.circuit.circuit_hash().hash(state);
    }
}

#[derive(PartialEq, Eq, Hash, Clone)]
enum MyAction {
    Move { dx: i32, dy: i32 },
}

impl<T> State for MyState<T> {
    type A = MyAction;

    fn reward(&self) -> f64 {
        let (tx, ty) = (10, 10);
        let d = (((tx - self.x).pow(2) + (ty - self.y).pow(2)) as f64).sqrt();
        -d
    }

    fn actions(&self) -> Vec<MyAction> {
        vec![
            MyAction::Move { dx: -1, dy: 0 },
            MyAction::Move { dx: 1, dy: 0 },
            MyAction::Move { dx: 0, dy: -1 },
            MyAction::Move { dx: 0, dy: 1 },
        ]
    }
}

struct MyAgent {
    state: MyState,
}

impl Agent<MyState> for MyAgent {
    fn current_state(&self) -> &MyState {
        &self.state
    }

    fn take_action(&mut self, action: &MyAction) {
        match action {
            &MyAction::Move { dx, dy } => {
                self.state = MyState {
                    x: (((self.state.x + dx) % self.state.maxx) + self.state.maxx)
                        % self.state.maxx,
                    y: (((self.state.y + dy) % self.state.maxy) + self.state.maxy)
                        % self.state.maxy,
                    ..self.state.clone()
                };
            }
        }
    }
}

fn main() {
    let initial_state = MyState {
        x: 0,
        y: 0,
        maxx: 21,
        maxy: 21,
    };
    let mut trainer = AgentTrainer::new();
    let mut agent = MyAgent {
        state: initial_state.clone(),
    };
    trainer.train(
        &mut agent,
        &QLearning::new(0.2, 0.01, 2.),
        &mut FixedIterations::new(100000),
        &RandomExploration::new(),
    );
    for j in 0..21 {
        for i in 0..21 {
            let entry: &HashMap<MyAction, f64> = trainer
                .expected_values(&MyState {
                    x: i,
                    y: j,
                    ..initial_state
                })
                .unwrap();
            let best_action = entry
                .iter()
                .max_by(|(_, v1), (_, v2)| v1.partial_cmp(v2).unwrap())
                .map(|(v, _)| v)
                .unwrap();
            match best_action {
                MyAction::Move { dx: -1, dy: 0 } => print!("<"),
                MyAction::Move { dx: 1, dy: 0 } => print!(">"),
                MyAction::Move { dx: 0, dy: -1 } => print!("^"),
                MyAction::Move { dx: 0, dy: 1 } => print!("v"),
                _ => unreachable!(),
            };
        }
        println!();
    }
}

impl<R, S> TasoOptimiser<R, S>
where
    R: Rewriter + Send + Clone + 'static,
    S: RewriteStrategy + Send + Sync + Clone + 'static,
    S::Cost: serde::Serialize + Send + Sync,
{
    #[tracing::instrument(target = "taso::metrics", skip(self, circ, logger))]
    fn qlearn(
        &self,
        circ: &Hugr,
        mut logger: TasoLogger,
        timeout: Option<u64>,
        queue_size: usize,
    ) -> Hugr {
        let start_time = Instant::now();

        let mut best_circ = circ.clone();
        let mut best_circ_cost = self.cost(circ);
        logger.log_best(&best_circ_cost);

        // Hash of seen circuits. Dot not store circuits as this map gets huge
        let mut seen_hashes = FxHashSet::default();
        seen_hashes.insert(circ.circuit_hash());

        // The priority queue of circuits to be processed (this should not get big)
        let cost_fn = {
            let strategy = self.strategy.clone();
            move |circ: &'_ Hugr| strategy.circuit_cost(circ)
        };
        let mut pq = HugrPQ::new(cost_fn, queue_size);
        pq.push(circ.clone());

        let mut circ_cnt = 0;
        let mut timeout_flag = false;
        while let Some(Entry { circ, cost, .. }) = pq.pop() {
            if cost < best_circ_cost {
                best_circ = circ.clone();
                best_circ_cost = cost.clone();
                logger.log_best(&best_circ_cost);
            }
            circ_cnt += 1;

            let rewrites = self.rewriter.get_rewrites(&circ);
            for (new_circ, cost_delta) in self.strategy.apply_rewrites(rewrites, &circ) {
                let new_circ_hash = new_circ.circuit_hash();
                if !seen_hashes.insert(new_circ_hash) {
                    // Ignore this circuit: we've already seen it
                    continue;
                }
                logger.log_progress(circ_cnt, Some(pq.len()), seen_hashes.len());
                let new_circ_cost = cost.add_delta(&cost_delta);
                pq.push_unchecked(new_circ, new_circ_hash, new_circ_cost);
            }

            if pq.len() >= queue_size {
                // Haircut to keep the queue size manageable
                pq.truncate(queue_size / 2);
            }

            if let Some(timeout) = timeout {
                if start_time.elapsed().as_secs() > timeout {
                    timeout_flag = true;
                    break;
                }
            }
        }

        logger.log_processing_end(
            circ_cnt,
            Some(seen_hashes.len()),
            best_circ_cost,
            false,
            timeout_flag,
        );
        best_circ
    }
}

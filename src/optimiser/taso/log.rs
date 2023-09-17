//! Logging utilities for the TASO optimiser.

use std::io;

use hugr::Hugr;

use crate::json::save_tk1_json_writer;

/// Logging configuration for the TASO optimiser.
#[derive(Default)]
pub struct TasoLogger<'w> {
    final_circ_writer: Option<Box<dyn io::Write + 'w>>,
    circ_candidates_csv: Option<csv::Writer<Box<dyn io::Write + 'w>>>,
}

/// The logging target for general events.
pub const LOG_TARGET: &str = "taso::log";
/// The logging target for progress events. More verbose than the general log.
pub const PROGRESS_TARGET: &str = "taso::progress";
/// The logging target for function spans.
pub const METRICS_TARGET: &str = "taso::metrics";

impl<'w> TasoLogger<'w> {
    /// Create a new logging configuration.
    ///
    /// Two writer objects must be provided:
    /// - best_circ_json_writer: for the final optimised circuit, in TK1 JSON
    ///   format,
    /// - best_progress_csv_writer: for a log of the successive best candidate
    ///   circuits,
    ///
    /// Regular events are logged with [`tracing`], with targets [`LOG_TARGET`]
    /// or [`PROGRESS_TARGET`].
    ///
    /// [`log`]: <https://docs.rs/log/latest/log/>
    pub fn new(
        best_circ_json_writer: impl io::Write + 'w,
        best_progress_csv_writer: impl io::Write + 'w,
    ) -> Self {
        let boxed_candidates_writer: Box<dyn io::Write> = Box::new(best_progress_csv_writer);
        Self {
            final_circ_writer: Some(Box::new(best_circ_json_writer)),
            circ_candidates_csv: Some(csv::Writer::from_writer(boxed_candidates_writer)),
        }
    }

    /// Log a new best candidate
    #[inline]
    pub fn log_best(&mut self, best_cost: usize) {
        self.log(format!("new best of size {}", best_cost));
        if let Some(csv_writer) = self.circ_candidates_csv.as_mut() {
            csv_writer.serialize(BestCircSer::new(best_cost)).unwrap();
            csv_writer.flush().unwrap();
        };
    }

    /// Log the final optimised circuit
    #[inline]
    pub fn log_processing_end(&self, circuit_count: usize, needs_joining: bool, timeout: bool) {
        if timeout {
            self.log("Timeout");
        }
        self.log("Optimisation finished");
        self.log(format!("Tried {circuit_count} circuits"));
        if needs_joining {
            self.log("Joining worker threads");
        }
    }

    /// Log the progress of the optimisation.
    #[inline(always)]
    pub fn log_progress(
        &mut self,
        circ_cnt: usize,
        workqueue_len: Option<usize>,
        seen_hashes: usize,
    ) {
        if circ_cnt % 1000 == 0 {
            self.progress(format!("{circ_cnt} circuits..."));
            if let Some(workqueue_len) = workqueue_len {
                self.progress(format!("Queue size: {workqueue_len} circuits"));
            }
            self.progress(format!("Total seen: {} circuits", seen_hashes));
        }
    }

    /// Log the final optimised circuit.
    #[inline]
    pub fn log_final(&mut self, best_circ: &Hugr, best_cost: usize) {
        self.log(format!("END RESULT: {}", best_cost));
        if let Some(circ_writer) = self.final_circ_writer.as_mut() {
            save_tk1_json_writer(best_circ, circ_writer).unwrap();
        }
    }

    /// Internal function to log general events, normally printed to stdout.
    #[inline]
    fn log(&self, msg: impl AsRef<str>) {
        tracing::info!(target: LOG_TARGET, "{}", msg.as_ref());
    }

    /// Internal function to log information on the progress of the optimization.
    #[inline]
    fn progress(&self, msg: impl AsRef<str>) {
        tracing::info!(target: PROGRESS_TARGET, "{}", msg.as_ref());
    }
}

/// A helper struct for logging improvements in circuit size seen during the
/// TASO execution.
//
// TODO: Replace this fixed logging. Report back intermediate results.
#[derive(serde::Serialize, Clone, Debug)]
struct BestCircSer {
    circ_len: usize,
    time: String,
}

impl BestCircSer {
    fn new(circ_len: usize) -> Self {
        let time = chrono::Local::now().to_rfc3339();
        Self { circ_len, time }
    }
}

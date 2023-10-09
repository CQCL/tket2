//! Logging utilities for the TASO optimiser.

use std::{fmt::Debug, io};

/// Logging configuration for the TASO optimiser.
#[derive(Default)]
pub struct TasoLogger<'w> {
    circ_candidates_csv: Option<csv::Writer<Box<dyn io::Write + 'w>>>,
    last_circ_processed: usize,
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
    /// - best_progress_csv_writer: for a log of the successive best candidate
    ///   circuits,
    ///
    /// Regular events are logged with [`tracing`], with targets [`LOG_TARGET`]
    /// or [`PROGRESS_TARGET`].
    ///
    /// [`log`]: <https://docs.rs/log/latest/log/>
    pub fn new(best_progress_csv_writer: impl io::Write + 'w) -> Self {
        let boxed_candidates_writer: Box<dyn io::Write> = Box::new(best_progress_csv_writer);
        Self {
            circ_candidates_csv: Some(csv::Writer::from_writer(boxed_candidates_writer)),
            last_circ_processed: 0,
        }
    }

    /// Log a new best candidate
    #[inline]
    pub fn log_best<C: Debug + serde::Serialize>(&mut self, best_cost: C) {
        self.log(format!("new best of size {:?}", best_cost));
        if let Some(csv_writer) = self.circ_candidates_csv.as_mut() {
            csv_writer.serialize(BestCircSer::new(best_cost)).unwrap();
            csv_writer.flush().unwrap();
        };
    }

    /// Log the final optimised circuit
    #[inline]
    pub fn log_processing_end<C: Debug>(
        &self,
        circuits_processed: usize,
        circuits_seen: Option<usize>,
        best_cost: C,
        needs_joining: bool,
        timeout: bool,
    ) {
        match timeout {
            true => self.log("Optimisation finished (timeout)."),
            false => self.log("Optimisation finished."),
        };
        match circuits_seen {
            Some(circuits_seen) => self.log(format!(
                "Processed {circuits_processed} circuits (out of {circuits_seen} seen)."
            )),
            None => self.log(format!("Processed {circuits_processed} circuits.")),
        }
        self.log(format!("---- END RESULT: {:?} ----", best_cost));
        if needs_joining {
            self.log("Joining worker threads.");
        }
    }

    /// Log the progress of the optimisation.
    #[inline(always)]
    pub fn log_progress(
        &mut self,
        circuits_processed: usize,
        workqueue_len: Option<usize>,
        seen_hashes: usize,
    ) {
        if circuits_processed > self.last_circ_processed && circuits_processed % 1000 == 0 {
            self.last_circ_processed = circuits_processed;

            self.progress(format!("Processed {circuits_processed} circuits..."));
            if let Some(workqueue_len) = workqueue_len {
                self.progress(format!("Queue size: {workqueue_len} circuits."));
            }
            self.progress(format!("Total seen: {} circuits.", seen_hashes));
        }
    }

    /// Log general events, normally printed to stdout.
    #[inline]
    pub fn log(&self, msg: impl AsRef<str>) {
        tracing::info!(target: LOG_TARGET, "{}", msg.as_ref());
    }

    /// Log verbose information on the progress of the optimization.
    #[inline]
    pub fn progress(&self, msg: impl AsRef<str>) {
        tracing::info!(target: PROGRESS_TARGET, "{}", msg.as_ref());
    }
}

/// A helper struct for logging improvements in circuit size seen during the
/// TASO execution.
//
// TODO: Replace this fixed logging. Report back intermediate results.
#[derive(serde::Serialize, Clone, Debug)]
struct BestCircSer<C> {
    circ_cost: C,
    time: String,
}

impl<C> BestCircSer<C> {
    fn new(circ_cost: C) -> Self {
        let time = chrono::Local::now().to_rfc3339();
        Self { circ_cost, time }
    }
}

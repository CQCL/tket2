// use crate::circuit::{
//     circuit::Circuit,
//     dag::{EdgeProperties, VertexProperties},
//     operation::Param,
// };

// use super::{
//     pattern::{Match, NodeCompClosure},
//     CircRewriteIter, RewriteGenerator,
// };
// use std::{
//     sync::{
//         mpsc::{channel, Receiver, Sender},
//         Arc,
//     },
//     thread::{self, JoinHandle},
// };
// pub struct MultiSearcher {
//     max_threads: u32,
//     r_main: Receiver<Option<Circuit>>,
//     thread_ts: Vec<Sender<Option<Arc<Circuit>>>>,
//     joins: Vec<JoinHandle<()>>,
//     // input_channel: (Sender<Arc<Circuit>>, Receiver<Arc<Circuit>>),
//     // return_channel: (Sender<Circuit>, Receiver<Circuit>),
// }

// pub fn searcher<T, F, G>(max_threads: u32, rewrite_gens: Vec<T>) -> MultiSearcher
// where
//     F: NodeCompClosure<VertexProperties, EdgeProperties> + Clone + Send + Sync + 'static,
//     G: Fn(Match) -> (Circuit, Param) + Clone + 'static,
//     T: RewriteGenerator<'static, CircRewriteIter<'static, F, G>> + Send + Sync + Clone + 'static,
// {
//     let n_threads = std::cmp::min(max_threads as usize, rewrite_gens.len());

//     let (t_main, r_main) = channel();

//     let mut chunks: Vec<Vec<_>> = (0..n_threads).map(|_| vec![]).collect();

//     for (count, p) in rewrite_gens.into_iter().enumerate() {
//         chunks[count % n_threads].push((count, p));
//     }

//     let (joins, thread_ts): (Vec<_>, Vec<_>) = chunks
//         .into_iter()
//         .enumerate()
//         .map(|(i, repsets)| {
//             // channel for sending circuits to each thread
//             let (t_this, r_this) = channel();
//             let tn = t_main.clone();
//             let jn = thread::spawn(move || {
//                 for received in r_this {
//                     let sent_circ: Arc<Circuit> = if let Some(hc) = received {
//                         hc
//                     } else {
//                         // main has signalled no more circuits will be sent
//                         return;
//                     };
//                     println!("thread {i} got one");
//                     for (set_i, rcs) in repsets.iter() {
//                         for rewrite in rcs.rewrites(&sent_circ) {
//                             // TODO here is where a optimal data-sharing copy would be handy
//                             let mut newc = sent_circ.as_ref().clone();
//                             newc.apply_rewrite(rewrite).expect("rewrite failure");

//                             tn.send(Some(newc)).unwrap();
//                             println!("thread {i} sent one back from set {set_i}");
//                         }
//                     }
//                     // no more circuits will be generated, tell main this thread is
//                     // done with this circuit
//                     tn.send(None).unwrap();
//                 }
//             });

//             (jn, t_this)
//         })
//         .unzip();

//     MultiSearcher {
//         max_threads,
//         r_main,
//         thread_ts,
//         joins,
//     }
// }
// impl MultiSearcher {
//     pub fn send_circ(&self, circ: Arc<Circuit>) {
//         for thread_t in &self.thread_ts {
//             thread_t.send(Some(circ.clone())).unwrap();
//         }
//     }

//     pub fn stop(self) {
//         for (join, tx) in self.joins.into_iter().zip(self.thread_ts.into_iter()) {
//             // tell all the threads we're done and join the threads
//             tx.send(None).unwrap();
//             join.join().unwrap();
//         }
//     }
// }

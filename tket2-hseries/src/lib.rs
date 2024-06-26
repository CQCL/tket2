//! Provides a preparation and validation workflow for Hugrs targeting
//! Quantinuum H-series quantum computers.

use hugr::Hugr;

#[cfg(feature = "cli")]
pub mod cli;

/// Modify a [Hugr] into a form that is acceptable for ingress into an H-series.
///
/// Returns an error if this cannot be done.
pub fn prepare_ngrte(#[allow(unused)] hugr: &mut Hugr) -> Result<(), Box<dyn std::error::Error>> {
    Ok(())
}

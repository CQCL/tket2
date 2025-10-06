//! Pass to resolve modifiers (control/dagger/power) in a Hugr.
use hugr::{Hugr, HugrView};

use crate::QSystemPassError;
use tket::modifier::modifier_resolver::resolve_modifier_with_entrypoints;

#[allow(missing_docs)]
#[derive(Default)]
pub struct ModifierResolverPass;

impl ModifierResolverPass {
    /// Run ModifierResolverPass.
    pub fn run(self, hugr: &mut Hugr) -> Result<(), QSystemPassError> {
        resolve_modifier_with_entrypoints(hugr, [hugr.entrypoint()]).unwrap();
        Ok(())
    }
}

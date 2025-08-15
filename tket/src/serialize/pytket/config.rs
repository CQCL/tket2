//! Configuration structs for pytket encoders and decoders.

mod decoder_config;
mod encoder_config;
mod type_translators;

pub use decoder_config::PytketDecoderConfig;
pub use encoder_config::PytketEncoderConfig;
pub use type_translators::TypeTranslatorSet;

use crate::serialize::pytket::extension::{
    BoolEmitter, FloatEmitter, PreludeEmitter, RotationEmitter, Tk1Emitter, TketOpEmitter,
};
use hugr::HugrView;

/// Default pytket decoder configuration for [`Circuit`][crate::Circuit]s.
///
/// Contains a list of custom decoders that define translations of legacy tket
/// primitives into HUGR operations.
pub fn default_decoder_config() -> PytketDecoderConfig {
    let mut config = PytketDecoderConfig::new();
    config.add_decoder(PreludeEmitter);
    config.add_decoder(BoolEmitter);
    config.add_decoder(TketOpEmitter);

    config.add_type_translator(PreludeEmitter);
    config.add_type_translator(BoolEmitter);
    config.add_type_translator(FloatEmitter);
    config.add_type_translator(RotationEmitter);

    config
}

/// Default encoder configuration for [`Circuit`][crate::Circuit]s.
///
/// Contains emitters for std and tket operations.
pub fn default_encoder_config<H: HugrView>() -> PytketEncoderConfig<H> {
    let mut config = PytketEncoderConfig::new();
    config.add_emitter(PreludeEmitter);
    config.add_emitter(BoolEmitter);
    config.add_emitter(FloatEmitter);
    config.add_emitter(RotationEmitter);
    config.add_emitter(Tk1Emitter);
    config.add_emitter(TketOpEmitter);

    config.add_type_translator(PreludeEmitter);
    config.add_type_translator(BoolEmitter);
    config.add_type_translator(FloatEmitter);
    config.add_type_translator(RotationEmitter);

    config
}

//! Sets of extensions for translating HUGR types into pytket registers.
//!
//! Defines sets of [`TypeTranslator`]s used by the encoders and decoders to
//! translate HUGR types into pytket primitives.

use std::collections::HashMap;
use std::sync::RwLock;

use hugr::extension::prelude::bool_t;
use hugr::extension::ExtensionId;
use hugr::types::{Type, TypeEnum};
use itertools::Itertools;

use crate::serialize::pytket::extension::{PytketTypeTranslator, RegisterCount};

/// A set of [`PytketTypeTranslator`]s that can be used to translate HUGR types
/// into pytket registers (qubits, bits, and parameter expressions).
#[derive(Default, derive_more::Debug)]
pub struct TypeTranslatorSet {
    /// Registered type translators
    #[debug(skip)]
    pub(super) type_translators: Vec<Box<dyn PytketTypeTranslator + Send + Sync>>,
    /// Pre-computed map from extension ids to corresponding type translators in
    /// `type_translators`, identified by their index.
    #[debug("{:?}", extension_translators.keys().collect_vec())]
    extension_translators: HashMap<ExtensionId, Vec<usize>>,
    /// A cache of translated types.
    #[debug(skip)]
    type_cache: RwLock<HashMap<Type, Option<RegisterCount>>>,
}

impl TypeTranslatorSet {
    /// Add a translator to the set.
    ///
    /// This operation invalidates the type translation cache.
    pub(super) fn add_type_translator(
        &mut self,
        translator: impl PytketTypeTranslator + Send + Sync + 'static,
    ) {
        let idx = self.type_translators.len();

        for ext in translator.extensions() {
            self.extension_translators.entry(ext).or_default().push(idx);
        }

        self.type_translators.push(Box::new(translator));

        // Clear the cache, or create a new one if it's poisoned.
        let cache = self.type_cache.write();
        if let Ok(mut cache) = cache {
            cache.clear();
        } else {
            drop(cache);
            self.type_cache = RwLock::new(HashMap::new());
        }
    }

    /// Translate a HUGR type into a count of qubits, bits, and parameters,
    /// using the registered custom translator.
    ///
    /// Only tuple sums, bools, and custom types are supported.
    /// Other types will return `None`.
    pub fn type_to_pytket(&self, typ: &Type) -> Option<RegisterCount> {
        let cache = self.type_cache.read().ok();
        if let Some(count) = cache.and_then(|c| c.get(typ).cloned()) {
            return count;
        }

        let res = match typ.as_type_enum() {
            TypeEnum::Sum(sum) => {
                if sum.num_variants() == 0 {
                    return Some(RegisterCount::default());
                }
                if typ == &bool_t() {
                    return Some(RegisterCount::only_bits(1));
                }
                if let Some(tuple) = sum.as_tuple() {
                    let count: Option<RegisterCount> = tuple
                        .iter()
                        .map(|ty| {
                            match ty.clone().try_into() {
                                Ok(ty) => self.type_to_pytket(&ty),
                                // Sum types with row variables (variable tuple lengths) are not supported.
                                Err(_) => None,
                            }
                        })
                        .sum();
                    count
                } else {
                    None
                }
            }
            TypeEnum::Extension(custom) => 'outer: {
                let type_ext = custom.extension();
                for encoder in self.translators_for_extension(type_ext) {
                    if let Some(count) = encoder.type_to_pytket(custom, self) {
                        break 'outer Some(count);
                    }
                }
                None
            }
            _ => None,
        };

        // Insert the result into the cache. Ignoring it if it's poisoned.
        if let Ok(mut cache) = self.type_cache.write() {
            cache.insert(typ.clone(), res);
        }

        res
    }

    /// Lists the translators that can handle a given extension.
    fn translators_for_extension(
        &self,
        ext: &ExtensionId,
    ) -> impl Iterator<Item = &Box<dyn PytketTypeTranslator + Send + Sync>> {
        self.extension_translators
            .get(ext)
            .into_iter()
            .flatten()
            .map(move |idx| &self.type_translators[*idx])
    }
}

#[cfg(test)]
mod tests {
    use hugr::extension::prelude::{qb_t, PRELUDE_ID};
    use hugr::types::SumType;

    use crate::extension::bool::BOOL_EXTENSION_ID;

    use super::*;

    struct TestBoolTranslator;

    impl PytketTypeTranslator for TestBoolTranslator {
        fn extensions(&self) -> Vec<ExtensionId> {
            vec![BOOL_EXTENSION_ID, PRELUDE_ID]
        }

        fn type_to_pytket(
            &self,
            typ: &hugr::types::CustomType,
            _set: &TypeTranslatorSet,
        ) -> Option<RegisterCount> {
            match typ.name().as_str() {
                "usize" => Some(RegisterCount::only_bits(64)),
                "qubit" => Some(RegisterCount::only_qubits(1)),
                "bool" => Some(RegisterCount::only_bits(1)),
                _ => None,
            }
        }
    }

    #[rstest::fixture]
    fn translator_set() -> TypeTranslatorSet {
        let mut set = TypeTranslatorSet::default();
        set.add_type_translator(TestBoolTranslator);
        set
    }

    #[rstest::rstest]
    #[case::empty(SumType::new_unary(0).into(), Some(RegisterCount::default()))]
    #[case::native_bool(SumType::new_unary(2).into(), Some(RegisterCount::only_bits(1)))]
    #[case::simple(bool_t(), Some(RegisterCount::only_bits(1)))]
    #[case::tuple(SumType::new_tuple(vec![bool_t(), qb_t(), bool_t()]).into(), Some(RegisterCount::new(1, 2, 0)))]
    #[case::unsupported(SumType::new([vec![bool_t(), qb_t()], vec![bool_t()]]).into(), None)]
    fn test_translations(
        translator_set: TypeTranslatorSet,
        #[case] typ: Type,
        #[case] count: Option<RegisterCount>,
    ) {
        assert_eq!(translator_set.type_to_pytket(&typ), count);
    }
}

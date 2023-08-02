//! Utility functions for the library.

use hugr::types::{ClassicType, SimpleType};

pub(crate) const QB: SimpleType = SimpleType::Qubit;
pub(crate) const F64: SimpleType = SimpleType::Classic(ClassicType::F64);

#[allow(dead_code)]
// Test only utils
#[cfg(test)]
pub(crate) mod test {
    /// Open a browser page to render a dot string graph.
    ///
    /// This can be used directly on the output of `Hugr::dot_string`
    #[cfg(not(ci_run))]
    pub(crate) fn viz_dotstr(dotstr: &str) {
        let mut base: String = "https://dreampuf.github.io/GraphvizOnline/#".into();
        base.push_str(&urlencoding::encode(dotstr));
        webbrowser::open(&base).unwrap();
    }
}

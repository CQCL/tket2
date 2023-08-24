//! Utility functions for the library.

use hugr::ops::LeafOp;
use hugr::std_extensions::quantum::EXTENSION as QUANTUM;

fn get_gate(gate_name: &str) -> LeafOp {
    QUANTUM
        .instantiate_extension_op(gate_name, [])
        .unwrap()
        .into()
}

pub(crate) fn h_gate() -> LeafOp {
    get_gate("H")
}

pub(crate) fn cx_gate() -> LeafOp {
    get_gate("CX")
}

#[allow(unused)]
pub(crate) fn measure() -> LeafOp {
    get_gate("Measure")
}

pub(crate) fn rz_f64() -> LeafOp {
    get_gate("RzF64")
}

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

use hugr::{
    builder::{DFGBuilder, Dataflow, DataflowHugr},
    extension::prelude::{qb_t, Barrier},
    hugr::hugrmut::HugrMut,
    types::Signature,
    Node, OutgoingPort,
};

use super::{LowerTk2Error, QSystemOpBuilder};

/// Insert [RuntimeBarrier] after every [Barrier] in the Hugr.
pub(super) fn insert_runtime_barrier(
    hugr: &mut impl HugrMut,
    node: Node,
    barrier: Barrier,
) -> Result<(), LowerTk2Error> {
    let qubit_ports: Vec<_> = barrier
        .type_row
        .iter()
        .enumerate()
        .filter_map(|(i, ty)| (ty == &qb_t()).then_some(i))
        .collect();
    // TODO extend with unpacked array ports
    let targets: Vec<Vec<_>> = qubit_ports
        .iter()
        .map(|i| hugr.linked_inputs(node, OutgoingPort::from(*i)).collect())
        .collect();

    let mut barr_builder = DFGBuilder::new(Signature::new_endo(vec![qb_t(); qubit_ports.len()]))?;
    let outs = barr_builder.build_wrapped_barrier(barr_builder.input_wires())?;
    let barr_hugr = barr_builder.finish_hugr_with_outputs(outs)?;

    // TODO use SimpleReplace once order bug fixed https://github.com/CQCL/hugr/issues/1974
    let parent = hugr.get_parent(node).expect("Barrier can't be root.");
    let insert_res = hugr.insert_hugr(parent, barr_hugr);
    let r_bar_n = insert_res.new_root;

    for (r_bar_port, (&bar_out, targets)) in qubit_ports.iter().zip(targets).enumerate() {
        hugr.connect(node, bar_out, r_bar_n, r_bar_port);
        for (targ_n, targ_p) in targets {
            hugr.disconnect(targ_n, targ_p);
            hugr.connect(r_bar_n, r_bar_port, targ_n, targ_p);
        }
    }

    Ok(())
}

#[cfg(test)]
mod test {
    use super::*;
    use crate::extension::qsystem::{self, lower_tk2_op};
    use hugr::{
        builder::DFGBuilder,
        extension::prelude::{bool_t, qb_t},
    };
    use rstest::rstest;

    #[rstest]
    #[case(vec![qb_t(), qb_t()])]
    #[case(vec![qb_t(), qb_t(), bool_t()])]
    fn test_barrier(#[case] type_row: Vec<hugr::types::Type>) {
        // build a dfg with a barrier

        use hugr::{
            ops::{handle::NodeHandle, NamedOp},
            HugrView,
        };

        let (mut h, barr_n) = {
            let mut b = DFGBuilder::new(Signature::new_endo(type_row)).unwrap();

            let barr_n = b.add_barrier(b.input_wires()).unwrap();

            (
                b.finish_hugr_with_outputs(barr_n.outputs()).unwrap(),
                barr_n.node(),
            )
        };

        // lower barrier to barrier + runtime barrier
        let lowered = lower_tk2_op(&mut h).unwrap_or_else(|e| panic!("{}", e));
        h.validate().unwrap_or_else(|e| panic!("{}", e));
        assert!(matches!(&lowered[..], [n] if barr_n == *n));

        let _barr_op: Barrier = h.get_optype(barr_n).cast().unwrap();

        // dfg containing runtime barrier wrapped by array construction/destruction
        let dfg = h.output_neighbours(barr_n).next().unwrap();
        assert!(h.get_optype(dfg).is_dfg());

        let r_barr = h.children(dfg).nth(3).unwrap(); // I, O, new_array, barrier
        assert!(h
            .get_optype(r_barr)
            .as_extension_op()
            .unwrap()
            .name()
            .contains(qsystem::RUNTIME_BARRIER_NAME.as_str()));
    }
}

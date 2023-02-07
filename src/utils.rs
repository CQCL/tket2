use crate::circuit::circuit::UnitID;

pub(crate) fn n_qbs(n: u32) -> Vec<UnitID> {
    (0..n)
        .map(|i| UnitID::Qubit {
            reg_name: "q".into(),
            index: vec![i],
        })
        .collect()
}

#[allow(dead_code)]
pub(crate) fn print_circ(c: &crate::circuit::circuit::Circuit) {
    println!("{}", c.dot_string());
}

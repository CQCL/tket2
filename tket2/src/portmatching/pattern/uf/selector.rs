use delegate::delegate;
use hugr::HugrView;
use portmatching::{self as pm};

use crate::{
    portmatching::{
        branch::find_shared_class, indexing::HugrVariableValue, BranchSelector, Constraint,
    },
    Circuit,
};

pub struct BranchSelectorUf(BranchSelector);

impl pm::BranchSelector for BranchSelectorUf {
    type Key = <BranchSelector as pm::BranchSelector>::Key;

    delegate! {
        to self.0 {
            fn required_bindings(&self) -> &[Self::Key];
        }
    }
}

impl<H: HugrView> pm::EvaluateBranchSelector<Circuit<H>, HugrVariableValue> for BranchSelectorUf {
    delegate! {
        to self.0 {
            fn eval(&self, bindings: &[Option<HugrVariableValue>], data: &Circuit<H>) -> Vec<usize>;
        }
    }
}

impl pm::CreateBranchSelector<Constraint> for BranchSelectorUf {
    fn create_branch_selector(constraints: Vec<Constraint>) -> Self {
        let class = find_shared_class(&constraints).expect("no shared branch class");

        use crate::portmatching::branch::BranchClass::*;
        match class {
            IsOpEqualClass(_)
            | IsLinearWireSinkClass(_)
            | OccupyOutgoingPortClass(_, _)
            | OccupyIncomingPortClass(_, _)
            | IsWireSourceClass(_) => {
                let sel = BranchSelector::new_det(&constraints);
                Self(sel)
            }
            IsDistinctFromClass(_) => {
                let sel = BranchSelector::new_dominant(&constraints);
                Self(sel)
            }
        }
    }
}

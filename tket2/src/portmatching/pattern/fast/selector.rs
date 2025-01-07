use delegate::delegate;
use hugr::HugrView;
use portmatching::{self as pm};

use crate::{
    portmatching::{
        branch::find_shared_class, indexing::HugrVariableValue, BranchSelector, Constraint,
    },
    Circuit,
};

#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct BranchSelectorFast(BranchSelector);

impl pm::BranchSelector for BranchSelectorFast {
    type Key = <BranchSelector as pm::BranchSelector>::Key;

    delegate! {
        to self.0 {
            fn required_bindings(&self) -> &[Self::Key];
        }
    }
}

impl<H: HugrView> pm::EvaluateBranchSelector<Circuit<H>, HugrVariableValue> for BranchSelectorFast {
    delegate! {
        to self.0 {
            fn eval(&self, bindings: &[Option<HugrVariableValue>], data: &Circuit<H>) -> Vec<usize>;
        }
    }
}

impl pm::branch_selector::DisplayBranchSelector for BranchSelectorFast {
    delegate! {
        to self.0 {
            fn fmt_class(&self) -> String;

            fn fmt_nth_constraint(&self, n: usize) -> String;
        }
    }
}

impl pm::CreateBranchSelector<Constraint> for BranchSelectorFast {
    fn create_branch_selector(constraints: Vec<Constraint>) -> Self {
        let class = find_shared_class(&constraints).expect("no shared branch class");

        use crate::portmatching::branch::BranchClass::*;
        match class {
            IsOpEqualClass(_) => {
                let sel = BranchSelector::new_det(&constraints);
                Self(sel)
            }
            IsLinearWireSinkClass(_)
            | OccupyOutgoingPortClass(_, _)
            | OccupyIncomingPortClass(_, _)
            | IsWireSourceClass(_) => {
                let sel = BranchSelector::new_non_det(&constraints);
                Self(sel)
            }
            IsDistinctFromClass(_) => {
                let sel = BranchSelector::new_dominant(&constraints);
                Self(sel)
            }
        }
    }
}

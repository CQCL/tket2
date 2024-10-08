//! Definitions for decoding parameter expressions from pytket operations

use hugr::ops::OpType;

use hugr::std_extensions::arithmetic::float_ops::FloatOps;
use pest::iterators::{Pair, Pairs};
use pest::pratt_parser::PrattParser;
use pest::Parser;
use pest_derive::Parser;

/// The parsed AST for a pytket operation parameter.
///
/// The leafs of the AST are either a constant value, a variable name, or an
/// unrecognized sympy expression.
///
/// Return type of [`parse_pytket_param`].
//
// TODO: We have to decide how to parse non-trivial sympy expressions.
//       Either by implementing a parser, or calling out to python.
//       For now, we just create a [`SympyOp`].
#[derive(Debug, Clone, PartialEq)]
pub(in crate::serialize::pytket) enum PytketParam<'a> {
    /// A constant value that can be loaded directly.
    Constant(f64),
    /// A variable that should be routed as an input.
    InputVariable {
        /// The variable name.
        name: &'a str,
    },
    /// Unrecognized sympy expression.
    /// Will be emitted as a [`SympyOp`].
    Sympy(&'a str),
    /// An operation on some nested expressions.
    Operation {
        op: OpType,
        args: Vec<PytketParam<'a>>,
    },
}

/// Parse a TKET1 parameter, add
///
/// Angle parameters in TKET1 are encoded as a number of half-turns,
/// whereas HUGR uses radians.
#[inline]
pub(in crate::serialize::pytket) fn parse_pytket_param(param: &str) -> PytketParam<'_> {
    let Ok(mut parsed) = ParamParser::parse(Rule::parameter, param) else {
        // The parameter could not be parsed, so we just return it as an opaque sympy expression.
        return PytketParam::Sympy(param);
    };
    let parsed = parsed
        .next()
        .expect("The `parameter` rule can only be matched once.");

    assert_eq!(
        parsed.as_rule(),
        Rule::expr,
        "`parameter` does not contain rule {:?}",
        parsed.as_rule()
    );

    parse_infix_ops(parsed.into_inner())
}

#[derive(Parser)]
#[grammar = "serialize/pytket/param/param.pest"]
struct ParamParser;

lazy_static::lazy_static! {
    /// Precedence parser used to define the order of infix operations.
    ///
    /// Based on the calculator example from `pest`.
    /// https://pest.rs/book/examples/calculator.html
    static ref PRATT_PARSER: PrattParser<Rule> = {
        use pest::pratt_parser::{Assoc::*, Op};
        use Rule::*;

        // Precedence is defined lowest to highest
        PrattParser::new()
            // Addition and subtract have equal precedence
            .op(Op::infix(add, Left) | Op::infix(subtract, Left))
            .op(Op::infix(multiply, Left) | Op::infix(divide, Left))
            .op(Op::infix(power, Left))
    };
}

/// Parse a match of the [`Rule::expr`] rule.
///
/// This takes a sequence of rule matches alternating [`Rule::term`]s and infix operations.
fn parse_infix_ops(pairs: Pairs<'_, Rule>) -> PytketParam<'_> {
    PRATT_PARSER
        .map_primary(|primary| parse_term(primary))
        .map_infix(|lhs, op, rhs| {
            let op = match op.as_rule() {
                Rule::add => FloatOps::fadd,
                Rule::subtract => FloatOps::fsub,
                Rule::multiply => FloatOps::fmul,
                Rule::divide => FloatOps::fdiv,
                Rule::power => FloatOps::fpow,
                rule => unreachable!("Expr::parse expected infix operation, found {:?}", rule),
            }
            .into();
            PytketParam::Operation {
                op,
                args: vec![lhs, rhs],
            }
        })
        .parse(pairs)
}

/// Parse a match of the silent [`Rule::term`] rule.
fn parse_term(pair: Pair<'_, Rule>) -> PytketParam<'_> {
    match pair.as_rule() {
        Rule::expr => parse_infix_ops(pair.into_inner()),
        Rule::num => parse_number(pair),
        Rule::unary_minus => PytketParam::Operation {
            op: FloatOps::fneg.into(),
            args: vec![parse_term(pair.into_inner().next().unwrap())],
        },
        Rule::function_call => parse_function_call(pair),
        Rule::ident => PytketParam::InputVariable {
            name: pair.as_str(),
        },
        rule => unreachable!("Term::parse expected a term, found {:?}", rule),
    }
}

/// Parse a match of the [`Rule::num`] rule.
fn parse_number(pair: Pair<'_, Rule>) -> PytketParam<'_> {
    let num = pair.as_str();
    let half_turns = num
        .parse::<f64>()
        .unwrap_or_else(|_| panic!("`num` rule matched invalid number \"{num}\""));
    PytketParam::Constant(half_turns)
}

/// Parse a match of the [`Rule::function_call`] rule.
fn parse_function_call(pair: Pair<'_, Rule>) -> PytketParam<'_> {
    let pair_str = pair.as_str();
    let mut args = pair.into_inner();
    let name = args
        .next()
        .expect("Function call must have a name")
        .as_str();
    let Some(op) = get_optype(name) else {
        // Unrecognized function name.
        // Treat it as an opaque sympy expression.
        return PytketParam::Sympy(pair_str);
    };

    let args = args.map(|arg| parse_term(arg)).collect::<Vec<_>>();
    PytketParam::Operation { op, args }
}

/// Returns the optype given a function name.
///
/// If the function name is not recognized, returns `None`.
fn get_optype(name: &str) -> Option<OpType> {
    match name {
        "max" => Some(FloatOps::fmax.into()),
        "min" => Some(FloatOps::fmin.into()),
        "abs" => Some(FloatOps::fabs.into()),
        "floor" => Some(FloatOps::ffloor.into()),
        "ceil" => Some(FloatOps::fceil.into()),
        "round" => Some(FloatOps::fround.into()),
        _ => None,
    }
}

#[cfg(test)]
mod test {
    use super::*;
    use rstest::rstest;

    #[rstest]
    #[case::int("42", PytketParam::Constant(42.0))]
    #[case::float("42.37", PytketParam::Constant(42.37))]
    #[case::float_pointless("37.", PytketParam::Constant(37.))]
    #[case::exp("42e4", PytketParam::Constant(42e4))]
    #[case::neg("-42.55", PytketParam::Constant(-42.55))]
    #[case::parens("(42)", PytketParam::Constant(42.))]
    #[case::var("f64", PytketParam::InputVariable{name: "f64"})]
    #[case::add("42 + f64", PytketParam::Operation {
        op: FloatOps::fadd.into(),
        args: vec![PytketParam::Constant(42.), PytketParam::InputVariable{name: "f64"}]
    })]
    #[case::max("max(42, f64)", PytketParam::Operation {
        op: FloatOps::fmax.into(),
        args: vec![PytketParam::Constant(42.), PytketParam::InputVariable{name: "f64"}]
    })]
    #[case::minus("-f64", PytketParam::Operation {
        op: FloatOps::fneg.into(),
        args: vec![PytketParam::InputVariable{name: "f64"}]
    })]
    #[case::unknown("unknown_op(42, f64)", PytketParam::Sympy("unknown_op(42, f64)"))]
    #[case::nested("max(42, unknown_op(37))", PytketParam::Operation {
        op: FloatOps::fmax.into(),
        args: vec![PytketParam::Constant(42.), PytketParam::Sympy("unknown_op(37)")]
    })]
    fn parse_param(#[case] param: &str, #[case] expected: PytketParam) {
        let parsed = parse_pytket_param(param);
        assert_eq!(parsed, expected)
    }
}

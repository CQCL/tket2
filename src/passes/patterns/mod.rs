use crate::circuit::circuit::CircuitRewrite;

pub mod fixed;
pub mod linear_fixed;

struct MatchFail();

pub trait PatternMatcher<'g> {
    type Match;
    fn find_matches(&self) -> Box<dyn Iterator<Item = Self::Match> + 'g>;
    fn into_matches(self) -> Box<dyn Iterator<Item = Self::Match> + 'g>;
}

pub trait PatternRewriter<'g> {
    fn find_rewrites(&self) -> Box<dyn Iterator<Item = CircuitRewrite> + 'g>;
    fn into_rewrites(self) -> Box<dyn Iterator<Item = CircuitRewrite> + 'g>;
}

/*pub trait PatternMatcherPar<'g> : PatternMatcher<'g> {
    fn find_par_matches(&self) -> Box<dyn ParallelIterator<Item = Match>>;
}*/

pub struct StructRewriter<'g, M, F>
where
    M: PatternMatcher<'g>,
    F: Fn(M::Match) -> CircuitRewrite,
{
    matcher: M,
    rewrite: F,
}

impl<'g, M, F> StructRewriter<'g, M, F>
where
    M: PatternMatcher<'g>,
    F: Fn(M::Match) -> CircuitRewrite,
{
    pub fn new(matcher: M, rewrite: F) -> Self {
        Self { matcher, rewrite }
    }
}

impl<'g, M, F> PatternRewriter<'g> for StructRewriter<'g, M, F>
where
    M: PatternMatcher<'g>,
    F: Fn(M::Match) -> CircuitRewrite,
{
    fn find_rewrites(&self) -> Box<dyn Iterator<Item = CircuitRewrite> + 'g> {
        Box::new(self.matcher.find_matches().map(self.rewrite))
    }

    fn into_rewrites(self) -> Box<dyn Iterator<Item = CircuitRewrite> + 'g> {
        Box::new(self.matcher.into_matches().map(self.rewrite))
    }
}

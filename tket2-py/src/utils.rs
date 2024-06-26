//! Utility functions for the python interface.

/// A trait for types wrapping rust errors that may be converted into python exception.
///
/// In addition to raw errors, this is implemented for wrapper types such as `Result`.
/// [`ConvertPyErr::convert_errors`] will be called on the internal error type.
pub trait ConvertPyErr {
    /// The output type after conversion.
    type Output;

    /// Convert any internal errors to python errors.
    fn convert_pyerrs(self) -> Self::Output;
}

impl ConvertPyErr for pyo3::PyErr {
    type Output = Self;

    fn convert_pyerrs(self) -> Self::Output {
        self
    }
}

impl<T, E> ConvertPyErr for Result<T, E>
where
    E: ConvertPyErr,
{
    type Output = Result<T, E::Output>;

    fn convert_pyerrs(self) -> Self::Output {
        self.map_err(|e| e.convert_pyerrs())
    }
}

macro_rules! create_py_exception {
    ($err:path, $py_err:ident, $doc:expr) => {
        pyo3::create_exception!(tket2, $py_err, pyo3::exceptions::PyException, $doc);

        impl $crate::utils::ConvertPyErr for $err {
            type Output = pyo3::PyErr;

            fn convert_pyerrs(self) -> Self::Output {
                $py_err::new_err(<Self as std::string::ToString>::to_string(&self))
            }
        }
    };
}
pub(crate) use create_py_exception;
use itertools::Itertools;

/// Convert an iterator of one type into vector of another type.
pub fn into_vec<T, S: From<T>>(v: impl IntoIterator<Item = T>) -> Vec<S> {
    v.into_iter().map_into().collect()
}

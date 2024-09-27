use hugr::Hugr;
use pyo3::{types::{PyAnyMethods, PyString, PyList}, PyResult, Python};
use strum::ParseError;

// fn init_pyo3_with_venv(env_dir: &str) {
//     use std::mem::size_of;
//     use std::ptr::addr_of_mut;

//     use libc::wchar_t;
//     use pyo3::ffi::*;

//     unsafe {
//         fn check_exception(status: PyStatus, config: &mut PyConfig) {
//             unsafe {
//                 if PyStatus_Exception(status) != 0 {
//                     PyConfig_Clear(config);
//                     if PyStatus_IsExit(status) != 0 {
//                         std::process::exit(status.exitcode);
//                     }
//                     Py_ExitStatusException(status);
//                 }
//             }
//         }

//         let mut config = std::mem::zeroed::<PyConfig>();
//         PyConfig_InitPythonConfig(&mut config);

//         config.install_signal_handlers = 0;

//         // `wchar_t` is a mess.
//         let env_dir_utf16;
//         let env_dir_utf32;
//         let env_dir_ptr;
//         if size_of::<wchar_t>() == size_of::<u16>() {
//             env_dir_utf16 = env_dir
//                 .encode_utf16()
//                 .chain(std::iter::once(0))
//                 .collect::<Vec<_>>();
//             env_dir_ptr = env_dir_utf16.as_ptr().cast::<wchar_t>();
//         } else if size_of::<wchar_t>() == size_of::<u32>() {
//             env_dir_utf32 = env_dir
//                 .chars()
//                 .chain(std::iter::once('\0'))
//                 .collect::<Vec<_>>();
//             env_dir_ptr = env_dir_utf32.as_ptr().cast::<wchar_t>();
//         } else {
//             panic!("unknown encoding for `wchar_t`");
//         }
//         check_exception(
//             PyConfig_SetString(
//                 addr_of_mut!(config),
//                 addr_of_mut!(config.prefix),
//                 env_dir_ptr,
//             ),
//             &mut config,
//         );

//         check_exception(Py_InitializeFromConfig(&config), &mut config);

//         PyConfig_Clear(&mut config);

//         PyEval_SaveThread();
//     }
// }

/// TODO docs
pub fn parse_expr(expr: &str, vars: impl AsRef<[String]>) -> Result<Hugr, Box<dyn std::error::Error>> {
    // if let Ok(env_dir) = std::env::var("VIRTUAL_ENV") {
    //     init_pyo3_with_venv(&env_dir);
    // } else {
        pyo3::prepare_freethreaded_python();
    // }
    pyo3::prepare_freethreaded_python();
    let json: PyResult<String> = Python::with_gil(|py| {
//         py.run_bound(r"
// import sys
// print(sys.executable, sys.path, sys.prefix)
// help('modules')
// ", None, None)?;
        let expr = PyString::new_bound(py, expr);
        let vars = PyList::new_bound(py, vars.as_ref());
        let expr_module = py.import_bound("tket2.expr")?;
        let parse_expr_fn = expr_module.getattr("parse_expr")?;
        let result = parse_expr_fn.call1((expr,vars))?;
        result.extract()
    });
    Ok(serde_json::from_str(&json?)?)
}

mod test {
    use rstest::rstest;
    #[rstest]
    #[case("1 + 2", [])]
    #[case("a * b", ["a".to_string(), "b".to_string()])]
    fn test_parse_expr(#[case] expr: String, #[case] vars: impl AsRef<[String]>) {
        let hugr = super::parse_expr(&expr, vars).unwrap();
    }
}

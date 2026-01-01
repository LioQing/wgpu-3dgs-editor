#[macro_export]
macro_rules! inline_wesl_pkg {
    (use $deps:expr, $name:literal => $module_name:literal: $($body:tt)+) => {
        wesl::CodegenPkg {
            crate_name: $name,
            root: &wesl::CodegenModule {
                name: $module_name,
                source: {
                    stringify!($($body)+)
                },
                submodules: &[],
            },
            dependencies: &$deps,
        }
    };
    (use $deps:expr, $name:literal: $($body:tt)+) => {
        inline_wesl_pkg!(use $deps, $name => $name: $($body)+)
    };
    ($name:literal: $($body:tt)+) => {
        inline_wesl_pkg!(use [], $name => $name: $($body)+)
    };
}

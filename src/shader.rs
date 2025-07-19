use wesl::PkgModule;

pub struct Mod;

impl PkgModule for Mod {
    fn name(&self) -> &'static str {
        "wgpu_3dgs_editor"
    }

    fn source(&self) -> &'static str {
        ""
    }

    fn submodules(&self) -> &[&dyn PkgModule] {
        static SUBMODULES: &[&dyn PkgModule] = &[&selection::Mod];
        SUBMODULES
    }

    fn submodule(&self, name: &str) -> Option<&dyn PkgModule> {
        match name {
            "selection" => Some(&selection::Mod),
            // TODO: Wait for wesl-rs support nested modules
            "ops" => Some(&selection::ops::Mod),
            "primitive_ops" => Some(&selection::primitive_ops::Mod),
            "utils" => Some(&selection::utils::Mod),
            "sphere" => Some(&selection::sphere::Mod),
            _ => None,
        }
    }
}

macro_rules! submodule {
    ($name:ident $(, $dir:literal)?) => {
        paste::paste! {
            pub mod $name {
                pub struct Mod;

                impl wesl::PkgModule for Mod {
                    fn name(&self) -> &'static str {
                        stringify!($name)
                    }

                    fn source(&self) -> &'static str {
                        include_str!(concat!("shader/", $($dir,)? stringify!($name), ".wesl"))
                    }

                    fn submodules(&self) -> &[&dyn wesl::PkgModule] {
                        &[]
                    }

                    fn submodule(&self, _name: &str) -> Option<&dyn wesl::PkgModule> {
                        None
                    }
                }
            }
        }
    };
}

pub mod selection {
    use super::*;

    macro_rules! selection_submodule {
        ($name:ident) => {
            submodule!($name, "selection/");
        };
    }

    pub struct Mod;

    impl PkgModule for Mod {
        fn name(&self) -> &'static str {
            "selection"
        }

        fn source(&self) -> &'static str {
            ""
        }

        fn submodules(&self) -> &[&dyn PkgModule] {
            static SUBMODULES: &[&dyn PkgModule] =
                &[&ops::Mod, &primitive_ops::Mod, &utils::Mod, &sphere::Mod];
            SUBMODULES
        }

        fn submodule(&self, name: &str) -> Option<&dyn PkgModule> {
            match name {
                "ops" => Some(&ops::Mod),
                "primitive_ops" => Some(&primitive_ops::Mod),
                "utils" => Some(&utils::Mod),
                "sphere" => Some(&sphere::Mod),
                _ => None,
            }
        }
    }

    selection_submodule!(ops);
    selection_submodule!(primitive_ops);
    selection_submodule!(utils);
    selection_submodule!(sphere);
}

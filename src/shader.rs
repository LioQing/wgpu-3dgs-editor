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

    pub struct Mod;

    impl PkgModule for Mod {
        fn name(&self) -> &'static str {
            "selection"
        }

        fn source(&self) -> &'static str {
            ""
        }

        fn submodules(&self) -> &[&dyn PkgModule] {
            static SUBMODULES: &[&dyn PkgModule] = &[&ops::Mod, &primitive_ops::Mod];
            SUBMODULES
        }

        fn submodule(&self, name: &str) -> Option<&dyn PkgModule> {
            match name {
                _ => None,
            }
        }
    }

    submodule!(ops, "selection/");
    submodule!(primitive_ops, "selection/");
}

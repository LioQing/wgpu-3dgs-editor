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
            "sphere" => Some(&selection::sphere::Mod),
            "box" => Some(&selection::r#box::Mod),
            _ => None,
        }
    }
}

macro_rules! submodule {
    ($name:ident $(, $dir:literal)? override $mod_name:ident) => {
        paste::paste! {
            pub mod $mod_name {
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
    ($name:ident $(, $dir:literal)?) => {
        submodule!($name $(, $dir)? override $name);
    };
}

pub mod selection {
    use super::*;

    macro_rules! selection_submodule {
        ($name:ident) => {
            submodule!($name, "selection/");
        };
        ($name:ident override $mod_name:ident) => {
            submodule!($name, "selection/" override $mod_name);
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
                &[&ops::Mod, &primitive_ops::Mod, &sphere::Mod, &r#box::Mod];
            SUBMODULES
        }

        fn submodule(&self, name: &str) -> Option<&dyn PkgModule> {
            match name {
                "ops" => Some(&ops::Mod),
                "primitive_ops" => Some(&primitive_ops::Mod),
                "sphere" => Some(&sphere::Mod),
                "box" => Some(&r#box::Mod),
                _ => None,
            }
        }
    }

    selection_submodule!(ops);
    selection_submodule!(primitive_ops);
    selection_submodule!(sphere);
    selection_submodule!(box override r#box);
}

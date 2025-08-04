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
        static SUBMODULES: &[&dyn PkgModule] = &[&selection::Mod, &modifier::Mod];
        SUBMODULES
    }

    fn submodule(&self, name: &str) -> Option<&dyn PkgModule> {
        match name {
            "selection" => Some(&selection::Mod),
            "modifier" => Some(&modifier::Mod),
            _ => selection::Mod
                .submodule(name)
                .or_else(|| modifier::Mod.submodule(name)),
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
                &[&consts::Mod, &primitive_ops::Mod, &sphere::Mod, &r#box::Mod];
            SUBMODULES
        }

        fn submodule(&self, name: &str) -> Option<&dyn PkgModule> {
            match name {
                "consts" => Some(&consts::Mod),
                "primitive_ops" => Some(&primitive_ops::Mod),
                "sphere" => Some(&sphere::Mod),
                "box" => Some(&r#box::Mod),
                _ => None,
            }
        }
    }

    selection_submodule!(consts);
    selection_submodule!(primitive_ops);
    selection_submodule!(sphere);
    selection_submodule!(box override r#box);
}

pub mod modifier {
    use super::*;

    macro_rules! modifier_submodule {
        ($name:ident) => {
            submodule!($name, "modifier/");
        };
        ($name:ident override $mod_name:ident) => {
            submodule!($name, "modifier/" override $mod_name);
        };
    }

    pub struct Mod;

    impl PkgModule for Mod {
        fn name(&self) -> &'static str {
            "modifier"
        }

        fn source(&self) -> &'static str {
            ""
        }

        fn submodules(&self) -> &[&dyn PkgModule] {
            static SUBMODULES: &[&dyn PkgModule] =
                &[&modifier_consts::Mod, &utils::Mod, &basic::Mod];
            SUBMODULES
        }

        fn submodule(&self, name: &str) -> Option<&dyn PkgModule> {
            match name {
                "modifier_consts" => Some(&modifier_consts::Mod),
                "utils" => Some(&utils::Mod),
                "basic" => Some(&basic::Mod),
                _ => None,
            }
        }
    }

    modifier_submodule!(modifier_consts);
    modifier_submodule!(utils);
    modifier_submodule!(basic);
}

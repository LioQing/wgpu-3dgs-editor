use wesl::{Pkg, PkgModule};

use crate::core;

pub const PACKAGE: Pkg = Pkg {
    crate_name: "wgpu-3dgs-editor",
    root: &MODULE,
    dependencies: &[&core::shader::PACKAGE],
};

pub const MODULE: PkgModule = PkgModule {
    name: "wgpu_3dgs_editor",
    source: "",
    submodules: &[&selection::MODULE, &modifier::MODULE],
};

pub mod selection {
    use super::PkgModule;

    pub const MODULE: PkgModule = PkgModule {
        name: "selection",
        source: "",
        submodules: &[
            &consts::MODULE,
            &primitive_ops::MODULE,
            &sphere::MODULE,
            &r#box::MODULE,
        ],
    };

    pub mod consts {
        use super::PkgModule;

        pub const MODULE: PkgModule = PkgModule {
            name: "consts",
            source: include_str!("shader/selection/consts.wesl"),
            submodules: &[],
        };
    }

    pub mod primitive_ops {
        use super::PkgModule;

        pub const MODULE: PkgModule = PkgModule {
            name: "primitive_ops",
            source: include_str!("shader/selection/primitive_ops.wesl"),
            submodules: &[],
        };
    }

    pub mod sphere {
        use super::PkgModule;

        pub const MODULE: PkgModule = PkgModule {
            name: "sphere",
            source: include_str!("shader/selection/sphere.wesl"),
            submodules: &[],
        };
    }

    pub mod r#box {
        use super::PkgModule;

        pub const MODULE: PkgModule = PkgModule {
            name: "box",
            source: include_str!("shader/selection/box.wesl"),
            submodules: &[],
        };
    }
}

pub mod modifier {
    use super::PkgModule;

    pub const MODULE: PkgModule = PkgModule {
        name: "modifier",
        source: "",
        submodules: &[&modifier_consts::MODULE, &utils::MODULE, &basic::MODULE],
    };

    pub mod modifier_consts {
        use super::PkgModule;

        pub const MODULE: PkgModule = PkgModule {
            name: "modifier_consts",
            source: include_str!("shader/modifier/modifier_consts.wesl"),
            submodules: &[],
        };
    }

    pub mod utils {
        use super::PkgModule;

        pub const MODULE: PkgModule = PkgModule {
            name: "utils",
            source: include_str!("shader/modifier/utils.wesl"),
            submodules: &[],
        };
    }

    pub mod basic {
        use super::PkgModule;

        pub const MODULE: PkgModule = PkgModule {
            name: "basic",
            source: include_str!("shader/modifier/basic.wesl"),
            submodules: &[],
        };
    }
}

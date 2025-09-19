//! Shader modules for the [`wesl::Pkg`] `wgpu-3dgs-editor`.
//!
//! See the documentation of each module for details.

use wesl::{Pkg, PkgModule};

use crate::core;

/// The `wgpu-3dgs-editor` [`wesl::Pkg`].
pub const PACKAGE: Pkg = Pkg {
    crate_name: "wgpu-3dgs-editor",
    root: &MODULE,
    dependencies: &[&core::shader::PACKAGE],
};

/// The root module of the `wgpu-3dgs-editor` package.
pub const MODULE: PkgModule = PkgModule {
    name: "wgpu_3dgs_editor",
    source: "",
    submodules: &[&selection::MODULE, &modifier::MODULE],
};

pub mod selection {
    use super::PkgModule;

    /// The root module of the selection shaders.
    pub const MODULE: PkgModule = PkgModule {
        name: "selection",
        source: "",
        submodules: &[
            &consts::MODULE,
            &primitive::MODULE,
            &sphere::MODULE,
            &r#box::MODULE,
        ],
    };

    #[doc = concat!("```wgsl\n", include_str!("shader/selection/consts.wesl"), "\n```")]
    pub mod consts {
        use super::PkgModule;

        pub const MODULE: PkgModule = PkgModule {
            name: "consts",
            source: include_str!("shader/selection/consts.wesl"),
            submodules: &[],
        };
    }

    #[doc = concat!("```wgsl\n", include_str!("shader/selection/primitive.wesl"), "\n```")]
    pub mod primitive {
        use super::PkgModule;

        pub const MODULE: PkgModule = PkgModule {
            name: "primitive",
            source: include_str!("shader/selection/primitive.wesl"),
            submodules: &[],
        };
    }

    #[doc = concat!("```wgsl\n", include_str!("shader/selection/sphere.wesl"), "\n```")]
    pub mod sphere {
        use super::PkgModule;

        pub const MODULE: PkgModule = PkgModule {
            name: "sphere",
            source: include_str!("shader/selection/sphere.wesl"),
            submodules: &[],
        };
    }

    #[doc = concat!("```wgsl\n", include_str!("shader/selection/box.wesl"), "\n```")]
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

    /// The root module of the modifier shaders.
    pub const MODULE: PkgModule = PkgModule {
        name: "modifier",
        source: "",
        submodules: &[&modifier_consts::MODULE, &utils::MODULE, &basic::MODULE],
    };

    #[doc = concat!("```wgsl\n", include_str!("shader/modifier/consts.wesl"), "\n```")]
    pub mod modifier_consts {
        use super::PkgModule;

        pub const MODULE: PkgModule = PkgModule {
            name: "consts",
            source: include_str!("shader/modifier/consts.wesl"),
            submodules: &[],
        };
    }

    #[doc = concat!("```wgsl\n", include_str!("shader/modifier/utils.wesl"), "\n```")]
    pub mod utils {
        use super::PkgModule;

        pub const MODULE: PkgModule = PkgModule {
            name: "utils",
            source: include_str!("shader/modifier/utils.wesl"),
            submodules: &[],
        };
    }

    #[doc = concat!("```wgsl\n", include_str!("shader/modifier/basic.wesl"), "\n```")]
    pub mod basic {
        use super::PkgModule;

        pub const MODULE: PkgModule = PkgModule {
            name: "basic",
            source: include_str!("shader/modifier/basic.wesl"),
            submodules: &[],
        };
    }
}

# Changelog

Please also check out the [`wgpu-3dgs-viewer` changelog](https://github.com/LioQing/wgpu-3dgs-viewer/blob/master/CHANGELOG.md) and [`wgpu-3dgs-core` changelog](https://github.com/LioQing/wgpu-3dgs-core/blob/master/CHANGELOG.md).

## [0.8.0](https://crates.io/crates/wgpu-3dgs-editor/0.8.0) - 2026-08-22

### Changed

- ⚡ Upgrade `wgpu` to 30.0, `wesl` to 0.4, `glam` to 0.33, `bitflags` to 2.13, and a few other internal dependencies. [#14](https://github.com/LioQing/wgpu-3dgs-editor/pull/14)

## [0.7.0](https://crates.io/crates/wgpu-3dgs-editor/0.7.0) - 2026-05-15

### Changed

- ⚡ Upgrade `wgpu` to 29.0, `glam` to 0.32, and a few other internal dependencies. [#13](https://github.com/LioQing/wgpu-3dgs-editor/pull/13)
- 📝 Remove the active development breaking changes warning in README.md. [#13](https://github.com/LioQing/wgpu-3dgs-editor/pull/13)

## [0.6.0](https://crates.io/crates/wgpu-3dgs-editor/0.6.0) - 2026-01-11

### Added

- 🤖 CI workflow. [#12](https://github.com/LioQing/wgpu-3dgs-editor/pull/12)

### Changed

- ⚡ Upgrade `wgpu` to 28.0, `wesl` to 0.3, `half` to 2.7, and `bytemuck` to 1.24. [#11](https://github.com/LioQing/wgpu-3dgs-editor/pull/11)

## [0.5.0](https://crates.io/crates/wgpu-3dgs-editor/0.5.0) - 2025-12-30

### Added

- 🎨 Add `BasicColorRgbOrHsvModifiersPod` to represent RGB or HSV color modifiers more explicitly. [#6](https://github.com/LioQing/wgpu-3dgs-editor/pull/6)

### Changed

- ⚡ Upgrade `wgpu` to 27.0 and `bitflags` to 2.10. [#10](https://github.com/LioQing/wgpu-3dgs-editor/pull/10)
- 📂 Examples now support both PLY and SPZ file formats with automatic detection. [#9](https://github.com/LioQing/wgpu-3dgs-editor/pull/9)

### Breaking Changes

- Remove `SelectionModifier::apply_with` method. [#8](https://github.com/LioQing/wgpu-3dgs-editor/pull/8)
- `BasicColorModifiersPod::rgb_or_hsv` now uses `BasicColorRgbOrHsvModifiersPod` instead of plain `Vec3`. [#6](https://github.com/LioQing/wgpu-3dgs-editor/pull/6)

## [0.4.1](https://crates.io/crates/wgpu-3dgs-editor/0.4.1) - 2025-10-05

### Added

- 🟰 Add Clone for modifier buffers.
- 📑 Add example modules documentations.
- ✅ Add coverage script and reports.

### Changed

- 🔨 `SelectionBuffer::DEFAULT_USAGES` now includes `COPY_DST`.
- 🐛 Fix `SelectionExpr::as_u32` mistakenly swapped `Difference` and `SymmetricDifference`.

## [0.4.0](https://crates.io/crates/wgpu-3dgs-editor/0.4.0) - 2025-09-20

### Added

- 🛬 Things are moved from `wgpu-3dgs-viewer` to here.
- 1️⃣ Unified how selection and modification are done on Gaussians, take a look at `Modifier`, `SelectionModifier`, or `NonDestructiveModifier` for the highest level API we currently provide.

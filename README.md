# 3D Gaussian Splatting Editor

...written in Rust using [wgpu](https://wgpu.rs/).

[![Crates.io](https://img.shields.io/crates/v/wgpu-3dgs-editor)](https://crates.io/crates/wgpu-3dgs-editor)
[![Docs.rs](https://img.shields.io/docsrs/wgpu-3dgs-editor)](https://docs.rs/wgpu-3dgs-editor/latest/wgpu_3dgs_editor)
[![Coverage](https://img.shields.io/endpoint?url=https%3A%2F%2Fraw.githubusercontent.com%2FLioQing%2Fwgpu-3dgs-editor%2Frefs%2Fheads%2Fmaster%2Fcoverage%2Fbadge.json)](https://github.com/LioQing/wgpu-3dgs-editor/tree/master/coverage)
[![License](https://img.shields.io/crates/l/wgpu-3dgs-editor)](https://crates.io/crates/wgpu-3dgs-editor)

## Overview

This library provides a set of tools to create and manipulate 3D Gaussian Splatting models. It includes:

- Selecting Gaussians using selection operations and expressions.
  - Primitive operations: union, intersection, difference, symmetric difference, and inversion.
  - Custom operations: define your own selection logic using WGSL shaders.
  - Built-in custom operations: sphere and box selections.
- Modifying Gaussian attributes.
  - Masked modifications: apply changes only to selected Gaussians from selection buffers.
  - Custom modifiers: define your own modification logic using WGSL/WESL shaders.
  - Built-in basic modifier: color adjustments (RGB override or HSV modifications), alpha, contrast, exposure, gamma, and transformations (scale and rotation).
- Shaders
  - WGSL shaders packaged with WESL, you can extend or replace them.

## Usage

This crate provides some basic selection and modifier, but you may also define your own custom selection and modifier using WGSL/WESL shaders.

You may read the documentation of the following traits and structs for more details:

- [`Editor`]: The struct that manages the necessary buffers and applies the selection and modification.
- [`Modifier`]: The trait for modification operations, may be used to define custom modifiers.
- [`BasicModifier`]: A built-in modifier that enables some basic color and transformation modifications.
- [`SelectionBundle`]: A struct can evaluate selection for each Gaussian, may also call custom selection operations.
- [`SelectionModifier`]: A struct that applies selection and then modification, may be used to define custom selection modifiers.
- [`NonDestructiveModifier`]: A struct that applies modifications without changing the original Gaussians, it achieves this by storing an original copy of the Gaussians.

> [!TIP]
>
> The design principles of this crate are to provide modularity and flexibility to the end user of the API, which means exposing low-level WebGPU APIs. However, this means that you have to take care of your code when accessing low-level components. You risk breaking things at run-time if you don't handle them properly.
>
> If you do not want to take the risk, consider using the higher-level wrappers and avoid any instances of passing `wgpu` types into functions.

### Basic Editor

You can use [`Editor`] and [`SelectionModifier::new_with_basic_modifier`] to manage the selection and modification of Gaussians with the built-in basic operations. Here's a simple example:

```rust ignore
use wgpu_3dgs_editor as gs;

// Setup wgpu...

// Read the Gaussians from the .ply file
let gaussians = gs::core::Gaussians::read_from_file(model_path, GaussiansSource::Ply)
    .expect("gaussians");

// Create an editor that creates the necessary buffers, you may also create the buffers manually
let editor = gs::Editor::<GaussianPod>::new(&device, &gaussians);

// Create a basic selection modifier with a box and a sphere selection bundle
let mut basic_selection_modifier =
    gs::SelectionModifier::<GaussianPod, _>::new_with_basic_modifier(
        &device,
        &editor.gaussians_buffer,
        &editor.model_transform_buffer,
        &editor.gaussian_transform_buffer,
        vec![
            gs::SelectionBundle::create_box_bundle::<GaussianPod>(&device),
            gs::SelectionBundle::create_sphere_bundle::<GaussianPod>(&device),
        ],
    );

// Create the buffer and bind group for the box selection's position, rotation, and scale
let box_selection_buffer = gs::InvTransformBuffer::new(&device);
box_selection_buffer.update_with_scale_rot_pos(&queue, box_scale, box_rot, box_pos);

let box_selection_bind_group = basic_selection_modifier.selection.bundles[0]
    .create_bind_group(
        &device,
        1, // The built-in box selection bundle uses index 1 for the inv transform buffer,
           // see documentation of `SelectionBundle::create_box_bundle`
        [box_selection_buffer.buffer().as_entire_binding()],
    )
    .unwrap();

// Create the buffer and bind group for the sphere selection's position, rotation, and scale
let sphere_selection_buffer = gs::InvTransformBuffer::new(&device);
sphere_selection_buffer.update_with_scale_rot_pos(&queue, sphere_scale, sphere_rot, sphere_pos);

let sphere_selection_bind_group = basic_selection_modifier.selection.bundles[1]
    .create_bind_group(
        &device,
        1, // The built-in sphere selection bundle uses index 1 for the inv transform buffer,
           // see documentation of `SelectionBundle::create_sphere_bundle`
        [sphere_selection_buffer.buffer().as_entire_binding()],
    )
    .unwrap();

// Set the selection expression
basic_selection_modifier.selection_expr = gs::SelectionExpr::selection(
    0, // The bundle index for the box selection during creation of the selection bundle
    vec![box_selection_bind_group],
).union( // Combine with other selection expressions using different functions
    gs::SelectionExpr::selection(
        1, // The bundle index for the sphere selection during creation of the selection bundle
        vec![sphere_selection_bind_group],
    ),
);

// Set the modification parameters, here we override the color
basic_selection_modifier
    .modifier
    .basic_color_modifiers_buffer
    .update(
        &queue,
        gs::BasicColorRgbOverrideOrHsvModifiersPod::new_rgb_override(color),
        alpha,
        contrast,
        exposure,
        gamma,
    );

// Create wgpu command encoder...

// Apply the selection and modification
editor.apply(
    &device,
    &mut encoder,
    [&basic_selection_modifier as &dyn gs::Modifier<GaussianPod>],
);

// Submit the commands...

// Download the modified Gaussians
let modified_gaussians = editor
    .gaussians_buffer
    .download_gaussians(&device, &queue)
    .await
    .unwrap()
    .map(|g| g.to_ply())
    .collect::<core::PlyGaussians>();

// Write the modified Gaussians to a .ply file
modified_gaussians.write_ply_file(&args.output).unwrap();
```

## Examples

See the [examples](https://github.com/LioQing/wgpu-3dgs-editor/tree/master/examples) directory for usage examples.

## Dependencies

This crate depends on the following crates:

| `wgpu-3dgs-editor` | `wgpu` | `glam` | `wesl` |
| ------------------ | ------ | ------ | ------ |
| 0.6                | 28.0   | 0.30   | 0.3    |
| 0.5                | 27.0   | 0.30   | 0.2    |
| 0.4                | 26.0   | 0.30   | 0.2    |
| 0.3                | 25.0   | 0.30   | N/A    |
| 0.1 - 0.2          | 24.0   | 0.29   | N/A    |

## Related Crates

- [wgpu-3dgs-viewer](https://crates.io/crates/wgpu-3dgs-viewer)
- [wgpu-3dgs-core](https://crates.io/crates/wgpu-3dgs-core)

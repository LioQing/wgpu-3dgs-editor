# 3D Gaussian Splatting Editor

...written in Rust using [wgpu](https://wgpu.rs/).

[![Crates.io](https://img.shields.io/crates/v/wgpu-3dgs-editor)](https://crates.io/crates/wgpu-3dgs-editor) [![Docs.rs](https://img.shields.io/docsrs/wgpu-3dgs-editor)](https://docs.rs/wgpu-3dgs-editor/latest/wgpu_3dgs_editor) [![License](https://img.shields.io/github/license/lioqing/wgpu-3dgs-editor)](./LICENSE)

## Overview

This library provides a set of tools to create and manipulate 3D Gaussian Splatting models. It includes:
- Selecting Gaussians using selection operations and expressions.
    - Primitive operations: union, intersection, difference, symmetric difference, and inversion.
    - Custom operations: define your own selection logic using WGSL shaders.
    - Built-in custom operations: sphere and box selections.
- Modifying Gaussian attributes.
    - Masked modifications: apply changes only to selected Gaussians from selection buffers.
    - Custom modifiers: define your own modification logic using WGSL/WESL shaders.
    - Built-in basic modifiers: color adjustments (RGB override or HSV modifications), alpha, contrast, exposure, gamma, and transformations (scale and rotation).

## Usage

This crate provides some basic selection and modifier, but you may also define your own custom selection and modifier using WGSL/WESL shaders.

### Basic Editor

You can use [`Editor`] and [`BasicSelectionModifier`] to manage the selection and modification of Gaussians with the built-in basic operations. Here's a simple example:

```rust
use wgpu_3dgs_editor as gs;

// Setup wgpu...

// Read the Gaussians from the .ply file
let f = std::fs::File::open(input_path).unwrap();
let mut reader = std::io::BufReader::new(f);
let gaussians = gs::core::Gaussians::read_ply(&mut reader).unwrap();

// Create an editor that creates the necessary buffers, you may also create the buffers manually
let editor = gs::Editor::<GaussianPod>::new(&device, &gaussians);

// Create a basic selection modifier with a box selection bundle
let mut basic_selection_modifier = gs::BasicSelectionModifier::new::<GaussianPod>(
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

let box_selection_bind_group = basic_selection_modifier.bundles[0]
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

let sphere_selection_bind_group = basic_selection_modifier.bundles[1]
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
    .basic_color_modifiers_buffer
    .update_with_override_rgb(
        &queue,
        color,
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
let modified_gaussians = gs::core::Gaussians {
    gaussians: editor
        .gaussians_buffer
        .download_gaussians(&device, &queue)
        .await
        .unwrap(),
};

// Write the modified Gaussians to a .ply file
let output_file = std::fs::File::create(output_path).unwrap();
let mut writer = std::io::BufWriter::new(output_file);
modified_gaussians.write_ply(&mut writer).unwrap();
```

## Examples

See the [examples](./examples) directory for usage examples.
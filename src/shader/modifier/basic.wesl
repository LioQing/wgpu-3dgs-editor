import wgpu_3dgs_core::{
    compute_bundle,
    gaussian::{Gaussian, gaussian_unpack_color},
    gaussian_transform::GaussianTransform,
    model_transform::ModelTransform,
};
import wgpu_3dgs_editor::modifier::{
    utils::{
        BasicColorModifiers,
        RotScale,
        apply_basic_color_modifiers,
        apply_basic_transform_modifiers,
    },
    modifier_consts,
};

@group(0) @binding(0)
var<storage, read_write> gaussians: array<Gaussian>;

@group(0) @binding(1)
var<uniform> model_transform: ModelTransform;

@group(0) @binding(2)
var<uniform> gaussian_transform: GaussianTransform;

@group(1) @binding(0)
var<uniform> transform_flags: u32;

@group(1) @binding(1)
var<uniform> color_modifiers: BasicColorModifiers;

@group(1) @binding(2)
var<uniform> rot_scale: RotScale;

@group(1) @binding(3) var<storage, read> selection: array<u32>;

override workgroup_size: u32;

@compute @workgroup_size(workgroup_size, 1, 1)
fn main(
    @builtin(workgroup_id) wid: vec3<u32>,
    @builtin(local_invocation_id) lid: vec3<u32>,
) {
    let index = compute_bundle::index(wid, workgroup_size, lid);

    if index >= arrayLength(&gaussians) {
        return;
    }
    
    @if(selection_buffer) {
        let word_index = index / 32u;
        let bit_index = index % 32u;
        let bit_mask = 1u << bit_index;
        if (selection[word_index] & bit_mask) == 0 {
            return;
        }
    }

    var gaussian = gaussians[index];

    gaussian.color = pack4x8unorm(apply_basic_color_modifiers(
        color_modifiers,
        gaussian_unpack_color(gaussian),
    ));

    var applied_model_transform = ModelTransform(
        vec3<f32>(0.0, 0.0, 0.0),
        vec4<f32>(0.0, 0.0, 0.0, 1.0),
        vec3<f32>(1.0, 1.0, 1.0),
    );
    if (transform_flags & modifier_consts::transform_flags_model) != 0 {
        applied_model_transform = model_transform;
    }

    var applied_rot_scale = rot_scale;
    if (transform_flags & modifier_consts::transform_flags_gaussian) != 0 {
        applied_rot_scale.scale *= gaussian_transform.size;
    }

    gaussian = apply_basic_transform_modifiers(
        applied_model_transform,
        applied_rot_scale,
        gaussian,
    );

    gaussians[index] = gaussian;
}
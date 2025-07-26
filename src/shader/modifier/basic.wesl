import wgpu_3dgs_core::{
    compute_bundle,
    gaussian::{Gaussian, gaussian_unpack_color},
    gaussian_transform::GaussianTransform,
    model_transform::ModelTransform,
};
import wgpu_3dgs_editor::utils::{
    BasicColorModifiers,
    apply_basic_color_modifiers,
    apply_basic_transform_modifiers,
};

@group(0) @binding(0)
var<storage, read> source: array<Gaussian>;

@group(0) @binding(1)
var<storage, read_write> dest: array<Gaussian>;

@group(0) @binding(2)
var<uniform> model_transform: ModelTransform;

@group(0) @binding(3)
var<uniform> gaussian_transform: GaussianTransform;

const transform_flags_model: u32 = 1;
const transform_flags_gaussian: u32 = 2;

@group(1) @binding(0)
var<uniform> transform_flags: u32;

@group(1) @binding(1)
var<uniform> color_modifiers: BasicColorModifiers;

@group(1) @binding(2)
var<uniform> scale_rotation: mat3x3<f32>;

override workgroup_size: u32;

@compute @workgroup_size(workgroup_size, 1, 1)
fn main(
    @builtin(workgroup_id) wid: vec3<u32>,
    @builtin(local_invocation_id) lid: vec3<u32>,
) {
    let index = compute_bundle::index(wid, workgroup_size, lid);

    if index >= arrayLength(&source) {
        return;
    }

    var gaussian = source[index];

    gaussian.color = pack4x8unorm(apply_basic_color_modifiers(
        color_modifiers,
        gaussian_unpack_color(gaussian),
    ));

    var applied_model_transform = ModelTransform(
        vec3<f32>(0.0, 0.0, 0.0),
        vec4<f32>(0.0, 0.0, 0.0, 1.0),
        vec3<f32>(1.0, 1.0, 1.0),
    );
    if (transform_flags & transform_flags_model) != 0 {
        applied_model_transform = model_transform;
    }

    var applied_scale_rotation = scale_rotation;
    if (transform_flags & transform_flags_gaussian) != 0 {
        applied_scale_rotation *= gaussian_transform.size;
    }

    gaussian = apply_basic_transform_modifiers(
        applied_model_transform,
        applied_scale_rotation,
        gaussian,
    );

    dest[index] = gaussian;
}
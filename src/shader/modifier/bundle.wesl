import wgpu_3dgs_core::{
    compute_bundle,
    gaussian::{Gaussian, gaussian_unpack_color},
    gaussian_transform::GaussianTransform,
    model_transform::ModelTransform,
};
import wgpu_3dgs_editor::utils::{
    ModifierBundleParams,
    apply_bundle_color_modifiers,
    apply_bundle_transform_modifiers,
};

@group(0) @binding(0)
var<uniform> op: u32;

@group(0) @binding(1)
var<storage, read> source: array<u32>;

@group(0) @binding(2)
var<storage, read_write> dest: array<atomic<u32>>;

@group(0) @binding(3)
var<uniform> model_transform: ModelTransform;

@group(0) @binding(4)
var<uniform> gaussian_transform: GaussianTransform;

@group(0) @binding(5)
var<storage, read> gaussians: array<Gaussian>;

@group(1) @binding(0)
var<uniform> color_params: ModifierBundleColorParams;

const transform_flags_apply_model: u32 = 1;
const transform_flags_apply_gaussian: u32 = 2;

@group(1) @binding(1)
var<uniform> transform_flags: u32;

@group(1) @binding(2)
var<uniform> scale_rotation: mat3x3<f32>;

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

    var gaussian = gaussians[index];

    gaussian.color = pack4x8unorm(apply_bundle_color_modifiers(
        color_params,
        gaussian_unpack_color(gaussian),
    ));

    var applied_model_transform = ModelTransform(
        vec3<f32>(0.0, 0.0, 0.0),
        vec4<f32>(0.0, 0.0, 0.0, 1.0),
        vec3<f32>(1.0, 1.0, 1.0),
    );
    if (transform_flags & transform_flags_apply_model) != 0 {
        applied_model_transform = model_transform;
    }

    var applied_scale_rotation = scale_rotation;
    if (transform_flags & transform_flags_apply_gaussian) != 0 {
        applied_scale_rotation *= gaussian_transform.size;
    }

    gaussian = apply_bundle_transform_modifiers(
        applied_model_transform,
        applied_scale_rotation,
        gaussian,
    );

    gaussians[index] = gaussian;
}
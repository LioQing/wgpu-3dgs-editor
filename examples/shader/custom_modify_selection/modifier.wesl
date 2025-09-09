import wgpu_3dgs_core::{
    gaussian::{gaussian_unpack_color, Gaussian},
    gaussian_transform::GaussianTransform,
    model_transform::ModelTransform,
};
import wgpu_3dgs_editor::modifier::utils::{rgb_to_hsv, hsv_to_rgb};

@group(0) @binding(0)
var<storage, read_write> gaussians: array<Gaussian>;

@group(0) @binding(1)
var<uniform> model_transform: ModelTransform;

@group(0) @binding(2)
var<uniform> gaussian_transform: GaussianTransform;

@group(1) @binding(0)
var<uniform> pos: vec2<f32>;

@group(1) @binding(1)
var<uniform> radius: f32;

@group(1) @binding(2)
var<storage, read> selection: array<u32>;

override workgroup_size: u32;

@compute @workgroup_size(workgroup_size)
fn main(@builtin(global_invocation_id) id: vec3<u32>) {
    let index = id.x;

    if index >= arrayLength(&gaussians) {
        return;
    }

    var gaussian = gaussians[index];
    
    let word_index = index / 32u;
    let bit_index = index % 32u;
    let bit_mask = 1u << bit_index;
    if (selection[word_index] & bit_mask) == 0 {
        gaussian.color = 0;
        gaussians[index] = gaussian;
        return;
    }

    let displacement = gaussian.pos.xy - pos;
    let distance = length(displacement);

    if distance <= radius {
        let old_color = gaussian_unpack_color(gaussian);
        let old_hsv = rgb_to_hsv(old_color.rgb);
        let new_hsv = vec3<f32>((old_hsv.x + atan2(displacement.y, displacement.x)) % 1.0, old_hsv.y, old_hsv.z);
        let new_rgb = hsv_to_rgb(new_hsv);
        let new_alpha = old_color.a * (1.0 - distance / radius);
        gaussian.color = pack4x8unorm(vec4<f32>(new_rgb, new_alpha));
    }

    gaussians[index] = gaussian;
}
import wgpu_3dgs_core::{
    compute_bundle,
    gaussian::Gaussian,
    gaussian_transform::GaussianTransform,
    model_transform::{model_to_world, ModelTransform},
};
import super::utils;

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

struct Sphere {
    inv_transform: mat4x4<f32>,
}

@group(1) @binding(0)
var<uniform> sphere: Sphere;

override workgroup_size: u32;

@compute @workgroup_size(workgroup_size, 1, 1)
fn main(@builtin(workgroup_id) wid: vec3<u32>, @builtin(local_invocation_id) lid: vec3<u32>) {
    let index = compute_bundle::index(wid, workgroup_size, lid);

    if (index >= arrayLength(&gaussians)) {
        return;
    }

    let gaussian = gaussians[index];

    let world_pos = model_to_world(model_transform, gaussian.pos);
    let sphere_pos = sphere.inv_transform * world_pos;

    if length(sphere_pos.xyz) <= 1.0 {
        atomicOr(&dest[index], utils::bit_flag(op));
    } else {
        atomicAnd(&dest[index], ~utils::bit_flag(op));
    }
}
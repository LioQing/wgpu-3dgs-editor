import wgpu_3dgs_core::{
    gaussian::Gaussian,
    gaussian_transform::GaussianTransform,
    model_transform::{model_to_world, ModelTransform},
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
var<uniform> pos: vec2<f32>;

@group(1) @binding(1)
var<uniform> radius: f32;

override workgroup_size: u32;

@compute @workgroup_size(workgroup_size)
fn main(@builtin(global_invocation_id) id: vec3<u32>) {
    let index = id.x;

    if index >= arrayLength(&gaussians) {
        return;
    }

    let gaussian = gaussians[index];

    let world_pos = model_to_world(model_transform, gaussian.pos);

    let displacement = world_pos.xy - pos;
    let distance = length(displacement);

    let word_index = index / 32u;
    let bit_index = index % 32u;
    let bit_mask = 1u << bit_index;
    if distance <= radius {
        atomicOr(&dest[word_index], bit_mask);
    } else {
        atomicAnd(&dest[word_index], ~bit_mask);
    }
}
import wgpu_3dgs_core::{
    compute_bundle,
    gaussian_transform::GaussianTransform,
    model_transform::ModelTransform,
};
import super::ops;

override workgroup_size: u32;
override gaussians_group_index: u32;

@group(gaussians_group_index) @binding(0)
var<uniform> op: u32;

@group(gaussians_group_index) @binding(1)
var<storage, read> source: array<u32>;

@group(gaussians_group_index) @binding(2)
var<storage, read_write> dest: array<atomic<u32>>;

@group(gaussians_group_index) @binding(3)
var<uniform> model_transform: ModelTransform;

@group(gaussians_group_index) @binding(4)
var<uniform> gaussian_transform: GaussianTransform;

@group(gaussians_group_index) @binding(5)
var<storage, read> gaussians: array<Gaussian>;

@compute @workgroup_size(workgroup_size, 1, 1)
fn main(@builtin(workgroup_id) wid: vec3<u32>, @builtin(local_invocation_id) lid: vec3<u32>) {
    let index = compute_bundle::index(wid, workgroup_size, lid);

    if (index >= arrayLength(&gaussians)) {
        return;
    }

    if op == ops::union {
        atomicOr(&dest[index], source[index]);
    } else if op == ops::intersection {
        atomicAnd(&dest[index], source[index]);
    } else if op == ops::symmetric_difference {
        atomicXor(&dest[index], source[index]);
    } else if op == ops::difference {
        atomicAnd(&dest[index], ~source[index]);
    } else if op == ops::complement {
        atomicStore(&dest[index], ~atomicLoad(&dest[index]));
    } // {{custom_ops}}
}
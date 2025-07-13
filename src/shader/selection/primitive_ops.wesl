import wgpu_3dgs_core::{
    compute_bundle,
    gaussian::Gaussian,
    gaussian_transform::GaussianTransform,
    model_transform::ModelTransform,
};
import super::ops;

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

override workgroup_size: u32;

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
    }
}
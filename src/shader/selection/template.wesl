import wgpu_3dgs_core::{
    compute_bundle,
    gaussian_transform::GaussianTransform,
    model_transform::ModelTransform,
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

override workgroup_size: u32;

@compute @workgroup_size(workgroup_size, 1, 1)
fn main(@builtin(workgroup_id) wid: vec3<u32>, @builtin(local_invocation_id) lid: vec3<u32>) {
    let index = compute_bundle::index(wid, workgroup_size, lid);

    if (index >= arrayLength(&gaussians)) {
        return;
    }

    if op == op_union {
        atomicOr(&dest[index], mask);
    } else if op == op_intersection {
        atomicAnd(&dest[index], mask);
    } else if op == op_symmetric_difference {
        atomicXor(&dest[index], mask);
    } else if op == op_difference {
        atomicAnd(&dest[index], ~mask);
    } else if op == op_complement {
        atomicStore(&dest[index], ~atomicLoad(&dest[index]));
    } // {{remaining_ops}}
}
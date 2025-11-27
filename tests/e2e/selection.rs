use wgpu::util::DeviceExt;
use wgpu_3dgs_editor::{
    Editor, InvTransformBuffer, SelectionBuffer, SelectionBundle, SelectionExpr,
    core::{
        BufferWrapper, ComputeBundle, ComputeBundleBuilder, Gaussian, GaussianPod,
        GaussianPodWithShSingleCov3dRotScaleConfigs, glam::*,
    },
};

use crate::{common::TestContext, inline_wesl_pkg};

type G = GaussianPodWithShSingleCov3dRotScaleConfigs;

fn given_test_selection_gaussians() -> Vec<Gaussian> {
    vec![
        Gaussian {
            pos: Vec3::new(0.0, 0.0, 0.0),
            rot: Quat::IDENTITY,
            color: U8Vec4::new(255, 0, 0, 255),
            sh: [Vec3::ZERO; 15],
            scale: Vec3::ONE,
        },
        Gaussian {
            pos: Vec3::new(1.0, 0.0, 0.0),
            rot: Quat::IDENTITY,
            color: U8Vec4::new(0, 255, 0, 255),
            sh: [Vec3::ZERO; 15],
            scale: Vec3::ONE,
        },
        Gaussian {
            pos: Vec3::new(2.0, 0.0, 0.0),
            rot: Quat::IDENTITY,
            color: U8Vec4::new(0, 0, 255, 255),
            sh: [Vec3::ZERO; 15],
            scale: Vec3::ONE,
        },
        Gaussian {
            pos: Vec3::new(3.0, 0.0, 0.0),
            rot: Quat::IDENTITY,
            color: U8Vec4::new(255, 255, 255, 255),
            sh: [Vec3::ZERO; 15],
            scale: Vec3::ONE,
        },
    ]
}

const TEST_LEFT_SELECTION: u32 = 0b0011; // Select first two gaussians
const TEST_RIGHT_SELECTION: u32 = 0b0110; // Select middle two gaussians

fn given_test_selection_buffers(ctx: &TestContext) -> (SelectionBuffer, SelectionBuffer) {
    (
        SelectionBuffer::from(
            ctx.device
                .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                    label: Some("Left Test Selection Buffer"),
                    contents: bytemuck::cast_slice(&[TEST_LEFT_SELECTION]),
                    usage: wgpu::BufferUsages::STORAGE
                        | wgpu::BufferUsages::COPY_DST
                        | wgpu::BufferUsages::COPY_SRC,
                }),
        ),
        SelectionBuffer::from(
            ctx.device
                .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                    label: Some("Right Test Selection Buffer"),
                    contents: bytemuck::cast_slice(&[TEST_RIGHT_SELECTION]),
                    usage: wgpu::BufferUsages::STORAGE
                        | wgpu::BufferUsages::COPY_DST
                        | wgpu::BufferUsages::COPY_SRC,
                }),
        ),
    )
}

fn test_selection_bundle_evaluate_helper(
    ctx: &TestContext,
    bundles: Vec<ComputeBundle<()>>,
    expr: impl FnOnce(SelectionExpr, SelectionExpr) -> SelectionExpr,
    expected: u32,
) {
    let gaussians = given_test_selection_gaussians();
    let editor = Editor::new(&ctx.device, &gaussians);
    let (left_buffer, right_buffer) = given_test_selection_buffers(ctx);
    let selection_buffer = SelectionBuffer::new(&ctx.device, gaussians.len() as u32);

    let selection_bundle = SelectionBundle::<G>::new(&ctx.device, bundles);
    let selection_expr = expr(
        SelectionExpr::buffer(left_buffer),
        SelectionExpr::buffer(right_buffer),
    );

    let mut encoder = ctx
        .device
        .create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("Test Command Encoder"),
        });

    selection_bundle.evaluate(
        &ctx.device,
        &mut encoder,
        &selection_expr,
        &selection_buffer,
        &editor.model_transform_buffer,
        &editor.gaussian_transform_buffer,
        &editor.gaussians_buffer,
    );

    ctx.queue.submit(Some(encoder.finish()));

    let downloaded = pollster::block_on(selection_buffer.download::<u32>(&ctx.device, &ctx.queue))
        .expect("download")[0];

    assert_eq!(
        downloaded & 0b1111,
        expected & 0b1111,
        "\n  left: {:#032b}\n right: {:#032b}",
        downloaded,
        expected,
    );
}

#[test]
fn test_selection_bundle_evaluate_when_expr_is_identity_should_do_nothing() {
    test_selection_bundle_evaluate_helper(
        &TestContext::new(),
        vec![],
        |_, _| SelectionExpr::identity(),
        0,
    );
}

#[test]
fn test_selection_bundle_evaluate_when_expr_is_buffer_should_copy_buffer() {
    test_selection_bundle_evaluate_helper(
        &TestContext::new(),
        vec![],
        |l, _| l,
        TEST_LEFT_SELECTION,
    );
}

#[test]
fn test_selection_bundle_evaluate_when_expr_is_union_should_combine_selections() {
    test_selection_bundle_evaluate_helper(
        &TestContext::new(),
        vec![],
        SelectionExpr::union,
        TEST_LEFT_SELECTION | TEST_RIGHT_SELECTION,
    );
}

#[test]
fn test_selection_bundle_evaluate_when_expr_is_intersection_should_intersect_selections() {
    test_selection_bundle_evaluate_helper(
        &TestContext::new(),
        vec![],
        SelectionExpr::intersection,
        TEST_LEFT_SELECTION & TEST_RIGHT_SELECTION,
    );
}

#[test]
fn test_selection_bundle_evaluate_when_expr_is_difference_should_subtract_selections() {
    test_selection_bundle_evaluate_helper(
        &TestContext::new(),
        vec![],
        SelectionExpr::difference,
        TEST_LEFT_SELECTION & !TEST_RIGHT_SELECTION,
    );
}

#[test]
fn test_selection_bundle_evaluate_when_expr_is_symmetric_difference_should_xor_selections() {
    test_selection_bundle_evaluate_helper(
        &TestContext::new(),
        vec![],
        SelectionExpr::symmetric_difference,
        TEST_LEFT_SELECTION ^ TEST_RIGHT_SELECTION,
    );
}

#[test]
fn test_selection_bundle_evaluate_when_expr_is_complement_should_invert_selection() {
    test_selection_bundle_evaluate_helper(
        &TestContext::new(),
        vec![],
        |l, _| l.complement(),
        !TEST_LEFT_SELECTION,
    );
}

const TEST_UNARY_PACKAGE: wesl::Pkg = inline_wesl_pkg!(
    use [&wgpu_3dgs_editor::core::shader::PACKAGE],

    "test_unary": // Always set odd indices
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

    override workgroup_size: u32;

    @compute @workgroup_size(workgroup_size)
    fn main(@builtin(global_invocation_id) id: vec3<u32>) {
        let index = id.x;

        if index >= arrayLength(&gaussians) {
            return;
        }

        let gaussian = gaussians[index];

        if index % 2u == 0u {
            return;
        }

        let word_index = index / 32u;
        let bit_index = index % 32u;
        let bit_mask = 1u << bit_index;
        atomicOr(&dest[word_index], bit_mask);
    }
);

#[test]
fn test_selection_bundle_evaluate_when_expr_is_unary_should_correctly_select_gaussians() {
    let ctx = TestContext::new();
    let bundle = ComputeBundleBuilder::new()
        .bind_group_layouts([&SelectionBundle::<G>::GAUSSIANS_BIND_GROUP_LAYOUT_DESCRIPTOR])
        .resolver({
            let mut resolver = wesl::PkgResolver::new();
            resolver.add_package(&TEST_UNARY_PACKAGE);
            resolver.add_package(&wgpu_3dgs_core::shader::PACKAGE);
            resolver
        })
        .main_shader("test_unary".parse().expect("parse"))
        .entry_point("main")
        .wesl_compile_options(wesl::CompileOptions {
            features: G::wesl_features(),
            ..Default::default()
        })
        .build_without_bind_groups(&ctx.device)
        .expect("build_without_bind_groups");

    test_selection_bundle_evaluate_helper(&ctx, vec![bundle], |l, _| l.unary(0, vec![]), 0b1011);
}

const TEST_BINARY_PACKAGE: wesl::Pkg = inline_wesl_pkg!(
    use [&wgpu_3dgs_editor::core::shader::PACKAGE],

    "test_binary": // Use source if first half, dest otherwise
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

    override workgroup_size: u32;

    @compute @workgroup_size(workgroup_size)
    fn main(@builtin(global_invocation_id) id: vec3<u32>) {
        let index = id.x;

        if index >= arrayLength(&gaussians) {
            return;
        }

        let gaussian = gaussians[index];

        let word_index = index / 32u;
        let bit_index = index % 32u;
        let bit_mask = 1u << bit_index;
        if index < arrayLength(&gaussians) / 2u {
            if (source[word_index] & bit_mask) != 0u {
                atomicOr(&dest[word_index], bit_mask);
            } else {
                atomicAnd(&dest[word_index], ~bit_mask);
            }
        }
    }
);

#[test]
fn test_selection_bundle_evaluate_when_expr_is_binary_should_correctly_select_gaussians() {
    let ctx = TestContext::new();
    let bundle = ComputeBundleBuilder::new()
        .bind_group_layouts([&SelectionBundle::<G>::GAUSSIANS_BIND_GROUP_LAYOUT_DESCRIPTOR])
        .resolver({
            let mut resolver = wesl::PkgResolver::new();
            resolver.add_package(&TEST_BINARY_PACKAGE);
            resolver.add_package(&wgpu_3dgs_core::shader::PACKAGE);
            resolver
        })
        .main_shader("test_binary".parse().expect("parse"))
        .entry_point("main")
        .wesl_compile_options(wesl::CompileOptions {
            features: G::wesl_features(),
            ..Default::default()
        })
        .build_without_bind_groups(&ctx.device)
        .expect("build_without_bind_groups");

    test_selection_bundle_evaluate_helper(
        &ctx,
        vec![bundle],
        |l, r| l.binary(0, r, vec![]),
        0b0111,
    );
}

const TEST_SELECTION_PACKAGE: wesl::Pkg = inline_wesl_pkg!(
    use [&wgpu_3dgs_editor::core::shader::PACKAGE],

    "test_selection": // Select by position
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
    var<uniform> selected_pos: vec3<f32>;

    override workgroup_size: u32;

    @compute @workgroup_size(workgroup_size)
    fn main(@builtin(global_invocation_id) id: vec3<u32>) {
        let index = id.x;

        if index >= arrayLength(&gaussians) {
            return;
        }

        let gaussian = gaussians[index];

        let world_pos = model_to_world(model_transform, gaussian.pos);

        let word_index = index / 32u;
        let bit_index = index % 32u;
        let bit_mask = 1u << bit_index;
        if all(abs(world_pos.xyz / world_pos.w - selected_pos) < vec3<f32>(0.1)) {
            atomicOr(&dest[word_index], bit_mask);
        } else {
            atomicAnd(&dest[word_index], ~bit_mask);
        }
    }
);

const TEST_SELECTION_PACKAGE_BIND_GROUP_LAYOUT: wgpu::BindGroupLayoutDescriptor<'static> =
    wgpu::BindGroupLayoutDescriptor {
        label: Some("Test Selection Package Bind Group Layout"),
        entries: &[wgpu::BindGroupLayoutEntry {
            binding: 0,
            visibility: wgpu::ShaderStages::COMPUTE,
            ty: wgpu::BindingType::Buffer {
                ty: wgpu::BufferBindingType::Uniform,
                has_dynamic_offset: false,
                min_binding_size: None,
            },
            count: None,
        }],
    };

#[test]
fn test_selection_bundle_evaluate_when_expr_is_selection_should_select_by_position() {
    let ctx = TestContext::new();
    let bundle = ComputeBundleBuilder::new()
        .bind_group_layouts([
            &SelectionBundle::<G>::GAUSSIANS_BIND_GROUP_LAYOUT_DESCRIPTOR,
            &TEST_SELECTION_PACKAGE_BIND_GROUP_LAYOUT,
        ])
        .resolver({
            let mut resolver = wesl::PkgResolver::new();
            resolver.add_package(&TEST_SELECTION_PACKAGE);
            resolver.add_package(&wgpu_3dgs_core::shader::PACKAGE);
            resolver
        })
        .main_shader("test_selection".parse().expect("parse"))
        .entry_point("main")
        .wesl_compile_options(wesl::CompileOptions {
            features: G::wesl_features(),
            ..Default::default()
        })
        .build_without_bind_groups(&ctx.device)
        .expect("build_without_bind_groups");

    let buffer = ctx
        .device
        .create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Test Selected Pos Buffer"),
            contents: bytemuck::bytes_of(&Vec3::new(2.0, 0.0, 0.0)),
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        });

    let bind_group = bundle
        .create_bind_group(&ctx.device, 1, [buffer.as_entire_binding()])
        .expect("create_bind_group");

    test_selection_bundle_evaluate_helper(
        &ctx,
        vec![bundle],
        |_, _| SelectionExpr::selection(0, vec![bind_group]),
        0b0100,
    );
}

#[test]
fn test_selection_bundle_evaluate_when_expr_is_box_selection_should_select_by_indices() {
    let ctx = TestContext::new();
    let bundle = SelectionBundle::<G>::create_box_bundle(&ctx.device);

    let buffer = InvTransformBuffer::new(&ctx.device);
    buffer.update_with_scale_rot_pos(
        &ctx.queue,
        Vec3::ONE * 0.5,
        Quat::IDENTITY,
        Vec3::new(2.0, 0.0, 0.0),
    );

    let bind_group = bundle
        .create_bind_group(&ctx.device, 1, [buffer.buffer().as_entire_binding()])
        .expect("create_bind_group");

    test_selection_bundle_evaluate_helper(
        &ctx,
        vec![bundle],
        |_, _| SelectionExpr::selection(0, vec![bind_group]),
        0b0100,
    );
}

#[test]
fn test_selection_bundle_evaluate_when_expr_is_sphere_selection_should_select_by_indices() {
    let ctx = TestContext::new();
    let bundle = SelectionBundle::<G>::create_sphere_bundle(&ctx.device);

    let buffer = InvTransformBuffer::new(&ctx.device);
    buffer.update_with_scale_rot_pos(
        &ctx.queue,
        Vec3::ONE * 0.5,
        Quat::IDENTITY,
        Vec3::new(2.0, 0.0, 0.0),
    );

    let bind_group = bundle
        .create_bind_group(&ctx.device, 1, [buffer.buffer().as_entire_binding()])
        .expect("create_bind_group");

    test_selection_bundle_evaluate_helper(
        &ctx,
        vec![bundle],
        |_, _| SelectionExpr::selection(0, vec![bind_group]),
        0b0100,
    );
}

#[test]
fn test_selection_expr_update_with_should_replace_expr_in_place() {
    let ctx = TestContext::new();
    let buffer1 = SelectionBuffer::new(&ctx.device, 10);
    let buffer2 = SelectionBuffer::new(&ctx.device, 10);

    let mut expr = SelectionExpr::identity();
    expr.update_with(|_| SelectionExpr::buffer(buffer1).union(SelectionExpr::buffer(buffer2)));
    assert!(matches!(expr, SelectionExpr::Union(_, _)));

    expr.update_with(|e| e.complement());
    assert!(matches!(expr, SelectionExpr::Complement(_)));
}

#[test]
fn test_selection_expr_is_identity_should_correctly_identify_identity() {
    let ctx = TestContext::new();
    let buffer = SelectionBuffer::new(&ctx.device, 10);

    assert!(SelectionExpr::identity().is_identity());

    assert!(
        !SelectionExpr::identity()
            .union(SelectionExpr::identity())
            .is_identity()
    );
    assert!(!SelectionExpr::buffer(buffer).is_identity());
}

#[test]
fn test_selection_expr_is_primitive_should_correctly_identify_primitive_operations() {
    assert!(
        SelectionExpr::identity()
            .union(SelectionExpr::identity())
            .is_primitive()
    );
    assert!(
        SelectionExpr::identity()
            .intersection(SelectionExpr::identity())
            .is_primitive()
    );
    assert!(
        SelectionExpr::identity()
            .difference(SelectionExpr::identity())
            .is_primitive()
    );
    assert!(
        SelectionExpr::identity()
            .symmetric_difference(SelectionExpr::identity())
            .is_primitive()
    );
    assert!(SelectionExpr::identity().complement().is_primitive());

    assert!(!SelectionExpr::identity().is_primitive());
    assert!(!SelectionExpr::identity().unary(0, vec![]).is_primitive());
}

#[test]
fn test_selection_expr_is_custom_should_correctly_identify_custom_operations() {
    assert!(SelectionExpr::identity().unary(0, vec![]).is_custom());
    assert!(
        SelectionExpr::identity()
            .binary(0, SelectionExpr::identity(), vec![])
            .is_custom()
    );
    assert!(SelectionExpr::selection(0, vec![]).is_custom());

    assert!(
        !SelectionExpr::identity()
            .union(SelectionExpr::identity())
            .is_custom()
    );
    assert!(!SelectionExpr::identity().is_custom());
}

#[test]
fn test_selection_expr_is_operation_should_correctly_identify_operations() {
    let ctx = TestContext::new();
    let buffer = SelectionBuffer::new(&ctx.device, 10);

    assert!(
        SelectionExpr::identity()
            .union(SelectionExpr::identity())
            .is_operation()
    );
    assert!(SelectionExpr::identity().unary(0, vec![]).is_operation());
    assert!(SelectionExpr::selection(0, vec![]).is_operation());

    assert!(!SelectionExpr::identity().is_operation());
    assert!(!SelectionExpr::buffer(buffer).is_operation());
}

#[test]
fn test_selection_expr_is_buffer_should_correctly_identify_buffer() {
    let ctx = TestContext::new();
    let buffer = SelectionBuffer::new(&ctx.device, 10);

    assert!(SelectionExpr::buffer(buffer).is_buffer());

    assert!(!SelectionExpr::identity().is_buffer());
    assert!(
        !SelectionExpr::identity()
            .union(SelectionExpr::identity())
            .is_buffer()
    );
}

#[test]
fn test_selection_expr_custom_op_index_should_return_correct_index() {
    assert_eq!(
        SelectionExpr::identity().unary(5, vec![]).custom_op_index(),
        Some(5)
    );
    assert_eq!(
        SelectionExpr::identity()
            .binary(3, SelectionExpr::identity(), vec![])
            .custom_op_index(),
        Some(3)
    );
    assert_eq!(
        SelectionExpr::selection(7, vec![]).custom_op_index(),
        Some(7)
    );

    assert_eq!(
        SelectionExpr::identity()
            .union(SelectionExpr::identity())
            .custom_op_index(),
        None
    );
    assert_eq!(SelectionExpr::identity().custom_op_index(), None);
}

#[test]
fn test_selection_expr_custom_bind_groups_should_return_correct_bind_groups() {
    let ctx = TestContext::new();
    let bind_group = ctx.device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("Test Bind Group"),
        layout: &ctx
            .device
            .create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some("Test Layout"),
                entries: &[],
            }),
        entries: &[],
    });

    let bind_groups = vec![bind_group];

    let unary_expr = SelectionExpr::identity().unary(0, bind_groups.clone());
    assert!(unary_expr.custom_bind_groups().is_some());
    assert_eq!(unary_expr.custom_bind_groups().unwrap().len(), 1);

    let binary_expr =
        SelectionExpr::identity().binary(0, SelectionExpr::identity(), bind_groups.clone());
    assert!(binary_expr.custom_bind_groups().is_some());
    assert_eq!(binary_expr.custom_bind_groups().unwrap().len(), 1);

    let selection_expr = SelectionExpr::selection(0, bind_groups.clone());
    assert!(selection_expr.custom_bind_groups().is_some());
    assert_eq!(selection_expr.custom_bind_groups().unwrap().len(), 1);

    assert!(
        SelectionExpr::identity()
            .union(SelectionExpr::identity())
            .custom_bind_groups()
            .is_none()
    );
    assert!(SelectionExpr::identity().custom_bind_groups().is_none());
}

use wgpu::util::DeviceExt;
use wgpu_3dgs_core::BufferWrapper;
use wgpu_3dgs_editor::{
    BasicColorModifiersBuffer, BasicColorModifiersPod, BasicColorRgbOverrideOrHsvModifiersPod,
    BasicModifier, BasicModifierBundle, Editor, Modifier, RotScaleBuffer, SelectionBuffer,
    TransformFlagsBuffer,
    core::{
        GaussianPodWithShSingleCov3dRotScaleConfigs, GaussianTransformBuffer, GaussiansBuffer,
        ModelTransformBuffer, glam::*,
    },
};

use crate::common::{TestContext, given};

type G = GaussianPodWithShSingleCov3dRotScaleConfigs;

#[test]
fn test_modifier_apply_when_impl_is_fn_should_call_self() {
    use std::{cell::Cell, rc::Rc};

    let ctx = TestContext::new();

    let called = Rc::<Cell<bool>>::new(false.into());
    let modifier = |_device: &wgpu::Device,
                    _encoder: &mut wgpu::CommandEncoder,
                    _gaussians: &GaussiansBuffer<G>,
                    _model_transform: &ModelTransformBuffer,
                    _gaussian_transform: &GaussianTransformBuffer| {
        called.set(true);
    };

    let gaussians = GaussiansBuffer::<G>::new_empty(&ctx.device, 1);
    let model_transform = ModelTransformBuffer::new(&ctx.device);
    let gaussian_transform = GaussianTransformBuffer::new(&ctx.device);

    let mut encoder = ctx
        .device
        .create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("Test Command Encoder"),
        });

    modifier.apply(
        &ctx.device,
        &mut encoder,
        &gaussians,
        &model_transform,
        &gaussian_transform,
    );
}

#[test]
fn test_basic_modifier_bundle_without_bind_group_apply_should_modify_gaussians() {
    let ctx = TestContext::new();
    let gaussians = given::gaussians();
    let editor = Editor::<G>::new(&ctx.device, &gaussians);

    let transform_flags_buffer = TransformFlagsBuffer::new(&ctx.device);
    let basic_color_modifiers_buffer = BasicColorModifiersBuffer::new(&ctx.device);
    let rot_scale_buffer = RotScaleBuffer::new(&ctx.device);

    let modifier_bundle = BasicModifierBundle::<G, _, _>::new_without_bind_group(&ctx.device);

    let gaussians_bind_group = modifier_bundle
        .bundle()
        .create_bind_group(
            &ctx.device,
            0,
            [
                editor.gaussians_buffer.buffer().as_entire_binding(),
                editor.model_transform_buffer.buffer().as_entire_binding(),
                editor
                    .gaussian_transform_buffer
                    .buffer()
                    .as_entire_binding(),
            ],
        )
        .expect("create_bind_group");
    let basic_modifier_bind_group = modifier_bundle
        .bundle()
        .create_bind_group(
            &ctx.device,
            1,
            [
                transform_flags_buffer.buffer().as_entire_binding(),
                basic_color_modifiers_buffer.buffer().as_entire_binding(),
                rot_scale_buffer.buffer().as_entire_binding(),
            ],
        )
        .expect("create_bind_group");

    basic_color_modifiers_buffer.update_with_pod(
        &ctx.queue,
        &BasicColorModifiersPod::new(
            BasicColorRgbOverrideOrHsvModifiersPod::new_rgb_override(Vec3::new(1.0, 0.0, 0.0)),
            1.0,
            0.0,
            0.0,
            1.0,
        ),
    );

    let mut encoder = ctx
        .device
        .create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("Test Command Encoder"),
        });

    modifier_bundle.apply_with_count(
        &mut encoder,
        &gaussians_bind_group,
        &basic_modifier_bind_group,
        editor.gaussians_buffer.len() as u32,
    );

    ctx.queue.submit(Some(encoder.finish()));

    let downloaded = pollster::block_on(
        editor
            .gaussians_buffer
            .download_gaussians(&ctx.device, &ctx.queue),
    )
    .expect("download_gaussians");

    assert_eq!(downloaded.len(), 2);
    assert_eq!(downloaded[0].color.xyz(), U8Vec3::new(255, 0, 0));
    assert_eq!(downloaded[1].color.xyz(), U8Vec3::new(255, 0, 0));
}

#[test]
fn test_basic_modifier_bundle_without_bind_group_with_selection_apply_should_modify_gaussians() {
    let ctx = TestContext::new();
    let gaussians = given::gaussians();
    let editor = Editor::<G>::new(&ctx.device, &gaussians);
    let selection_buffer = SelectionBuffer::from(ctx.device.create_buffer_init(
        &wgpu::util::BufferInitDescriptor {
            label: Some("Selection Buffer"),
            contents: bytemuck::cast_slice(&[1u32]),
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
        },
    ));

    let transform_flags_buffer = TransformFlagsBuffer::new(&ctx.device);
    let basic_color_modifiers_buffer = BasicColorModifiersBuffer::new(&ctx.device);
    let rot_scale_buffer = RotScaleBuffer::new(&ctx.device);

    let modifier_bundle =
        BasicModifierBundle::<G, _, _>::new_without_bind_group_with_selection(&ctx.device);

    let gaussians_bind_group = modifier_bundle
        .bundle()
        .create_bind_group(
            &ctx.device,
            0,
            [
                editor.gaussians_buffer.buffer().as_entire_binding(),
                editor.model_transform_buffer.buffer().as_entire_binding(),
                editor
                    .gaussian_transform_buffer
                    .buffer()
                    .as_entire_binding(),
            ],
        )
        .expect("create_bind_group");
    let basic_modifier_bind_group = modifier_bundle
        .bundle()
        .create_bind_group(
            &ctx.device,
            1,
            [
                transform_flags_buffer.buffer().as_entire_binding(),
                basic_color_modifiers_buffer.buffer().as_entire_binding(),
                rot_scale_buffer.buffer().as_entire_binding(),
                selection_buffer.buffer().as_entire_binding(),
            ],
        )
        .expect("create_bind_group");

    basic_color_modifiers_buffer.update_with_pod(
        &ctx.queue,
        &BasicColorModifiersPod::new(
            BasicColorRgbOverrideOrHsvModifiersPod::new_rgb_override(Vec3::new(1.0, 0.0, 0.0)),
            1.0,
            0.0,
            0.0,
            1.0,
        ),
    );

    let mut encoder = ctx
        .device
        .create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("Test Command Encoder"),
        });

    modifier_bundle.apply_with_count(
        &mut encoder,
        &gaussians_bind_group,
        &basic_modifier_bind_group,
        editor.gaussians_buffer.len() as u32,
    );

    ctx.queue.submit(Some(encoder.finish()));

    let downloaded = pollster::block_on(
        editor
            .gaussians_buffer
            .download_gaussians(&ctx.device, &ctx.queue),
    )
    .expect("download_gaussians");

    assert_eq!(downloaded.len(), 2);
    assert_eq!(downloaded[0].color.xyz(), U8Vec3::new(255, 0, 0));
    assert_eq!(downloaded[1].color.xyz(), gaussians[1].color.xyz());
}

#[test]
fn test_basic_modifier_apply_should_modify_gaussians() {
    let ctx = TestContext::new();
    let gaussians = given::gaussians();
    let editor = Editor::<G>::new(&ctx.device, &gaussians);

    let modifier = BasicModifier::new(
        &ctx.device,
        &editor.gaussians_buffer,
        &editor.model_transform_buffer,
        &editor.gaussian_transform_buffer,
    );

    modifier.basic_color_modifiers_buffer.update_with_pod(
        &ctx.queue,
        &BasicColorModifiersPod::new(
            BasicColorRgbOverrideOrHsvModifiersPod::new_rgb_override(Vec3::new(1.0, 0.0, 0.0)),
            1.0,
            0.0,
            0.0,
            1.0,
        ),
    );

    let mut encoder = ctx
        .device
        .create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("Test Command Encoder"),
        });

    modifier.apply(
        &ctx.device,
        &mut encoder,
        &editor.gaussians_buffer,
        &editor.model_transform_buffer,
        &editor.gaussian_transform_buffer,
    );

    ctx.queue.submit(Some(encoder.finish()));

    let downloaded = pollster::block_on(
        editor
            .gaussians_buffer
            .download_gaussians(&ctx.device, &ctx.queue),
    )
    .expect("download_gaussians");

    assert_eq!(downloaded.len(), 2);
    assert_eq!(downloaded[0].color.xyz(), U8Vec3::new(255, 0, 0));
    assert_eq!(downloaded[1].color.xyz(), U8Vec3::new(255, 0, 0));
}

#[test]
fn test_basic_modifier_with_selection_apply_should_modify_selected_gaussians() {
    let ctx = TestContext::new();
    let gaussians = given::gaussians();
    let editor = Editor::<G>::new(&ctx.device, &gaussians);
    let selection_buffer = SelectionBuffer::from(ctx.device.create_buffer_init(
        &wgpu::util::BufferInitDescriptor {
            label: Some("Selection Buffer"),
            contents: bytemuck::cast_slice(&[1u32]),
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
        },
    ));

    let modifier = BasicModifier::new_with_selection(
        &ctx.device,
        &editor.gaussians_buffer,
        &editor.model_transform_buffer,
        &editor.gaussian_transform_buffer,
        &selection_buffer,
    );

    modifier.basic_color_modifiers_buffer.update_with_pod(
        &ctx.queue,
        &BasicColorModifiersPod::new(
            BasicColorRgbOverrideOrHsvModifiersPod::new_rgb_override(Vec3::new(1.0, 0.0, 0.0)),
            1.0,
            0.0,
            0.0,
            1.0,
        ),
    );

    let mut encoder = ctx
        .device
        .create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("Test Command Encoder"),
        });

    modifier.apply(
        &ctx.device,
        &mut encoder,
        &editor.gaussians_buffer,
        &editor.model_transform_buffer,
        &editor.gaussian_transform_buffer,
    );

    ctx.queue.submit(Some(encoder.finish()));

    let downloaded = pollster::block_on(
        editor
            .gaussians_buffer
            .download_gaussians(&ctx.device, &ctx.queue),
    )
    .expect("download_gaussians");

    assert_eq!(downloaded.len(), 2);
    assert_eq!(downloaded[0].color.xyz(), U8Vec3::new(255, 0, 0));
    assert_eq!(downloaded[1].color.xyz(), gaussians[1].color.xyz());
}

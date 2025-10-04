use assert_matches::assert_matches;
use wgpu_3dgs_editor::{
    BasicColorModifiersPod, BasicModifier, Editor, Modifier, NonDestructiveModifier,
    NonDestructiveModifierCreateError,
    core::{
        GaussianPodWithShSingleCov3dRotScaleConfigs, Gaussians, GaussiansBuffer,
        GaussiansBufferUpdateError, glam::*,
    },
};

use crate::common::{TestContext, given};

type G = GaussianPodWithShSingleCov3dRotScaleConfigs;

#[test]
fn test_non_destructive_modifier_new_should_create_copy_of_original_gaussians() {
    let ctx = TestContext::new();
    let gaussians = given::gaussians();
    let editor = Editor::<G>::new(&ctx.device, &gaussians);

    let modifier = BasicModifier::new(
        &ctx.device,
        &editor.gaussians_buffer,
        &editor.model_transform_buffer,
        &editor.gaussian_transform_buffer,
    );

    let non_destructive_modifier =
        NonDestructiveModifier::new(&ctx.device, &ctx.queue, modifier, &editor.gaussians_buffer)
            .expect("non-destructive modifier");

    assert_eq!(
        non_destructive_modifier.original_gaussians.len(),
        editor.gaussians_buffer.len()
    );

    let original_downloaded = pollster::block_on(
        non_destructive_modifier
            .original_gaussians
            .download_gaussians(&ctx.device, &ctx.queue),
    )
    .expect("download_gaussians");

    assert_eq!(original_downloaded.len(), 2);
    assert_eq!(original_downloaded[0].pos, gaussians.gaussians[0].pos);
    assert_eq!(original_downloaded[1].pos, gaussians.gaussians[1].pos);
    assert_eq!(
        original_downloaded[0].color.xyz(),
        gaussians.gaussians[0].color.xyz()
    );
    assert_eq!(
        original_downloaded[1].color.xyz(),
        gaussians.gaussians[1].color.xyz()
    );
}

#[test]
fn test_non_destructive_modifier_new_without_copy_src_usage_should_return_error() {
    let ctx = TestContext::new();
    let gaussians = given::gaussians();

    let gaussians_buffer = GaussiansBuffer::<G>::new_with_usage(
        &ctx.device,
        &gaussians.gaussians,
        wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
    );

    let model_transform_buffer = wgpu_3dgs_editor::core::ModelTransformBuffer::new(&ctx.device);
    let gaussian_transform_buffer =
        wgpu_3dgs_editor::core::GaussianTransformBuffer::new(&ctx.device);

    let modifier = BasicModifier::new(
        &ctx.device,
        &gaussians_buffer,
        &model_transform_buffer,
        &gaussian_transform_buffer,
    );

    let result = NonDestructiveModifier::new(&ctx.device, &ctx.queue, modifier, &gaussians_buffer);

    assert_matches!(
        result,
        Err(NonDestructiveModifierCreateError::MissingCopySrcBufferUsage)
    );
}

#[test]
fn test_non_destructive_modifier_apply_should_not_modify_original_gaussians() {
    let ctx = TestContext::new();
    let gaussians = given::gaussians();
    let editor = Editor::<G>::new(&ctx.device, &gaussians);

    let modifier = BasicModifier::new(
        &ctx.device,
        &editor.gaussians_buffer,
        &editor.model_transform_buffer,
        &editor.gaussian_transform_buffer,
    );

    let non_destructive_modifier =
        NonDestructiveModifier::new(&ctx.device, &ctx.queue, modifier, &editor.gaussians_buffer)
            .expect("non-destructive modifier");

    non_destructive_modifier
        .modifier
        .basic_color_modifiers_buffer
        .update_with_pod(
            &ctx.queue,
            &BasicColorModifiersPod::new_with_override_rgb(
                Vec3::new(1.0, 0.0, 0.0),
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

    non_destructive_modifier.apply(
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

    let original_downloaded = pollster::block_on(
        non_destructive_modifier
            .original_gaussians
            .download_gaussians(&ctx.device, &ctx.queue),
    )
    .expect("download_gaussians");

    assert_eq!(original_downloaded.len(), 2);
    assert_eq!(
        original_downloaded[0].color.xyz(),
        gaussians.gaussians[0].color.xyz()
    );
    assert_eq!(
        original_downloaded[1].color.xyz(),
        gaussians.gaussians[1].color.xyz()
    );
}

#[test]
fn test_non_destructive_modifier_apply_multiple_times_should_only_have_result_from_latest() {
    let ctx = TestContext::new();
    let gaussians = given::gaussians();
    let editor = Editor::<G>::new(&ctx.device, &gaussians);

    let modifier = BasicModifier::new(
        &ctx.device,
        &editor.gaussians_buffer,
        &editor.model_transform_buffer,
        &editor.gaussian_transform_buffer,
    );

    let non_destructive_modifier =
        NonDestructiveModifier::new(&ctx.device, &ctx.queue, modifier, &editor.gaussians_buffer)
            .expect("non-destructive modifier");

    non_destructive_modifier
        .modifier
        .basic_color_modifiers_buffer
        .update_with_pod(
            &ctx.queue,
            &BasicColorModifiersPod::new_with_override_rgb(
                Vec3::new(1.0, 0.0, 0.0),
                1.0,
                0.0,
                0.0,
                1.0,
            ),
        );

    let mut encoder = ctx
        .device
        .create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("Test Command Encoder 1"),
        });

    non_destructive_modifier.apply(
        &ctx.device,
        &mut encoder,
        &editor.gaussians_buffer,
        &editor.model_transform_buffer,
        &editor.gaussian_transform_buffer,
    );

    ctx.queue.submit(Some(encoder.finish()));

    let downloaded_red = pollster::block_on(
        editor
            .gaussians_buffer
            .download_gaussians(&ctx.device, &ctx.queue),
    )
    .expect("download_gaussians");

    assert_eq!(downloaded_red[0].color.xyz(), U8Vec3::new(255, 0, 0));
    assert_eq!(downloaded_red[1].color.xyz(), U8Vec3::new(255, 0, 0));

    non_destructive_modifier
        .modifier
        .basic_color_modifiers_buffer
        .update_with_pod(
            &ctx.queue,
            &BasicColorModifiersPod::new_with_override_rgb(
                Vec3::new(0.0, 1.0, 0.0),
                1.0,
                0.0,
                0.0,
                1.0,
            ),
        );

    let mut encoder = ctx
        .device
        .create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("Test Command Encoder 2"),
        });

    non_destructive_modifier.apply(
        &ctx.device,
        &mut encoder,
        &editor.gaussians_buffer,
        &editor.model_transform_buffer,
        &editor.gaussian_transform_buffer,
    );

    ctx.queue.submit(Some(encoder.finish()));

    let downloaded_green = pollster::block_on(
        editor
            .gaussians_buffer
            .download_gaussians(&ctx.device, &ctx.queue),
    )
    .expect("download_gaussians");

    assert_eq!(downloaded_green[0].color.xyz(), U8Vec3::new(0, 255, 0));
    assert_eq!(downloaded_green[1].color.xyz(), U8Vec3::new(0, 255, 0));

    let original_downloaded = pollster::block_on(
        non_destructive_modifier
            .original_gaussians
            .download_gaussians(&ctx.device, &ctx.queue),
    )
    .expect("download_gaussians");

    assert_eq!(
        original_downloaded[0].color.xyz(),
        gaussians.gaussians[0].color.xyz()
    );
    assert_eq!(
        original_downloaded[1].color.xyz(),
        gaussians.gaussians[1].color.xyz()
    );
}

#[test]
fn test_non_destructive_modifier_try_apply_with_mismatched_count_should_return_error() {
    let ctx = TestContext::new();
    let gaussians = given::gaussians();
    let editor = Editor::<G>::new(&ctx.device, &gaussians);

    let modifier = BasicModifier::new(
        &ctx.device,
        &editor.gaussians_buffer,
        &editor.model_transform_buffer,
        &editor.gaussian_transform_buffer,
    );

    let non_destructive_modifier =
        NonDestructiveModifier::new(&ctx.device, &ctx.queue, modifier, &editor.gaussians_buffer)
            .expect("non-destructive modifier");

    let different_gaussians = Gaussians {
        gaussians: vec![
            given::gaussian_with_seed(1),
            given::gaussian_with_seed(2),
            given::gaussian_with_seed(3),
        ],
    };
    let different_gaussians_buffer =
        GaussiansBuffer::<G>::new(&ctx.device, &different_gaussians.gaussians);

    let mut encoder = ctx
        .device
        .create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("Test Command Encoder"),
        });

    let result = non_destructive_modifier.try_apply(
        &ctx.device,
        &mut encoder,
        &different_gaussians_buffer,
        &editor.model_transform_buffer,
        &editor.gaussian_transform_buffer,
    );

    assert_matches!(
        result,
        Err(GaussiansBufferUpdateError::CountMismatch {
            count: 3,
            expected_count: 2,
        })
    )
}

#[test]
fn test_non_destructive_modifier_try_apply_with_should_use_custom_apply_function() {
    use std::{cell::Cell, rc::Rc};

    let ctx = TestContext::new();
    let gaussians = given::gaussians();
    let editor = Editor::<G>::new(&ctx.device, &gaussians);

    let modifier = BasicModifier::new(
        &ctx.device,
        &editor.gaussians_buffer,
        &editor.model_transform_buffer,
        &editor.gaussian_transform_buffer,
    );

    let non_destructive_modifier =
        NonDestructiveModifier::new(&ctx.device, &ctx.queue, modifier, &editor.gaussians_buffer)
            .expect("non-destructive modifier");

    let mut encoder = ctx
        .device
        .create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("Test Command Encoder"),
        });

    let called = Rc::new(Cell::new(false));
    let result = non_destructive_modifier.try_apply_with(
        &mut encoder,
        &editor.gaussians_buffer,
        |_encoder, _modifier, _gaussians| called.set(true),
    );

    assert!(result.is_ok());

    ctx.queue.submit(Some(encoder.finish()));

    let downloaded = pollster::block_on(
        editor
            .gaussians_buffer
            .download_gaussians(&ctx.device, &ctx.queue),
    )
    .expect("download_gaussians");

    assert_eq!(downloaded.len(), 2);
    assert_eq!(
        downloaded[0].color.xyz(),
        gaussians.gaussians[0].color.xyz()
    );
    assert_eq!(
        downloaded[1].color.xyz(),
        gaussians.gaussians[1].color.xyz()
    );
}

#[test]
fn test_non_destructive_modifier_as_modifier_trait_should_work_correctly() {
    let ctx = TestContext::new();
    let gaussians = given::gaussians();
    let editor = Editor::<G>::new(&ctx.device, &gaussians);

    let modifier = BasicModifier::new(
        &ctx.device,
        &editor.gaussians_buffer,
        &editor.model_transform_buffer,
        &editor.gaussian_transform_buffer,
    );

    let non_destructive_modifier =
        NonDestructiveModifier::new(&ctx.device, &ctx.queue, modifier, &editor.gaussians_buffer)
            .expect("non-destructive modifier");

    non_destructive_modifier
        .modifier
        .basic_color_modifiers_buffer
        .update_with_pod(
            &ctx.queue,
            &BasicColorModifiersPod::new_with_override_rgb(
                Vec3::new(0.0, 0.0, 1.0),
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

    let modifier_trait: &dyn Modifier<G> = &non_destructive_modifier;
    modifier_trait.apply(
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
    assert_eq!(downloaded[0].color.xyz(), U8Vec3::new(0, 0, 255));
    assert_eq!(downloaded[1].color.xyz(), U8Vec3::new(0, 0, 255));
}

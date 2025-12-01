use wgpu_3dgs_editor::{
    BasicColorModifiersPod, BasicColorRgbOverrideOrHsvModifiersPod, BasicModifier, Editor,
    InvTransformBuffer, Modifier, SelectionBundle, SelectionExpr, SelectionModifier,
    core::{BufferWrapper, GaussianPodWithShSingleCov3dRotScaleConfigs, glam::*},
};

use crate::common::{TestContext, given};

type G = GaussianPodWithShSingleCov3dRotScaleConfigs;

#[test]
fn test_selection_modifier_apply_should_correctly_select_and_modify_gaussians() {
    let ctx = TestContext::new();
    let gaussians = given::gaussians();
    let editor = Editor::<G>::new(&ctx.device, &gaussians);

    let mut selection_modifier = SelectionModifier::new(
        &ctx.device,
        &editor.gaussians_buffer,
        vec![SelectionBundle::<G>::create_sphere_bundle(&ctx.device)],
        |selection_buffer| {
            BasicModifier::new_with_selection(
                &ctx.device,
                &editor.gaussians_buffer,
                &editor.model_transform_buffer,
                &editor.gaussian_transform_buffer,
                selection_buffer,
            )
        },
    );

    selection_modifier
        .modifier
        .basic_color_modifiers_buffer
        .update_with_pod(
            &ctx.queue,
            &BasicColorModifiersPod::new(
                BasicColorRgbOverrideOrHsvModifiersPod::new_rgb_override(Vec3::new(1.0, 0.0, 0.0)),
                1.0,
                0.0,
                0.0,
                1.0,
            ),
        );

    let inv_transform_buffer = InvTransformBuffer::new(&ctx.device);
    inv_transform_buffer.update_with_scale_rot_pos(
        &ctx.queue,
        Vec3::ONE,
        Quat::IDENTITY,
        gaussians[0].pos,
    );

    selection_modifier.selection_expr = SelectionExpr::selection(
        0,
        vec![
            selection_modifier.selection.bundles[0]
                .create_bind_group(
                    &ctx.device,
                    1,
                    [inv_transform_buffer.buffer().as_entire_binding()],
                )
                .expect("create_bind_group"),
        ],
    );

    let mut encoder = ctx
        .device
        .create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("Test Command Encoder"),
        });

    selection_modifier.apply(
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

#[test]
fn test_basic_selection_modifier_apply_should_correctly_select_and_modify_gaussians() {
    let ctx = TestContext::new();
    let gaussians = given::gaussians();
    let editor = Editor::<G>::new(&ctx.device, &gaussians);

    let mut selection_modifier = SelectionModifier::new_with_basic_modifier(
        &ctx.device,
        &editor.gaussians_buffer,
        &editor.model_transform_buffer,
        &editor.gaussian_transform_buffer,
        vec![SelectionBundle::<G>::create_sphere_bundle(&ctx.device)],
    );

    selection_modifier
        .modifier
        .basic_color_modifiers_buffer
        .update_with_pod(
            &ctx.queue,
            &BasicColorModifiersPod::new(
                BasicColorRgbOverrideOrHsvModifiersPod::new_rgb_override(Vec3::new(1.0, 0.0, 0.0)),
                1.0,
                0.0,
                0.0,
                1.0,
            ),
        );

    let inv_transform_buffer = InvTransformBuffer::new(&ctx.device);
    inv_transform_buffer.update_with_scale_rot_pos(
        &ctx.queue,
        Vec3::ONE,
        Quat::IDENTITY,
        gaussians[0].pos,
    );

    selection_modifier.selection_expr = SelectionExpr::selection(
        0,
        vec![
            selection_modifier.selection.bundles[0]
                .create_bind_group(
                    &ctx.device,
                    1,
                    [inv_transform_buffer.buffer().as_entire_binding()],
                )
                .expect("create_bind_group"),
        ],
    );

    let mut encoder = ctx
        .device
        .create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("Test Command Encoder"),
        });

    selection_modifier.apply(
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

use glam::*;
use pollster::FutureExt;
use wgpu::util::DeviceExt;
use wgpu_3dgs_editor::{
    BasicColorModifiersBuffer, BasicColorModifiersPod, RotScaleBuffer, RotScalePod, TransformFlags,
    TransformFlagsBuffer, core::BufferWrapper, core::FixedSizeBufferWrapper,
};

use crate::common::TestContext;

#[test]
fn test_basic_color_modifiers_buffer_new_should_return_correct_buffer() {
    let ctx = TestContext::new();
    let buffer = BasicColorModifiersBuffer::new(&ctx.device);

    assert_eq!(
        buffer.buffer().size(),
        std::mem::size_of::<BasicColorModifiersPod>() as wgpu::BufferAddress
    );
}

#[test]
fn test_basic_color_modifiers_buffer_update_with_override_rgb_should_update_buffer_correctly() {
    let ctx = TestContext::new();
    let buffer =
        BasicColorModifiersBuffer::try_from(ctx.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Test Basic Color Modifiers Buffer"),
            size: std::mem::size_of::<BasicColorModifiersPod>() as wgpu::BufferAddress,
            usage: BasicColorModifiersBuffer::DEFAULT_USAGES | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        }))
        .expect("try_from");

    let rgb = Vec3::new(1.0, 0.5, 0.25);
    let alpha = 0.8;
    let contrast = 0.2;
    let exposure = 0.5;
    let gamma = 2.2;
    let pod = BasicColorModifiersPod::new_with_override_rgb(rgb, alpha, contrast, exposure, gamma);

    buffer.update_with_override_rgb(&ctx.queue, rgb, alpha, contrast, exposure, gamma);

    let downloaded = buffer
        .download_single(&ctx.device, &ctx.queue)
        .block_on()
        .expect("download");

    assert_eq!(downloaded, pod);
}

#[test]
fn test_basic_color_modifiers_buffer_update_with_hsv_modifiers_should_update_buffer_correctly() {
    let ctx = TestContext::new();
    let buffer =
        BasicColorModifiersBuffer::try_from(ctx.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Test Basic Color Modifiers Buffer"),
            size: std::mem::size_of::<BasicColorModifiersPod>() as wgpu::BufferAddress,
            usage: BasicColorModifiersBuffer::DEFAULT_USAGES | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        }))
        .expect("try_from");

    let hsv = Vec3::new(0.5, 1.0, 1.0);
    let alpha = 0.9;
    let contrast = 0.1;
    let exposure = 0.3;
    let gamma = 1.8;
    let pod = BasicColorModifiersPod::new_with_hsv_modifiers(hsv, alpha, contrast, exposure, gamma);

    buffer.update_with_hsv_modifiers(&ctx.queue, hsv, alpha, contrast, exposure, gamma);

    let downloaded = buffer
        .download_single(&ctx.device, &ctx.queue)
        .block_on()
        .expect("download");

    assert_eq!(downloaded, pod);
}

#[test]
fn test_basic_color_modifiers_buffer_try_from_and_into_wgpu_buffer_should_be_equal() {
    let ctx = TestContext::new();
    let pod = BasicColorModifiersPod::new_with_override_rgb(
        Vec3::new(1.0, 0.5, 0.25),
        0.8,
        0.2,
        0.5,
        2.2,
    );
    let wgpu_buffer = ctx
        .device
        .create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Test Basic Color Modifiers Buffer"),
            contents: bytemuck::bytes_of(&pod),
            usage: BasicColorModifiersBuffer::DEFAULT_USAGES | wgpu::BufferUsages::COPY_SRC,
        });

    let converted_buffer =
        BasicColorModifiersBuffer::try_from(wgpu_buffer.clone()).expect("try_from");
    let wgpu_converted_buffer = wgpu::Buffer::from(converted_buffer.clone());

    let wgpu_downloaded = wgpu_converted_buffer
        .download::<BasicColorModifiersPod>(&ctx.device, &ctx.queue)
        .block_on()
        .expect("download");
    let converted_downloaded = converted_buffer
        .download::<BasicColorModifiersPod>(&ctx.device, &ctx.queue)
        .block_on()
        .expect("download");
    let wgpu_converted_downloaded = wgpu_buffer
        .download::<BasicColorModifiersPod>(&ctx.device, &ctx.queue)
        .block_on()
        .expect("download");

    assert_eq!(wgpu_downloaded, converted_downloaded);
    assert_eq!(wgpu_downloaded, wgpu_converted_downloaded);
}

#[test]
fn test_basic_color_modifiers_pod_new_with_override_rgb_should_return_correct_pod() {
    let rgb = Vec3::new(1.0, 0.5, 0.25);
    let alpha = 0.8;
    let contrast = 0.2;
    let exposure = 0.5;
    let gamma = 2.2;
    let pod = BasicColorModifiersPod::new_with_override_rgb(rgb, alpha, contrast, exposure, gamma);

    assert_eq!(pod.rgb_or_hsv, -rgb);
    assert_eq!(pod.alpha, alpha);
    assert_eq!(pod.contrast, contrast);
    assert_eq!(pod.exposure, exposure);
    assert_eq!(pod.gamma, gamma);
}

#[test]
fn test_basic_color_modifiers_pod_new_with_hsv_modifiers_should_return_correct_pod() {
    let hsv = Vec3::new(0.5, 1.0, 1.0);
    let alpha = 0.9;
    let contrast = 0.1;
    let exposure = 0.3;
    let gamma = 1.8;
    let pod = BasicColorModifiersPod::new_with_hsv_modifiers(hsv, alpha, contrast, exposure, gamma);

    assert_eq!(pod.rgb_or_hsv, hsv);
    assert_eq!(pod.alpha, alpha);
    assert_eq!(pod.contrast, contrast);
    assert_eq!(pod.exposure, exposure);
    assert_eq!(pod.gamma, gamma);
}

#[test]
fn test_transform_flags_buffer_new_should_return_correct_buffer() {
    let ctx = TestContext::new();
    let buffer = TransformFlagsBuffer::new(&ctx.device);

    assert_eq!(
        buffer.buffer().size(),
        std::mem::size_of::<u32>() as wgpu::BufferAddress
    );
}

#[test]
fn test_transform_flags_buffer_update_should_update_buffer_correctly() {
    let ctx = TestContext::new();
    let buffer =
        TransformFlagsBuffer::try_from(ctx.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Test Transform Flags Buffer"),
            size: std::mem::size_of::<u32>() as wgpu::BufferAddress,
            usage: TransformFlagsBuffer::DEFAULT_USAGES | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        }))
        .expect("try_from");

    let flags = TransformFlags::MODEL | TransformFlags::GAUSSIAN;

    buffer.update(&ctx.queue, flags);

    let downloaded = buffer
        .download_single(&ctx.device, &ctx.queue)
        .block_on()
        .expect("download");

    assert_eq!(downloaded, flags.bits());
}

#[test]
fn test_transform_flags_buffer_try_from_and_into_wgpu_buffer_should_be_equal() {
    let ctx = TestContext::new();
    let flags = TransformFlags::MODEL;
    let wgpu_buffer = ctx
        .device
        .create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Test Transform Flags Buffer"),
            contents: bytemuck::bytes_of(&flags.bits()),
            usage: TransformFlagsBuffer::DEFAULT_USAGES | wgpu::BufferUsages::COPY_SRC,
        });

    let converted_buffer = TransformFlagsBuffer::try_from(wgpu_buffer.clone()).expect("try_from");
    let wgpu_converted_buffer = wgpu::Buffer::from(converted_buffer.clone());

    let wgpu_downloaded = wgpu_converted_buffer
        .download::<u32>(&ctx.device, &ctx.queue)
        .block_on()
        .expect("download");
    let converted_downloaded = converted_buffer
        .download::<u32>(&ctx.device, &ctx.queue)
        .block_on()
        .expect("download");
    let wgpu_converted_downloaded = wgpu_buffer
        .download::<u32>(&ctx.device, &ctx.queue)
        .block_on()
        .expect("download");

    assert_eq!(wgpu_downloaded, converted_downloaded);
    assert_eq!(wgpu_downloaded, wgpu_converted_downloaded);
}

#[test]
fn test_rot_scale_buffer_new_should_return_correct_buffer() {
    let ctx = TestContext::new();
    let buffer = RotScaleBuffer::new(&ctx.device);

    assert_eq!(
        buffer.buffer().size(),
        std::mem::size_of::<RotScalePod>() as wgpu::BufferAddress
    );
}

#[test]
fn test_rot_scale_buffer_update_should_update_buffer_correctly() {
    let ctx = TestContext::new();
    let buffer = RotScaleBuffer::try_from(ctx.device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("Test Rot Scale Buffer"),
        size: std::mem::size_of::<RotScalePod>() as wgpu::BufferAddress,
        usage: RotScaleBuffer::DEFAULT_USAGES | wgpu::BufferUsages::COPY_SRC,
        mapped_at_creation: false,
    }))
    .expect("try_from");

    let rot = Quat::from_rotation_y(std::f32::consts::PI / 4.0);
    let scale = Vec3::new(2.0, 3.0, 4.0);
    let pod = RotScalePod::new(rot, scale);

    buffer.update(&ctx.queue, rot, scale);

    let downloaded = buffer
        .download_single(&ctx.device, &ctx.queue)
        .block_on()
        .expect("download");

    assert_eq!(downloaded, pod);
}

#[test]
fn test_rot_scale_buffer_try_from_and_into_wgpu_buffer_should_be_equal() {
    let ctx = TestContext::new();
    let pod = RotScalePod::new(
        Quat::from_rotation_y(std::f32::consts::PI / 4.0),
        Vec3::new(2.0, 3.0, 4.0),
    );
    let wgpu_buffer = ctx
        .device
        .create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Test Rot Scale Buffer"),
            contents: bytemuck::bytes_of(&pod),
            usage: RotScaleBuffer::DEFAULT_USAGES | wgpu::BufferUsages::COPY_SRC,
        });

    let converted_buffer = RotScaleBuffer::try_from(wgpu_buffer.clone()).expect("try_from");
    let wgpu_converted_buffer = wgpu::Buffer::from(converted_buffer.clone());

    let wgpu_downloaded = wgpu_converted_buffer
        .download::<RotScalePod>(&ctx.device, &ctx.queue)
        .block_on()
        .expect("download");
    let converted_downloaded = converted_buffer
        .download::<RotScalePod>(&ctx.device, &ctx.queue)
        .block_on()
        .expect("download");
    let wgpu_converted_downloaded = wgpu_buffer
        .download::<RotScalePod>(&ctx.device, &ctx.queue)
        .block_on()
        .expect("download");

    assert_eq!(wgpu_downloaded, converted_downloaded);
    assert_eq!(wgpu_downloaded, wgpu_converted_downloaded);
}

#[test]
fn test_rot_scale_pod_new_should_return_correct_pod() {
    let rot = Quat::from_rotation_y(std::f32::consts::PI / 4.0);
    let scale = Vec3::new(2.0, 3.0, 4.0);
    let pod = RotScalePod::new(rot, scale);

    assert_eq!(pod.rot, rot);
    assert_eq!(pod.scale, scale.to_vec3a());
}

use glam::*;
use wgpu::util::DeviceExt;
use wgpu_3dgs_editor::{
    InvTransformBuffer, SelectionBuffer, SelectionOpBuffer,
    core::{BufferWrapper, DownloadableBufferWrapper},
};

use crate::common::TestContext;

#[test]
fn test_selection_buffer_new_should_return_correct_buffer() {
    let ctx = TestContext::new();
    let gaussian_count = 100;
    let buffer = SelectionBuffer::new(&ctx.device, gaussian_count);

    let expected_size =
        (gaussian_count.div_ceil(32) * std::mem::size_of::<u32>() as u32) as wgpu::BufferAddress;
    assert_eq!(buffer.buffer().size(), expected_size);
}

#[test]
fn test_selection_buffer_new_with_label_should_return_correct_buffer() {
    let ctx = TestContext::new();
    let gaussian_count = 64;
    let label = "Test";
    let buffer = SelectionBuffer::new_with_label(&ctx.device, label, gaussian_count);

    let expected_size =
        (gaussian_count.div_ceil(32) * std::mem::size_of::<u32>() as u32) as wgpu::BufferAddress;
    assert_eq!(buffer.buffer().size(), expected_size);
}

#[test]
fn test_selection_buffer_from_and_into_wgpu_buffer_should_be_equal() {
    let ctx = TestContext::new();
    let selection: [u32; 10] = std::array::from_fn(|i| i as u32);
    let wgpu_buffer = ctx
        .device
        .create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Test Selection Buffer"),
            contents: bytemuck::cast_slice(&selection),
            usage: SelectionBuffer::DEFAULT_USAGES | wgpu::BufferUsages::COPY_SRC,
        });

    let converted_buffer = SelectionBuffer::from(wgpu_buffer.clone());
    let wgpu_converted_buffer = wgpu::Buffer::from(converted_buffer.clone());

    let wgpu_downloaded =
        pollster::block_on(wgpu_converted_buffer.download::<u32>(&ctx.device, &ctx.queue))
            .expect("download");
    let converted_downloaded =
        pollster::block_on(converted_buffer.download::<u32>(&ctx.device, &ctx.queue))
            .expect("download");
    let wgpu_converted_downloaded =
        pollster::block_on(wgpu_buffer.download::<u32>(&ctx.device, &ctx.queue)).expect("download");

    assert_eq!(wgpu_downloaded, converted_downloaded);
    assert_eq!(wgpu_downloaded, wgpu_converted_downloaded);
}

#[test]
fn test_selection_op_buffer_new_should_return_correct_buffer() {
    let ctx = TestContext::new();
    let op = 42u32;
    let buffer = SelectionOpBuffer::new(&ctx.device, op);

    assert_eq!(
        buffer.buffer().size(),
        std::mem::size_of::<u32>() as wgpu::BufferAddress
    );
}

#[test]
fn test_selection_op_buffer_update_should_update_buffer_correctly() {
    let ctx = TestContext::new();
    let initial_op = 1u32;
    let buffer = SelectionOpBuffer::try_from(ctx.device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("Test Selection Op Buffer"),
        size: std::mem::size_of::<u32>() as wgpu::BufferAddress,
        usage: SelectionOpBuffer::DEFAULT_USAGES | wgpu::BufferUsages::COPY_SRC,
        mapped_at_creation: false,
    }))
    .expect("try_from");

    buffer.update(&ctx.queue, initial_op);
    let downloaded =
        pollster::block_on(buffer.download::<u32>(&ctx.device, &ctx.queue)).expect("download")[0];
    assert_eq!(downloaded, initial_op);

    let new_op = 5u32;
    buffer.update(&ctx.queue, new_op);
    let downloaded =
        pollster::block_on(buffer.download::<u32>(&ctx.device, &ctx.queue)).expect("download")[0];
    assert_eq!(downloaded, new_op);
}

#[test]
fn test_selection_op_buffer_try_from_and_into_wgpu_buffer_should_be_equal() {
    let ctx = TestContext::new();
    let wgpu_buffer = ctx
        .device
        .create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Test Selection Op Buffer"),
            contents: bytemuck::bytes_of(&3u32),
            usage: SelectionOpBuffer::DEFAULT_USAGES | wgpu::BufferUsages::COPY_SRC,
        });

    let converted_buffer = SelectionOpBuffer::try_from(wgpu_buffer.clone()).expect("try_from");
    let wgpu_converted_buffer = wgpu::Buffer::from(converted_buffer.clone());

    let op_value = 7u32;
    converted_buffer.update(&ctx.queue, op_value);

    let wgpu_downloaded =
        pollster::block_on(wgpu_converted_buffer.download::<u32>(&ctx.device, &ctx.queue))
            .expect("download");
    let converted_downloaded =
        pollster::block_on(converted_buffer.download::<u32>(&ctx.device, &ctx.queue))
            .expect("download");
    let wgpu_converted_downloaded =
        pollster::block_on(wgpu_buffer.download::<u32>(&ctx.device, &ctx.queue)).expect("download");

    assert_eq!(wgpu_downloaded, converted_downloaded);
    assert_eq!(wgpu_downloaded, wgpu_converted_downloaded);
}

#[test]
fn test_inv_transform_buffer_new_should_return_correct_buffer() {
    let ctx = TestContext::new();
    let buffer = InvTransformBuffer::new(&ctx.device);

    assert_eq!(
        buffer.buffer().size(),
        std::mem::size_of::<Mat4>() as wgpu::BufferAddress
    );
}

#[test]
fn test_inv_transform_buffer_update_should_update_buffer_correctly() {
    let ctx = TestContext::new();
    let buffer = InvTransformBuffer::try_from(ctx.device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("Test Inv Transform Buffer"),
        size: std::mem::size_of::<Mat4>() as wgpu::BufferAddress,
        usage: InvTransformBuffer::DEFAULT_USAGES | wgpu::BufferUsages::COPY_SRC,
        mapped_at_creation: false,
    }))
    .expect("try_from");

    let inv_transform = Mat4::from_rotation_translation(
        Quat::from_rotation_y(std::f32::consts::PI / 4.0),
        Vec3::new(1.0, 2.0, 3.0),
    );

    buffer.update(&ctx.queue, inv_transform);

    let downloaded =
        pollster::block_on(buffer.download::<Mat4>(&ctx.device, &ctx.queue)).expect("download")[0];

    assert_eq!(downloaded, inv_transform);
}

#[test]
fn test_inv_transform_buffer_update_with_scale_rot_pos_should_update_buffer_correctly() {
    let ctx = TestContext::new();
    let buffer = InvTransformBuffer::try_from(ctx.device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("Test Inv Transform Buffer"),
        size: std::mem::size_of::<Mat4>() as wgpu::BufferAddress,
        usage: InvTransformBuffer::DEFAULT_USAGES | wgpu::BufferUsages::COPY_SRC,
        mapped_at_creation: false,
    }))
    .expect("try_from");

    let scale = Vec3::new(2.0, 3.0, 4.0);
    let rot = Quat::from_rotation_y(std::f32::consts::PI / 4.0);
    let pos = Vec3::new(1.0, 2.0, 3.0);
    let expected_inv_transform = Mat4::from_scale_rotation_translation(scale, rot, pos).inverse();

    buffer.update_with_scale_rot_pos(&ctx.queue, scale, rot, pos);

    let downloaded =
        pollster::block_on(buffer.download::<Mat4>(&ctx.device, &ctx.queue)).expect("download")[0];

    assert_eq!(downloaded, expected_inv_transform);
}

#[test]
fn test_inv_transform_buffer_try_from_and_into_wgpu_buffer_should_be_equal() {
    let ctx = TestContext::new();
    let wgpu_buffer = ctx
        .device
        .create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Test Inv Transform Buffer"),
            contents: bytemuck::bytes_of(&Mat4::IDENTITY),
            usage: InvTransformBuffer::DEFAULT_USAGES | wgpu::BufferUsages::COPY_SRC,
        });

    let converted_buffer = InvTransformBuffer::try_from(wgpu_buffer.clone()).expect("try_from");
    let wgpu_converted_buffer = wgpu::Buffer::from(converted_buffer.clone());

    let inv_transform = Mat4::IDENTITY;
    converted_buffer.update(&ctx.queue, inv_transform);

    let wgpu_downloaded =
        pollster::block_on(wgpu_converted_buffer.download::<Mat4>(&ctx.device, &ctx.queue))
            .expect("download");
    let converted_downloaded =
        pollster::block_on(converted_buffer.download::<Mat4>(&ctx.device, &ctx.queue))
            .expect("download");
    let wgpu_converted_downloaded =
        pollster::block_on(wgpu_buffer.download::<Mat4>(&ctx.device, &ctx.queue))
            .expect("download");

    assert_eq!(wgpu_downloaded, converted_downloaded);
    assert_eq!(wgpu_downloaded, wgpu_converted_downloaded);
}

//! This example selects parts of the model, then filters the model to keep only the selected parts
//! using [`SelectionBundle`](wgpu_3dgs_editor::SelectionBundle) and [`SelectionExpr`](wgpu_3dgs_editor::SelectionExpr).
//!
//! For example, to select a sphere at (0.5, 1.0, 0.5) with scale (1, 1, 1),
//! repeat the selection 2 times with an offset (2, 0, 0), and keep only the selected Gaussians:
//!
//! ```sh
//! cargo run --example filter-selection -- \
//!     -m "path/to/model.ply" \
//!     -p 0.5 1.0 0.5 -s 1 1 1 --repeat 2 --offset 2.0 0.0 0.0
//! ```

use clap::{Parser, ValueEnum};
use glam::*;

use wgpu_3dgs_core::IterGaussian;
use wgpu_3dgs_editor::{self as gs, core::BufferWrapper};

/// The command line arguments.
#[derive(Parser, Debug)]
#[command(
    version,
    about,
    long_about = "\
    A 3D Gaussian splatting editor to filter selected Gaussians in a model.
    "
)]
struct Args {
    /// Path to the .ply file.
    #[arg(short, long, default_value = "examples/model.ply")]
    model: String,

    /// The output path for the modified .ply file.
    #[arg(short, long, default_value = "target/output.ply")]
    output: String,

    /// The position of the selection shape.
    #[arg(
        short,
        long,
        allow_hyphen_values = true,
        num_args = 3,
        value_delimiter = ',',
        default_value = "0.0,0.0,0.0"
    )]
    pos: Vec<f32>,

    /// The rotation of the selection shape.
    #[arg(
        short,
        long,
        allow_hyphen_values = true,
        num_args = 4,
        value_delimiter = ',',
        default_value = "0.0,0.0,0.0,1.0"
    )]
    rot: Vec<f32>,

    /// The scale of the selection shape.
    #[arg(
        short,
        long,
        allow_hyphen_values = true,
        num_args = 3,
        value_delimiter = ',',
        default_value = "0.5,1.0,2.0"
    )]
    scale: Vec<f32>,

    /// The shape of the selection.
    #[arg(long, value_enum, default_value_t = Shape::Sphere, ignore_case = true)]
    shape: Shape,

    /// The number of times to run the selection.
    #[arg(long, default_value = "1")]
    repeat: u32,

    /// The offset of each selection.
    #[arg(
        long,
        allow_hyphen_values = true,
        num_args = 3,
        value_delimiter = ',',
        default_value = "2.0,0.0,0.0"
    )]
    offset: Vec<f32>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, ValueEnum)]
enum Shape {
    Sphere,
    Box,
}

type GaussianPod = gs::core::GaussianPodWithShSingleCov3dSingleConfigs;

#[pollster::main]
async fn main() {
    env_logger::Builder::from_env(env_logger::Env::default().default_filter_or("info")).init();

    let args = Args::parse();
    let model_path = &args.model;
    let pos = Vec3::from_slice(&args.pos);
    let rot = Quat::from_slice(&args.rot);
    let scale = Vec3::from_slice(&args.scale);
    let shape = match args.shape {
        Shape::Sphere => gs::SelectionBundle::<GaussianPod>::create_sphere_bundle,
        Shape::Box => gs::SelectionBundle::<GaussianPod>::create_box_bundle,
    };
    let repeat = args.repeat;
    let offset = Vec3::from_slice(&args.offset);

    log::debug!("Creating wgpu instance");
    let instance = wgpu::Instance::new(&wgpu::InstanceDescriptor::default());

    log::debug!("Requesting adapter");
    let adapter = instance
        .request_adapter(&wgpu::RequestAdapterOptions::default())
        .await
        .expect("adapter");

    log::debug!("Requesting device");
    let (device, queue) = adapter
        .request_device(&wgpu::DeviceDescriptor {
            label: Some("Device"),
            required_limits: adapter.limits(),
            ..Default::default()
        })
        .await
        .expect("device");

    log::debug!("Creating gaussians");
    let gaussians = [
        gs::core::GaussiansSource::Ply,
        gs::core::GaussiansSource::Spz,
    ]
    .into_iter()
    .find_map(|source| gs::core::Gaussians::read_from_file(model_path, source).ok())
    .expect("gaussians");

    log::debug!("Creating gaussians buffer");
    let gaussians_buffer = gs::core::GaussiansBuffer::<GaussianPod>::new(&device, &gaussians);

    log::debug!("Creating model transform buffer");
    let model_transform = gs::core::ModelTransformBuffer::new(&device);

    log::debug!("Creating Gaussian transform buffer");
    let gaussian_transform = gs::core::GaussianTransformBuffer::new(&device);

    log::debug!("Creating shape selection compute bundle");
    let shape_selection = shape(&device);

    log::debug!("Creating selection bundle");
    let selection_bundle = gs::SelectionBundle::<GaussianPod>::new(&device, vec![shape_selection]);

    log::debug!("Creating shape selection buffers");
    let shape_selection_buffers = (0..repeat)
        .map(|i| {
            let offset_pos = pos + offset * i as f32;
            let buffer = gs::InvTransformBuffer::new(&device);
            buffer.update_with_scale_rot_pos(&queue, scale, rot, offset_pos);
            buffer
        })
        .collect::<Vec<_>>();

    log::debug!("Creating shape selection bind groups");
    let shape_selection_bind_groups = shape_selection_buffers
        .iter()
        .map(|buffer| {
            selection_bundle.bundles[0]
                .create_bind_group(
                    &device,
                    // index 0 is the Gaussians buffer, so we use 1,
                    // see docs of create_sphere_bundle or create_box_bundle
                    1,
                    [buffer.buffer().as_entire_binding()],
                )
                .expect("bind group")
        })
        .collect::<Vec<_>>();

    log::debug!("Creating selection expression");
    let selection_expr = shape_selection_bind_groups.into_iter().fold(
        gs::SelectionExpr::Identity,
        |acc, bind_group| {
            acc.union(gs::SelectionExpr::selection(
                0, // the 0 here is the bundle index in the selection bundle
                vec![bind_group],
            ))
        },
    );

    log::debug!("Creating destination buffer");
    let dest = gs::SelectionBuffer::new(&device, gaussians_buffer.len() as u32);

    log::info!("Starting selection process");
    let time = std::time::Instant::now();

    log::debug!("Selecting Gaussians");
    let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
        label: Some("Selection Encoder"),
    });

    selection_bundle.evaluate(
        &device,
        &mut encoder,
        &selection_expr,
        &dest,
        &model_transform,
        &gaussian_transform,
        &gaussians_buffer,
    );

    queue.submit(Some(encoder.finish()));

    device
        .poll(wgpu::PollType::wait_indefinitely())
        .expect("poll");

    log::info!("Editing process completed in {:?}", time.elapsed());

    log::debug!("Filtering Gaussians");
    let selected_gaussians = dest
        .download::<u32>(&device, &queue)
        .await
        .expect("selected download")
        .iter()
        .flat_map(|group| {
            std::iter::repeat_n(group, 32)
                .enumerate()
                .map(|(i, g)| g & (1 << i) != 0)
        })
        .zip(gaussians.iter_gaussian())
        .filter(|(selected, _)| *selected)
        .map(|(_, g)| g)
        .collect::<Vec<_>>();

    let selected_gaussians = match &args.output[args.output.len().saturating_sub(4)..] {
        ".ply" => gs::core::Gaussians::Ply(gs::core::PlyGaussians::from_iter(
            selected_gaussians.into_iter(),
        )),
        ".spz" => {
            gs::core::Gaussians::Spz(
                gs::core::SpzGaussians::from_gaussians_with_options(
                    selected_gaussians,
                    &gs::core::SpzGaussiansFromGaussianSliceOptions {
                        version: 2, // Version 2 is more widely supported as of now
                        ..Default::default()
                    },
                )
                .expect("SpzGaussians from gaussians"),
            )
        }
        _ => panic!("Unsupported output file extension, expected .ply or .spz"),
    };

    log::debug!("Writing modified Gaussians to output file");
    selected_gaussians
        .write_to_file(&args.output)
        .expect("write modified Gaussians to output file");

    log::info!("Filtered Gaussians written to {}", args.output);
}

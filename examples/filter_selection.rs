use clap::{Parser, ValueEnum};
use glam::*;

use wgpu_3dgs_core::{DownloadableBufferWrapper, GaussianPodWithShSingleCov3dSingleConfigs};
use wgpu_3dgs_editor as gs;

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
    #[arg(short, long)]
    model: String,

    /// The output path for the modified .ply file.
    #[arg(short, long, default_value = "target/output.ply")]
    output: String,

    /// The position of the selection shape.
    #[arg(
        short,
        long,
        num_args = 3,
        value_delimiter = ',',
        default_value = "0.0,0.0,0.0"
    )]
    position: Vec<f32>,

    /// The rotation of the selection shape.
    #[arg(
        short,
        long,
        num_args = 4,
        value_delimiter = ',',
        default_value = "0.0,0.0,0.0,1.0"
    )]
    rotation: Vec<f32>,

    /// The scale of the selection shape.
    #[arg(
        short,
        long,
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

type GaussianPod = GaussianPodWithShSingleCov3dSingleConfigs;

#[tokio::main]
async fn main() {
    env_logger::Builder::from_env(env_logger::Env::default().default_filter_or("info")).init();

    let args = Args::parse();
    let model_path = &args.model;
    let position = Vec3::from_slice(&args.position);
    let rotation = Quat::from_slice(&args.rotation);
    let scale = Vec3::from_slice(&args.scale);
    let shape = match args.shape {
        Shape::Sphere => gs::selection_bundle::sphere::<GaussianPod>,
        Shape::Box => gs::selection_bundle::r#box::<GaussianPod>,
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
            required_features: wgpu::Features::empty(),
            required_limits: adapter.limits(),
            memory_hints: wgpu::MemoryHints::default(),
            trace: wgpu::Trace::Off,
        })
        .await
        .expect("device");

    log::debug!("Creating gaussians");
    let f = std::fs::File::open(model_path).expect("ply file");
    let mut reader = std::io::BufReader::new(f);
    let gaussians = gs::core::Gaussians::read_ply(&mut reader).expect("gaussians");

    log::debug!("Creating gaussians buffer");
    let gaussians_buffer =
        gs::core::GaussiansBuffer::<GaussianPod>::new(&device, &gaussians.gaussians);

    log::debug!("Creating model transform buffer");
    let model_transform = gs::core::ModelTransformBuffer::new(&device);

    log::debug!("Creating Gaussian transform buffer");
    let gaussian_transform = gs::core::GaussianTransformBuffer::new(&device);

    log::debug!("Creating shape selection compute bundle");
    let shape_selection = shape(&device);

    log::debug!("Creating selection bundle");
    let selection_bundle = gs::SelectionBundle::new::<GaussianPod>(&device, vec![shape_selection]);

    log::debug!("Creating shape selection buffers");
    let shape_selection_buffers = (0..repeat)
        .map(|i| {
            let offset_pos = position + offset * i as f32;
            let buffer = gs::InvTransformBuffer::new(&device);
            buffer.update_with_scale_rotation_position(&queue, scale, rotation, offset_pos);
            buffer
        })
        .collect::<Vec<_>>();

    log::debug!("Creating shape selection bind groups");
    let shape_selection_bind_groups = shape_selection_buffers
        .iter()
        .map(|buffer| {
            selection_bundle.bundles[0]
                .create_bind_group(&device, 1, [buffer as &dyn gs::core::BufferWrapper])
                .expect("bind group")
        })
        .collect::<Vec<_>>();

    log::debug!("Creating selection expression");
    let selection_expr = shape_selection_bind_groups
        .into_iter()
        .fold(gs::SelectionExpr::Identity, |acc, bind_group| {
            acc.union(gs::SelectionExpr::selection(0, vec![bind_group]))
        });

    log::debug!("Creating destination buffer");
    let dest = gs::SelectionBuffer::new(&device, gaussians_buffer.len() as u32);

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

    log::debug!("Filtering Gaussians");
    let selected_gaussians = gs::core::Gaussians {
        gaussians: dest
            .download(&device, &queue)
            .await
            .expect("selected download")
            .iter()
            .flat_map(|group| {
                std::iter::repeat_n(group, 32)
                    .enumerate()
                    .map(|(i, g)| g & (1 << i) != 0)
            })
            .zip(gaussians.gaussians.iter())
            .filter(|(selected, _)| *selected)
            .map(|(_, g)| g.clone())
            .collect::<Vec<_>>(),
    };

    log::debug!("Writing modified Gaussians to output file");
    let output_file = std::fs::File::create(&args.output).expect("output file");
    let mut writer = std::io::BufWriter::new(output_file);
    selected_gaussians
        .write_ply(&mut writer)
        .expect("write modified Gaussians to output file");
}

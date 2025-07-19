use clap::Parser;
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
    #[arg(short, long, default_value = "output.ply")]
    output: String,

    /// The radii of the selection.
    #[arg(
        short,
        long,
        num_args = 3,
        value_delimiter = ',',
        default_value = "0.0,0.0,0.0"
    )]
    pos: Vec<f32>,

    /// The radii of the selection.
    #[arg(
        short,
        long,
        num_args = 4,
        value_delimiter = ',',
        default_value = "-0.324783,-0.324783,-0.1623915,-0.8733046"
    )]
    rot: Vec<f32>,

    /// The radii/scale of the selection.
    #[arg(
        short,
        long,
        num_args = 3,
        value_delimiter = ',',
        default_value = "3.0,4.0,5.0"
    )]
    scale: Vec<f32>,

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

type GaussianPod = GaussianPodWithShSingleCov3dSingleConfigs;

#[tokio::main]
async fn main() {
    env_logger::Builder::from_env(env_logger::Env::default().default_filter_or("info")).init();

    let args = Args::parse();
    let model_path = &args.model;
    let pos = Vec3::from_slice(&args.pos);
    let rot = Quat::from_slice(&args.rot);
    let radii = Vec3::from_slice(&args.scale);
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
    let mut gaussians = gs::core::Gaussians::read_ply(&mut reader).expect("gaussians");

    log::debug!("Creating gaussians buffer");
    let gaussians_buffer =
        gs::core::GaussiansBuffer::<GaussianPod>::new(&device, &gaussians.gaussians);

    log::debug!("Creating model transform buffer");
    let model_transform = gs::core::ModelTransformBuffer::new(&device);

    log::debug!("Creating Gaussian transform buffer");
    let gaussian_transform = gs::core::GaussianTransformBuffer::new(&device);

    log::debug!("Creating sphere selection compute bundle");
    let sphere_selection = gs::ops::sphere::<GaussianPod>(&device);

    log::debug!("Creating selection bundle");
    let selection_bundle = gs::SelectionBundle::new::<GaussianPod>(&device, vec![sphere_selection]);

    log::debug!("Creating sphere selection buffers");
    let sphere_selection_buffers = (0..repeat)
        .map(|i| {
            let offset_pos = pos + offset * i as f32;
            let buffer = gs::SphereSelectionBuffer::new(&device);
            buffer.update_with_pos_rot_radii(&queue, offset_pos, rot, radii);
            buffer
        })
        .collect::<Vec<_>>();

    log::debug!("Creating sphere selection bind groups");
    let sphere_selection_bind_groups = sphere_selection_buffers
        .iter()
        .map(|buffer| {
            selection_bundle.bundles[0]
                .create_bind_group(&device, 1, [buffer as &dyn gs::core::BufferWrapper])
                .expect("bind group")
        })
        .collect::<Vec<_>>();

    log::debug!("Creating selection expression");
    let selection_expr = sphere_selection_bind_groups
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

    log::debug!("Removing unslected Gaussians");
    let selected_download = dest
        .download(&device, &queue)
        .await
        .expect("selected download");

    gaussians.gaussians = selected_download
        .iter()
        .flat_map(|group| {
            std::iter::repeat_n(group, 32)
                .enumerate()
                .map(|(i, g)| g & (1 << i) != 0)
        })
        .zip(gaussians.gaussians.into_iter())
        .filter(|(selected, _)| *selected)
        .map(|(_, g)| g)
        .collect::<Vec<_>>();

    log::debug!("Writing modified Gaussians to output file");
    let output_file = std::fs::File::create(&args.output).expect("output file");
    let mut writer = std::io::BufWriter::new(output_file);
    gaussians
        .write_ply(&mut writer)
        .expect("write modified Gaussians to output file");
}

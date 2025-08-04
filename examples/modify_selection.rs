use clap::{Parser, ValueEnum};
use glam::*;

use wgpu_3dgs_editor::{self as gs};

/// The command line arguments.
#[derive(Parser, Debug)]
#[command(
    version,
    about,
    long_about = "\
    A 3D Gaussian splatting editor to apply basic modifiers to selected Gaussians in a model.
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

    /// Whether to override the RGB color of the selected Gaussians.
    #[arg(long)]
    override_rgb: bool,

    /// If [`Args::override_rgb`], then it is used to override the RGB color,
    /// otherwise it is used to apply HSV modifications.
    ///
    /// Normally hue (H) is in [0, 1], saturation (S) and value (V) are in [0, 2].
    /// This function adds the hue and multiplies saturation and value.
    #[arg(
        long,
        allow_hyphen_values = true,
        num_args = 3,
        value_delimiter = ',',
        default_value = "0.0,1.0,1.0"
    )]
    rgb_or_hsv: Vec<f32>,

    /// Alpha is multiplied with the original alpha.
    #[arg(long, allow_hyphen_values = true, default_value = "1.0")]
    alpha: f32,

    /// Contrast is applied to the RGB color.
    ///
    /// Normally the range is [-1, 1].
    #[arg(long, allow_hyphen_values = true, default_value = "0.0")]
    contrast: f32,

    /// Exposure is applied to the RGB color.
    ///
    /// Normally the range is [-5, 5].
    #[arg(long, allow_hyphen_values = true, default_value = "0.0")]
    exposure: f32,

    /// Gamma is applied to the RGB color.
    ///
    /// Normally the range is [0, 5].
    #[arg(long, allow_hyphen_values = true, default_value = "1.0")]
    gamma: f32,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, ValueEnum)]
enum Shape {
    Sphere,
    Box,
}

type GaussianPod = gs::core::GaussianPodWithShSingleCov3dRotScaleConfigs;

#[tokio::main]
async fn main() {
    env_logger::Builder::from_env(env_logger::Env::default().default_filter_or("info")).init();

    let args = Args::parse();
    let model_path = &args.model;
    let pos = Vec3::from_slice(&args.pos);
    let rot = Quat::from_slice(&args.rot);
    let scale = Vec3::from_slice(&args.scale);
    let shape = match args.shape {
        Shape::Sphere => gs::SelectionBundle::create_sphere_bundle::<GaussianPod>,
        Shape::Box => gs::SelectionBundle::create_box_bundle::<GaussianPod>,
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

    log::debug!("Creating editor");
    let editor = gs::Editor::<GaussianPod>::new(&device, &gaussians);

    log::debug!("Creating shape selection compute bundle");
    let shape_selection = shape(&device);

    log::debug!("Creating basic selection modifier");
    let mut basic_selection_modifier = gs::BasicSelectionModifier::new::<GaussianPod>(
        &device,
        &editor.gaussians_buffer,
        &editor.model_transform_buffer,
        &editor.gaussian_transform_buffer,
        vec![shape_selection],
    );

    log::debug!("Configuring modifiers");
    match args.override_rgb {
        true => basic_selection_modifier
            .basic_color_modifiers_buffer
            .update_with_override_rgb(
                &queue,
                Vec3::from_slice(&args.rgb_or_hsv),
                args.alpha,
                args.contrast,
                args.exposure,
                args.gamma,
            ),
        false => basic_selection_modifier
            .basic_color_modifiers_buffer
            .update_with_hsv_modifiers(
                &queue,
                Vec3::from_slice(&args.rgb_or_hsv),
                args.alpha,
                args.contrast,
                args.exposure,
                args.gamma,
            ),
    }

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
            basic_selection_modifier.selection.bundles[0]
                .create_bind_group(
                    &device,
                    // index 0 is the Gaussians buffer, so we use 1,
                    // see docs of create_sphere_bundle or create_box_bundle
                    1,
                    gs::core::buffer_wrapper_arr![buffer],
                )
                .expect("bind group")
        })
        .collect::<Vec<_>>();

    log::debug!("Creating selection expression");
    basic_selection_modifier.selection_expr = shape_selection_bind_groups.into_iter().fold(
        gs::SelectionExpr::Identity,
        |acc, bind_group| {
            acc.union(gs::SelectionExpr::selection(
                0, // the 0 here is the bundle index in the selection bundle
                vec![bind_group],
            ))
        },
    );

    log::info!("Starting editing process");
    let time = std::time::Instant::now();

    log::debug!("Editing Gaussians");
    let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
        label: Some("Edit Encoder"),
    });

    editor.apply(
        &device,
        &mut encoder,
        [&basic_selection_modifier as &dyn gs::Modifier<GaussianPod>],
    );

    queue.submit(Some(encoder.finish()));

    #[allow(unused_must_use)]
    device.poll(wgpu::PollType::Wait);

    log::info!("Editing process completed in {:?}", time.elapsed());

    log::debug!("Downloading Gaussians");
    let modified_gaussians = gs::core::Gaussians {
        gaussians: editor
            .gaussians_buffer
            .download_gaussians(&device, &queue)
            .await
            .expect("gaussians download"),
    };

    log::debug!("Writing modified Gaussians to output file");
    let output_file = std::fs::File::create(&args.output).expect("output file");
    let mut writer = std::io::BufWriter::new(output_file);
    modified_gaussians
        .write_ply(&mut writer)
        .expect("write modified Gaussians to output file");

    log::info!("Modified Gaussians written to {}", args.output);
}

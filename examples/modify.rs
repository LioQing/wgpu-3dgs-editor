//! This example modifies the entire model using [`BasicModifier`](wgpu_3dgs_editor::BasicModifier).
//!
//! For example, to decrease the contrast of the model:
//!
//! ```sh
//! cargo run --example modify -- -m "path/to/model.ply" --contrast "-1.0"
//! ```

use clap::Parser;
use glam::*;

use wgpu_3dgs_editor::{self as gs};

/// The command line arguments.
#[derive(Parser, Debug)]
#[command(
    version,
    about,
    long_about = "\
    A 3D Gaussian splatting editor to apply basic modifier to all Gaussians in a model.
    "
)]
struct Args {
    /// Path to the .ply file.
    #[arg(short, long, default_value = "examples/model.ply")]
    model: String,

    /// The output path for the modified .ply file.
    #[arg(short, long, default_value = "target/output.ply")]
    output: String,

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

type GaussianPod = gs::core::GaussianPodWithShSingleCov3dRotScaleConfigs;

#[pollster::main]
async fn main() {
    env_logger::Builder::from_env(env_logger::Env::default().default_filter_or("info")).init();

    let args = Args::parse();
    let model_path = &args.model;

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
    let gaussians = gs::core::PlyGaussians::read_ply_file(model_path).expect("gaussians");

    log::debug!("Creating editor");
    let editor = gs::Editor::<GaussianPod>::new(&device, &gaussians);

    log::debug!("Creating basic modifier");
    let basic_modifier = gs::BasicModifier::<GaussianPod>::new(
        &device,
        &editor.gaussians_buffer,
        &editor.model_transform_buffer,
        &editor.gaussian_transform_buffer,
    );

    log::debug!("Configuring modifiers");
    match args.override_rgb {
        true => basic_modifier
            .basic_color_modifiers_buffer
            .update_with_override_rgb(
                &queue,
                Vec3::from_slice(&args.rgb_or_hsv),
                args.alpha,
                args.contrast,
                args.exposure,
                args.gamma,
            ),
        false => basic_modifier
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

    log::info!("Starting editing process");
    let time = std::time::Instant::now();

    log::debug!("Editing Gaussians");
    let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
        label: Some("Edit Encoder"),
    });

    editor.apply(
        &device,
        &mut encoder,
        [&basic_modifier as &dyn gs::Modifier<GaussianPod>],
    );

    queue.submit(Some(encoder.finish()));

    #[allow(unused_must_use)]
    device.poll(wgpu::PollType::Wait);

    log::info!("Editing process completed in {:?}", time.elapsed());

    log::debug!("Downloading Gaussians");
    let modified_gaussians = editor
        .gaussians_buffer
        .download_gaussians(&device, &queue)
        .await
        .expect("gaussians download")
        .into_iter()
        .map(|g| g.to_ply())
        .collect::<gs::core::PlyGaussians>();

    log::debug!("Writing modified Gaussians to output file");
    modified_gaussians
        .write_ply_file(&args.output)
        .expect("write modified Gaussians to output file");

    log::info!("Modified Gaussians written to {}", args.output);
}

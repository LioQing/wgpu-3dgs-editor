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
    /// Path to the .ply or .spz file.
    #[arg(short, long, default_value = "examples/model.ply")]
    model: String,

    /// The output path for the modified .ply or .spz file.
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
    basic_modifier.basic_color_modifiers_buffer.update(
        &queue,
        match args.override_rgb {
            true => gs::BasicColorRgbOverrideOrHsvModifiersPod::new_rgb_override,
            false => gs::BasicColorRgbOverrideOrHsvModifiersPod::new_hsv_modifiers,
        }(Vec3::from_slice(&args.rgb_or_hsv)),
        args.alpha,
        args.contrast,
        args.exposure,
        args.gamma,
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
        [&basic_modifier as &dyn gs::Modifier<GaussianPod>],
    );

    queue.submit(Some(encoder.finish()));

    device
        .poll(wgpu::PollType::wait_indefinitely())
        .expect("poll");

    log::info!("Editing process completed in {:?}", time.elapsed());

    log::debug!("Downloading Gaussians");
    let modified_gaussians = editor
        .gaussians_buffer
        .download_gaussians(&device, &queue)
        .await
        .map(|gs| {
            match &args.output[args.output.len().saturating_sub(4)..] {
                ".ply" => {
                    gs::core::Gaussians::Ply(gs::core::PlyGaussians::from_iter(gs.into_iter()))
                }
                ".spz" => {
                    gs::core::Gaussians::Spz(
                        gs::core::SpzGaussians::from_gaussians_with_options(
                            gs,
                            &gs::core::SpzGaussiansFromGaussianSliceOptions {
                                version: 2, // Version 2 is more widely supported as of now
                                ..Default::default()
                            },
                        )
                        .expect("SpzGaussians from gaussians"),
                    )
                }
                _ => panic!("Unsupported output file extension, expected .ply or .spz"),
            }
        })
        .expect("gaussians download");

    log::debug!("Writing modified Gaussians to output file");
    modified_gaussians
        .write_to_file(&args.output)
        .expect("write modified Gaussians to output file");

    log::info!("Modified Gaussians written to {}", args.output);
}

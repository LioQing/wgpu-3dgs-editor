//! This example selects parts of the model, then applies a custom modifier
//! using [`SelectionModifier`](wgpu_3dgs_editor::SelectionModifier) and a user-defined
//! compute pipeline built with [`ComputeBundleBuilder`](wgpu_3dgs_editor::core::ComputeBundleBuilder).
//!
//! The custom selection is a cylinder aligned with the z-axis, centered at `pos` with a given `radius`.
//!
//! The custom modifier shifts the hue of the selected Gaussians according to their x and y coordinates
//! about the axis of the cylinder selection.
//!
//! For example, to use the defaults (pos = (0, 0, 0), radius = 3.0) and apply the custom modifier:
//!
//! ```sh
//! cargo run --example custom-modify-selection -- -m "path/to/model.ply"
//! ```

use clap::{Parser, ValueEnum};
use glam::*;

use wgpu::util::DeviceExt;
use wgpu_3dgs_editor::{
    self as gs,
    core::{BufferWrapper, GaussianPod as _},
};

/// The command line arguments.
#[derive(Parser, Debug)]
#[command(
    version,
    about,
    long_about = "\
    A 3D Gaussian splatting editor to apply custom modifier to Gaussians in a model selected by a cylinder along the z-axis.
    "
)]
struct Args {
    /// Path to the .ply file.
    #[arg(short, long)]
    model: String,

    /// The output path for the modified .ply file.
    #[arg(short, long, default_value = "target/output.ply")]
    output: String,

    /// The position of the selection cylinder.
    #[arg(
        short,
        long,
        allow_hyphen_values = true,
        num_args = 3,
        value_delimiter = ',',
        default_value = "0.0,0.0,0.0"
    )]
    pos: Vec<f32>,

    /// The radius of the selection cylinder.
    #[arg(short, long, default_value_t = 3.0)]
    radius: f32,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, ValueEnum)]
enum Factory {
    Struct,
    Closure,
}

type GaussianPod = gs::core::GaussianPodWithShSingleCov3dRotScaleConfigs;

#[pollster::main]
async fn main() {
    env_logger::Builder::from_env(env_logger::Env::default().default_filter_or("info")).init();

    let args = Args::parse();
    let model_path = &args.model;
    let pos = Vec3::from_slice(&args.pos);
    let radius = args.radius;

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

    log::debug!("Creating buffers");
    let pos_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("Position Buffer"),
        contents: bytemuck::bytes_of(&pos),
        usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
    });

    let radius_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("Radius Buffer"),
        contents: bytemuck::bytes_of(&radius),
        usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
    });

    const BIND_GROUP_LAYOUT_DESCRIPTOR: wgpu::BindGroupLayoutDescriptor =
        wgpu::BindGroupLayoutDescriptor {
            label: Some("Bind Group Layout"),
            entries: &[
                // Position uniform buffer
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                // Radius uniform buffer
                wgpu::BindGroupLayoutEntry {
                    binding: 1,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                // Selection buffer (only in modifier pipeline)
                wgpu::BindGroupLayoutEntry {
                    binding: 2,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
            ],
        };

    log::debug!("Creating cylinder selection compute bundle");
    let cylinder_selection_bundle = gs::core::ComputeBundleBuilder::new()
        .label("Selection")
        .bind_group_layouts([
            &gs::SelectionBundle::<GaussianPod>::GAUSSIANS_BIND_GROUP_LAYOUT_DESCRIPTOR,
            &wgpu::BindGroupLayoutDescriptor {
                entries: &BIND_GROUP_LAYOUT_DESCRIPTOR.entries[..2],
                ..BIND_GROUP_LAYOUT_DESCRIPTOR
            },
        ])
        .main_shader("package::selection".parse().unwrap())
        .entry_point("main")
        .wesl_compile_options(wesl::CompileOptions {
            features: GaussianPod::wesl_features(),
            ..Default::default()
        })
        .resolver({
            let mut resolver =
                wesl::StandardResolver::new("examples/shader/custom_modify_selection");
            resolver.add_package(&gs::core::shader::PACKAGE);
            resolver
        })
        .build_without_bind_groups(&device)
        .map_err(|e| log::error!("{e}"))
        .expect("selection bundle");

    log::debug!("Creating custom modifier");
    #[allow(dead_code)]
    let modifier_factory = |selection_buffer: &gs::SelectionBuffer| /* -> impl gs::Modifier<GaussianPod> */ {
            log::debug!("Creating custom modifier compute bundle");
            let modifier_bundle = gs::core::ComputeBundleBuilder::new()
                .label("Modifier")
                .bind_group_layouts([
                    &gs::MODIFIER_GAUSSIANS_BIND_GROUP_LAYOUT_DESCRIPTOR,
                    &BIND_GROUP_LAYOUT_DESCRIPTOR,
                ])
                .resolver({
                    let mut resolver =
                        wesl::StandardResolver::new("examples/shader/custom_modify_selection");
                    resolver.add_package(&gs::core::shader::PACKAGE);
                    resolver.add_package(&gs::shader::PACKAGE);
                    resolver
                })
                .main_shader("package::modifier".parse().unwrap())
                .entry_point("main")
                .wesl_compile_options(wesl::CompileOptions {
                    features: GaussianPod::wesl_features(),
                    ..Default::default()
                })
                .build(
                    &device,
                    [
                        vec![
                            editor.gaussians_buffer.buffer().as_entire_binding(),
                            editor.model_transform_buffer.buffer().as_entire_binding(),
                            editor
                                .gaussian_transform_buffer
                                .buffer()
                                .as_entire_binding(),
                        ],
                        vec![
                            pos_buffer.as_entire_binding(),
                            radius_buffer.as_entire_binding(),
                            selection_buffer.buffer().as_entire_binding(),
                        ],
                    ],
                )
                .map_err(|e| log::error!("{e}"))
                .expect("modifier bundle");

            // This is a modifier closure because this function signature has blanket impl of the modifier trait
            move |_device: &wgpu::Device,
                  encoder: &mut wgpu::CommandEncoder,
                  gaussians: &gs::core::GaussiansBuffer<GaussianPod>,
                  _model_transform: &gs::core::ModelTransformBuffer,
                  _gaussian_transform: &gs::core::GaussianTransformBuffer| {
                  modifier_bundle.dispatch(encoder, gaussians.len() as u32);
            }
        };

    #[allow(dead_code)]
    struct Modifier<G: gs::core::GaussianPod>(gs::core::ComputeBundle, std::marker::PhantomData<G>);

    impl<G: gs::core::GaussianPod> Modifier<G> {
        #[allow(dead_code)]
        fn new(
            device: &wgpu::Device,
            editor: &gs::Editor<G>,
            pos_buffer: &wgpu::Buffer,
            radius_buffer: &wgpu::Buffer,
            selection_buffer: &gs::SelectionBuffer,
        ) -> Self {
            log::debug!("Creating custom modifier compute bundle");
            let modifier_bundle = gs::core::ComputeBundleBuilder::new()
                .label("Modifier")
                .bind_group_layouts([
                    &gs::MODIFIER_GAUSSIANS_BIND_GROUP_LAYOUT_DESCRIPTOR,
                    &BIND_GROUP_LAYOUT_DESCRIPTOR,
                ])
                .resolver({
                    let mut resolver =
                        wesl::StandardResolver::new("examples/shader/custom_modify_selection");
                    resolver.add_package(&gs::core::shader::PACKAGE);
                    resolver.add_package(&gs::shader::PACKAGE);
                    resolver
                })
                .main_shader("package::modifier".parse().unwrap())
                .entry_point("main")
                .wesl_compile_options(wesl::CompileOptions {
                    features: GaussianPod::wesl_features(),
                    ..Default::default()
                })
                .build(
                    device,
                    [
                        vec![
                            editor.gaussians_buffer.buffer().as_entire_binding(),
                            editor.model_transform_buffer.buffer().as_entire_binding(),
                            editor
                                .gaussian_transform_buffer
                                .buffer()
                                .as_entire_binding(),
                        ],
                        vec![
                            pos_buffer.as_entire_binding(),
                            radius_buffer.as_entire_binding(),
                            selection_buffer.buffer().as_entire_binding(),
                        ],
                    ],
                )
                .map_err(|e| log::error!("{e}"))
                .expect("modifier bundle");

            Self(modifier_bundle, std::marker::PhantomData)
        }
    }

    impl<G: gs::core::GaussianPod> gs::Modifier<G> for Modifier<G> {
        fn apply(
            &self,
            _device: &wgpu::Device,
            encoder: &mut wgpu::CommandEncoder,
            gaussians: &gs::core::GaussiansBuffer<G>,
            _model_transform: &gs::core::ModelTransformBuffer,
            _gaussian_transform: &gs::core::GaussianTransformBuffer,
        ) {
            self.0.dispatch(encoder, gaussians.len() as u32);
        }
    }

    log::debug!("Creating selection modifier");
    let mut selection_modifier = gs::SelectionModifier::<GaussianPod, _>::new(
        &device,
        &editor.gaussians_buffer,
        vec![cylinder_selection_bundle],
        modifier_factory,
        // Uncomment the following line to use modifier struct instead of closure
        // |selection_buffer| {
        //     Modifier::new(
        //         &device,
        //         &editor,
        //         &pos_buffer,
        //         &radius_buffer,
        //         selection_buffer,
        //     )
        // },
    );

    log::debug!("Creating selection expression");
    selection_modifier.selection_expr = gs::SelectionExpr::selection(
        0,
        vec![
            selection_modifier.selection.bundles[0]
                .create_bind_group(
                    &device,
                    1, // index 0 is the Gaussians buffer, so we use 1,
                    [
                        pos_buffer.as_entire_binding(),
                        radius_buffer.as_entire_binding(),
                    ],
                )
                .expect("selection expr bind group"),
        ],
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
        [&selection_modifier as &dyn gs::Modifier<GaussianPod>],
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

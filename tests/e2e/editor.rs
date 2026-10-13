use wgpu_3dgs_editor::{
    BasicColorRgbOverrideOrHsvModifiersPod, BasicModifier, BasicModifierBundle, Editor,
    SelectionBundle,
    core::{
        BufferWrapper, CovHalf, CovRotScale, CovSingle, GaussianPod, Gaussians, InternalGaussians,
        PackedGaussian, PlyGaussians, ShDegree0, ShDegree1, ShDegree2, ShDegree3, ShDegree4,
        ShHalf, ShNone, ShNorm8, ShSingle, SpzGaussians, glam::*,
    },
};

use crate::common::{TestContext, given};

macro_rules! test_editor_degree {
    ($name:ident, $degree:ty) => {
        #[test]
        fn $name() {
            let ctx = TestContext::new();
            type G = PackedGaussian<ShSingle<$degree>, CovRotScale>;
            let mut gaussian = given::gaussian_with_seed(42).convert_sh_degree::<$degree>();
            // Also check coefficients added beyond the old degree-three storage.
            gaussian.sh.fill(Vec3::splat(0.25));
            let gaussians = vec![gaussian];
            let editor = Editor::<G>::new(&ctx.device, &gaussians).expect("editor");
            assert!(
                editor
                    .gaussians_buffer
                    .buffer()
                    .usage()
                    .contains(wgpu::BufferUsages::COPY_SRC)
            );

            let runtime = Gaussians::Ply(
                PlyGaussians::from_gaussians(gaussians.clone()).expect("PlyGaussians"),
            );
            let runtime_editor = Editor::<G>::new(&ctx.device, &runtime).expect("editor");
            let typed_pods = pollster::block_on(
                editor
                    .gaussians_buffer
                    .download::<G>(&ctx.device, &ctx.queue),
            )
            .expect("typed download");
            let runtime_pods = pollster::block_on(
                runtime_editor
                    .gaussians_buffer
                    .download::<G>(&ctx.device, &ctx.queue),
            )
            .expect("runtime download");
            assert_eq!(typed_pods.len(), runtime_pods.len());
            assert_eq!(typed_pods[0].pos, runtime_pods[0].pos);
            assert_eq!(typed_pods[0].color, runtime_pods[0].color);
            assert_eq!(typed_pods[0].sh, runtime_pods[0].sh);
            // PLY stores logarithmic scales, so round-tripping can introduce rounding.
            for (typed, runtime) in typed_pods[0].cov3d.iter().zip(runtime_pods[0].cov3d) {
                assert!((typed - runtime).abs() < 1.0e-5);
            }

            let modifier = BasicModifier::new(
                &ctx.device,
                &editor.gaussians_buffer,
                &editor.model_transform_buffer,
                &editor.gaussian_transform_buffer,
            );
            modifier.basic_color_modifiers_buffer.update(
                &ctx.queue,
                BasicColorRgbOverrideOrHsvModifiersPod::new_rgb_override(Vec3::X),
                1.0,
                0.0,
                0.0,
                1.0,
            );
            let mut encoder = ctx.device.create_command_encoder(&Default::default());
            editor.apply(
                &ctx.device,
                &mut encoder,
                [&modifier as &dyn wgpu_3dgs_editor::Modifier<G>],
            );
            ctx.queue.submit(Some(encoder.finish()));
            let downloaded = pollster::block_on(
                editor
                    .gaussians_buffer
                    .download_gaussians(&ctx.device, &ctx.queue),
            )
            .expect("download_gaussians");
            assert_eq!(downloaded[0].color.xyz(), Vec3::X);
            assert_eq!(downloaded[0].sh, gaussians[0].sh);
        }
    };
}

test_editor_degree!(test_editor_degree_zero_should_upload_and_modify, ShDegree0);
test_editor_degree!(test_editor_degree_one_should_upload_and_modify, ShDegree1);
test_editor_degree!(test_editor_degree_two_should_upload_and_modify, ShDegree2);
test_editor_degree!(test_editor_degree_three_should_upload_and_modify, ShDegree3);
test_editor_degree!(test_editor_degree_four_should_upload_and_modify, ShDegree4);

#[test]
fn test_editor_new_should_accept_unified_sources() {
    let ctx = TestContext::new();
    type G = PackedGaussian<ShSingle, CovRotScale>;
    let gaussians = given::gaussians();
    let sources = [
        Gaussians::Internal(InternalGaussians::from_gaussians(gaussians.clone()).unwrap()),
        Gaussians::Ply(PlyGaussians::from_gaussians(gaussians.clone()).unwrap()),
        Gaussians::Spz(SpzGaussians::from_gaussians(gaussians).unwrap()),
    ];
    for source in sources {
        let editor = Editor::<G>::new(&ctx.device, &source).expect("editor");
        assert_eq!(editor.gaussians_buffer.len(), 2);
        assert!(
            editor
                .gaussians_buffer
                .buffer()
                .usage()
                .contains(wgpu::BufferUsages::COPY_SRC)
        );
    }
}

#[test]
fn test_editor_new_with_mismatched_degree_should_return_error() {
    let ctx = TestContext::new();
    type G = PackedGaussian<ShNone, CovRotScale>;
    let gaussians = given::gaussians();
    let source = Gaussians::Ply(PlyGaussians::from_gaussians(gaussians).unwrap());
    let Err(error) = Editor::<G>::new(&ctx.device, &source) else {
        panic!("expected a degree mismatch");
    };
    assert_eq!(error.actual_degree, 3);
    assert_eq!(error.expected_degree, 0);
}

#[test]
fn test_editor_new_with_mismatched_typed_degree_should_return_error() {
    let ctx = TestContext::new();
    type G = PackedGaussian<ShNone, CovRotScale>;
    for gaussians in [given::gaussians(), vec![]] {
        let Err(error) = Editor::<G>::new(&ctx.device, &gaussians) else {
            panic!("expected a degree mismatch");
        };
        assert_eq!(error.actual_degree, 3);
        assert_eq!(error.expected_degree, 0);
    }
}

#[test]
fn test_editor_new_with_empty_source_should_check_declared_degree() {
    let ctx = TestContext::new();
    type G = PackedGaussian<ShNone, CovRotScale>;
    let source = Gaussians::Ply(PlyGaussians::new(vec![], 0).unwrap());
    let editor = Editor::<G>::new(&ctx.device, &source).expect("editor");
    assert!(editor.gaussians_buffer.is_empty());

    let mismatched_source = Gaussians::Ply(PlyGaussians::new(vec![], 4).unwrap());
    let Err(error) = Editor::<G>::new(&ctx.device, &mismatched_source) else {
        panic!("expected a degree mismatch");
    };
    assert_eq!(error.actual_degree, 4);
    assert_eq!(error.expected_degree, 0);
}

fn check_shader_layout<G: GaussianPod>(ctx: &TestContext) {
    BasicModifierBundle::<G, _, _>::new_without_bind_group(&ctx.device);
    BasicModifierBundle::<G, _, _>::new_without_bind_group_with_selection(&ctx.device);
    SelectionBundle::<G>::create_box_bundle(&ctx.device);
    SelectionBundle::<G>::create_sphere_bundle(&ctx.device);
}

#[test]
fn test_editor_shaders_should_compile_for_all_storage_layouts() {
    let ctx = TestContext::new();
    macro_rules! check_degree {
        ($degree:ty) => {
            check_shader_layout::<PackedGaussian<ShSingle<$degree>, CovRotScale>>(&ctx);
            check_shader_layout::<PackedGaussian<ShSingle<$degree>, CovSingle>>(&ctx);
            check_shader_layout::<PackedGaussian<ShSingle<$degree>, CovHalf>>(&ctx);
            check_shader_layout::<PackedGaussian<ShHalf<$degree>, CovRotScale>>(&ctx);
            check_shader_layout::<PackedGaussian<ShHalf<$degree>, CovSingle>>(&ctx);
            check_shader_layout::<PackedGaussian<ShHalf<$degree>, CovHalf>>(&ctx);
            check_shader_layout::<PackedGaussian<ShNorm8<$degree>, CovRotScale>>(&ctx);
            check_shader_layout::<PackedGaussian<ShNorm8<$degree>, CovSingle>>(&ctx);
            check_shader_layout::<PackedGaussian<ShNorm8<$degree>, CovHalf>>(&ctx);
        };
    }
    check_degree!(ShDegree0);
    check_degree!(ShDegree1);
    check_degree!(ShDegree2);
    check_degree!(ShDegree3);
    check_degree!(ShDegree4);
    check_shader_layout::<PackedGaussian<ShNone, CovRotScale>>(&ctx);
    check_shader_layout::<PackedGaussian<ShNone, CovSingle>>(&ctx);
    check_shader_layout::<PackedGaussian<ShNone, CovHalf>>(&ctx);
}

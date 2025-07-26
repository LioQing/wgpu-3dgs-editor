import wgpu_3dgs_core::{
    gaussian::{Gaussian, gaussian_unpack_cov3d},
    model_transform::{ModelTransform, model_to_world, model_scale_rotation_mat},
};

fn rgb_to_hsv(c: vec3<f32>) -> vec3<f32> {
    const k = vec4<f32>(0.0, -1.0 / 3.0, 2.0 / 3.0, -1.0);
    let p = select(vec4<f32>(c.bg, k.wz), vec4<f32>(c.gb, k.xy), c.b < c.g);
    let q = select(vec4<f32>(p.xyw, c.r), vec4<f32>(c.r, p.yzx), p.x < c.r);

    let d = q.x - min(q.w, q.y);
    const e = 1.0e-10;
    return vec3<f32>(abs(q.z + (q.w - q.y) / (6.0 * d + e)), d / (q.x + e), q.x);
}

fn hsv_to_rgb(c: vec3<f32>) -> vec3<f32> {
    const k = vec4<f32>(1.0, 2.0 / 3.0, 1.0 / 3.0, 3.0);
    let p = abs(fract(c.xxx + k.xyz) * 6.0 - k.www);
    return c.z * mix(k.xxx, saturate(p - k.xxx), c.y);
}

// Apply HSV modification to a color.
//
// Normally hue (H) is in [0, 1], saturation (S) and value (V) are in [0, 2].
// This function adds the hue and multiplies saturation and value.
//
// This function modifies HSV and returns HSV.
fn apply_hsv(hsv: vec3<f32>, apply_hsv: vec3<f32>) -> vec3<f32> {
    return saturate(vec3<f32>(
        (hsv.x + apply_hsv.x) % 1.0,
        hsv.y * apply_hsv.y,
        hsv.z * apply_hsv.z,
    ));
}

// Apply contrast to a color.
//
// Normally the range is [-1, 1].
fn apply_contrast(rgb: vec3<f32>, contrast: f32) -> vec3<f32> {
    const contrast_const = 259.0 / 255.0;
    let contrast_factor = contrast_const * (contrast + 1.0) / min(contrast_const - contrast, 1e-6);
    return (rgb - 0.5) * contrast_factor + 0.5;
}

// Apply exposure to a color.
//
// Normally the range is [-5, 5].
fn apply_exposure(rgb: vec3<f32>, exposure: f32) -> vec3<f32> {
    return rgb * exp2(exposure);
}

// Apply gamma correction to a color.
//
// Normally the range is [0, 5].
fn apply_gamma(rgb: vec3<f32>, gamma: f32) -> vec3<f32> {
    return pow(rgb, vec3<f32>(1.0 / min(gamma, 1e-6)));
}

// Apply a scale rotation matrix to a Gaussian.
fn apply_scale_rotation(
    scale_rotation: mat3x3<f32>,
    cov3d: array<f32, 6>,
) -> Gaussian {
    let vrk = scale_rotation * mat3x3<f32>(
        cov3d[0], cov3d[1], cov3d[2],
        cov3d[1], cov3d[3], cov3d[4],
        cov3d[2], cov3d[4], cov3d[5],
    ) * transpose(scale_rotation);

    let cov3d = array<f32>(
        vrk[0][0], vrk[0][1], vrk[0][2],
        vrk[1][1], vrk[1][2], vrk[2][2],
    );

    @if(cov3d_single)
        return cov3d;

    @if(cov3d_half)
        return array<f32>(
            pack2x16float(cov3d[0], cov3d[1]),
            pack2x16float(cov3d[2], cov3d[3]),
            pack2x16float(cov3d[4], cov3d[5]),
        );
}

// A struct for all the basic color related modifiers.
struct BasicColorModifiers {
    // If any value is negative, then it is used to override the RGB color,
    // otherwise it is used to apply HSV modifications.
    rgb_or_hsv: vec3<f32>,
    // Alpha is multiplied with the original alpha.
    alpha: f32,
    // Contrast is applied to the RGB color.
    contrast: f32,
    // Exposure is applied to the RGB color.
    exposure: f32,
    // Gamma is applied to the RGB color.
    gamma: f32,
}

// Apply the basic color modifiers.
//
// This is the default implementation for the ModifierBundle.
fn apply_basic_color_modifiers(
    modifiers: BasicColorModifiers,
    in_color: vec4<f32>,
) -> vec4<f32> {
    var color = in_color;

    if any(modifiers.rgb_or_hsv < vec3<f32>(0.0)) {
        color.rgb = modifiers.rgb_or_hsv;
    } else {
        let hsv = rgb_to_hsv(color.rgb);
        let modified_hsv = apply_hsv(hsv, modifiers.rgb_or_hsv);
        color.rgb = hsv_to_rgb(modified_hsv);
    }

    color.rgb = apply_contrast(color.rgb, modifiers.contrast);
    color.rgb = apply_exposure(color.rgb, modifiers.exposure);
    color.rgb = apply_gamma(color.rgb, modifiers.gamma);

    color.a *= modifiers.alpha;

    return color;
}

// Apply the basic transform modifiers.
//
// Transform related parameters are applied using model and per Gaussian transforms.
// The model transform is applied with ModelTransform struct,
// and the per Gaussian transform is applied with a scale and rotation transform matrix.
fn apply_basic_transform_modifiers(
    model_transform: ModelTransform,
    scale_rotation: mat3x3<f32>,
    in_gaussian: Gaussian,
) -> Gaussian {
    var gaussian = in_gaussian;

    gaussian.pos = model_to_world(model_transform, gaussian.pos);

    let sr = model_scale_rotation_mat(model_transform);
    let cov3d = gaussian_unpack_cov3d(gaussian);

    gaussian.cov3d = apply_scale_rotation(scale_rotation * sr, cov3d);

    return gaussian;
}
import wgpu_3dgs_core::{
    gaussian::{Gaussian, gaussian_unpack_cov3d},
    model_transform::{ModelTransform, model_to_world, model_scale_rot_mat},
};

// Convert RGB color to HSV color.
fn rgb_to_hsv(c: vec3<f32>) -> vec3<f32> {
    const k = vec4<f32>(0.0, -1.0 / 3.0, 2.0 / 3.0, -1.0);
    let p = select(vec4<f32>(c.bg, k.wz), vec4<f32>(c.gb, k.xy), c.b < c.g);
    let q = select(vec4<f32>(p.xyw, c.r), vec4<f32>(c.r, p.yzx), p.x < c.r);

    let d = q.x - min(q.w, q.y);
    const e = 1.0e-10;
    return vec3<f32>(abs(q.z + (q.w - q.y) / (6.0 * d + e)), d / (q.x + e), q.x);
}

// Convert HSV color to RGB color.
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
    let contrast_factor = contrast_const * (contrast + 1.0) / (contrast_const - contrast);
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
    return pow(rgb, vec3<f32>(1.0 / gamma));
}

// Helper function to do quaternion multiplication.
fn quat_mul(a: vec4<f32>, b: vec4<f32>) -> vec4<f32> {
    return vec4<f32>(
        a.w * b.x + a.x * b.w + a.y * b.z - a.z * b.y,
        a.w * b.y - a.x * b.z + a.y * b.w + a.z * b.x,
        a.w * b.z + a.x * b.y - a.y * b.x + a.z * b.w,
        a.w * b.w - a.x * b.x - a.y * b.y - a.z * b.z,
    );
}

// Apply the model transform and rotation scale to a Gaussian.
@if(cov3d_rot_scale) fn apply_scale_rot(
    model_transform: ModelTransform,
    rot_scale: RotScale,
    gaussian: Gaussian,
) -> array<f32, 7> {
    let rot = vec4<f32>(gaussian.cov3d[0], gaussian.cov3d[1], gaussian.cov3d[2], gaussian.cov3d[3]);
    let scale = vec3<f32>(gaussian.cov3d[4], gaussian.cov3d[5], gaussian.cov3d[6]);

    let modified_rot = quat_mul(quat_mul(model_transform.rot, rot_scale.rot), rot);
    let modified_scale = model_transform.scale * rot_scale.scale * scale;

    return array<f32, 7>(
        modified_rot.x, modified_rot.y, modified_rot.z, modified_rot.w,
        modified_scale.x, modified_scale.y, modified_scale.z,
    );
}

// Helper method to apply a scale rotation matrix to a covariance matrix.
//
// This is identical to `apply_scale_rot` for `cov3d_single` configuration with
// the second parameter being a compact covariance matrix.
fn apply_scale_rot_cov3d(
    scale_rot: mat3x3<f32>,
    cov3d: array<f32, 6>,
) -> array<f32, 6> {
    let vrk = scale_rot * mat3x3<f32>(
        cov3d[0], cov3d[1], cov3d[2],
        cov3d[1], cov3d[3], cov3d[4],
        cov3d[2], cov3d[4], cov3d[5],
    ) * transpose(scale_rot);

    return array<f32, 6>(
        vrk[0][0], vrk[0][1], vrk[0][2],
        vrk[1][1], vrk[1][2], vrk[2][2],
    );
}

// Apply a scale rotation matrix to a Gaussian.
@if(cov3d_single) fn apply_scale_rot(
    scale_rot: mat3x3<f32>,
    gaussian: Gaussian,
) -> array<f32, 6> {
    return apply_scale_rot_cov3d(scale_rot, gaussian_unpack_cov3d(gaussian));
}

// Apply a scale rotation matrix to a Gaussian.
@if(cov3d_half) fn apply_scale_rot(
    scale_rot: mat3x3<f32>,
    gaussian: Gaussian,
) -> array<u32, 3> {
    let modified_cov3d = apply_scale_rot_cov3d(scale_rot, gaussian_unpack_cov3d(gaussian));
 
    return array<u32, 3>(
        pack2x16float(vec2<f32>(modified_cov3d[0], modified_cov3d[1])),
        pack2x16float(vec2<f32>(modified_cov3d[2], modified_cov3d[3])),
        pack2x16float(vec2<f32>(modified_cov3d[4], modified_cov3d[5])),
    );
}

// A struct for all the basic color related modifiers.
//
// Corresponds to `BasicColorModifiersPod`.
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
    var rgb = in_color.rgb;

    if any(sign(modifiers.rgb_or_hsv) == vec3<f32>(-1.0)) {
        rgb = abs(modifiers.rgb_or_hsv);
    } else {
        let hsv = rgb_to_hsv(rgb);
        let modified_hsv = apply_hsv(hsv, modifiers.rgb_or_hsv);
        rgb = hsv_to_rgb(modified_hsv);
    }

    rgb = apply_contrast(rgb, modifiers.contrast);
    rgb = apply_exposure(rgb, modifiers.exposure);
    rgb = apply_gamma(rgb, modifiers.gamma);

    let a = in_color.a * modifiers.alpha;

    return vec4<f32>(rgb, a);
}

// The rotation and scale for per Gaussian transform.
//
// Corresponds to `RotScalePod`.
struct RotScale {
    rot: vec4<f32>,
    scale: vec3<f32>,
}

// Convert a rotation scale to a scale rotation matrix.
fn rot_scale_scale_rot_mat(rot_scale: RotScale) -> mat3x3<f32> {
    let rot = rot_scale.rot;
    let scale = rot_scale.scale;

    let x2 = rot.x + rot.x;
    let y2 = rot.y + rot.y;
    let z2 = rot.z + rot.z;
    let xx = rot.x * x2;
    let xy = rot.x * y2;
    let xz = rot.x * z2;
    let yy = rot.y * y2;
    let yz = rot.y * z2;
    let zz = rot.z * z2;
    let wx = rot.w * x2;
    let wy = rot.w * y2;
    let wz = rot.w * z2;

    let sx = scale.x;
    let sy = scale.y;
    let sz = scale.z;

    return mat3x3<f32>(
        vec3<f32>(
            (1.0 - (yy + zz)) * sx,
            (xy + wz) * sx,
            (xz - wy) * sx,
        ),
        vec3<f32>(
            (xy - wz) * sy,
            (1.0 - (xx + zz)) * sy,
            (yz + wx) * sy,
        ),
        vec3<f32>(
            (xz + wy) * sz,
            (yz - wx) * sz,
            (1.0 - (xx + yy)) * sz,
        ),
    );
}

// Apply the basic transform modifiers.
//
// Transform related parameters are applied using model and per Gaussian transforms.
// The model transform is applied with ModelTransform struct,
// and the per Gaussian transform is applied with RotScale struct.
fn apply_basic_transform_modifiers(
    model_transform: ModelTransform,
    rot_scale: RotScale,
    in_gaussian: Gaussian,
) -> Gaussian {
    var gaussian = in_gaussian;

    let pos = model_to_world(model_transform, gaussian.pos);
    gaussian.pos = pos.xyz / pos.w;

    // Special logic for rot_scale configuration because it does not use a covariance matrix.
    @if(cov3d_rot_scale) {
        gaussian.cov3d = apply_scale_rot(model_transform, rot_scale, gaussian);
    }

    let model_sr = model_scale_rot_mat(model_transform);
    let rot_scale_sr = rot_scale_scale_rot_mat(rot_scale);
    @if(cov3d_single) {
        gaussian.cov3d = apply_scale_rot(model_sr * rot_scale_sr, gaussian);
    }
    @if(cov3d_half) {
        gaussian.cov3d = apply_scale_rot(model_sr * rot_scale_sr, gaussian);
    }

    return gaussian;
}
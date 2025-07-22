import wgpu_3dgs_core::gaussian::Gaussian;

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
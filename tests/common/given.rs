use wgpu_3dgs_core::{Gaussian, glam::*};

pub fn gaussian_with_seed(seed: u32) -> Gaussian {
    let base = seed as f32;

    let rot_x = base + 0.1;
    let rot_y = base + 0.2;
    let rot_z = base + 0.3;
    let rot_w = base + 0.4;
    let rot = Quat::from_xyzw(rot_x, rot_y, rot_z, rot_w).normalize();

    let pos = Vec3::new(base + 1.0, base + 2.0, base + 3.0);

    let color = U8Vec4::new(
        ((base + 10.0) % 256.0) as u8,
        ((base + 20.0) % 256.0) as u8,
        ((base + 30.0) % 256.0) as u8,
        ((base + 40.0) % 256.0) as u8,
    );

    let mut sh = [Vec3::ZERO; 15];
    for (i, sh) in sh.iter_mut().enumerate() {
        let sh_base = base + (i as f32);
        *sh = Vec3::new(sh_base + 0.1, sh_base + 0.2, sh_base + 0.3);
    }

    let scale = Vec3::new(base + 0.1, base + 0.2, base + 0.3);

    Gaussian {
        rot,
        pos,
        color,
        sh,
        scale,
    }
}

pub fn gaussians() -> Vec<Gaussian> {
    vec![gaussian_with_seed(42), gaussian_with_seed(123)]
}

use image::imageops::FilterType;

const MEAN: [f32; 3] = [0.485, 0.456, 0.406];
const STD: [f32; 3] = [0.229, 0.224, 0.225];

/// ImageNet-normalized CHW planes, converted to the output element as each
/// f32 value is computed — `f16::from_f32` fills the encoders' tensor
/// buffer directly (the identity closure is the generic f32 path), so the
/// intermediate f32 plane never has to exist.
pub(crate) fn normalize_chw<T: Clone>(canvas: &[u8], into: impl Fn(f32) -> T) -> Vec<T> {
    let n = canvas.len() / 3;
    let mut image = vec![into(0.0); 3 * n];
    for (i, px) in canvas.as_chunks::<3>().0.iter().enumerate() {
        for c in 0..3 {
            image[c * n + i] = into((px[c] as f32 / 255.0 - MEAN[c]) / STD[c]);
        }
    }
    image
}

/// Bilinear resize, single-channel f32 - SAM2's soft masks to source size.
pub(crate) fn resize_bilinear_f32(
    src: &[f32],
    (sw, sh): (u32, u32),
    (dw, dh): (u32, u32),
) -> Vec<f32> {
    let src = image::ImageBuffer::<image::Luma<f32>, &[f32]>::from_raw(sw, sh, src)
        .expect("src holds sw*sh samples");
    image::imageops::resize(&src, dw, dh, FilterType::Triangle).into_raw()
}

/// Bilinear resize, interleaved RGB u8 - the encoders' squashed canvas.
pub(crate) fn resize_rgb8(src: &[u8], (sw, sh): (u32, u32), (dw, dh): (u32, u32)) -> Vec<u8> {
    let src = image::ImageBuffer::<image::Rgb<u8>, &[u8]>::from_raw(sw, sh, src)
        .expect("src holds sw*sh*3 bytes");
    image::imageops::resize(&src, dw, dh, FilterType::Triangle).into_raw()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn resize_f32_preserves_constant() {
        let src = vec![0.75f32; 6 * 4];
        let out = resize_bilinear_f32(&src, (6, 4), (3, 8));
        assert!(out.iter().all(|&x| (x - 0.75).abs() < 1e-6));
    }

    #[test]
    fn resize_is_identity_at_same_size() {
        let src: Vec<u8> = (0..5 * 3 * 3).map(|i| (i * 13 % 251) as u8).collect();
        assert_eq!(resize_rgb8(&src, (5, 3), (5, 3)), src);
    }

    #[test]
    fn squash_resize_geometry() {
        let src = vec![7u8; 4 * 2 * 3];
        let out = resize_rgb8(&src, (4, 2), (8, 8));
        assert_eq!(out.len(), 8 * 8 * 3);
        assert!(out.iter().all(|&x| x == 7));
    }

    #[test]
    fn normalize_layout_is_chw() {
        let canvas = [0u8, 255, 128, 0, 255, 128];
        let out: Vec<f32> = normalize_chw(&canvas, |x| x);
        let want = |c: usize, v: u8| (v as f32 / 255.0 - MEAN[c]) / STD[c];
        assert_eq!(out.len(), 6);
        assert_eq!(out[0], want(0, 0));
        assert_eq!(out[1], want(0, 0));
        assert_eq!(out[2], want(1, 255));
        assert_eq!(out[3], want(1, 255));
        assert_eq!(out[4], want(2, 128));
        assert_eq!(out[5], want(2, 128));
    }
}

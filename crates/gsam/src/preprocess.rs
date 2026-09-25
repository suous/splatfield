use image::imageops::FilterType;

const MEAN: [f32; 3] = [0.485, 0.456, 0.406];
const STD: [f32; 3] = [0.229, 0.224, 0.225];

/// ImageNet-normalized CHW planes, converted to the output element as each
/// f32 value is computed — the f16 closure fills the encoders' u16 bit
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

/// Bilinear resize, interleaved RGB u8 - the encoders' squashed canvas.
pub(crate) fn resize_rgb8(src: &[u8], (sw, sh): (u32, u32), (dw, dh): (u32, u32)) -> Vec<u8> {
    let src = image::ImageBuffer::<image::Rgb<u8>, &[u8]>::from_raw(sw, sh, src)
        .expect("src holds sw*sh*3 bytes");
    image::imageops::resize(&src, dw, dh, FilterType::Triangle).into_raw()
}

/// The encoders' shared input: squash-resize RGB to the graph's fixed net
/// size, ImageNet-normalize to CHW, wrap in the session's element type.
/// The f16 graphs (both releases are q4f16) normalize straight into the f16
/// bit buffer — the f32 CHW plane never exists on that path.
///
/// The host ort path has its own ort-typed twin (`crate::encoder_input`,
/// which reads the element type from session metadata); backend tensor
/// types force the split — this half is what the ortweb Detector/Sam2 call.
#[cfg_attr(not(target_arch = "wasm32"), allow(dead_code))] // host callers are the tests below
pub(crate) fn encoder_input(
    rgb: &[u8],
    width: u32,
    height: u32,
    net: u32,
    dtype: crate::ortweb::DType,
) -> anyhow::Result<crate::ortweb::OrtTensor> {
    use crate::ortweb::DType;
    let canvas = resize_rgb8(rgb, (width, height), (net, net));
    let dims = [1, 3, net as i64, net as i64];
    match dtype {
        DType::F16 => {
            let bits: Vec<u16> = normalize_chw(&canvas, |x| half::f16::from_f32(x).to_bits());
            Ok(crate::ortweb::OrtTensor::from_f16_bits(bits, &dims)?)
        }
        DType::F32 => Ok(crate::ortweb::OrtTensor::from_f32(
            normalize_chw(&canvas, |x| x),
            &dims,
        )?),
        DType::I64 => anyhow::bail!("pixel_values as int64 is not a valid encoder input"),
    }
}

#[cfg(test)]
mod encoder_input_tests {
    use super::*;
    use crate::ortweb::{DType, OrtTensor};

    /// Same-size squash is the identity canvas, so the tensor is exactly
    /// `normalize_chw` of the source pixels, plane-major C-H-W.
    #[test]
    fn encoder_input_f32_is_identity_chw_at_net_size() {
        let rgb: Vec<u8> = (0..2 * 2 * 3).map(|i| (i * 37 % 251) as u8).collect();
        let t = encoder_input(&rgb, 2, 2, 2, DType::F32).unwrap();
        let OrtTensor::F32(data, dims) = t else {
            panic!("expected F32 tensor, got {t:?}")
        };
        assert_eq!(dims, vec![1, 3, 2, 2]);
        assert_eq!(data, normalize_chw(&rgb, |x| x));
    }

    /// One pixel, hand-computed: ((v/255) - mean) / std per channel.
    #[test]
    fn encoder_input_normalizes_one_pixel_exactly() {
        let t = encoder_input(&[255u8, 0, 128], 1, 1, 1, DType::F32).unwrap();
        let OrtTensor::F32(data, dims) = t else {
            panic!("expected F32 tensor, got {t:?}")
        };
        assert_eq!(dims, vec![1, 3, 1, 1]);
        let want = |c: usize, v: u8| (v as f32 / 255.0 - MEAN[c]) / STD[c];
        assert_eq!(data, vec![want(0, 255), want(1, 0), want(2, 128)]);
    }

    /// The f16 path carries raw bit patterns — one u16 per element, so a
    /// constant canvas yields per-channel constant bit planes of the right
    /// length (one ImageNet-normalized value per channel, not one global).
    #[test]
    fn encoder_input_f16_carries_bit_patterns_and_len() {
        let rgb = vec![200u8; 2 * 3 * 3]; // 2×3 image
        let t = encoder_input(&rgb, 2, 3, 4, DType::F16).unwrap();
        let OrtTensor::F16(bits, dims) = t else {
            panic!("expected F16 tensor, got {t:?}")
        };
        assert_eq!(dims, vec![1, 3, 4, 4]);
        assert_eq!(bits.len(), 3 * 4 * 4);
        let normalized = |c: usize| (200f32 / 255.0 - MEAN[c]) / STD[c];
        for (c, plane) in bits.chunks(4 * 4).enumerate() {
            let want = half::f16::from_f32(normalized(c)).to_bits();
            assert!(plane.iter().all(|&b| b == want), "channel {c} not constant");
        }
    }

    #[test]
    fn encoder_input_rejects_i64_pixel_values() {
        assert!(encoder_input(&[0u8; 3], 1, 1, 1, DType::I64).is_err());
    }
}

#[cfg(test)]
mod tests {
    use super::*;

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

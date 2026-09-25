//! Minimal Rust bridge to onnxruntime-web.
//!
//! No script tag loads ORT anywhere: `index.html` loads none (the main
//! thread never touches ORT) — the worker dynamic-imports the pinned ort
//! ES module and publishes it as `globalThis.ort` before the first
//! session create (`load_ort_once` in src/worker_main.rs owns the pin
//! and the non-JSEP reasoning). Sessions are created from in-memory model bytes
//! (external `.onnx_data` weights passed alongside), inference is awaited
//! per call. Only f32 / f16 / i64 tensors are supported — both shipped
//! graphs declare nothing else, and anything else must fail loud, not
//! silently reinterp.

#![deny(unreachable_pub)]

use anyhow::{Result, bail, ensure};
use half::slice::{HalfBitsSliceExt, HalfFloatSliceExt};
use half::vec::HalfFloatVecExt;

/// Element types the shipped graphs declare. Anything else in a graph is a
/// load-time error, not a silent f32 cast.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DType {
    F32,
    F16,
    I64,
}

impl DType {
    /// onnxruntime-web's tensor-type strings.
    pub fn from_js_name(name: &str) -> Option<Self> {
        match name {
            "float32" => Some(Self::F32),
            "float16" => Some(Self::F16),
            "int64" => Some(Self::I64),
            _ => None,
        }
    }
}

fn product(dims: &[i64]) -> usize {
    dims.iter().product::<i64>().max(0) as usize
}

/// Which ORT execution provider ONE session runs on. The constraint is
/// per-session: ORT #27291 breaks a session whose graph mixes EPs, so
/// each session keeps exactly one. The worker probes ONE EP and every
/// session of the pipeline follows it (see `Sam2::load`).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Ep {
    Wasm,
    WebGpu,
}

impl Ep {
    /// ort-web's provider name strings.
    pub fn as_str(self) -> &'static str {
        match self {
            Ep::Wasm => "wasm",
            Ep::WebGpu => "webgpu",
        }
    }
}

/// A dense tensor. F16 carries raw `half::f16` bit patterns (u16) — the same
/// layout onnxruntime-web's `float16` Uint16Array tensors expect.
#[derive(Debug, Clone, PartialEq)]
pub enum OrtTensor {
    F32(Vec<f32>, Vec<i64>),
    F16(Vec<u16>, Vec<i64>),
    I64(Vec<i64>, Vec<i64>),
}

impl OrtTensor {
    pub fn from_f32(data: Vec<f32>, dims: &[i64]) -> Result<Self> {
        ensure!(
            data.len() == product(dims),
            "f32 len {} != dims {:?}",
            data.len(),
            dims
        );
        Ok(Self::F32(data, dims.to_vec()))
    }

    pub fn from_f16_bits(bits: Vec<u16>, dims: &[i64]) -> Result<Self> {
        ensure!(
            bits.len() == product(dims),
            "f16 len {} != dims {:?}",
            bits.len(),
            dims
        );
        Ok(Self::F16(bits, dims.to_vec()))
    }

    pub fn from_i64(data: Vec<i64>, dims: &[i64]) -> Result<Self> {
        ensure!(
            data.len() == product(dims),
            "i64 len {} != dims {:?}",
            data.len(),
            dims
        );
        Ok(Self::I64(data, dims.to_vec()))
    }

    pub fn dims(&self) -> &[i64] {
        match self {
            Self::F32(_, d) | Self::F16(_, d) | Self::I64(_, d) => d,
        }
    }

    /// (dims, data) widened to f32 — f16 bit patterns widen, i64 is an
    /// error (no postprocess path consumes integer outputs).
    pub fn into_f32(self) -> Result<(Vec<i64>, Vec<f32>)> {
        self.into_f32_slice(|_, len| 0..len)
    }

    /// [`into_f32`] with the flat range picked from the dims/len before any
    /// widening: a [.., C, H, W] output whose one used channel is `chan`
    /// widens one H·W channel instead of all C (SAM2's argmax-IoU mask).
    pub fn into_f32_slice(
        self,
        take: impl FnOnce(&[i64], usize) -> std::ops::Range<usize>,
    ) -> Result<(Vec<i64>, Vec<f32>)> {
        match self {
            Self::F32(d, dims) => {
                let range = take(&dims, d.len());
                Ok((dims, d[range].to_vec()))
            }
            Self::F16(bits, dims) => {
                let range = take(&dims, bits.len());
                let data = bits[range].reinterpret_cast::<half::f16>().to_f32_vec();
                Ok((dims, data))
            }
            Self::I64(..) => bail!("i64 tensor cannot widen to f32"),
        }
    }
}

/// Build an input tensor in the element type the session declares, from f32
/// source values (ids travel as f32-converted ints, mirroring the desktop
/// make_input).
// Production callers are the wasm Detector/Sam2; on host only the tests
// below reach this.
#[cfg_attr(not(target_arch = "wasm32"), allow(dead_code))]
pub(crate) fn make_input(shape: &[i64], data: Vec<f32>, want: DType) -> Result<OrtTensor> {
    match want {
        DType::F32 => OrtTensor::from_f32(data, shape),
        DType::F16 => OrtTensor::from_f16_bits(
            Vec::<half::f16>::from_f32_slice(&data).reinterpret_into(),
            shape,
        ),
        DType::I64 => OrtTensor::from_i64(data.into_iter().map(|x| x as i64).collect(), shape),
    }
}

/// The graphs' output counts are fixed; a mismatch is a load/config bug, not
/// a per-call condition. Destructures for positional access.
// Production callers are the wasm Detector/Sam2; on host only the tests
// below reach this.
#[cfg_attr(not(target_arch = "wasm32"), allow(dead_code))]
pub(crate) fn fixed_outputs<const N: usize>(
    outputs: Vec<OrtTensor>,
    graph: &str,
) -> Result<[OrtTensor; N]> {
    outputs.try_into().map_err(|o: Vec<OrtTensor>| {
        anyhow::anyhow!("{graph} returned {} outputs, expected {N}", o.len())
    })
}

/// The real bridge. Sessions hold the ort-web `InferenceSession` object plus
/// its name lists (session metadata never changes after create).
#[cfg(target_arch = "wasm32")]
pub struct Session {
    obj: js_sys::Object,
    input_names: Vec<String>,
    output_names: Vec<String>,
}

#[cfg(target_arch = "wasm32")]
use wasm_bindgen::{JsCast, JsValue};
#[cfg(target_arch = "wasm32")]
use wasm_bindgen_futures::JsFuture;

// The ort binding layer, hand-rolled over js_sys: wasm-bindgen turns every
// JS module (inline or file) into a snippets-dir import, and the no-modules
// glue that trunk's data-type="worker" build produces cannot carry imports
// — no bindgen JS module can be shared with the worker, while js_sys calls
// work on every target from one source.
// ort resolves at call time, so import order never matters.
#[cfg(target_arch = "wasm32")]
fn ort_global() -> Result<JsValue> {
    let ort = js_sys::Reflect::get(&js_sys::global(), &"ort".into()).map_err(js_err)?;
    anyhow::ensure!(
        !ort.is_undefined(),
        "onnxruntime-web not loaded: globalThis.ort is undefined (the worker's dynamic-imported ort module must be published as globalThis.ort first — see load_ort_once in src/worker_main.rs)"
    );
    Ok(ort)
}

/// `ort.InferenceSession.create(modelBytes, {executionProviders: [ep],
/// externalData})` — awaited. `external` pairs are (location-in-graph,
/// bytes); `path` must equal the location string the ONNX proto references.
#[cfg(target_arch = "wasm32")]
async fn bridge_create_session(
    model: &[u8],
    external: &[(&str, &[u8])],
    ep: Ep,
) -> Result<js_sys::Object> {
    let ort = ort_global()?;
    let session_class = js_get(&ort, "InferenceSession")?;
    let create: js_sys::Function = js_cast(
        js_get(&session_class, "create")?,
        "InferenceSession.create is not a function",
    )?;
    let opts = js_sys::Object::new();
    js_set(
        &opts,
        "executionProviders",
        &js_sys::Array::of1(&ep.as_str().into()),
    )?;
    let external_data = js_sys::Array::new();
    for (path, bytes) in external {
        let entry = js_sys::Object::new();
        js_set(&entry, "path", &(*path).into())?;
        // Uint8Array::from copies out of wasm memory.
        js_set(&entry, "data", &js_sys::Uint8Array::from(*bytes))?;
        external_data.push(&entry);
    }
    js_set(&opts, "externalData", &external_data)?;
    let promise: js_sys::Promise = create
        .call2(&session_class, &js_sys::Uint8Array::from(model), &opts)
        .map_err(js_err)?
        .into();
    promise_obj(promise, "InferenceSession.create did not yield an object").await
}

/// `session.run(feeds)` — awaited; returns the outputs record.
#[cfg(target_arch = "wasm32")]
async fn bridge_run(session: &js_sys::Object, feeds: js_sys::Object) -> Result<js_sys::Object> {
    let run: js_sys::Function = js_cast(js_get(session, "run")?, "session.run is not a function")?;
    let promise: js_sys::Promise = run.call1(session, &feeds).map_err(js_err)?.into();
    promise_obj(promise, "session.run did not yield an object").await
}

/// `new (ort.Tensor)(dtype, data, dims)` — Reflect::construct performs the
/// `new` binding (a bare call on a class constructor throws).
#[cfg(target_arch = "wasm32")]
fn bridge_make_tensor(dtype: &str, data: JsValue, dims: &[i64]) -> Result<js_sys::Object> {
    let ort = ort_global()?;
    let ctor: js_sys::Function =
        js_cast(js_get(&ort, "Tensor")?, "ort.Tensor is not a constructor")?;
    let dims_js: js_sys::Array = dims.iter().map(|&d| JsValue::from_f64(d as f64)).collect();
    let args = js_sys::Array::of3(&dtype.into(), &data, &dims_js);
    let obj = js_sys::Reflect::construct(&ctor, &args).map_err(js_err)?;
    obj.dyn_into::<js_sys::Object>()
        .map_err(|_| anyhow::anyhow!("ort.Tensor constructor did not yield an object"))
}

/// String-array session property (`inputNames` / `outputNames`).
#[cfg(target_arch = "wasm32")]
fn bridge_names(session: &js_sys::Object, key: &str) -> Result<Vec<String>> {
    let arr: js_sys::Array = js_cast(
        js_get(session, key)?,
        &format!("session.{key} is not an array"),
    )?;
    arr.iter()
        .map(|v| {
            v.as_string()
                .ok_or_else(|| anyhow::anyhow!("session.{key} has a non-string entry"))
        })
        .collect()
}

/// `session.inputMetadata?.map((m) => m.type)` — ort-web versions differ on
/// whether inputMetadata is exposed; None makes the Rust side fail loud
/// instead of guessing element types.
#[cfg(target_arch = "wasm32")]
fn bridge_input_types(session: &js_sys::Object) -> Result<Option<Vec<DType>>> {
    let meta = js_get(session, "inputMetadata")?;
    if meta.is_undefined() || meta.is_null() {
        return Ok(None);
    }
    let arr: js_sys::Array = js_cast(meta, "session.inputMetadata is not an array")?;
    let types = arr
        .iter()
        .map(|m| {
            let name = js_get(&m, "type")?
                .as_string()
                .ok_or_else(|| anyhow::anyhow!("non-string type in inputMetadata"))?;
            DType::from_js_name(&name)
                .ok_or_else(|| anyhow::anyhow!("unsupported input type {name:?}"))
        })
        .collect::<Result<Vec<_>>>()?;
    Ok(Some(types))
}

#[cfg(target_arch = "wasm32")]
impl Session {
    /// Create from model bytes on `ep`. `external` pairs are
    /// (location-in-graph, bytes) for external-weights siblings (the
    /// `.onnx_data` files); the location must match the path string the ONNX
    /// proto references.
    pub async fn load(model: &[u8], external: &[(&str, &[u8])], ep: Ep) -> Result<Self> {
        let obj = bridge_create_session(model, external, ep).await?;
        let input_names = bridge_names(&obj, "inputNames")?;
        let output_names = bridge_names(&obj, "outputNames")?;
        Ok(Self {
            obj,
            input_names,
            output_names,
        })
    }

    /// Element type per input, in positional order, from
    /// `session.inputMetadata`. ort-web versions differ on whether that is
    /// exposed — when it is not, this fails loud (the caller falls back to
    /// graph-specific defaults); there is no probe-run guessing here.
    pub async fn input_dtypes(&self) -> Result<Vec<DType>> {
        let Some(types) = bridge_input_types(&self.obj)? else {
            bail!(
                "this onnxruntime-web build does not expose session.inputMetadata; \
                 input dtypes cannot be resolved"
            );
        };
        Ok(types)
    }

    /// Positional feed: `feeds[i]` binds to `input_names()[i]`. Returns
    /// outputs in graph order.
    pub async fn run(&self, feeds: &[&OrtTensor]) -> Result<Vec<OrtTensor>> {
        ensure!(
            feeds.len() == self.input_names.len(),
            "feed count {} != session input count {}",
            feeds.len(),
            self.input_names.len()
        );
        let record = js_sys::Object::new();
        for (tensor, name) in feeds.iter().zip(&self.input_names) {
            let t = tensor_to_js(tensor)?;
            js_set(&record, name, &t)?;
        }
        let outputs = bridge_run(&self.obj, record).await?;
        self.output_names
            .iter()
            .map(|name| {
                let t = js_get(&outputs, name)?;
                ensure!(!t.is_undefined(), "output {name} missing from run result");
                tensor_from_js(&t)
            })
            .collect()
    }
}

#[cfg(target_arch = "wasm32")]
fn js_err(e: JsValue) -> anyhow::Error {
    let msg = if e.is_instance_of::<js_sys::Error>() {
        js_sys::Error::from(e).to_string().into()
    } else {
        e.as_string().unwrap_or_else(|| format!("{e:?}"))
    };
    anyhow::anyhow!("ort-web: {msg}")
}

/// `Reflect::get` with this module's [`js_err`] mapping.
#[cfg(target_arch = "wasm32")]
fn js_get(obj: &JsValue, key: &str) -> Result<JsValue> {
    js_sys::Reflect::get(obj, &key.into()).map_err(js_err)
}

/// `Reflect::set` with this module's [`js_err`] mapping.
#[cfg(target_arch = "wasm32")]
fn js_set(obj: &JsValue, key: &str, val: &JsValue) -> Result<()> {
    js_sys::Reflect::set(obj, &key.into(), val)
        .map(|_| ())
        .map_err(js_err)
}

/// `dyn_into` with the site's failure message — the JsValue is discarded,
/// the message names what was expected.
#[cfg(target_arch = "wasm32")]
fn js_cast<T: JsCast>(v: JsValue, what: &str) -> Result<T> {
    v.dyn_into().map_err(|_| anyhow::anyhow!("{what}"))
}

/// Await a promise and cast the resolved value, with the site's failure
/// message for the cast.
#[cfg(target_arch = "wasm32")]
async fn promise_obj(promise: js_sys::Promise, what: &str) -> Result<js_sys::Object> {
    js_cast(JsFuture::from(promise).await.map_err(js_err)?, what)
}

#[cfg(target_arch = "wasm32")]
fn tensor_to_js(t: &OrtTensor) -> Result<js_sys::Object> {
    let (dtype, typed_array, dims) = match t {
        OrtTensor::F32(d, dims) => (
            "float32",
            JsValue::from(js_sys::Float32Array::from(d.as_slice())),
            dims,
        ),
        OrtTensor::F16(bits, dims) => (
            "float16",
            JsValue::from(js_sys::Uint16Array::from(bits.as_slice())),
            dims,
        ),
        OrtTensor::I64(d, dims) => (
            "int64",
            JsValue::from(js_sys::BigInt64Array::from(d.as_slice())),
            dims,
        ),
    };
    bridge_make_tensor(dtype, typed_array, dims)
}

#[cfg(target_arch = "wasm32")]
fn tensor_from_js(t: &JsValue) -> Result<OrtTensor> {
    let ty = js_get(t, "type")?
        .as_string()
        .ok_or_else(|| anyhow::anyhow!("ort tensor without a string .type"))?;
    let dims_arr: js_sys::Array = js_cast(js_get(t, "dims")?, "ort tensor .dims is not an array")?;
    let dims: Vec<i64> = dims_arr
        .iter()
        .map(|d| {
            let f = d
                .as_f64()
                .ok_or_else(|| anyhow::anyhow!("non-numeric dim in {ty} tensor"))?;
            Ok(f as i64)
        })
        .collect::<Result<_>>()?;
    let data = js_get(t, "data")?;
    match ty.as_str() {
        // ort-web hands float16 data back as Uint16Array bit patterns — the
        // same layout this module ships out.
        "float32" => {
            let a: js_sys::Float32Array = js_cast(data, "float32 output data is not Float32Array")?;
            Ok(OrtTensor::F32(a.to_vec(), dims))
        }
        "float16" => {
            let a: js_sys::Uint16Array = js_cast(data, "float16 output data is not Uint16Array")?;
            Ok(OrtTensor::F16(a.to_vec(), dims))
        }
        "int64" => {
            let a: js_sys::BigInt64Array = js_cast(data, "int64 output data is not BigInt64Array")?;
            Ok(OrtTensor::I64(a.to_vec(), dims))
        }
        other => bail!("unsupported output tensor type {other:?}"),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The EP wire strings ort-web expects are exact and lowercase — a
    /// typo here would silently create sessions on some other (or no)
    /// provider, so they are pinned character-for-character.
    #[test]
    fn ep_wire_strings_match_ort_web_exactly() {
        assert_eq!(Ep::Wasm.as_str(), "wasm");
        assert_eq!(Ep::WebGpu.as_str(), "webgpu");
    }

    #[test]
    fn dtype_from_js_name_accepts_only_ort_dtypes() {
        for name in ["float32", "float16", "int64"] {
            assert_eq!(
                DType::from_js_name(name),
                Some(match name {
                    "float32" => DType::F32,
                    "float16" => DType::F16,
                    _ => DType::I64,
                })
            );
        }
        assert_eq!(DType::from_js_name("int8"), None);
        assert_eq!(DType::from_js_name("bool"), None);
    }

    #[test]
    fn tensor_builders_validate_length_against_dims() {
        assert!(OrtTensor::from_f32(vec![0.0; 6], &[1, 2, 3]).is_ok());
        assert!(OrtTensor::from_f32(vec![0.0; 5], &[1, 2, 3]).is_err());
        assert!(OrtTensor::from_f16_bits(vec![0; 4], &[2, 2]).is_ok());
        assert!(OrtTensor::from_f16_bits(vec![0; 3], &[2, 2]).is_err());
        assert!(OrtTensor::from_i64(vec![0; 2], &[1, 2]).is_ok());
        assert!(OrtTensor::from_i64(vec![0; 3], &[1, 2]).is_err());
        // Zero-sized and scalar shapes are legal.
        assert!(OrtTensor::from_f32(vec![], &[0]).is_ok());
        assert!(OrtTensor::from_f32(vec![1.0], &[]).is_ok());
    }

    #[test]
    fn tensor_accessors() {
        let t = OrtTensor::from_i64(vec![1, 2, 3, 4], &[2, 2]).unwrap();
        assert_eq!(t.dims(), &[2, 2]);
    }

    #[test]
    fn into_f32_widens_f16_and_rejects_i64() {
        let bits: Vec<u16> = [1.0f32, -2.0, 0.5]
            .iter()
            .map(|&x| half::f16::from_f32(x).to_bits())
            .collect();
        let (dims, data) = OrtTensor::from_f16_bits(bits, &[3])
            .unwrap()
            .into_f32()
            .unwrap();
        assert_eq!(dims, vec![3]);
        assert_eq!(data, vec![1.0, -2.0, 0.5]);
        assert_eq!(
            OrtTensor::from_f32(vec![3.5], &[1])
                .unwrap()
                .into_f32()
                .unwrap()
                .1,
            vec![3.5]
        );
        assert!(
            OrtTensor::from_i64(vec![1], &[1])
                .unwrap()
                .into_f32()
                .is_err()
        );
    }

    /// The SAM2 mask slice: [1, 1, C, mh, mw] widens channel `chan` only,
    /// while the dims stay full so mh/mw stay readable.
    #[test]
    fn into_f32_slice_widens_only_the_picked_range() {
        let bits: Vec<u16> = [1.0f32, 2.0, 3.0, 4.0]
            .iter()
            .map(|&x| half::f16::from_f32(x).to_bits())
            .collect();
        let (dims, data) = OrtTensor::from_f16_bits(bits, &[1, 1, 2, 2])
            .unwrap()
            .into_f32_slice(|shape, len| {
                let px = len / shape[2] as usize;
                px..2 * px
            })
            .unwrap();
        assert_eq!(dims, vec![1, 1, 2, 2]);
        assert_eq!(data, vec![3.0, 4.0]);
    }

    #[test]
    fn make_input_converts_f32_source_to_each_dtype() {
        let mk = |dt| make_input(&[1, 2], vec![1.0, 2.0], dt).unwrap();
        assert_eq!(mk(DType::F32), OrtTensor::F32(vec![1.0, 2.0], vec![1, 2]));
        let bits: Vec<u16> = [1.0f32, 2.0]
            .iter()
            .map(|&x| half::f16::from_f32(x).to_bits())
            .collect();
        assert_eq!(mk(DType::F16), OrtTensor::F16(bits, vec![1, 2]));
        assert_eq!(mk(DType::I64), OrtTensor::I64(vec![1, 2], vec![1, 2]));
    }

    #[test]
    fn fixed_outputs_destructures_and_counts_loud() {
        let two = vec![
            OrtTensor::from_f32(vec![1.0], &[]).unwrap(),
            OrtTensor::from_f32(vec![2.0], &[]).unwrap(),
        ];
        let [a, b] = fixed_outputs::<2>(two, "graph").unwrap();
        assert_eq!(a.into_f32().unwrap().1, vec![1.0]);
        assert_eq!(b.into_f32().unwrap().1, vec![2.0]);
        let one = vec![OrtTensor::from_f32(vec![1.0], &[]).unwrap()];
        assert!(fixed_outputs::<2>(one, "graph").is_err());
    }
}

//! The file-picker bridge (src/picker.rs): web file pickers only exist as a
//! DOM `<input type=file>`, so the landing zone's Browse action
//! programmatic-clicks the hidden input from index.html and POLLS its
//! FileList every frame — the poll stays the single drain. The one listener
//! is a `change` handler whose ONLY job is `ctx.request_repaint()`: a file
//! dialog fires no canvas event, so a page that never repaints after the
//! dialog closed would leave the picked file waiting indefinitely (verified
//! experimentally). The closure is `mem::forget`-kept — a page-lifetime
//! listener on a page-lifetime input; the egui borrow never outlives a
//! frame because the handler only touches the cloned context. The host
//! build gets a no-op stub (the tombstone never draws a UI).

pub(crate) struct Picker {
    #[cfg(target_arch = "wasm32")]
    input: web_sys::HtmlInputElement,
}

impl Picker {
    /// Pin the repaint-only `change` listener. Missing input = a page
    /// built from a stale index.html, same class as the missing canvas
    /// panic.
    pub(crate) fn new(ctx: eframe::egui::Context) -> Self {
        #[cfg(target_arch = "wasm32")]
        {
            use eframe::wasm_bindgen::JsCast;
            use eframe::wasm_bindgen::closure::Closure;
            let document = web_sys::window()
                .expect("no window")
                .document()
                .expect("no document");
            let input = document
                .get_element_by_id("scene_picker")
                .expect("missing #scene_picker")
                .dyn_into::<web_sys::HtmlInputElement>()
                .expect("#scene_picker is not an <input>");
            let on_change = Closure::<dyn FnMut(wasm_bindgen::JsValue)>::new(move |_event| {
                ctx.request_repaint();
            });
            // Best-effort: a failed registration only falls back to the old
            // possible stall — startup must not brick over an optional
            // wake-up.
            let _ = input
                .add_event_listener_with_callback("change", on_change.as_ref().unchecked_ref());
            // Page-lifetime input, page-lifetime listener: forgetting the
            // closure is the keep-alive. The drain stays `take_picked`'s
            // poll — this handler repaints, it never touches the FileList
            // (draining from a callback would race the frame that reads it).
            on_change.forget();
            Self { input }
        }
        #[cfg(not(target_arch = "wasm32"))]
        {
            let _ = ctx;
            Self {}
        }
    }

    /// Open the browser's file dialog. Runs inside the egui click frame,
    /// within the user-activation window a canvas click provides.
    pub(crate) fn open(&self) {
        #[cfg(target_arch = "wasm32")]
        self.input.click();
    }

    /// Files picked since the last frame; the input resets on drain.
    #[cfg(target_arch = "wasm32")]
    pub(crate) fn take_picked(&self) -> Vec<web_sys::File> {
        let Some(files) = self.input.files() else {
            return Vec::new();
        };
        let picked: Vec<_> = (0..files.length()).filter_map(|i| files.get(i)).collect();
        if !picked.is_empty() {
            self.input.set_value("");
        }
        picked
    }
}

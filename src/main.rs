#![cfg_attr(debug_assertions, allow(dead_code, unused_variables,))]
#![cfg_attr(not(debug_assertions), windows_subsystem = "windows")]
use std::{
    borrow::Cow,
    collections::{hash_map::Entry, HashMap},
    io::{Read, Seek},
    path::{Path, PathBuf},
    sync::{
        atomic::{AtomicUsize, Ordering},
        Arc,
    },
};

use anyhow::Context;
use eframe::{
    egui::{SelectableLabel, Sense, Widget},
    emath::{lerp, Align, Align2, NumExt},
    epaint::{vec2, Color32, Rect, Rgba, Rounding, Stroke},
    epi::{backend::RepaintSignal, IconData},
};

#[derive(Default)]
struct Selected(HashMap<usize, ()>);

impl std::fmt::Debug for Selected {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_set().entries(self.0.keys()).finish()
    }
}

impl Selected {
    fn is_selected(&self, id: usize) -> bool {
        self.0.contains_key(&id)
    }

    fn unset(&mut self, id: usize) {
        self.0.remove(&id);
    }

    fn set(&mut self, id: usize) {
        self.0.insert(id, ());
    }

    fn drain(&mut self) -> impl Iterator<Item = usize> + '_ {
        self.0.drain().map(|(d, _)| d)
    }

    fn toggle(&mut self, id: usize) {
        match self.0.entry(id) {
            Entry::Occupied(e) => {
                e.remove_entry();
            }
            Entry::Vacant(e) => {
                e.insert(());
            }
        }
    }
}

struct Application<T>
where
    T: Uploader + Send + Sync + 'static,
{
    jobs: Vec<Job<<T as Uploader>::UploadResult>>,
    selected: Selected,
    last_clicked: Option<usize>,
    id: usize,
    repaint: Option<Arc<dyn RepaintSignal>>,
    uploader: T,
}

impl<T> eframe::epi::App for Application<T>
where
    T: Uploader + Send + Sync + 'static,
{
    fn name(&self) -> &str {
        "showit"
    }

    fn update(&mut self, ctx: &eframe::egui::Context, frame: &eframe::epi::Frame) {
        eframe::egui::CentralPanel::default().show(ctx, |ui| {
            for d in ctx.input().raw.dropped_files.iter().flat_map(|d| &d.path) {
                self.add_files(d)
            }

            let max = self.jobs.len();

            if let Err(err) = self.with_ctx(ctx, frame).check_key() {
                self.with_ctx(ctx, frame).report_error(err)
            }
            let jobs_changed = max != self.jobs.len();

            let modifiers = ctx.input().modifiers;
            eframe::egui::ScrollArea::vertical()
                .auto_shrink([false, true])
                .show(ui, |ui| {
                    let max = self.jobs.len();
                    for (id, job) in self.jobs.iter_mut().map(|j| (j.id(), j)) {
                        if job.check_done() {
                            if !ui
                                .add(SelectableLabel::new(
                                    self.selected.is_selected(id),
                                    &*job.title(),
                                ))
                                .clicked()
                            {
                                continue;
                            }

                            // set the finished job's selection state
                            match (modifiers.shift, modifiers.ctrl) {
                                (true, false) => {
                                    if let Some(last) = self.last_clicked {
                                        for n in std::cmp::min(last, id)..=std::cmp::max(last, id) {
                                            self.selected.set(n)
                                        }
                                    }
                                }
                                (false, true) => self.selected.toggle(id),
                                (false, false) => {
                                    for n in (0..max).filter(|&t| t != id) {
                                        self.selected.unset(n)
                                    }
                                    self.selected.set(id)
                                }
                                _ => continue,
                            }

                            self.last_clicked = Some(id);
                        }

                        // NOTE: this has to happen 2nd so the running->finish transition can occur
                        if job.is_running() {
                            ui.add(CustomProgress::<T> { job });
                        }
                    }

                    if jobs_changed {
                        ui.scroll_to_cursor(Some(Align::BOTTOM));
                    }
                });
        });
    }

    fn setup(
        &mut self,
        ctx: &eframe::egui::Context,
        frame: &eframe::epi::Frame,
        _storage: Option<&dyn eframe::epi::Storage>,
    ) {
        self.repaint
            .get_or_insert(frame.0.lock().unwrap().repaint_signal.clone());
        ctx.set_pixels_per_point(1.2);
    }
}

struct ApplicationCtx<'b, 'a: 'b, T>
where
    T: Uploader + Send + Sync + 'static,
{
    app: &'a mut Application<T>,
    ctx: &'b eframe::egui::Context,
    frame: &'b eframe::epi::Frame,
}

impl<'b, 'a: 'b, T> ApplicationCtx<'b, 'a, T>
where
    T: Uploader + Send + Sync + 'static,
{
    fn report_error(&self, err: impl std::fmt::Display) {
        // TODO render this in a modal window
        eprintln!("err: {}", err);
    }

    fn check_key(&mut self) -> anyhow::Result<()> {
        let input = self.ctx.input();

        macro_rules! key {
            ($key:expr, $modifier:tt) => {
                key!($key) && input.modifiers.$modifier == true
            };
            ($key:expr) => {
                input.key_pressed($key)
            };
        }

        use eframe::egui::Key;
        match () {
            _ if key!(Key::C, ctrl) => self.copy_to_clipboard(),
            _ if key!(Key::V, ctrl) => self.app.paste_from_clipboard(),
            _ if key!(Key::A, ctrl) => self.app.select_all(),
            _ if key!(Key::Delete) => self.app.remove_selected(),
            _ => return Ok(()),
        }

        Ok(())
    }

    fn copy_to_clipboard(&mut self) {
        match self.app.copy_to_clipboard() {
            Ok(Some(links)) => {
                eprintln!("copied:\n{}", links)
                // TODO show what links were copied in a modal window
            }
            Ok(..) => {}
            Err(err) => self.report_error(err),
        }
    }
}

impl<T> Application<T>
where
    T: Uploader + Send + Sync + 'static,
{
    fn new(uploader: T) -> Self {
        let (jobs, selected, last_clicked, id, repaint) = <_>::default();
        Self {
            uploader,
            jobs,
            selected,
            last_clicked,
            id,
            repaint,
        }
    }

    fn with_ctx<'b, 'a: 'b>(
        &'a mut self,
        ctx: &'b eframe::egui::Context,
        frame: &'b eframe::epi::Frame,
    ) -> ApplicationCtx<'b, 'a, T> {
        ApplicationCtx {
            app: self,
            ctx,
            frame,
        }
    }

    fn repaint_signal(&self) -> Arc<dyn RepaintSignal> {
        self.repaint
            .clone()
            .expect("application must be initialized first")
    }

    fn check_file_from_disk(path: &std::path::Path) -> anyhow::Result<(String, Vec<u8>)> {
        const LIKELY_SIGNATURE_SIZE: usize = 64;

        anyhow::ensure!(path.is_file(), "cannot parse directory as an image");

        let mut fi = std::fs::OpenOptions::new()
            .read(true)
            .write(false)
            .open(path)?;
        let len = fi.metadata()?.len();

        let mut buf = [0; LIKELY_SIGNATURE_SIZE];
        let max = std::cmp::min(buf.len(), len as usize);
        fi.read_exact(&mut buf[..max])?;
        image::guess_format(buf.as_slice())?;
        fi.rewind()?;

        let mut buf = Vec::with_capacity(len as _);
        std::io::BufReader::new(&mut fi).read_to_end(&mut buf)?;
        let name = path
            .file_name()
            .with_context(|| "cannot get filename")?
            .to_string_lossy()
            .to_string();
        Ok((name, buf))
    }

    fn add_files(&mut self, path: &Path) {
        if !path.is_dir() {
            self.add_file(path);
            return;
        }

        if let Ok(dir) = std::fs::read_dir(path) {
            for file in dir.into_iter().flatten().map(|de| de.path()) {
                self.add_file(&file)
            }
        }
    }

    fn add_file(&mut self, path: &Path) {
        if let Ok((name, data)) = Self::check_file_from_disk(path) {
            self.add_image(ImageSource::dropped(name, data))
        }
    }

    fn add_image(&mut self, source: ImageSource) {
        let id = self.id;
        self.id += 1;

        let (tx, rx) = oneshot::channel();
        let mut job = Job::running(id, source, rx);

        let data = job.take_data();
        if data.is_empty() {
            eprintln!("cannot upload '{}'. empty data", job.title());
            return;
        }

        // TODO store this for later
        // TODO thread pool
        let _handle = std::thread::spawn({
            let mut uploader = self.uploader.clone();
            let progress = DataProgress {
                repaint: self.repaint_signal(),
                amount: job.read_amount(),
            };

            let title = job.title().to_string();
            move || {
                match uploader.upload(data, title, progress) {
                    Ok(resp) => {
                        let _ = tx.send(resp);
                    }
                    Err(err) => {
                        // TODO report this
                    }
                }
            }
        });

        self.jobs.push(job);
    }

    fn paste_from_clipboard(&mut self) {
        if let Some(cb) = Clipboard::try_get() {
            match cb {
                Clipboard::Files { list } => {
                    for file in list {
                        match Self::check_file_from_disk(&*file) {
                            Ok((name, data)) => {
                                self.add_image(ImageSource::dropped(name, data));
                            }
                            Err(err) => eprintln!("cannot read dropped file: {}", err),
                        }
                    }
                }
                Clipboard::Image { data } => {
                    self.add_image(ImageSource::clipboard(data));
                }
                Clipboard::Text { text } => match ImageSource::url(&*text) {
                    Ok(source) => {
                        self.add_image(source);
                    }
                    Err(err) => eprintln!("cannot lookup url: {}", err),
                },
            }
        }
    }

    fn copy_to_clipboard(&mut self) -> anyhow::Result<Option<String>> {
        let links = self
            .jobs
            .iter()
            .filter_map(|job| {
                if self.selected.is_selected(job.id()) {
                    if let Some(res) = job.get_result() {
                        return Some(res.link());
                    }
                }
                None
            })
            .filter(|s| !s.is_empty())
            .fold(String::new(), |mut out, s| {
                if !out.is_empty() {
                    out.push('\n');
                }
                out.push_str(&*s);
                out
            });

        if links.is_empty() {
            return Ok(None);
        }

        clipboard_win::set_clipboard_string(&links)
            .map(|_| Some(links))
            .map_err(Into::into)
    }

    fn select_all(&mut self) {
        for (i, job) in self.jobs.iter().map(|j| (j.id(), j)) {
            if job.is_finished() {
                self.selected.set(i)
            }
        }
    }

    fn remove_selected(&mut self) {
        for selected in self.selected.drain() {
            if let Some(p) = self.jobs.iter().position(|j| j.id() == selected) {
                self.jobs.swap_remove(p);
            }
        }
        self.jobs.sort_by_key(|j| j.id());
    }
}

struct CustomProgress<'a, R>
where
    R: Uploader + 'static,
{
    job: &'a Job<R::UploadResult>,
}

impl<'a, R> Widget for CustomProgress<'a, R>
where
    R: Uploader + 'static,
{
    fn ui(self, ui: &mut eframe::egui::Ui) -> eframe::egui::Response {
        // adapted from `https://github.com/emilk/egui/blob/833829e3d857835e5b2983b3ec6297c5e06f0197/egui/src/widgets/progress_bar.rs`
        let w = ui.available_size_before_wrap().x.at_least(96.0);
        let h = ui.spacing().interact_size.y;
        let (outer_rect, resp) = ui.allocate_exact_size(vec2(w, h), Sense::hover());
        if ui.is_rect_visible(resp.rect) {
            ui.ctx().request_repaint();
            let visuals = ui.style().visuals.clone();
            ui.painter().rect(
                outer_rect,
                Rounding::none(),
                visuals.extreme_bg_color,
                Stroke::none(),
            );

            let inner_rect = Rect::from_min_size(
                outer_rect.min,
                vec2(
                    (outer_rect.width() * self.job.progress()).at_least(outer_rect.height()),
                    outer_rect.height(),
                ),
            );

            let (dark, bright) = (0.7, 1.0);
            let color_factor = lerp(dark..=bright, ui.input().time.cos().abs());

            ui.painter().rect(
                inner_rect,
                Rounding::none(),
                Color32::from(Rgba::from(visuals.selection.bg_fill) * color_factor as f32),
                Stroke::none(),
            );
        }

        // and the custom text overlay
        let rect = resp.rect;
        let painter = ui.painter_at(rect);

        painter.text(
            rect.left_center(),
            Align2::LEFT_CENTER,
            self.job.title(),
            eframe::egui::TextStyle::Button.resolve(&*ui.ctx().style()),
            ui.style().visuals.strong_text_color(),
        );

        painter.text(
            rect.center(),
            Align2::CENTER_CENTER,
            self.job.formatted_size(),
            eframe::egui::TextStyle::Monospace.resolve(&*ui.ctx().style()),
            ui.style().visuals.text_color(),
        );

        painter.text(
            rect.right_center(),
            Align2::RIGHT_CENTER,
            format_args!("{:.2}%", 100.0 * self.job.progress()),
            eframe::egui::TextStyle::Monospace.resolve(&*ui.ctx().style()),
            ui.style().visuals.text_color(),
        );

        resp
    }
}

struct Progress<R, C>(R, C);
impl<R: Read, C: FnMut(usize)> Read for Progress<R, C> {
    fn read(&mut self, buf: &mut [u8]) -> std::io::Result<usize> {
        let n = self.0.read(buf)?;
        (self.1)(n);
        Ok(n)
    }
}

struct NullRepaint;
impl RepaintSignal for NullRepaint {
    fn request_repaint(&self) {}
}

#[derive(Clone)]
struct DataProgress {
    pub repaint: Arc<dyn RepaintSignal>,
    pub amount: Arc<AtomicUsize>,
}

impl Default for DataProgress {
    fn default() -> Self {
        Self {
            repaint: Arc::new(NullRepaint),
            amount: <Arc<AtomicUsize>>::default(),
        }
    }
}

trait UploadResult: std::fmt::Debug + Send + Sync + 'static {
    fn link(&self) -> Cow<'_, str>;
    fn delete_link(&self) -> Option<Cow<'_, str>>;
}

trait Uploader: Clone {
    type UploadResult: UploadResult;
    fn upload(
        &mut self,
        data: Vec<u8>,
        name: String,
        progress: DataProgress,
    ) -> anyhow::Result<Self::UploadResult>;
}

#[derive(Default, Copy, Clone)]
struct ImgurUploader;

impl ImgurUploader {
    const ENDPOINT: &'static str = "https://api.imgur.com/3/upload";
    const USER_AGENT: &'static str =
        concat!(env!("CARGO_PKG_NAME"), "/", env!("CARGO_PKG_VERSION"));
}

impl Uploader for ImgurUploader {
    type UploadResult = ImgurResponse;

    fn upload(
        &mut self,
        data: Vec<u8>,
        name: String,
        progress: DataProgress,
    ) -> anyhow::Result<Self::UploadResult> {
        use reqwest::blocking::{multipart::*, Client};
        use std::io::Cursor;

        #[derive(serde::Deserialize)]
        struct Resp {
            data: <ImgurUploader as Uploader>::UploadResult,
            success: bool,
            status: u16,
        }

        let len = data.len();
        let boxed = Cursor::new(data);

        let part = Part::reader_with_length(
            Progress(boxed, {
                let repaint = progress.repaint.clone();
                let amount = progress.amount.clone();
                move |d| {
                    amount.fetch_add(d, Ordering::SeqCst);
                    repaint.request_repaint();
                }
            }),
            len as _,
        )
        .file_name(name);

        let resp = Client::new()
            .post(Self::ENDPOINT)
            .header("User-Agent", Self::USER_AGENT)
            .header(reqwest::header::AUTHORIZATION, "Client-ID c9ae1602addbd9c") // TODO don't hardcode this
            .multipart(Form::new().part("image", part))
            .send()?;

        anyhow::ensure!(
            resp.status() == reqwest::StatusCode::OK,
            "cannot upload image"
        );
        let resp = resp.json::<Resp>()?;
        anyhow::ensure!(resp.success, "upload failed");
        anyhow::ensure!(resp.status == 200, "upload failed");
        Ok(resp.data)
    }
}

#[derive(Debug, serde::Deserialize)]
#[allow(dead_code)]
struct ImgurResponse {
    id: String,
    datetime: u64,
    #[serde(rename = "type")]
    ty: String,
    width: u64,
    height: u64,
    deletehash: String,
    link: String,
}

impl UploadResult for ImgurResponse {
    fn link(&self) -> Cow<'_, str> {
        Cow::Borrowed(&*self.link)
    }

    fn delete_link(&self) -> Option<Cow<'_, str>> {
        Some(Cow::Owned(format!(
            "https://api.imgur.com/3/image/{}",
            self.deletehash
        )))
    }
}

fn humanize(size: usize) -> String {
    const SIZES: [&str; 5] = ["B", "KB", "MB", "GB", "TB"];
    const SCALE: f64 = 1024.0;

    let (mut order, mut size) = (0, size as f64);
    while size >= SCALE && order + 1 < SIZES.len() {
        order += 1;
        size /= SCALE;
    }

    format!("{:.2} {}", size, SIZES[order])
}

#[derive(Default, Debug)]
struct ImageSource {
    kind: ImageKind,
    size: u64,
}

impl ImageSource {
    fn clipboard(data: impl Into<Box<[u8]>>) -> Self {
        let data = data.into();
        Self {
            size: data.len() as _,
            kind: ImageKind::Clipboard { data },
        }
    }

    fn dropped(name: impl Into<Box<str>>, data: impl Into<Box<[u8]>>) -> Self {
        let data = data.into();
        Self {
            size: data.len() as _,
            kind: ImageKind::Dropped {
                name: name.into(),
                data,
            },
        }
    }

    fn url(url: &str) -> anyhow::Result<Self> {
        anyhow::ensure!(
            matches!(url::Url::parse(url)?.scheme(), "http" | "https"),
            "must be an http/https url"
        );

        let resp = reqwest::blocking::Client::new().head(url).send()?;
        let size = resp
            .headers()
            .get("Content-Length")
            .map(|h| h.to_str())
            .transpose()?
            .map(<u64 as std::str::FromStr>::from_str)
            .transpose()?
            .with_context(|| "unsupported url")?;

        resp.headers()
            .get("Content-Type")
            .map(|h| h.to_str())
            .transpose()?
            .and_then(|s| s.split_once('/').map(|(ty, _)| ty))
            .filter(|&ty| ty == "image")
            .with_context(|| "only image urls are supported")?;

        let resp = reqwest::blocking::Client::new()
            .get(&*url)
            .header("User-Agent", ImgurUploader::USER_AGENT)
            .send()?;

        // TODO: probably a better way of doing this
        let body = resp.bytes()?.as_ref().to_vec();

        Ok(Self {
            kind: ImageKind::Url {
                url: url.into(),
                data: body.into(),
            },
            size,
        })
    }
}

enum ImageKind {
    Clipboard { data: Box<[u8]> },
    Dropped { name: Box<str>, data: Box<[u8]> },
    Url { url: Box<str>, data: Box<[u8]> },
    Null,
}

impl Default for ImageKind {
    fn default() -> Self {
        Self::Null
    }
}

impl ImageKind {
    const fn title(&self) -> Cow<'_, str> {
        match self {
            Self::Clipboard { .. } => Cow::Borrowed("<clipboard>"),
            Self::Dropped { name, .. } => Cow::Borrowed(&*name),
            Self::Url { url, .. } => Cow::Borrowed(&*url),
            _ => unreachable!(),
        }
    }
}

impl std::fmt::Debug for ImageKind {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Clipboard { data } => f
                .debug_struct("Clipboard")
                .field("size", &data.len())
                .finish_non_exhaustive(),
            Self::Dropped { name, data } => f
                .debug_struct("Dropped")
                .field("name", name)
                .field("size", &data.len())
                .finish_non_exhaustive(),
            Self::Url { url, data } => f
                .debug_struct("Url")
                .field("url", url)
                .field("size", &data.len())
                .finish_non_exhaustive(),
            Self::Null => f.debug_struct("Null").finish_non_exhaustive(),
        }
    }
}

#[derive(Default)]
struct JobInner {
    id: usize,
    source: ImageSource,
    read_amount: Arc<AtomicUsize>,
    total: usize,
    formatted_size: String,
}

enum Job<R> {
    Running {
        inner: JobInner,
        rx: oneshot::Receiver<R>,
    },
    Finished {
        inner: JobInner,
        result: R,
    },
}

impl<R> Job<R>
where
    R: UploadResult,
{
    fn get_result(&self) -> Option<&'_ R> {
        match self {
            Self::Finished { result, .. } => Some(result),
            _ => None,
        }
    }

    // TODO this should return something other than a bool, (e.g. it should consume self)
    fn check_done(&mut self) -> bool {
        match self {
            Job::Running { inner, rx } => match rx.try_recv() {
                Ok(result) => {
                    assert!(
                        matches!(rx.try_recv(), Err(oneshot::TryRecvError::Disconnected)),
                        "job should be finished before transitioning to the finished state"
                    );
                    *self = Self::Finished {
                        inner: std::mem::take(inner),
                        result,
                    };
                    assert!(self.is_finished(), "job should be finished");
                    true
                }
                _ => false,
            },
            _ => true,
        }
    }
}

impl<R> Job<R> {
    fn running(id: usize, source: ImageSource, rx: oneshot::Receiver<R>) -> Self {
        Self::Running {
            inner: JobInner {
                id,
                read_amount: <Arc<AtomicUsize>>::default(),
                total: source.size as _,
                formatted_size: humanize(source.size as _),
                source,
            },
            rx,
        }
    }

    const fn is_running(&self) -> bool {
        matches!(self, Self::Running { .. })
    }

    const fn is_finished(&self) -> bool {
        matches!(self, Self::Finished { .. })
    }

    #[doc(hidden)]
    fn inner(&self) -> &JobInner {
        match self {
            Self::Running { inner, .. } | Self::Finished { inner, .. } => inner,
        }
    }

    #[doc(hidden)]
    fn inner_mut(&mut self) -> &mut JobInner {
        match self {
            Self::Running { inner, .. } | Self::Finished { inner, .. } => inner,
        }
    }

    fn id(&self) -> usize {
        self.inner().id
    }

    fn title(&self) -> Cow<'_, str> {
        self.inner().source.kind.title()
    }

    fn read_amount(&self) -> Arc<AtomicUsize> {
        self.inner().read_amount.clone()
    }

    fn formatted_size(&self) -> &str {
        &*self.inner().formatted_size
    }

    fn total(&self) -> usize {
        self.inner().total
    }

    fn progress(&self) -> f32 {
        self.read_amount().load(Ordering::SeqCst) as f32 / self.total() as f32
    }

    fn take_data(&mut self) -> Vec<u8> {
        match &mut self.inner_mut().source.kind {
            ImageKind::Clipboard { data }
            | ImageKind::Dropped { data, .. }
            | ImageKind::Url { data, .. } => std::mem::take(data).into(),
            _ => vec![],
        }
    }
}

#[derive(Debug)]
enum Clipboard {
    Text { text: String },
    Files { list: Vec<PathBuf> },
    Image { data: Vec<u8> },
}

impl Clipboard {
    fn try_get() -> Option<Self> {
        {
            let mut text = String::new();
            if let Ok(..) = clipboard_win::with_clipboard(|| {
                clipboard_win::get::<String, _>(clipboard_win::formats::Unicode)
                    .ok()
                    .map(|t| text = t)
                    .unwrap_or_default();
            }) {
                if !text.is_empty() {
                    return Some(Self::Text { text });
                }
            }
        }

        {
            let mut data = Vec::new();
            if let Ok(..) = clipboard_win::with_clipboard(|| {
                clipboard_win::get::<Vec<u8>, _>(clipboard_win::formats::Bitmap)
                    .ok()
                    .map(|t| data = t)
                    .unwrap_or_default();
            }) {
                if !data.is_empty() {
                    return Some(Self::Image { data });
                }
            }
        }

        {
            let mut data = Vec::new();
            if let Ok(..) = clipboard_win::with_clipboard(|| {
                clipboard_win::get::<Vec<String>, _>(clipboard_win::formats::FileList)
                    .ok()
                    .map(|t| data = t)
                    .unwrap_or_default();
            }) {
                if !data.is_empty() {
                    let list = data.into_iter().map(Into::into).collect();
                    return Some(Self::Files { list });
                }
            }
        }
        None
    }
}

#[cfg(feature = "mock")]
mod mock;

const THE_BEST_ICON: &[u8] = include_bytes!("../icon.png");

fn main() {
    #[cfg(feature = "mock")]
    mock::start_test_server();

    let img = image::load_from_memory(THE_BEST_ICON).unwrap().to_rgba8();
    let (width, height) = img.dimensions();

    let opts = eframe::NativeOptions {
        icon_data: Some(IconData {
            rgba: img.into_raw(),
            width,
            height,
        }),
        drag_and_drop_support: true,
        initial_window_size: Some(eframe::epaint::Vec2 { x: 400.0, y: 600.0 }),
        ..Default::default()
    };
    if cfg!(feature = "mock") {
        #[cfg(feature = "mock")]
        eframe::run_native(Box::new(Application::new(mock::TestUploader)), opts);
    } else {
        eframe::run_native(Box::new(Application::new(ImgurUploader)), opts);
    };
}

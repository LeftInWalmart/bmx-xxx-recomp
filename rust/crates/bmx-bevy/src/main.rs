//! Bevy 3D host for the recompiled BMX XXX executable.
//!
//! The recompiled game runs on its own threads (one host thread per guest OS thread). Every
//! frame the game finishes on its GX pipe is decoded by `gc_runtime::gx` into clip-space
//! triangle batches, each carrying its full TEV state, and replayed on an emulated EFB by the
//! render-world GX pass ([`gxrender`]). Audio produced by the DSP emulation and the game's DAC
//! mixer is played through the default output device.
//!
//! Controls (port 1): arrows = stick, X = A, Z = B, C = X, V = Y, Enter = Start,
//! A / S = L / R, D = Z, IJKL = C-stick, or any connected gamepad. F1 toggles the overlay,
//! F2 opens the controls screen, P the settings menu (volume, window mode, gamma, VSync).

mod gxrender;
mod input;
mod settings;

use bevy::core_pipeline::tonemapping::Tonemapping;
use bevy::prelude::*;
use bevy::render::view::screenshot::{save_to_disk, Screenshot};
use gc_runtime::gx::FrameShared;
use gc_runtime::{audio::AudioShared, diag, hw, pad, CALLS, FN_ENTRIES, LAST_FN, UNKNOWN_TARGETS};
use std::sync::atomic::Ordering::Relaxed;
use std::sync::Arc;

#[derive(Resource)]
struct GameLink {
    frames: Arc<FrameShared>,
    #[allow(dead_code)]
    pad: Arc<pad::PadShared>,
    shown: u64,
    draws: u32,
    tris: usize,
    batches: usize,
}

#[derive(Component)]
struct StatusText;

#[derive(Resource)]
struct PendingLink(Option<(Arc<FrameShared>, Arc<pad::PadShared>)>);

fn main() {
    let mut cpu = bmx_recomp::boot();
    let m = gc_runtime::machine();
    let (frames, pad, audio) = (m.frames.clone(), m.pad.clone(), m.audio.clone());
    std::thread::Builder::new()
        .name("bmx-cpu".into())
        .stack_size(256 << 20)
        .spawn(move || bmx_recomp::run(&mut cpu))
        .expect("spawn cpu thread");
    start_audio(audio);

    let mut app = App::new();
    app.insert_resource(ClearColor(Color::BLACK))
        .add_plugins(DefaultPlugins.set(WindowPlugin {
            primary_window: Some(Window {
                title: "BMX XXX (recomp)".into(),
                resolution: (1280, 960).into(),
                ..default()
            }),
            ..default()
        }));
    app.add_plugins(gxrender::GxRenderPlugin)
        .insert_resource(gxrender::GxLink(frames.clone()))
        .insert_resource(PendingLink(Some((frames, pad.clone()))))
        .insert_resource(input::Input::new(pad.clone()))
        .insert_resource(settings::Settings::load())
        .init_resource::<settings::MenuState>()
        .add_systems(Startup, (setup, input::setup_menu, settings::setup, settings::setup_fps))
        .add_systems(Update, (show_frame, input::read_input, input::menu, settings::menu, fps_counter, update_status, toggle_overlay, probe_exit, timed_screenshots))
        .run();
}

/// `BMX_SCREENSHOTS=10,30,60`: save `bmx_shot_<secs>.png` at those app times (lets automated
/// probe runs check what is actually on screen).
fn timed_screenshots(time: Res<Time>, mut commands: Commands, mut next: Local<usize>) {
    static AT: std::sync::OnceLock<Vec<f32>> = std::sync::OnceLock::new();
    let at = AT.get_or_init(|| {
        std::env::var("BMX_SCREENSHOTS")
            .map(|v| v.split(',').filter_map(|x| x.trim().parse().ok()).collect())
            .unwrap_or_default()
    });
    while *next < at.len() && time.elapsed_secs() >= at[*next] {
        let path = format!("bmx_shot_{:03}.png", at[*next] as u32);
        commands.spawn(Screenshot::primary_window()).observe(save_to_disk(path));
        *next += 1;
    }
}

/// `BMX_PROBE_SECS=n`: probe run — write the pacing summary to bmx_probe.txt and exit.
fn probe_exit(time: Res<Time>, mut exit: MessageWriter<AppExit>) {
    static LIMIT: std::sync::OnceLock<Option<f32>> = std::sync::OnceLock::new();
    let limit = *LIMIT.get_or_init(|| {
        let s = std::env::var("BMX_PROBE_SECS").ok()?.parse().ok()?;
        gc_runtime::probe::start(gc_runtime::machine());
        Some(s)
    });
    if limit.is_some_and(|l| time.elapsed_secs() > l) {
        use std::io::Write;
        if let Ok(mut f) = std::fs::OpenOptions::new().append(true).create(true).open("bmx_probe.txt") {
            let _ = writeln!(f, "=== summary ===
{}", gc_runtime::probe::report());
        }
        exit.write(AppExit::Success);
    }
}

/// Play the AI output through the default device (resampled to its rate).
fn start_audio(src: Arc<AudioShared>) {
    use cpal::traits::{DeviceTrait, HostTrait, StreamTrait};
    let result = (|| -> Result<(), String> {
        let host = cpal::default_host();
        let dev = host.default_output_device().ok_or("no audio output device")?;
        let cfg = dev.default_output_config().map_err(|e| e.to_string())?;
        let channels = cfg.channels() as usize;
        let out_rate = cfg.sample_rate() as f64;
        let config: cpal::StreamConfig = cfg.into();
        let mut pos = 0f64;
        let mut cur = [0i16; 2];
        let mut next = [0i16; 2];
        let mut buf = Vec::new();
        let stream = dev
            .build_output_stream(
                &config,
                move |data: &mut [f32], _| {
                    // Source/device rate ratio, trimmed to keep the jitter buffer near its target.
                    let step = src.rate() as f64 / out_rate * src.rate_trim();
                    for frame in data.chunks_mut(channels) {
                        pos += step;
                        while pos >= 1.0 {
                            pos -= 1.0;
                            cur = next;
                            buf.clear();
                            src.pop(1, &mut buf);
                            next = buf[0];
                        }
                        let t = pos as f32;
                        let vol = f32::from_bits(settings::VOLUME.load(Relaxed));
                        // AI DMA frames are (right, left), as the AX ucode writes them.
                        let r = (cur[0] as f32 * (1.0 - t) + next[0] as f32 * t) / 32768.0 * vol;
                        let l = (cur[1] as f32 * (1.0 - t) + next[1] as f32 * t) / 32768.0 * vol;
                        for (i, s) in frame.iter_mut().enumerate() {
                            *s = if i % 2 == 0 { l } else { r };
                        }
                    }
                },
                |e| eprintln!("[bmx] audio stream error: {e}"),
                None,
            )
            .map_err(|e| e.to_string())?;
        stream.play().map_err(|e| e.to_string())?;
        std::mem::forget(stream); // keep playing for the life of the process
        Ok(())
    })();
    if let Err(e) = result {
        diag::log(format!("audio output unavailable: {e}"));
    }
}

fn setup(mut commands: Commands, mut pending: ResMut<PendingLink>) {
    commands.spawn((Camera3d::default(), Tonemapping::None));
    commands.spawn((
        Text::new(""),
        TextFont { font_size: FontSize::Px(14.0), ..default() },
        TextColor(Color::srgb(0.9, 1.0, 0.6)),
        Node { position_type: PositionType::Absolute, top: px(8), left: px(8), ..default() },
        // Debug overlay: hidden unless F1 is pressed.
        Visibility::Hidden,
        StatusText,
    ));
    let (frames, pad) = pending.0.take().expect("game link");
    commands.insert_resource(GameLink { frames, pad, shown: 0, draws: 0, tris: 0, batches: 0 });
}

/// Track the newest guest frame for the status overlay and the probe (the drawing itself
/// happens in the render world, gxrender.rs).
fn show_frame(mut link: ResMut<GameLink>) {
    let n = link.frames.count();
    gc_runtime::probe::host_frame(n != link.shown);
    if n == link.shown {
        return;
    }
    link.shown = n;
    let f = link.frames.latest();
    link.draws = f.draws;
    link.tris = f.triangles();
    link.batches = f.batches.len();
}

/// Top-right FPS counter (toggle in the P menu).
fn fps_counter(
    link: Option<Res<GameLink>>,
    settings: Res<settings::Settings>,
    mut w: Local<settings::FpsWindow>,
    mut q: Query<(&mut Text, &mut Visibility), With<settings::FpsText>>,
) {
    let Some(link) = link else { return };
    settings::update_fps(link.frames.count(), &settings, &mut w, &mut q);
}

fn update_status(link: Res<GameLink>, mut q: Query<(&mut Text, &Visibility), With<StatusText>>) {
    if q.iter().all(|(_, v)| *v == Visibility::Hidden) {
        return;
    }
    let last = LAST_FN.load(Relaxed);
    let mut s = format!(
        "BMX XXX recomp - {} functions | GX frames {} (draws {}, batches {}, tris {})\n\
         functions entered {} | indirect calls {} (unresolved {})\n\
         last function {:#010x} {}\nMMIO r/w {}/{} | GX FIFO bytes {} | VI {}\n",
        bmx_recomp::FUNCTION_COUNT,
        link.frames.count(),
        link.draws,
        link.batches,
        link.tris,
        FN_ENTRIES.load(Relaxed),
        CALLS.load(Relaxed),
        UNKNOWN_TARGETS.load(Relaxed),
        last,
        bmx_recomp::function_name(last).unwrap_or("?"),
        hw::MMIO_READS.load(Relaxed),
        hw::MMIO_WRITES.load(Relaxed),
        hw::GX_BYTES.load(Relaxed),
        gc_runtime::sched::VI_COUNT.load(Relaxed),
    );
    for l in diag::tail(6) {
        s += &format!("> {l}\n");
    }
    for (mut t, _) in &mut q {
        t.0 = s.clone();
    }
}

fn toggle_overlay(keys: Res<ButtonInput<KeyCode>>, mut q: Query<&mut Visibility, With<StatusText>>) {
    if keys.just_pressed(KeyCode::F1) {
        for mut v in &mut q {
            v.toggle_visible_hidden();
        }
    }
}

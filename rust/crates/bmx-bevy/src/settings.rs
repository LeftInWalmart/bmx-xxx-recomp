//! Player settings and the P menu: volume, window mode, gamma, VSync, internal resolution
//! (anti-aliasing), the FPS counter and "Unlock all".
//!
//! Saved as JSON next to the controls (`%APPDATA%\BMX-XXX-Recomp\settings.json`, or the
//! `BMX_SETTINGS` path), so each player keeps their own. Every change applies and saves
//! immediately.
//!
//! The game runs at the console's steady 59.94 frames per second: its logic, physics and
//! animation are frame-locked to the video interrupt, as on the GameCube, and each finished frame
//! is shown at the next retrace. VSync only decides whether the window waits for the monitor.

use bevy::prelude::*;
use bevy::render::extract_resource::ExtractResource;
use bevy::window::{MonitorSelection, PresentMode, PrimaryWindow, VideoModeSelection, WindowMode};
use serde::{Deserialize, Serialize};
use std::path::PathBuf;
use std::sync::atomic::{AtomicBool, AtomicU32, Ordering::Relaxed};

/// Output volume as f32 bits (read by the audio callback).
pub static VOLUME: AtomicU32 = AtomicU32::new(0x3F80_0000); // 1.0
/// The P menu is open (game input is held neutral).
pub static MENU_OPEN: AtomicBool = AtomicBool::new(false);

#[derive(Serialize, Deserialize, Clone, Copy, PartialEq, Debug)]
pub enum WindowKind {
    Windowed,
    Borderless,
    Fullscreen,
}

#[derive(Serialize, Deserialize, Clone, Resource, ExtractResource, Debug)]
#[serde(default)]
pub struct Settings {
    pub version: u32,
    /// 0.0 ..= 1.5
    pub volume: f32,
    pub window: WindowKind,
    /// Display gamma applied to the final image (1.0 = as the console outputs it).
    pub gamma: f32,
    pub vsync: bool,
    /// Internal resolution multiplier (1 = console native 640x528). Higher values render the
    /// frame larger and downsample it to the window: smooth, anti-aliased edges.
    pub internal_res: u32,
    /// FPS counter in the top-right corner.
    pub show_fps: bool,
}

impl Default for Settings {
    fn default() -> Self {
        Settings { version: 1, volume: 1.0, window: WindowKind::Windowed, gamma: 1.0, vsync: true, internal_res: 4, show_fps: true }
    }
}

pub fn settings_path() -> PathBuf {
    if let Some(p) = std::env::var_os("BMX_SETTINGS") {
        return PathBuf::from(p);
    }
    crate::input::config_path().with_file_name("settings.json")
}

impl Settings {
    pub fn load() -> Settings {
        let s: Settings = std::fs::read_to_string(settings_path())
            .ok()
            .and_then(|t| serde_json::from_str(&t).ok())
            .unwrap_or_default();
        s.sanitized()
    }
    fn sanitized(mut self) -> Settings {
        self.volume = self.volume.clamp(0.0, 1.5);
        self.gamma = self.gamma.clamp(0.5, 2.0);
        self.internal_res = self.internal_res.clamp(1, 6);
        self
    }
    pub fn save(&self) {
        let p = settings_path();
        if let Some(d) = p.parent() {
            let _ = std::fs::create_dir_all(d);
        }
        if let Ok(t) = serde_json::to_string_pretty(self) {
            let _ = std::fs::write(p, t);
        }
    }
    fn window_mode(&self) -> WindowMode {
        match self.window {
            WindowKind::Windowed => WindowMode::Windowed,
            WindowKind::Borderless => WindowMode::BorderlessFullscreen(MonitorSelection::Current),
            WindowKind::Fullscreen => WindowMode::Fullscreen(MonitorSelection::Current, VideoModeSelection::Current),
        }
    }
    fn present_mode(&self) -> PresentMode {
        if self.vsync { PresentMode::AutoVsync } else { PresentMode::AutoNoVsync }
    }
}

#[derive(Component)]
pub struct SettingsText;

#[derive(Resource, Default)]
pub struct MenuState {
    open: bool,
    selected: usize,
}

const ITEMS: usize = 8;
const UNLOCK_ROW: usize = 6;

pub fn setup(mut commands: Commands, settings: Res<Settings>, mut windows: Query<&mut Window, With<PrimaryWindow>>) {
    commands.spawn((
        Text::new(""),
        TextFont { font_size: FontSize::Px(18.0), ..default() },
        TextColor(Color::WHITE),
        BackgroundColor(Color::srgba(0.0, 0.0, 0.0, 0.85)),
        Node {
            position_type: PositionType::Absolute,
            top: Val::Percent(30.0),
            left: Val::Percent(30.0),
            padding: UiRect::all(Val::Px(16.0)),
            ..default()
        },
        Visibility::Hidden,
        SettingsText,
    ));
    VOLUME.store(settings.volume.to_bits(), Relaxed);
    if let Ok(mut w) = windows.single_mut() {
        w.mode = settings.window_mode();
        w.present_mode = settings.present_mode();
    }
}

/// P: open/close (Esc also closes). Up/Down select, Left/Right change, Enter on "Reset" restores
/// the defaults. F11 / Alt+Enter toggle borderless fullscreen at any time.
pub fn menu(
    keys: Res<ButtonInput<KeyCode>>,
    mut settings: ResMut<Settings>,
    mut state: ResMut<MenuState>,
    mut text: Query<(&mut Text, &mut Visibility), With<SettingsText>>,
    mut windows: Query<&mut Window, With<PrimaryWindow>>,
) {
    let mut changed = false;
    let alt = keys.pressed(KeyCode::AltLeft) || keys.pressed(KeyCode::AltRight);
    if keys.just_pressed(KeyCode::F11) || (alt && keys.just_pressed(KeyCode::Enter)) {
        settings.window = if settings.window == WindowKind::Windowed { WindowKind::Borderless } else { WindowKind::Windowed };
        changed = true;
    }
    if keys.just_pressed(KeyCode::KeyP) || (state.open && keys.just_pressed(KeyCode::Escape)) {
        state.open = !state.open;
    }
    MENU_OPEN.store(state.open, Relaxed);
    if let Ok((mut t, mut vis)) = text.single_mut() {
        *vis = if state.open { Visibility::Visible } else { Visibility::Hidden };
        if state.open {
            if keys.just_pressed(KeyCode::ArrowDown) {
                state.selected = (state.selected + 1) % ITEMS;
            }
            if keys.just_pressed(KeyCode::ArrowUp) {
                state.selected = (state.selected + ITEMS - 1) % ITEMS;
            }
            let dir = keys.just_pressed(KeyCode::ArrowRight) as i32 - keys.just_pressed(KeyCode::ArrowLeft) as i32;
            let enter = keys.just_pressed(KeyCode::Enter) && !alt;
            if state.selected == UNLOCK_ROW && (enter || dir != 0) {
                // Session only: not saved, never written to the memory card (cheats.rs).
                gc_runtime::cheats::unlock_all();
            } else if dir != 0 || (state.selected == ITEMS - 1 && enter) {
                let s = &mut *settings;
                match state.selected {
                    0 => s.volume = ((s.volume + 0.05 * dir as f32) * 20.0).round() / 20.0,
                    1 => {
                        let order = [WindowKind::Windowed, WindowKind::Borderless, WindowKind::Fullscreen];
                        let i = order.iter().position(|k| *k == s.window).unwrap_or(0) as i32;
                        s.window = order[(i + dir).rem_euclid(3) as usize];
                    }
                    2 => s.gamma = ((s.gamma + 0.05 * dir as f32) * 20.0).round() / 20.0,
                    3 => s.vsync = !s.vsync,
                    4 => s.internal_res = (s.internal_res as i32 + dir).clamp(1, 6) as u32,
                    5 => s.show_fps = !s.show_fps,
                    _ => *s = Settings::default(),
                }
                *s = s.clone().sanitized();
                changed = true;
            }
            let rows = [
                format!("Volume        < {:>3.0}% >", settings.volume * 100.0),
                format!("Window        < {:?} >", settings.window),
                format!("Gamma         < {:.2} >", settings.gamma),
                format!("VSync         < {} >", if settings.vsync { "On" } else { "Off" }),
                format!(
                    "Anti-aliasing < {}x internal resolution ({}x{}) >",
                    settings.internal_res, 640 * settings.internal_res, 528 * settings.internal_res
                ),
                format!("FPS counter   < {} >", if settings.show_fps { "On" } else { "Off" }),
                if gc_runtime::cheats::unlock_active() {
                    "Unlock all  (everything unlocked this session)".to_string()
                } else {
                    "Unlock all  [Enter]  (this session only, never saved)".to_string()
                },
                "Reset to defaults  [Enter]".to_string(),
            ];
            let mut s = String::from("SETTINGS\nUp/Down select | Left/Right change | P / Esc close\n\n");
            for (i, r) in rows.iter().enumerate() {
                s += if i == state.selected { "> " } else { "  " };
                s += r;
                s += "\n";
            }
            t.0 = s;
        }
    }
    // Keeps "Unlock all" applied for this session if the game rewrites an unlock byte.
    gc_runtime::cheats::tick();
    if changed {
        settings.save();
        VOLUME.store(settings.volume.to_bits(), Relaxed);
        if let Ok(mut w) = windows.single_mut() {
            let mode = settings.window_mode();
            if w.mode != mode {
                w.mode = mode;
            }
            let pm = settings.present_mode();
            if w.present_mode != pm {
                w.present_mode = pm;
            }
        }
    }
}

#[derive(Component)]
pub struct FpsText;

pub fn setup_fps(mut commands: Commands) {
    commands.spawn((
        Text::new(""),
        TextFont { font_size: FontSize::Px(16.0), ..default() },
        TextColor(Color::srgb(1.0, 1.0, 0.4)),
        BackgroundColor(Color::srgba(0.0, 0.0, 0.0, 0.55)),
        Node {
            position_type: PositionType::Absolute,
            top: Val::Px(8.0),
            right: Val::Px(8.0),
            padding: UiRect::axes(Val::Px(8.0), Val::Px(4.0)),
            ..default()
        },
        Visibility::Hidden,
        FpsText,
    ));
}

/// Window counter for the FPS display: game frames (the console's frames actually produced,
/// i.e. the real game speed; 60 = full speed), averaged over ~0.5 s.
#[derive(Default)]
pub struct FpsWindow {
    start: Option<std::time::Instant>,
    game0: u64,
    text: String,
}

pub fn update_fps(game_frames: u64, settings: &Settings, w: &mut FpsWindow, q: &mut Query<(&mut Text, &mut Visibility), With<FpsText>>) {
    let Ok((mut t, mut vis)) = q.single_mut() else { return };
    *vis = if settings.show_fps { Visibility::Visible } else { Visibility::Hidden };
    let now = std::time::Instant::now();
    let start = *w.start.get_or_insert_with(|| {
        w.game0 = game_frames;
        now
    });
    let dt = now.duration_since(start).as_secs_f64();
    if dt >= 0.5 {
        let game = (game_frames - w.game0) as f64 / dt;
        w.text = format!("{game:.0} FPS");
        w.start = Some(now);
        w.game0 = game_frames;
    }
    if settings.show_fps && t.0 != w.text {
        t.0 = w.text.clone();
    }
}

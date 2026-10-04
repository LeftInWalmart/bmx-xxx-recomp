//! Controls: keyboard, mouse and gamepads mapped onto the GameCube controller in port 1.
//!
//! Bindings live in a per-user JSON file (`controls.json` in the platform config folder,
//! e.g. `%APPDATA%\BMX-XXX-Recomp\controls.json`), so everyone who plays a shared copy keeps
//! their own. On first run the file is created from the defaults; if a Dolphin GCPad profile
//! (`BMX.ini`) is found, its gamepad mapping is imported. F2 opens the in-game rebinding screen.
//!
//! A binding is a list of sources per GameCube control. Sources are strings:
//! `Key:KeyX`, `Mouse:Left`, `MouseMove:X+`, `Pad:South`, `Axis:LeftStickX+`.
//! Gamepads are read through Bevy (gilrs: XInput / HID on Windows), so any common controller
//! works without extra drivers.
//!
//! Origin calibration: like the GameCube's PAD library (which reports sticks and triggers
//! relative to the position captured when a controller connects), every gamepad axis and analog
//! trigger that rests far from zero right after connecting is re-centred on that rest value.
//! Many DirectInput pads expose a trigger as an axis resting at -1, which would otherwise read as
//! a fully tilted C-stick / pressed trigger (e.g. the pause menu scrolling by itself).

use bevy::input::gamepad::{GamepadAxis, GamepadButton};
use bevy::input::mouse::AccumulatedMouseMotion;
use bevy::prelude::*;
use gc_runtime::pad::{self, PadShared, PadStatus};
use serde::{Deserialize, Serialize};
use std::collections::{BTreeMap, HashMap};
use std::path::PathBuf;
use std::sync::Arc;

/// GameCube controls in display order: (name, kind).
pub const CONTROLS: &[&str] = &[
    "A", "B", "X", "Y", "Z", "Start", "L", "R", "L-Analog", "R-Analog",
    "Stick Up", "Stick Down", "Stick Left", "Stick Right", "Stick Modifier",
    "C-Stick Up", "C-Stick Down", "C-Stick Left", "C-Stick Right",
    "D-Pad Up", "D-Pad Down", "D-Pad Left", "D-Pad Right",
];

#[derive(Serialize, Deserialize, Clone, Debug)]
#[serde(default)]
pub struct Controls {
    pub version: u32,
    /// Stick inputs below this are ignored (0..1).
    pub deadzone: f32,
    /// Main-stick tilt while "Stick Modifier" is held (Dolphin-style walk modifier).
    pub modifier_scale: f32,
    /// Mouse movement -> virtual stick: tilt per pixel, and how fast it recentres (0..1 per frame).
    pub mouse_sensitivity: f32,
    pub mouse_return: f32,
    pub bindings: BTreeMap<String, Vec<String>>,
}

impl Default for Controls {
    fn default() -> Self {
        let b = |v: &[&str]| v.iter().map(|s| s.to_string()).collect::<Vec<_>>();
        let mut m = BTreeMap::new();
        for (k, v) in [
            ("A", b(&["Key:KeyX", "Mouse:Left", "Pad:South"])),
            ("B", b(&["Key:KeyZ", "Mouse:Right", "Pad:East"])),
            ("X", b(&["Key:KeyC", "Pad:West"])),
            ("Y", b(&["Key:KeyV", "Pad:North"])),
            ("Z", b(&["Key:KeyD", "Pad:Select"])),
            ("Start", b(&["Key:Enter", "Pad:Start"])),
            ("L", b(&["Key:KeyA", "Pad:LeftTrigger"])),
            ("R", b(&["Key:KeyS", "Mouse:Middle", "Pad:RightTrigger"])),
            ("L-Analog", b(&["Key:KeyA", "Pad:LeftTrigger2"])),
            ("R-Analog", b(&["Key:KeyS", "Pad:RightTrigger2"])),
            ("Stick Up", b(&["Key:ArrowUp", "Axis:LeftStickY+"])),
            ("Stick Down", b(&["Key:ArrowDown", "Axis:LeftStickY-"])),
            ("Stick Left", b(&["Key:ArrowLeft", "Axis:LeftStickX-"])),
            ("Stick Right", b(&["Key:ArrowRight", "Axis:LeftStickX+"])),
            ("Stick Modifier", b(&["Key:ShiftLeft"])),
            ("C-Stick Up", b(&["Key:KeyI", "Axis:RightStickY+"])),
            ("C-Stick Down", b(&["Key:KeyK", "Axis:RightStickY-"])),
            ("C-Stick Left", b(&["Key:KeyJ", "Axis:RightStickX-"])),
            ("C-Stick Right", b(&["Key:KeyL", "Axis:RightStickX+"])),
            ("D-Pad Up", b(&["Key:KeyT", "Pad:DPadUp"])),
            ("D-Pad Down", b(&["Key:KeyG", "Pad:DPadDown"])),
            ("D-Pad Left", b(&["Key:KeyF", "Pad:DPadLeft"])),
            ("D-Pad Right", b(&["Key:KeyH", "Pad:DPadRight"])),
        ] {
            m.insert(k.to_string(), v);
        }
        Controls { version: 1, deadzone: 0.15, modifier_scale: 0.5, mouse_sensitivity: 0.02, mouse_return: 0.15, bindings: m }
    }
}

/// `controls.json` in the per-user config folder (overridable with `BMX_CONTROLS`).
pub fn config_path() -> PathBuf {
    if let Some(p) = std::env::var_os("BMX_CONTROLS") {
        return PathBuf::from(p);
    }
    let base = std::env::var_os("APPDATA")
        .map(PathBuf::from)
        .or_else(|| std::env::var_os("XDG_CONFIG_HOME").map(PathBuf::from))
        .or_else(|| std::env::var_os("HOME").map(|h| PathBuf::from(h).join(".config")))
        .unwrap_or_else(|| PathBuf::from("."));
    base.join("BMX-XXX-Recomp").join("controls.json")
}

/// Dolphin's GCPad profile for this game, if the user has one.
fn dolphin_profile() -> Option<PathBuf> {
    let p = PathBuf::from(std::env::var_os("APPDATA")?)
        .join("Dolphin Emulator").join("Config").join("Profiles").join("GCPad").join("BMX.ini");
    p.exists().then_some(p)
}

impl Controls {
    /// Load the user's controls, creating the file (from Dolphin's profile if present) on first run.
    pub fn load_or_create() -> Controls {
        let path = config_path();
        if let Ok(text) = std::fs::read_to_string(&path) {
            match serde_json::from_str::<Controls>(&text) {
                Ok(mut c) => {
                    // Controls added in later versions get their defaults.
                    for (k, v) in Controls::default().bindings {
                        c.bindings.entry(k).or_insert(v);
                    }
                    return c;
                }
                Err(e) => gc_runtime::diag::log(format!("controls: {} is invalid ({e}); using defaults", path.display())),
            }
        }
        let mut c = Controls::default();
        if let Some(ini) = dolphin_profile() {
            if let Ok(text) = std::fs::read_to_string(&ini) {
                c.import_dolphin(&text);
                gc_runtime::diag::log(format!("controls: imported gamepad mapping from {}", ini.display()));
            }
        }
        c.save();
        c
    }

    pub fn save(&self) {
        let path = config_path();
        if let Some(dir) = path.parent() {
            let _ = std::fs::create_dir_all(dir);
        }
        match serde_json::to_string_pretty(self) {
            Ok(text) => {
                if let Err(e) = std::fs::write(&path, text) {
                    gc_runtime::diag::log(format!("controls: cannot save {}: {e}", path.display()));
                }
            }
            Err(e) => gc_runtime::diag::log(format!("controls: cannot serialize: {e}")),
        }
    }

    /// Replace gamepad sources with those from a Dolphin GCPad profile (keyboard/mouse stay).
    pub fn import_dolphin(&mut self, ini: &str) {
        let map: &[(&str, &str)] = &[
            ("Buttons/A", "A"), ("Buttons/B", "B"), ("Buttons/X", "X"), ("Buttons/Y", "Y"),
            ("Buttons/Z", "Z"), ("Buttons/Start", "Start"), ("Triggers/L", "L"), ("Triggers/R", "R"),
            ("Triggers/L-Analog", "L-Analog"), ("Triggers/R-Analog", "R-Analog"),
            ("Main Stick/Up", "Stick Up"), ("Main Stick/Down", "Stick Down"),
            ("Main Stick/Left", "Stick Left"), ("Main Stick/Right", "Stick Right"),
            ("Main Stick/Modifier", "Stick Modifier"),
            ("C-Stick/Up", "C-Stick Up"), ("C-Stick/Down", "C-Stick Down"),
            ("C-Stick/Left", "C-Stick Left"), ("C-Stick/Right", "C-Stick Right"),
            ("D-Pad/Up", "D-Pad Up"), ("D-Pad/Down", "D-Pad Down"),
            ("D-Pad/Left", "D-Pad Left"), ("D-Pad/Right", "D-Pad Right"),
        ];
        for line in ini.lines() {
            let Some((k, v)) = line.split_once('=') else { continue };
            let Some(&(_, ctl)) = map.iter().find(|(d, _)| *d == k.trim()) else { continue };
            let sources: Vec<String> = v.split('|').filter_map(|e| dolphin_source(e.trim().trim_matches('`'))).collect();
            if sources.is_empty() {
                continue;
            }
            let list = self.bindings.entry(ctl.to_string()).or_default();
            list.retain(|s| !s.starts_with("Pad:") && !s.starts_with("Axis:"));
            for s in sources {
                if !list.contains(&s) {
                    list.push(s);
                }
            }
        }
    }
}

/// One Dolphin input expression (SDL/XInput naming) -> our source string.
fn dolphin_source(e: &str) -> Option<String> {
    let pad = |s: &str| Some(format!("Pad:{s}"));
    let axis = |s: &str| Some(format!("Axis:{s}"));
    match e {
        "Button S" | "Button A" => pad("South"),
        "Button E" | "Button B" => pad("East"),
        "Button W" | "Button X" => pad("West"),
        "Button N" | "Button Y" => pad("North"),
        "Back" => pad("Select"),
        "Start" => pad("Start"),
        "Guide" => pad("Mode"),
        "Shoulder L" => pad("LeftTrigger"),
        "Shoulder R" => pad("RightTrigger"),
        "Trigger L" => pad("LeftTrigger2"),
        "Trigger R" => pad("RightTrigger2"),
        "Thumb L" => pad("LeftThumb"),
        "Thumb R" => pad("RightThumb"),
        "Pad N" => pad("DPadUp"),
        "Pad S" => pad("DPadDown"),
        "Pad W" => pad("DPadLeft"),
        "Pad E" => pad("DPadRight"),
        // SDL axes: 0/1 left stick (Y grows downwards), 2/3 right stick.
        "Axis 0-" | "Left X-" => axis("LeftStickX-"),
        "Axis 0+" | "Left X+" => axis("LeftStickX+"),
        "Axis 1-" | "Left Y+" => axis("LeftStickY+"),
        "Axis 1+" | "Left Y-" => axis("LeftStickY-"),
        "Axis 2-" | "Right X-" => axis("RightStickX-"),
        "Axis 2+" | "Right X+" => axis("RightStickX+"),
        "Axis 3-" | "Right Y+" => axis("RightStickY+"),
        "Axis 3+" | "Right Y-" => axis("RightStickY-"),
        "Shift" | "LSHIFT" => Some("Key:ShiftLeft".into()),
        "Ctrl" | "LCONTROL" => Some("Key:ControlLeft".into()),
        "Alt" | "LMENU" => Some("Key:AltLeft".into()),
        k if k.len() == 1 && k.chars().all(|c| c.is_ascii_alphabetic()) => Some(format!("Key:Key{}", k.to_ascii_uppercase())),
        _ => None,
    }
}

// ---- source names <-> Bevy inputs ----

const KEYS: &[KeyCode] = &[
    KeyCode::KeyA, KeyCode::KeyB, KeyCode::KeyC, KeyCode::KeyD, KeyCode::KeyE, KeyCode::KeyF, KeyCode::KeyG,
    KeyCode::KeyH, KeyCode::KeyI, KeyCode::KeyJ, KeyCode::KeyK, KeyCode::KeyL, KeyCode::KeyM, KeyCode::KeyN,
    KeyCode::KeyO, KeyCode::KeyP, KeyCode::KeyQ, KeyCode::KeyR, KeyCode::KeyS, KeyCode::KeyT, KeyCode::KeyU,
    KeyCode::KeyV, KeyCode::KeyW, KeyCode::KeyX, KeyCode::KeyY, KeyCode::KeyZ, KeyCode::Digit0, KeyCode::Digit1,
    KeyCode::Digit2, KeyCode::Digit3, KeyCode::Digit4, KeyCode::Digit5, KeyCode::Digit6, KeyCode::Digit7,
    KeyCode::Digit8, KeyCode::Digit9, KeyCode::ArrowUp, KeyCode::ArrowDown, KeyCode::ArrowLeft,
    KeyCode::ArrowRight, KeyCode::Space, KeyCode::Enter, KeyCode::Tab, KeyCode::Backquote, KeyCode::ShiftLeft,
    KeyCode::ShiftRight, KeyCode::ControlLeft, KeyCode::ControlRight, KeyCode::AltLeft, KeyCode::AltRight,
    KeyCode::Comma, KeyCode::Period, KeyCode::Slash, KeyCode::Semicolon, KeyCode::Quote, KeyCode::BracketLeft,
    KeyCode::BracketRight, KeyCode::Minus, KeyCode::Equal, KeyCode::Numpad0, KeyCode::Numpad1, KeyCode::Numpad2,
    KeyCode::Numpad3, KeyCode::Numpad4, KeyCode::Numpad5, KeyCode::Numpad6, KeyCode::Numpad7, KeyCode::Numpad8,
    KeyCode::Numpad9, KeyCode::Home, KeyCode::End, KeyCode::PageUp, KeyCode::PageDown, KeyCode::Insert,
    KeyCode::Delete,
];
const PAD_BUTTONS: &[GamepadButton] = &[
    GamepadButton::South, GamepadButton::East, GamepadButton::North, GamepadButton::West, GamepadButton::C,
    GamepadButton::Z, GamepadButton::LeftTrigger, GamepadButton::LeftTrigger2, GamepadButton::RightTrigger,
    GamepadButton::RightTrigger2, GamepadButton::Select, GamepadButton::Start, GamepadButton::Mode,
    GamepadButton::LeftThumb, GamepadButton::RightThumb, GamepadButton::DPadUp, GamepadButton::DPadDown,
    GamepadButton::DPadLeft, GamepadButton::DPadRight,
];
const PAD_AXES: &[GamepadAxis] = &[
    GamepadAxis::LeftStickX, GamepadAxis::LeftStickY, GamepadAxis::RightStickX, GamepadAxis::RightStickY,
    GamepadAxis::LeftZ, GamepadAxis::RightZ,
];
const MOUSE: &[MouseButton] = &[MouseButton::Left, MouseButton::Right, MouseButton::Middle, MouseButton::Back, MouseButton::Forward];

fn name<T: std::fmt::Debug>(v: &T) -> String {
    format!("{v:?}")
}

#[derive(Clone, Copy)]
enum Source {
    Key(KeyCode),
    Mouse(MouseButton),
    MouseMove(bool, f32), // (x axis?, sign)
    Pad(GamepadButton),
    Axis(GamepadAxis, f32),
}

fn parse(s: &str) -> Option<Source> {
    let (kind, rest) = s.split_once(':')?;
    let signed = |r: &str| -> Option<(String, f32)> {
        let sign = if r.ends_with('+') { 1.0 } else if r.ends_with('-') { -1.0 } else { return None };
        Some((r[..r.len() - 1].to_string(), sign))
    };
    match kind {
        "Key" => KEYS.iter().find(|k| name(*k) == rest).map(|k| Source::Key(*k)),
        "Mouse" => MOUSE.iter().find(|b| name(*b) == rest).map(|b| Source::Mouse(*b)),
        "MouseMove" => signed(rest).map(|(a, s)| Source::MouseMove(a == "X", s)),
        "Pad" => PAD_BUTTONS.iter().find(|b| name(*b) == rest).map(|b| Source::Pad(*b)),
        "Axis" => {
            let (a, s) = signed(rest)?;
            PAD_AXES.iter().find(|x| name(*x) == a).map(|x| Source::Axis(*x, s))
        }
        _ => None,
    }
}

/// Rest-position detection per gamepad input (see the module doc).
#[derive(Default)]
struct Origins {
    /// When each gamepad was first seen.
    connected: HashMap<Entity, std::time::Instant>,
    /// (gamepad, input) -> (candidate value, since when it has been steady)
    candidate: HashMap<(Entity, String), (f32, std::time::Instant)>,
    /// (gamepad, input) -> rest value
    origin: HashMap<(Entity, String), f32>,
}

/// Inputs only count as resting off-centre if they hold still this long right after connecting.
const ORIGIN_WINDOW: std::time::Duration = std::time::Duration::from_millis(2000);
const ORIGIN_STEADY: std::time::Duration = std::time::Duration::from_millis(400);

impl Origins {
    fn observe(&mut self, now: std::time::Instant, pads: &[(Entity, &Gamepad)]) {
        for &(e, g) in pads {
            let since = *self.connected.entry(e).or_insert(now);
            if now.duration_since(since) > ORIGIN_WINDOW {
                continue;
            }
            let inputs = PAD_AXES.iter().map(|a| (name(a), g.get(*a)))
                .chain([GamepadButton::LeftTrigger2, GamepadButton::RightTrigger2].iter().map(|b| (name(b), g.get(*b))));
            for (n, v) in inputs {
                let Some(v) = v else { continue };
                let key = (e, n);
                if self.origin.contains_key(&key) {
                    continue;
                }
                let c = self.candidate.entry(key.clone()).or_insert((v, now));
                if (c.0 - v).abs() > 0.05 {
                    *c = (v, now);
                } else if v.abs() > 0.5 && now.duration_since(c.1) >= ORIGIN_STEADY {
                    gc_runtime::diag::log(format!(
                        "gamepad {} ({}): {} rests at {v:.2}; using it as the origin",
                        e, g.vendor_id().unwrap_or(0), key.1
                    ));
                    self.origin.insert(key, v);
                }
            }
        }
        self.connected.retain(|e, _| pads.iter().any(|(p, _)| p == e));
    }

    /// Raw value re-centred on the input's origin (unchanged when it rests at zero).
    fn adjust(&self, e: Entity, input: &str, v: f32) -> f32 {
        match self.origin.get(&(e, input.to_string())) {
            Some(&o) => ((v - o) / (1.0 + o.abs())).clamp(-1.0, 1.0),
            None => v,
        }
    }
}

#[derive(Resource)]
pub struct Input {
    pub controls: Controls,
    parsed: Vec<(String, Vec<Source>)>,
    mouse_stick: Vec2,
    origins: Origins,
    pad: Arc<PadShared>,
    // Rebinding screen.
    pub menu_open: bool,
    selected: usize,
    capturing: bool,
}

impl Input {
    pub fn new(pad: Arc<PadShared>) -> Input {
        let controls = Controls::load_or_create();
        let mut i = Input { controls, parsed: Vec::new(), mouse_stick: Vec2::ZERO, origins: Origins::default(), pad, menu_open: false, selected: 0, capturing: false };
        i.reparse();
        i
    }
    fn reparse(&mut self) {
        self.parsed = CONTROLS
            .iter()
            .map(|c| {
                let v = self.controls.bindings.get(*c).map(|l| l.iter().filter_map(|s| parse(s)).collect()).unwrap_or_default();
                (c.to_string(), v)
            })
            .collect();
    }
}

struct Devices<'a> {
    keys: &'a ButtonInput<KeyCode>,
    mouse: &'a ButtonInput<MouseButton>,
    pads: Vec<(Entity, &'a Gamepad)>,
    origins: &'a Origins,
    mouse_stick: Vec2,
    deadzone: f32,
}

impl Devices<'_> {
    /// 0..1 strength of one source.
    fn value(&self, s: &Source) -> f32 {
        match *s {
            Source::Key(k) => self.keys.pressed(k) as u8 as f32,
            Source::Mouse(b) => self.mouse.pressed(b) as u8 as f32,
            Source::MouseMove(x, sign) => {
                let v = if x { self.mouse_stick.x } else { -self.mouse_stick.y };
                (v * sign).clamp(0.0, 1.0)
            }
            Source::Pad(b) => self
                .pads
                .iter()
                .map(|&(e, g)| self.origins.adjust(e, &name(&b), g.get(b).unwrap_or(g.pressed(b) as u8 as f32)))
                .fold(0.0, f32::max),
            Source::Axis(a, sign) => self
                .pads
                .iter()
                .map(|&(e, g)| (self.origins.adjust(e, &name(&a), g.get(a).unwrap_or(0.0)) * sign).clamp(0.0, 1.0))
                .map(|v| if v < self.deadzone { 0.0 } else { v })
                .fold(0.0, f32::max),
        }
    }
}

fn axis(v: f32) -> i8 {
    (v.clamp(-1.0, 1.0) * 100.0) as i8
}

/// Read every bound source and write the GameCube pad state for port 1.
pub fn read_input(
    keys: Res<ButtonInput<KeyCode>>,
    mouse: Res<ButtonInput<MouseButton>>,
    motion: Res<AccumulatedMouseMotion>,
    pads: Query<(Entity, &Gamepad)>,
    mut input: ResMut<Input>,
) {
    let pads: Vec<(Entity, &Gamepad)> = pads.iter().collect();
    input.origins.observe(std::time::Instant::now(), &pads);
    let c = input.controls.clone();
    // Mouse movement drives a virtual stick that recentres when the mouse stops.
    let ms = (input.mouse_stick + motion.delta * c.mouse_sensitivity).clamp(Vec2::splat(-1.0), Vec2::splat(1.0));
    input.mouse_stick = ms * (1.0 - c.mouse_return.clamp(0.0, 1.0));
    let dev = Devices { keys: &keys, mouse: &mouse, pads, origins: &input.origins, mouse_stick: ms, deadzone: c.deadzone };
    if input.menu_open || crate::settings::MENU_OPEN.load(std::sync::atomic::Ordering::Relaxed) {
        input.pad.ports.lock().unwrap()[0] = PadStatus { connected: true, ..default() };
        return;
    }
    let v = |name: &str| -> f32 {
        input.parsed.iter().find(|(n, _)| n == name).map_or(0.0, |(_, l)| l.iter().map(|s| dev.value(s)).fold(0.0, f32::max))
    };
    let mut s = PadStatus { connected: true, ..default() };
    for (n, bit) in [
        ("A", pad::BUTTON_A), ("B", pad::BUTTON_B), ("X", pad::BUTTON_X), ("Y", pad::BUTTON_Y),
        ("Z", pad::TRIGGER_Z), ("Start", pad::BUTTON_START), ("L", pad::TRIGGER_L), ("R", pad::TRIGGER_R),
        ("D-Pad Up", pad::BUTTON_UP), ("D-Pad Down", pad::BUTTON_DOWN), ("D-Pad Left", pad::BUTTON_LEFT),
        ("D-Pad Right", pad::BUTTON_RIGHT),
    ] {
        if v(n) > 0.5 {
            s.button |= bit;
        }
    }
    let scale = if v("Stick Modifier") > 0.5 { c.modifier_scale } else { 1.0 };
    s.stick_x = axis((v("Stick Right") - v("Stick Left")) * scale);
    s.stick_y = axis((v("Stick Up") - v("Stick Down")) * scale);
    s.substick_x = axis(v("C-Stick Right") - v("C-Stick Left"));
    s.substick_y = axis(v("C-Stick Up") - v("C-Stick Down"));
    // Analog triggers also follow the digital L/R click.
    let tl = v("L-Analog").max(if s.button & pad::TRIGGER_L != 0 { 1.0 } else { 0.0 });
    let tr = v("R-Analog").max(if s.button & pad::TRIGGER_R != 0 { 1.0 } else { 0.0 });
    s.trigger_l = (tl.clamp(0.0, 1.0) * 255.0) as u8;
    s.trigger_r = (tr.clamp(0.0, 1.0) * 255.0) as u8;
    s.analog_a = if s.button & pad::BUTTON_A != 0 { 255 } else { 0 };
    s.analog_b = if s.button & pad::BUTTON_B != 0 { 255 } else { 0 };
    input.pad.ports.lock().unwrap()[0] = s;
}

// ---- rebinding screen (F2) ----

#[derive(Component)]
pub struct ControlsText;

pub fn setup_menu(mut commands: Commands) {
    commands.spawn((
        Text::new(""),
        TextFont { font_size: FontSize::Px(15.0), ..default() },
        TextColor(Color::WHITE),
        BackgroundColor(Color::srgba(0.0, 0.0, 0.0, 0.85)),
        Node { position_type: PositionType::Absolute, top: Val::Px(10.0), right: Val::Px(10.0), padding: UiRect::all(Val::Px(10.0)), ..default() },
        Visibility::Hidden,
        ControlsText,
    ));
}

/// F2: open/close. Up/Down select, Enter: add a binding (press the input), Backspace: clear,
/// R: reset all to defaults. Every change is saved to controls.json immediately.
pub fn menu(
    keys: Res<ButtonInput<KeyCode>>,
    mouse: Res<ButtonInput<MouseButton>>,
    pads: Query<&Gamepad>,
    mut input: ResMut<Input>,
    mut text: Query<(&mut Text, &mut Visibility), With<ControlsText>>,
) {
    if keys.just_pressed(KeyCode::F2) {
        input.menu_open = !input.menu_open;
        input.capturing = false;
    }
    let Ok((mut t, mut vis)) = text.single_mut() else { return };
    *vis = if input.menu_open { Visibility::Visible } else { Visibility::Hidden };
    if !input.menu_open {
        return;
    }
    let sel = input.selected;
    let control = CONTROLS[sel].to_string();
    let mut changed = false;
    if input.capturing {
        // First newly pressed key / mouse button / pad button / pushed axis becomes a binding.
        let mut got: Option<String> = None;
        if let Some(k) = keys.get_just_pressed().find(|k| **k != KeyCode::Enter && **k != KeyCode::F2) {
            if KEYS.contains(k) {
                got = Some(format!("Key:{}", name(k)));
            }
        }
        if got.is_none() {
            got = mouse.get_just_pressed().find(|b| MOUSE.contains(b)).map(|b| format!("Mouse:{}", name(b)));
        }
        for g in &pads {
            if got.is_some() {
                break;
            }
            got = g.get_just_pressed().find(|b| PAD_BUTTONS.contains(b)).map(|b| format!("Pad:{}", name(b)));
            if got.is_none() {
                for a in PAD_AXES {
                    let x = g.get(*a).unwrap_or(0.0);
                    if x.abs() > 0.7 {
                        got = Some(format!("Axis:{}{}", name(a), if x > 0.0 { "+" } else { "-" }));
                        break;
                    }
                }
            }
        }
        if keys.just_pressed(KeyCode::Escape) {
            input.capturing = false;
        } else if let Some(g) = got {
            let list = input.controls.bindings.entry(control.clone()).or_default();
            if !list.contains(&g) {
                list.push(g);
            }
            input.capturing = false;
            changed = true;
        }
    } else {
        if keys.just_pressed(KeyCode::ArrowDown) {
            input.selected = (sel + 1) % CONTROLS.len();
        }
        if keys.just_pressed(KeyCode::ArrowUp) {
            input.selected = (sel + CONTROLS.len() - 1) % CONTROLS.len();
        }
        if keys.just_pressed(KeyCode::Enter) {
            input.capturing = true;
        }
        if keys.just_pressed(KeyCode::Backspace) {
            input.controls.bindings.insert(control.clone(), Vec::new());
            changed = true;
        }
        if keys.just_pressed(KeyCode::KeyR) {
            input.controls.bindings = Controls::default().bindings;
            changed = true;
        }
    }
    if changed {
        input.controls.save();
        input.reparse();
    }
    let mut s = format!("CONTROLS  (saved to {})\nUp/Down select | Enter add input | Backspace clear | R reset all | F2 close\n\n", config_path().display());
    for (i, c) in CONTROLS.iter().enumerate() {
        let b = input.controls.bindings.get(*c).map(|l| l.join(", ")).unwrap_or_default();
        let mark = if i == input.selected { if input.capturing { ">> press an input <<  " } else { "> " } } else { "  " };
        s += &format!("{mark}{c:<15} {b}\n");
    }
    t.0 = s;
}

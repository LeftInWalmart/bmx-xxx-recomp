//! Controller state shared between the Bevy input system and the guest `PADRead` HLE.

use std::sync::Mutex;

pub const BUTTON_LEFT: u16 = 0x0001;
pub const BUTTON_RIGHT: u16 = 0x0002;
pub const BUTTON_DOWN: u16 = 0x0004;
pub const BUTTON_UP: u16 = 0x0008;
pub const TRIGGER_Z: u16 = 0x0010;
pub const TRIGGER_R: u16 = 0x0020;
pub const TRIGGER_L: u16 = 0x0040;
pub const BUTTON_A: u16 = 0x0100;
pub const BUTTON_B: u16 = 0x0200;
pub const BUTTON_X: u16 = 0x0400;
pub const BUTTON_Y: u16 = 0x0800;
pub const BUTTON_START: u16 = 0x1000;

/// Mirrors the guest `PADStatus` struct (12 bytes).
#[derive(Clone, Copy, Default, Debug)]
pub struct PadStatus {
    pub button: u16,
    pub stick_x: i8,
    pub stick_y: i8,
    pub substick_x: i8,
    pub substick_y: i8,
    pub trigger_l: u8,
    pub trigger_r: u8,
    pub analog_a: u8,
    pub analog_b: u8,
    pub connected: bool,
}

impl PadStatus {
    pub fn to_bytes(self) -> [u8; 12] {
        let b = self.button.to_be_bytes();
        [
            b[0], b[1],
            self.stick_x as u8, self.stick_y as u8,
            self.substick_x as u8, self.substick_y as u8,
            self.trigger_l, self.trigger_r,
            self.analog_a, self.analog_b,
            if self.connected { 0 } else { 0xFF }, // err: PAD_ERR_NONE / PAD_ERR_NO_CONTROLLER
            0,
        ]
    }
}

#[derive(Default)]
pub struct PadShared {
    pub ports: Mutex<[PadStatus; 4]>,
}

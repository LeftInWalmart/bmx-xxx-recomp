//! "Unlock all" (P menu): a session-only cheat that never reaches the memory card.
//!
//! It applies exactly what the game's own `XXX RATED CHEAT` code does (cheat table at
//! 0x80193270, applied by `bmxmenuscriptCheat` 0x80127A24):
//! * every rider's 12 items: bit 7 of `0x8019D960 + rider*0x1FF4 + item*0x206 + 0x7B7`;
//! * the 20 entries of the table at 0x801BF8C0 (stride 4): bit 7;
//! * every rider: bit 7 of `0x8019D984 + rider*0x1FF4` cleared;
//! * every rider: `0x8019D9C7 + rider*0x1FF4` = 5.
//!
//! The player's own value of each of those bytes is kept aside. The two places the record
//! crosses the memory card are hooked (hle.csv):
//! * `bmxSaveSerialize` (record -> card buffer): the player's own values are put back for the copy
//!   and the unlocks re-applied afterwards, so the card only ever holds real progress;
//! * `bmxSaveDeserialize` (card buffer -> record): the loaded values become the player's own,
//!   then the unlocks are re-applied.
//!
//! Nothing is remembered across launches: the option has to be chosen again each session.

use std::sync::Mutex;

const RIDERS: u32 = 17;
const RIDER_STRIDE: u32 = 0x1FF4;

#[derive(Clone, Copy)]
enum Want {
    Set(u8),
    Clear(u8),
    Value(u8),
}

/// Every byte the unlock touches, with what it wants there.
fn targets() -> Vec<(u32, Want)> {
    let mut v = Vec::new();
    for r in 0..RIDERS {
        for i in 0..12 {
            v.push((0x8019_D960 + r * RIDER_STRIDE + i * 0x206 + 0x7B7, Want::Set(0x80)));
        }
        v.push((0x8019_D984 + r * RIDER_STRIDE, Want::Clear(0x80)));
        v.push((0x8019_D9C7 + r * RIDER_STRIDE, Want::Value(5)));
    }
    for i in 0..20 {
        v.push((0x801B_F8C0 + i * 4, Want::Set(0x80)));
    }
    v
}

fn unlocked(cur: u8, w: Want) -> u8 {
    match w {
        Want::Set(b) => cur | b,
        Want::Clear(b) => cur & !b,
        Want::Value(x) => x,
    }
}

struct Unlock {
    active: bool,
    /// Save/load hook running: the record is temporarily in the player's own state.
    busy: bool,
    /// The player's own value per target byte.
    earned: Vec<u8>,
}

static UNLOCK: Mutex<Unlock> = Mutex::new(Unlock { active: false, busy: false, earned: Vec::new() });

fn read(addr: u32) -> u8 {
    crate::machine().read_be(addr, 1) as u8
}

fn write(addr: u32, v: u8) {
    crate::machine().write_be(addr, 1, v as u64);
}

/// Force every target to its unlocked value. A byte that doesn't hold its unlocked value was
/// last written by the game itself (load, reset, progress), so that value is the player's own.
fn apply(u: &mut Unlock) {
    for (k, (addr, w)) in targets().into_iter().enumerate() {
        let cur = read(addr);
        if cur != unlocked(cur, w) || u.earned.len() <= k {
            if u.earned.len() <= k {
                u.earned.push(cur);
            } else {
                u.earned[k] = cur;
            }
            write(addr, unlocked(cur, w));
        }
    }
}

/// P menu: turn the unlock on for this session (no-op if it already is).
pub fn unlock_all() {
    if crate::try_machine().is_none() {
        return;
    }
    let mut u = UNLOCK.lock().unwrap();
    if !u.active {
        u.active = true;
        apply(&mut u);
        crate::diag::log("cheat: unlock all applied for this session (never saved)".into());
    }
}

pub fn unlock_active() -> bool {
    UNLOCK.lock().unwrap().active
}

/// Once per host frame: keep the unlock applied if the game rewrote any of the bytes.
pub fn tick() {
    if crate::try_machine().is_none() {
        return;
    }
    let mut u = UNLOCK.lock().unwrap();
    if u.active && !u.busy {
        apply(&mut u);
    }
}

/// Run the save serializer with the record holding only the player's own progress.
pub fn around_save(run: impl FnOnce()) {
    let saved: Option<Vec<u8>> = {
        let mut u = UNLOCK.lock().unwrap();
        if u.active {
            u.busy = true;
            let t = targets();
            let cur: Vec<u8> = t.iter().map(|(a, _)| read(*a)).collect();
            for ((addr, _), e) in t.iter().zip(&u.earned) {
                write(*addr, *e);
            }
            Some(cur)
        } else {
            None
        }
    };
    run();
    if let Some(cur) = saved {
        let mut u = UNLOCK.lock().unwrap();
        for ((addr, _), v) in targets().iter().zip(cur) {
            write(*addr, v);
        }
        u.busy = false;
    }
}

/// After the save loader filled the record: what it loaded is the player's own progress.
pub fn around_load(run: impl FnOnce()) {
    {
        let mut u = UNLOCK.lock().unwrap();
        u.busy = u.active;
    }
    run();
    let mut u = UNLOCK.lock().unwrap();
    if u.active {
        u.earned = targets().iter().map(|(a, _)| read(*a)).collect();
        u.busy = false;
        apply(&mut u);
    }
}

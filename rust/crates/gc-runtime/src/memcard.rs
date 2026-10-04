//! Memory card emulation at the CARD API level (slot A present, slot B empty).
//!
//! The SDK's EXI-level driver can't be used as-is: unlocking a card runs a challenge on the
//! DSP. Instead `hle.rs` routes the public CARD* functions here. Files live in memory and are
//! persisted to `memcard/slotA.bin` (next to the working directory) after every change.
//!
//! Card: 4 Mbit (59 user blocks of 8 KiB), like the standard "Memory Card 59".

use crate::{Ctx, CtxExt, Machine};
use std::path::PathBuf;
use std::sync::Mutex;

pub const SECTOR_SIZE: u32 = 8192;
pub const USER_BLOCKS: u32 = 59;
pub const MAX_FILES: usize = 127;
const MEM_SIZE_MBIT: u32 = 4;

pub const READY: i32 = 0;
pub const WRONGDEVICE: i32 = -2;
pub const NOCARD: i32 = -3;
pub const FATAL: i32 = -128;
pub const NOFILE: i32 = -4;
pub const EXIST: i32 = -7;
pub const INSSPACE: i32 = -9;
pub const NOPERM: i32 = -10;
pub const LIMIT: i32 = -11;
pub const NAMETOOLONG: i32 = -12;

/// Mirrors the user-settable part of CARDStat.
#[derive(Clone, Default)]
pub struct Entry {
    pub name: [u8; 32],
    pub game: [u8; 4],
    pub company: [u8; 2],
    pub time: u32,
    pub banner_format: u8,
    pub icon_addr: u32,
    pub icon_format: u16,
    pub icon_speed: u16,
    pub comment_addr: u32,
    pub data: Vec<u8>,
}

impl Entry {
    fn blocks(&self) -> u32 {
        (self.data.len() as u32).div_ceil(SECTOR_SIZE)
    }
}

#[derive(Default)]
pub struct Card {
    pub files: Vec<Option<Entry>>,
    pub mounted: bool,
}

fn card() -> &'static Mutex<Card> {
    static C: std::sync::OnceLock<Mutex<Card>> = std::sync::OnceLock::new();
    C.get_or_init(|| Mutex::new(load().unwrap_or_else(|| Card { files: vec![None; MAX_FILES], mounted: false })))
}

fn path() -> PathBuf {
    std::env::var_os("BMX_MEMCARD").map(PathBuf::from).unwrap_or_else(|| PathBuf::from("memcard/slotA.bin"))
}

// ---- persistence: "BMXXCARD" v1, then per slot: present u8, header fields, data ----

fn load() -> Option<Card> {
    let b = std::fs::read(path()).ok()?;
    if b.len() < 12 || &b[..8] != b"BMXXCARD" {
        return None;
    }
    let mut p = 12;
    let mut take = |n: usize| -> Option<&[u8]> {
        let s = b.get(p..p + n)?;
        p += n;
        Some(s)
    };
    let u32_ = |s: &[u8]| u32::from_le_bytes(s.try_into().unwrap());
    let u16_ = |s: &[u8]| u16::from_le_bytes(s.try_into().unwrap());
    let mut files = vec![None; MAX_FILES];
    for slot in files.iter_mut() {
        if take(1)?[0] == 0 {
            continue;
        }
        let mut e = Entry::default();
        e.name.copy_from_slice(take(32)?);
        e.game.copy_from_slice(take(4)?);
        e.company.copy_from_slice(take(2)?);
        e.time = u32_(take(4)?);
        e.banner_format = take(1)?[0];
        e.icon_addr = u32_(take(4)?);
        e.icon_format = u16_(take(2)?);
        e.icon_speed = u16_(take(2)?);
        e.comment_addr = u32_(take(4)?);
        let len = u32_(take(4)?) as usize;
        e.data = take(len)?.to_vec();
        *slot = Some(e);
    }
    Some(Card { files, mounted: false })
}

fn save(c: &Card) {
    let mut b = Vec::new();
    b.extend_from_slice(b"BMXXCARD");
    b.extend_from_slice(&1u32.to_le_bytes());
    for f in &c.files {
        match f {
            None => b.push(0),
            Some(e) => {
                b.push(1);
                b.extend_from_slice(&e.name);
                b.extend_from_slice(&e.game);
                b.extend_from_slice(&e.company);
                b.extend_from_slice(&e.time.to_le_bytes());
                b.push(e.banner_format);
                b.extend_from_slice(&e.icon_addr.to_le_bytes());
                b.extend_from_slice(&e.icon_format.to_le_bytes());
                b.extend_from_slice(&e.icon_speed.to_le_bytes());
                b.extend_from_slice(&e.comment_addr.to_le_bytes());
                b.extend_from_slice(&(e.data.len() as u32).to_le_bytes());
                b.extend_from_slice(&e.data);
            }
        }
    }
    let p = path();
    if let Some(dir) = p.parent() {
        let _ = std::fs::create_dir_all(dir);
    }
    let tmp = p.with_extension("tmp");
    if std::fs::write(&tmp, &b).is_ok() {
        let _ = std::fs::rename(&tmp, &p);
    }
}

// ---- helpers over guest memory ----

fn read_name(m: &Machine, addr: u32) -> [u8; 32] {
    let mut n = [0u8; 32];
    for (i, b) in n.iter_mut().enumerate() {
        *b = m.read_be(addr + i as u32, 1) as u8;
        if *b == 0 {
            break;
        }
    }
    n
}

fn name_len(addr: u32, m: &Machine) -> usize {
    (0..64).find(|i| m.read_be(addr + i, 1) == 0).unwrap_or(64) as usize
}

fn this_game(m: &Machine) -> ([u8; 4], [u8; 2]) {
    let mut g = [0u8; 4];
    let mut c = [0u8; 2];
    for i in 0..4 {
        g[i] = m.read_be(0x8000_0000 + i as u32, 1) as u8;
    }
    for i in 0..2 {
        c[i] = m.read_be(0x8000_0004 + i as u32, 1) as u8;
    }
    (g, c)
}

fn now_seconds_since_2000() -> u32 {
    let secs = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_secs())
        .unwrap_or(0);
    secs.saturating_sub(946_684_800) as u32
}

/// CARDFileInfo: chan, fileNo, offset, length, iBlock.
fn write_fileinfo(m: &Machine, fi: u32, chan: u32, no: u32, len: u32) {
    m.write_be(fi, 4, chan as u64);
    m.write_be(fi + 4, 4, no as u64);
    m.write_be(fi + 8, 4, 0);
    m.write_be(fi + 12, 4, len as u64);
    m.write_be(fi + 16, 2, (5 + no) as u64);
}

fn find(c: &Card, name: &[u8; 32], game: [u8; 4], company: [u8; 2]) -> Option<usize> {
    c.files.iter().position(|f| {
        f.as_ref().is_some_and(|e| e.name == *name && e.game == game && e.company == company)
    })
}

fn free_blocks(c: &Card) -> u32 {
    USER_BLOCKS.saturating_sub(c.files.iter().flatten().map(Entry::blocks).sum())
}

/// Fill a CARDStat (0x6C bytes), including the derived icon/banner offsets.
fn write_stat(m: &Machine, st: u32, e: &Entry) {
    for (i, b) in e.name.iter().enumerate() {
        m.write_be(st + i as u32, 1, *b as u64);
    }
    m.write_be(st + 0x20, 4, e.data.len() as u64);
    m.write_be(st + 0x24, 4, e.time as u64);
    for i in 0..4 {
        m.write_be(st + 0x28 + i, 1, e.game[i as usize] as u64);
    }
    for i in 0..2 {
        m.write_be(st + 0x2C + i, 1, e.company[i as usize] as u64);
    }
    m.write_be(st + 0x2E, 1, e.banner_format as u64);
    m.write_be(st + 0x30, 4, e.icon_addr as u64);
    m.write_be(st + 0x34, 2, e.icon_format as u64);
    m.write_be(st + 0x36, 2, e.icon_speed as u64);
    m.write_be(st + 0x38, 4, e.comment_addr as u64);
    // Same layout rules as the SDK's __CARDUpdateIconOffsets.
    const NONE: u32 = 0xFFFF_FFFF;
    let (mut banner_format, mut icon_format) = (e.banner_format, e.icon_format);
    let mut off = if e.icon_addr == NONE {
        banner_format = 0;
        icon_format = 0;
        0
    } else {
        e.icon_addr
    };
    let (banner, banner_tlut) = match banner_format & 3 {
        1 => {
            let r = (off, off + 96 * 32);
            off += 96 * 32 + 512;
            r
        }
        2 => {
            let r = (off, NONE);
            off += 96 * 32 * 2;
            r
        }
        _ => (NONE, NONE),
    };
    m.write_be(st + 0x3C, 4, banner as u64);
    m.write_be(st + 0x40, 4, banner_tlut as u64);
    let mut tlut = false;
    for i in 0..8u32 {
        let icon = match (icon_format >> (2 * i)) & 3 {
            1 => {
                tlut = true;
                let r = off;
                off += 32 * 32;
                r
            }
            2 => {
                let r = off;
                off += 32 * 32 * 2;
                r
            }
            _ => NONE,
        };
        m.write_be(st + 0x44 + 4 * i, 4, icon as u64);
    }
    let icon_tlut = if tlut {
        let r = off;
        off += 512;
        r
    } else {
        NONE
    };
    m.write_be(st + 0x64, 4, icon_tlut as u64);
    m.write_be(st + 0x68, 4, off as u64);
}

// ---- API (called from hle.rs; arguments in r3.., result in r3) ----

/// Last result per channel, what `CARDGetResultCode` reports (the guest control blocks are never
/// touched, since the whole CARD API is emulated here).
static RESULT: std::sync::Mutex<[i32; 2]> = std::sync::Mutex::new([READY, NOCARD]);

/// Run an async callback `(chan, result)` if present, and return the result.
fn done(c: &mut Ctx, cb: u32, chan: u32, result: i32) -> i32 {
    if let Some(r) = RESULT.lock().unwrap().get_mut(chan as usize) {
        *r = result;
    }
    if cb != 0 {
        c.call_guest(cb, &[chan, result as u32]);
    }
    result
}

fn present(chan: u32) -> bool {
    chan == 0
}

pub fn call(c: &mut Ctx, name: &str) -> Option<i32> {
    let m = c.m();
    let args: [u32; 6] = std::array::from_fn(|i| c.gpr(3 + i));
    let a = |i: usize| args[i];
    let r = match name {
        "CARDInit" => READY,
        "CARDGetResultCode" => match RESULT.lock().unwrap().get(a(0) as usize) {
            Some(&r) => r,
            None => FATAL,
        },
        "CARDProbeEx" => {
            if present(a(0)) {
                if a(1) != 0 {
                    m.write_be(a(1), 4, MEM_SIZE_MBIT as u64);
                }
                if a(2) != 0 {
                    m.write_be(a(2), 4, SECTOR_SIZE as u64);
                }
                READY
            } else {
                NOCARD
            }
        }
        "CARDMount" | "CARDMountAsync" => {
            let chan = a(0);
            let cb = if name == "CARDMountAsync" { a(3) } else { 0 };
            let r = if present(chan) {
                card().lock().unwrap().mounted = true;
                READY
            } else {
                NOCARD
            };
            done(c, cb, chan, r)
        }
        "CARDUnmount" => {
            if present(a(0)) {
                card().lock().unwrap().mounted = false;
            }
            READY
        }
        "CARDCheck" => if present(a(0)) { READY } else { NOCARD },
        "CARDCheckExAsync" => {
            let (chan, xfer, cb) = (a(0), a(1), a(2));
            if xfer != 0 {
                m.write_be(xfer, 4, 0);
            }
            let r = if present(chan) { READY } else { NOCARD };
            done(c, cb, chan, r)
        }
        "CARDFormat" => {
            if !present(a(0)) {
                NOCARD
            } else {
                let mut k = card().lock().unwrap();
                k.files = vec![None; MAX_FILES];
                save(&k);
                READY
            }
        }
        "CARDFreeBlocks" => {
            if !present(a(0)) {
                NOCARD
            } else {
                let k = card().lock().unwrap();
                if a(1) != 0 {
                    m.write_be(a(1), 4, (free_blocks(&k) * SECTOR_SIZE) as u64);
                }
                if a(2) != 0 {
                    let used = k.files.iter().flatten().count();
                    m.write_be(a(2), 4, (MAX_FILES - used) as u64);
                }
                READY
            }
        }
        "CARDOpen" => {
            let (chan, name_addr, fi) = (a(0), a(1), a(2));
            if !present(chan) {
                NOCARD
            } else {
                let (g, co) = this_game(m);
                let k = card().lock().unwrap();
                match find(&k, &read_name(m, name_addr), g, co) {
                    Some(no) => {
                        let len = k.files[no].as_ref().unwrap().data.len() as u32;
                        write_fileinfo(m, fi, chan, no as u32, len);
                        READY
                    }
                    None => NOFILE,
                }
            }
        }
        "CARDFastOpen" => {
            let (chan, no, fi) = (a(0), a(1) as usize, a(2));
            let k = card().lock().unwrap();
            match k.files.get(no).and_then(|f| f.as_ref()) {
                Some(e) if present(chan) => {
                    write_fileinfo(m, fi, chan, no as u32, e.data.len() as u32);
                    READY
                }
                _ if !present(chan) => NOCARD,
                _ => NOFILE,
            }
        }
        "CARDClose" => READY,
        "CARDCreate" | "CARDCreateAsync" => {
            let (chan, name_addr, size, fi) = (a(0), a(1), a(2), a(3));
            let cb = if name == "CARDCreateAsync" { a(4) } else { 0 };
            let r = if !present(chan) {
                NOCARD
            } else if name_len(name_addr, m) > 32 {
                NAMETOOLONG
            } else {
                let (g, co) = this_game(m);
                let nm = read_name(m, name_addr);
                let mut k = card().lock().unwrap();
                let blocks = size.div_ceil(SECTOR_SIZE);
                if find(&k, &nm, g, co).is_some() {
                    EXIST
                } else if blocks > free_blocks(&k) {
                    INSSPACE
                } else if let Some(no) = k.files.iter().position(|f| f.is_none()) {
                    k.files[no] = Some(Entry {
                        name: nm,
                        game: g,
                        company: co,
                        time: now_seconds_since_2000(),
                        icon_addr: 0xFFFF_FFFF,
                        comment_addr: 0xFFFF_FFFF,
                        data: vec![0; (blocks * SECTOR_SIZE) as usize],
                        ..Default::default()
                    });
                    save(&k);
                    write_fileinfo(m, fi, chan, no as u32, blocks * SECTOR_SIZE);
                    READY
                } else {
                    LIMIT
                }
            };
            done(c, cb, chan, r)
        }
        "CARDRead" | "CARDReadAsync" | "CARDWrite" | "CARDWriteAsync" => {
            let (fi, buf, len, off) = (a(0), a(1), a(2), a(3));
            let cb = if name.ends_with("Async") { a(4) } else { 0 };
            let (chan, no) = (m.read_be(fi, 4) as u32, m.read_be(fi + 4, 4) as usize);
            let write = name.starts_with("CARDWrite");
            let r = {
                let mut k = card().lock().unwrap();
                match k.files.get_mut(no).and_then(|f| f.as_mut()) {
                    None => NOFILE,
                    Some(e) => {
                        let (off, len) = (off as usize, len as usize);
                        if off + len > e.data.len() {
                            crate::diag::log(format!("CARD {name} out of range {off:#x}+{len:#x}"));
                            NOPERM
                        } else {
                            for i in 0..len {
                                if write {
                                    e.data[off + i] = m.read_be(buf + i as u32, 1) as u8;
                                } else {
                                    m.write_be(buf + i as u32, 1, e.data[off + i] as u64);
                                }
                            }
                            if write {
                                e.time = now_seconds_since_2000();
                                save(&k);
                            }
                            m.write_be(fi + 8, 4, (off + len) as u64);
                            READY
                        }
                    }
                }
            };
            done(c, cb, chan, r)
        }
        "CARDDelete" | "CARDDeleteAsync" | "CARDFastDelete" | "CARDFastDeleteAsync" => {
            let chan = a(0);
            let cb = if name.ends_with("Async") { a(2) } else { 0 };
            let r = if !present(chan) {
                NOCARD
            } else {
                let mut k = card().lock().unwrap();
                let no = if name.contains("Fast") {
                    Some(a(1) as usize).filter(|n| k.files.get(*n).is_some_and(|f| f.is_some()))
                } else {
                    let (g, co) = this_game(m);
                    find(&k, &read_name(m, a(1)), g, co)
                };
                match no {
                    Some(n) => {
                        k.files[n] = None;
                        save(&k);
                        READY
                    }
                    None => NOFILE,
                }
            };
            done(c, cb, chan, r)
        }
        "CARDGetStatus" => {
            let (chan, no, st) = (a(0), a(1) as usize, a(2));
            let k = card().lock().unwrap();
            match k.files.get(no).and_then(|f| f.as_ref()) {
                Some(e) if present(chan) => {
                    write_stat(m, st, e);
                    READY
                }
                _ if !present(chan) => NOCARD,
                _ => NOFILE,
            }
        }
        "CARDSetStatus" | "CARDSetStatusAsync" => {
            let (chan, no, st) = (a(0), a(1) as usize, a(2));
            let cb = if name.ends_with("Async") { a(3) } else { 0 };
            let r = {
                let mut k = card().lock().unwrap();
                match k.files.get_mut(no).and_then(|f| f.as_mut()) {
                    Some(e) if present(chan) => {
                        e.banner_format = m.read_be(st + 0x2E, 1) as u8;
                        e.icon_addr = m.read_be(st + 0x30, 4) as u32;
                        e.icon_format = m.read_be(st + 0x34, 2) as u16;
                        e.icon_speed = m.read_be(st + 0x36, 2) as u16;
                        e.comment_addr = m.read_be(st + 0x38, 4) as u32;
                        e.time = now_seconds_since_2000();
                        let e2 = e.clone();
                        save(&k);
                        write_stat(m, st, &e2);
                        READY
                    }
                    _ if !present(chan) => NOCARD,
                    _ => NOFILE,
                }
            };
            done(c, cb, chan, r)
        }
        "CARDRename" | "CARDRenameAsync" => {
            let (chan, old, new) = (a(0), a(1), a(2));
            let cb = if name.ends_with("Async") { a(3) } else { 0 };
            let r = if !present(chan) {
                NOCARD
            } else if name_len(new, m) > 32 {
                NAMETOOLONG
            } else {
                let (g, co) = this_game(m);
                let mut k = card().lock().unwrap();
                let (o, n) = (read_name(m, old), read_name(m, new));
                if find(&k, &n, g, co).is_some() {
                    EXIST
                } else if let Some(i) = find(&k, &o, g, co) {
                    k.files[i].as_mut().unwrap().name = n;
                    save(&k);
                    READY
                } else {
                    NOFILE
                }
            };
            done(c, cb, chan, r)
        }
        _ => return None,
    };
    let _ = WRONGDEVICE;
    Some(r)
}

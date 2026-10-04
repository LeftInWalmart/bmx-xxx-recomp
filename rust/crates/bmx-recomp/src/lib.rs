//! Recompiled BMX XXX (GB3E51) main.dol.
//!
//! `table.rs` and the `bmx-part-NN` crates are generated from the sheets by
//! `tools/gen_rust.py`; this file wires them together and boots the game.

mod table;

use gc_runtime::{disc::Disc, Ctx, Machine};
use gc_core::install_hooks;
use std::path::PathBuf;
use std::sync::OnceLock;
pub use table::{ENTRY_POINT, FUNCTION_COUNT};

/// Default disc image location, relative to the user's profile folder (`%USERPROFILE%`);
/// override with the `BMX_DISC` environment variable.
pub const DEFAULT_DISC: &str = r"Downloads\Emus\GAMECUBE\BMX XXX (USA).ciso";

type Entry = (u32, u32, fn(&mut Ctx, u32));

/// All functions sorted by start address, merged from every part.
fn functions() -> &'static [Entry] {
    static ALL: OnceLock<Vec<Entry>> = OnceLock::new();
    ALL.get_or_init(|| {
        let mut v: Vec<Entry> = table::parts().iter().flat_map(|p| p.iter().copied()).collect();
        v.sort_by_key(|e| e.0);
        v
    })
}

/// Direct-mapped dispatch table: for every instruction word of the code range, 1 + the index
/// (in [`functions`]) of the innermost function whose range contains it, 0 for none.
struct DispatchMap {
    base: u32,
    slots: Vec<u16>,
}

fn dispatch_map() -> &'static DispatchMap {
    static MAP: OnceLock<DispatchMap> = OnceLock::new();
    MAP.get_or_init(|| {
        let fs = functions();
        let base = fs.first().map_or(0, |e| e.0) & !3;
        let end = fs.iter().map(|e| e.1).max().unwrap_or(base);
        let mut slots = vec![0u16; ((end - base) / 4 + 1) as usize];
        // Ranges can nest: Ghidra leaves a few instructions inside some functions unowned (e.g.
        // around switch tables), and those become small orphan pseudo-functions. Filling in
        // start order lets an inner range override its enclosing function only where it lies,
        // so a computed jump (`bctr` through a jump table) to a case label past such an orphan
        // still resolves to the enclosing function.
        assert!(fs.len() < u16::MAX as usize);
        for (i, &(start, end, _)) in fs.iter().enumerate() {
            for a in (start..end).step_by(4) {
                slots[((a - base) / 4) as usize] = i as u16 + 1;
            }
        }
        DispatchMap { base, slots }
    })
}

/// Resolve `addr` to the function whose range contains it and run from there. Called for
/// every indirect branch / call, so this is one table lookup.
fn dispatch(c: &mut Ctx, addr: u32) -> bool {
    let map = dispatch_map();
    let slot = (addr.wrapping_sub(map.base) / 4) as usize;
    match map.slots.get(slot) {
        Some(&i) if i != 0 && addr & 3 == 0 => {
            (functions()[i as usize - 1].2)(c, addr);
            true
        }
        _ => false,
    }
}

pub fn function_name(addr: u32) -> Option<&'static str> {
    table::NAMES.binary_search_by_key(&addr, |e| e.0).ok().map(|i| table::NAMES[i].1)
}

pub fn function_addr(name: &str) -> Option<u32> {
    table::NAMES.iter().find(|e| e.1 == name).map(|e| e.0)
}

pub fn disc_path() -> PathBuf {
    std::env::var_os("BMX_DISC").map(PathBuf::from).unwrap_or_else(|| {
        let home = std::env::var_os("USERPROFILE").or_else(|| std::env::var_os("HOME")).unwrap_or_default();
        PathBuf::from(home).join(DEFAULT_DISC)
    })
}

/// Build the console: load the DOL, the disc's FST/BI2 and the low-memory globals the IPL and
/// apploader would normally set, then return the boot CPU context.
pub fn boot() -> Ctx {
    // Panics on guest threads would otherwise vanish; record them for bmx_trace.log.
    let default_hook = std::panic::take_hook();
    std::panic::set_hook(Box::new(move |info| {
        let thread = std::thread::current().name().unwrap_or("?").to_string();
        let last = gc_core::LAST_FN.load(std::sync::atomic::Ordering::Relaxed);
        gc_runtime::diag::log(format!(
            "PANIC on {thread} (last fn {last:#010x} {}): {info}",
            function_name(last).unwrap_or("?")
        ));
        default_hook(info);
    }));
    let path = disc_path();
    let disc = match Disc::open(&path) {
        Ok(d) => Some(d),
        Err(e) => {
            gc_runtime::diag::log(format!("could not open disc image {}: {e} (set BMX_DISC)", path.display()));
            None
        }
    };
    let m = Machine::new(table::REG_SPACE, dispatch, function_name, function_addr, disc);
    for (addr, bytes) in table::SECTIONS {
        m.load(*addr, bytes);
    }

    let mut arena_hi = 0x8180_0000u32;
    if let Some(d) = &m.disc {
        let mut header = [0u8; 0x20];
        d.read(0, &mut header);
        m.load(0x8000_0000, &header);
        let (fst_off, fst_size, fst_max) = (d.read_u32(0x424), d.read_u32(0x428), d.read_u32(0x42C));
        let fst_addr = (0x8180_0000 - fst_max.max(fst_size)) & !0x1F;
        let mut fst = vec![0u8; fst_size as usize];
        d.read(fst_off as u64, &mut fst);
        m.load(fst_addr, &fst);
        m.write_be(0x8000_0038, 4, fst_addr as u64);
        m.write_be(0x8000_003C, 4, fst_max as u64);
        let bi2_addr = (fst_addr - 0x2000) & !0x1F;
        let mut bi2 = vec![0u8; 0x2000];
        d.read(0x440, &mut bi2);
        m.load(bi2_addr, &bi2);
        m.write_be(0x8000_00F4, 4, bi2_addr as u64);
        // 64 KiB below BI2 is the stack for interrupt handlers run while every thread sleeps.
        arena_hi = bi2_addr - 0x1_0000;
        gc_runtime::sched::set_idle_stack(bi2_addr - 0x20);
    } else {
        m.load(0x8000_0000, b"GB3E51");
    }
    for (addr, v) in [
        (0x8000_0020, 0x0D15_EA5E),            // normal boot magic
        (0x8000_0024, 1),                      // boot version
        (0x8000_0028, gc_runtime::RAM_SIZE as u64), // physical memory size
        (0x8000_002C, 1),                      // console type: retail
        (0x8000_0034, arena_hi as u64),        // ArenaHi
        (0x8000_00CC, 0),                      // video mode: NTSC
        (0x8000_00D0, gc_runtime::ARAM_SIZE as u64),
        (0x8000_00F0, gc_runtime::RAM_SIZE as u64), // simulated memory size
        (0x8000_00F8, 162_000_000),            // bus clock
        (0x8000_00FC, 486_000_000),            // cpu clock
    ] {
        m.write_be(addr, 4, v);
    }

    let hle: Vec<u32> = table::HLE.iter().map(|e| e.0).collect();
    install_hooks(&hle);
    gc_runtime::gx::start_gpu(m);
    gc_runtime::sched::start_idle(m);
    gc_runtime::diag::start_watchdog(m);
    gc_runtime::probe::from_env(m);
    let mut c = m.new_ctx();
    c.thread = gc_runtime::sched::BOOT;
    c.wr(gc_runtime::REG_MSR, 4, 0x2032); // FP, IR/DR on; external interrupts off until OSInit
    c
}

/// Run the game from the DOL entry point (`__start`). Does not return while the game runs.
pub fn run(c: &mut Ctx) {
    gc_runtime::hostprof::register(gc_runtime::sched::BOOT);
    c.call(ENTRY_POINT as u64);
    gc_runtime::diag::log("entry point returned".into());
}

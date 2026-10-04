//! Host-level sampling profiler for the probe (Windows): periodically suspends the host thread
//! that holds the guest CPU token (and the GPU thread), reads its instruction pointer, and at
//! report time symbolizes the samples with dbghelp. Generated functions are public symbols, so
//! the report names them (`f_<ADDR>`) along with runtime functions. Only active with the probe.

use std::collections::HashMap;
use std::sync::Mutex;

/// Key for the GPU thread in the thread table.
pub const GPU: u32 = u32::MAX - 1;

#[derive(Default)]
struct St {
    tids: HashMap<u32, u32>,      // guest thread id (or GPU) -> OS thread id
    handles: HashMap<u32, usize>, // OS thread id -> HANDLE
    cpu: HashMap<u64, u32>,       // RIP -> samples (token holder)
    gpu: HashMap<u64, u32>,       // RIP -> samples (GPU thread)
}

fn st() -> &'static Mutex<St> {
    static S: std::sync::OnceLock<Mutex<St>> = std::sync::OnceLock::new();
    S.get_or_init(Default::default)
}

/// Record that the calling OS thread runs guest thread `guest` (or `GPU`).
///
/// These are the threads that set the game's pace (the guest CPU and the GX decoder), so they
/// also get above-normal OS priority: Bevy keeps a worker on every core, and being preempted
/// by one of those mid-frame shows up as a dropped frame.
pub fn register(guest: u32) {
    #[cfg(windows)]
    {
        use windows_sys::Win32::System::Threading::{GetCurrentThread, GetCurrentThreadId, SetThreadPriority, THREAD_PRIORITY_ABOVE_NORMAL};
        let tid = unsafe { GetCurrentThreadId() };
        unsafe { SetThreadPriority(GetCurrentThread(), THREAD_PRIORITY_ABOVE_NORMAL) };
        st().lock().unwrap().tids.insert(guest, tid);
    }
    #[cfg(not(windows))]
    let _ = guest;
}

/// The boot context became the OS default thread `to`.
pub fn rekey(from: u32, to: u32) {
    let mut s = st().lock().unwrap();
    if let Some(t) = s.tids.remove(&from) {
        s.tids.insert(to, t);
    }
}

#[cfg(windows)]
fn rip_of(tid: u32, handles: &mut HashMap<u32, usize>) -> Option<u64> {
    use windows_sys::Win32::System::Diagnostics::Debug::{GetThreadContext, CONTEXT, CONTEXT_CONTROL_AMD64};
    use windows_sys::Win32::System::Threading::{
        OpenThread, ResumeThread, SuspendThread, THREAD_GET_CONTEXT, THREAD_SUSPEND_RESUME,
    };
    #[repr(C, align(16))]
    struct Ctx16(CONTEXT);
    let h = *handles
        .entry(tid)
        .or_insert_with(|| unsafe { OpenThread(THREAD_SUSPEND_RESUME | THREAD_GET_CONTEXT, 0, tid) as usize });
    if h == 0 {
        return None;
    }
    let mut ctx: Ctx16 = unsafe { std::mem::zeroed() };
    ctx.0.ContextFlags = CONTEXT_CONTROL_AMD64;
    // Nothing between suspend and resume may allocate or take a lock the target might hold.
    unsafe {
        if SuspendThread(h as _) == u32::MAX {
            return None;
        }
        let ok = GetThreadContext(h as _, &mut ctx.0);
        ResumeThread(h as _);
        (ok != 0).then_some(ctx.0.Rip)
    }
}

/// Take one sample of the token holder `running` and of the GPU thread.
pub fn sample(running: u32) {
    #[cfg(windows)]
    {
        let mut guard = st().lock().unwrap();
        let s = &mut *guard;
        if let Some(&tid) = s.tids.get(&running) {
            if let Some(rip) = rip_of(tid, &mut s.handles) {
                *s.cpu.entry(rip).or_default() += 1;
            }
        }
        if let Some(&tid) = s.tids.get(&GPU) {
            if let Some(rip) = rip_of(tid, &mut s.handles) {
                *s.gpu.entry(rip).or_default() += 1;
            }
        }
    }
    #[cfg(not(windows))]
    let _ = running;
}

#[cfg(windows)]
fn symbolize(addrs: &[u64]) -> HashMap<u64, String> {
    use windows_sys::Win32::System::Diagnostics::Debug::{SymFromAddr, SymInitialize, SymSetOptions, SYMBOL_INFO};
    static INIT: std::sync::Once = std::sync::Once::new();
    let process = usize::MAX as *mut core::ffi::c_void; // GetCurrentProcess() pseudo-handle
    INIT.call_once(|| unsafe {
        SymSetOptions(0x2 | 0x4); // UNDNAME | DEFERRED_LOADS
        SymInitialize(process, std::ptr::null(), 1);
    });
    #[repr(C, align(8))]
    struct Buf {
        info: SYMBOL_INFO,
        name: [u8; 512],
    }
    let mut out = HashMap::new();
    for &a in addrs {
        let mut b: Buf = unsafe { std::mem::zeroed() };
        b.info.SizeOfStruct = std::mem::size_of::<SYMBOL_INFO>() as u32;
        b.info.MaxNameLen = 511;
        let mut disp = 0u64;
        let name = if unsafe { SymFromAddr(process, a, &mut disp, &mut b.info) } != 0 {
            let p = b.info.Name.as_ptr() as *const u8;
            let n = (b.info.NameLen as usize).min(511);
            String::from_utf8_lossy(unsafe { std::slice::from_raw_parts(p, n) }).into_owned()
        } else {
            let base = unsafe { windows_sys::Win32::System::LibraryLoader::GetModuleHandleW(std::ptr::null()) } as u64;
            if a >= base && a - base < 0x4000_0000 { format!("exe+{:#x}", a - base) } else { format!("{a:#x}") }
        };
        out.insert(a, name);
    }
    out
}

/// Shorten Rust symbol paths (drop hash suffixes).
fn clean(s: &str) -> String {
    let s = s.split("::h").next().unwrap_or(s);
    s.chars().take(110).collect()
}

fn top(samples: HashMap<u64, u32>, n: usize) -> String {
    let total: u32 = samples.values().sum();
    if total == 0 {
        return "  (no samples)\n".into();
    }
    #[cfg(windows)]
    let names = symbolize(&samples.keys().copied().collect::<Vec<_>>());
    #[cfg(not(windows))]
    let names: HashMap<u64, String> = samples.keys().map(|a| (*a, format!("{a:#x}"))).collect();
    let mut by: HashMap<String, u32> = HashMap::new();
    for (a, k) in samples {
        *by.entry(clean(&names[&a])).or_default() += k;
    }
    let mut v: Vec<_> = by.into_iter().collect();
    v.sort_by(|a, b| b.1.cmp(&a.1));
    v.iter()
        .take(n)
        .map(|(s, k)| format!("  {:5.1}%  {s}\n", *k as f64 * 100.0 / total as f64))
        .collect()
}

/// Report and reset: host profiles of the guest CPU thread and the GPU thread.
pub fn report() -> String {
    let (cpu, gpu) = {
        let mut s = st().lock().unwrap();
        (std::mem::take(&mut s.cpu), std::mem::take(&mut s.gpu))
    };
    // Raw samples as exe-relative offsets, for offline symbolization (tools/symbolize_probe.py).
    #[cfg(windows)]
    {
        let base = unsafe { windows_sys::Win32::System::LibraryLoader::GetModuleHandleW(std::ptr::null()) } as u64;
        let mut out = String::new();
        for (tag, map) in [("cpu", &cpu), ("gpu", &gpu)] {
            for (a, k) in map.iter() {
                if *a >= base && *a - base < 0x4000_0000 {
                    out += &format!("{tag} {:#x} {k}
", a - base);
                } else {
                    out += &format!("{tag} abs{a:#x} {k}
");
                }
            }
        }
        let _ = std::fs::write("bmx_hostprof_raw.txt", out);
    }
    format!(
        "host profile, guest CPU thread (native code incl. generated parts):\n{}host profile, GPU thread:\n{}",
        top(cpu, 25),
        top(gpu, 12)
    )
}

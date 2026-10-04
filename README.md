# bmx-xxx-recomp — BMX XXX (GB3E51), Spreadsheet Method

> **This repository contains no game code.** It holds the pipeline (Ghidra scripts and Python
> tools), the hand-written console runtime and Bevy host (`rust/crates/gc-core`, `gc-runtime`,
> `bmx-bevy`, `bmx-recomp/src`), and three hand-curated metadata sheets (`sheets/hle.csv`,
> `sheets/manual_names.csv`, `sheets/meta.csv`: names and addresses only). Everything derived from
> the game's executable is not here: the other sheets, `src/`, the 24 generated `rust/crates/recomp`
> crates, the memory image and the generated manifests. Rebuild them from **your own** BMX XXX
> (USA) disc image with the steps below, then `cargo build`. `sheets/meta.csv` holds the expected
> `main.dol` SHA-1.
>
> Setup notes:
> * Folder layout: the Ghidra work folder defaults to `../BMX-XXX/`; set `BMX_WORK` to use
>   another path. `import_sdk_names.py` names SDK functions by matching them against a named
>   Mario Kart: Double Dash!! build: `$MKDD_DECOMP/sheets/functions.csv` and
>   `$MKDD_WORK/extracted/main.dol`, defaulting to `../MKDD-Decomp` and `../MKDD`. That reference
>   is optional. Without it, only `sheets/manual_names.csv` is applied. The runtime's HLE hooks come
>   from `sheets/hle.csv`, by address and name, so the game runs either way.
> * Licence: GPL-2.0-or-later (see `LICENSE`). The AX audio DSP emulation in
>   `gc-runtime/src/dsp.rs` follows Dolphin's GPL-2.0+ DSP HLE.


The CSVs in `sheets/` are the **single source of truth** for the whole executable.
Everything else is a projection: `src/` (Ghidra C, for reading) and `rust/` (compilable
Rust, for running) are generated from the sheets and never edited by hand.

```
C:\...\BMX XXX (USA).ciso
   │ tools/ciso_extract.py
   ▼
../BMX-XXX/extracted/main.dol ──ghidra-cli import+analyze──► ../BMX-XXX/ghidra_project/BMX.gpr
   │ tools/FixNoReturn.java, CreateGapFunctions.java, ExportPcode.java, ExportSheets.java (via `ghidra script run`)
   ▼
../BMX-XXX/export/ ──tools/build_sheets.py──► sheets/*.csv ──► src/units/*.c   (readable C)
MKDD-Decomp (named SDK) ──tools/import_sdk_names.py──┘  │
                                                         └─tools/gen_rust.py──► rust/  (Bevy app)
```

Rebuild everything from the disc:

```bat
python tools\ciso_extract.py "%USERPROFILE%\Downloads\Emus\GAMECUBE\BMX XXX (USA).ciso" ..\BMX-XXX\extracted
ghidra --projects-dir ..\BMX-XXX\ghidra_project import ..\BMX-XXX\extracted\main.dol --project BMX --program main.dol
ghidra --projects-dir ..\BMX-XXX\ghidra_project script run tools\FixNoReturn.java --project BMX --program main.dol
ghidra --projects-dir ..\BMX-XXX\ghidra_project script run tools\CreateGapFunctions.java --project BMX --program main.dol
ghidra --projects-dir ..\BMX-XXX\ghidra_project script run tools\ExportPcode.java --project BMX --program main.dol -- ..\BMX-XXX\export
ghidra --projects-dir ..\BMX-XXX\ghidra_project script run tools\ExportSheets.java --project BMX --program main.dol -- ..\BMX-XXX\export
python tools\build_sheets.py && python tools\import_sdk_names.py && python tools\build_sheets.py
python tools\gen_rust.py
```

## Sheets

| Sheet | Rows | Key | Contents |
|---|---|---|---|
| `meta.csv` | 8 | `key` | Game ID, entry point (`0x800051CC`), BSS, DOL hash, toolchain |
| `sections.csv` | 23 | `name` | DOL sections + MMIO blocks |
| `functions.csv` | 5,276 | `addr` | name, size, section, **unit, status, owner**, fan-in/out, signature, C path, notes |
| `units.csv` | 307 | `unit` | Work partitions (161 named after real source files, the rest `blk_<addr>` of ≤40 functions) with status roll-ups |
| `source_files.csv` | 26 | `string_addr` | Strings naming Z-Axis source files (`rider.c`, `physics.c`, `zobj.c`, `gamecubeMovie.c`, `hvqm4read.c`, ...) |
| `symbol_matches.csv` | 691 | `addr` | SDK / runtime names ported from the MKDD sheets, with the reference address and match method |
| `manual_names.csv` | 11 | `addr` | Hand-identified functions with the evidence (OSInit, PADRead, CARDOpen, ...) |
| `hle.csv` | 69 | `name` | Functions replaced by high-level emulation in the Rust runtime (threads, alarms, DVD, PAD, CARD, interrupts, MTX) |
| `calls.csv` | 17,738 | `caller,callee` | Static call graph |
| `instructions.csv` | 374,711 | `addr` | Every instruction: owning function, bytes, disassembly, **raw p-code semantics** |
| `memory.csv` | 26,417 | `addr` | Every initialized byte of the DOL, 64 bytes/row hex |
| `registers.csv` | 2,220 | `name` | Gekko register file layout (offsets used by p-code `R` varnodes) |
| `strings.csv` | 3,184 | `addr` | Defined strings + referencing functions |
| `globals.csv` | 8,040 | `addr` | Data labels, types, sizes, xref counts |
| `disc_files.csv` | 337 | `path` | Every file on the disc (assets: `.zal`, `.zsd` sound, `.h4m` HVQM4 movies, levels) with offset and size |

`instructions.csv` + `memory.csv` + `registers.csv` + `meta.csv` fully describe the program;
that is what `gen_rust.py` consumes. P-code varnodes are `<space><hexoffset>:<size>`
with space `R` register, `U` temporary, `C` constant, `M` RAM.

### Names (`tools/import_sdk_names.py`)
BMX XXX ships without symbols, but it links the Metrowerks runtime and a Dolphin SDK of the same
family as Mario Kart: Double Dash!! (whose names come from doldecomp/mkdd). Its SDK is the
September 2002 build (0x2301), older than MKDD's, so the importer combines several passes:
1. **masked-code** (440): identical instruction words once relocated fields are masked
   (branch targets, `@ha/@l` halves, SDA offsets), unique on both sides.
2. **call-graph** (88): matched functions with the same number of calls name their callees by position.
3. **link-order** (74): equal-sized gaps between matched functions pair up positionally.
4. **aligned** (71): sequence alignment by masked-code similarity inside each gap.
5. **near-code** (7): same length, ≥85 % identical words.
6. **manual** (11): `sheets/manual_names.csv`, with the evidence for each.

The 4,570 `raw` functions are the game (Z-Axis engine and BMX XXX itself). The 161 source-named
units come from the game's assert / error strings (`rider.c`, `weapon.c`, `zgui.c`, ...) and the
SDK split names.

### Status lifecycle (`functions.csv:status`)
`raw` (auto name `FUN_…`) → `named` → `typed` → `clean` → `verified`.
"What is left?" → the per-status columns in `units.csv`.

## Rust / Bevy (`rust/`)

| Crate | What |
|---|---|
| `gc-core` | Stable ABI for generated code: CPU context, p-code ops, paired singles, inline MEM1 access and the `Host` function table. The only crate the generated parts depend on |
| `gc-runtime` | The console behind `Host`: MMIO devices, scheduler, HLE, GX decoder, **AX DSP**, disc, pads, memory card, diagnostics. Editing it rebuilds only the small crates |
| `recomp/part_00..23` | **Generated.** One `pub fn f_<ADDR>(c, entry)` per function, from `instructions.csv`, with one match arm per basic block. Loop back-edges call `c.poll()`, so interrupts arrive inside loops as on the CPU (the AX init spins on a flag the DSP interrupt sets) |
| `bmx-recomp` | Dispatch table, memory image (`mem/*.bin` from `memory.csv`), boot (apploader-equivalent low memory, FST, BI2), `preflight` binary |
| `bmx-bevy` | Bevy 0.19 app: runs the game, draws its GX output on an emulated EFB, plays its audio, feeds keyboard/gamepad into `PADRead` |

Run with `play.bat` (or `cd rust && cargo run -p bmx-bevy`). `preflight.bat` boots the game
headlessly and writes a pass/fail report to `rust/preflight_report.txt`. The disc image is read
from `BMX_DISC` (default: the .ciso in Downloads/Emus/GAMECUBE); `.iso` and `.ciso` both work.

### Audio: AX DSP (`gc-runtime/src/dsp.rs`)
BMX XXX uses the SDK's AX library (MKDD uses JAudio). The DSP is emulated at the mailbox level, so the
game's own `DSP` driver and `AX` code run unmodified:
- **ROM**: mails `0x8071FEED` after reset. It takes `__DSP_boot_task`'s boot parameters and identifies the
  task by its Ector checksum (logged in `bmx_trace.log`).
- **AX ucode**: executes the 5 ms command lists (setup, PB processing, aux A/B send/return, compressor,
  output). For each voice it decodes ADPCM, PCM8 and PCM16 like the hardware accelerator, with loop and
  stream handling. Then it applies linear/nearest sample-rate conversion, the volume envelope, the low-pass
  filter and per-bus volume ramps, and writes depop values back. Both PB layouts are handled: the oldest
  ucode `0x4E8A8B21` has no LPF words and uses the old mixer-control bits.
- Polyphase resampling falls back to linear (the DSP ROM coefficient table isn't available), like Dolphin
  without `dsp_coef.bin`.

### Controls, settings, rendering
Inherited from the MKDD port:
- Keyboard, mouse and gamepads all work at once. F2 rebinds controls (`%APPDATA%\BMX-XXX-Recomp\controls.json`).
- P opens the settings: volume, window mode, gamma, VSync, internal resolution 1x-6x and the FPS counter.
- **Unlock all** (P menu) applies what the game's own `XXX RATED CHEAT` code applies: every rider's items,
  the 20 entries of the table at 0x801BF8C0, every rider unlocked, and the per-rider value of 5. It lasts
  for this session only. Saves made meanwhile keep only the player's own progress: `bmxSaveSerialize` /
  `bmxSaveDeserialize` are hooked in `cheats.rs`. After a restart, choose it again or earn things normally.
- Gamepad origin calibration: an axis or analog trigger that rests far from 0 right after the pad connects
  (common on DirectInput pads) is re-centred on that rest value, like the GameCube PAD origin. It is
  logged in `bmx_trace.log`.
- F1 shows the debug overlay. F11 or Alt+Enter toggles borderless fullscreen.
- Rendering replays every decoded GX frame on an emulated EFB with TEV shaders. The GX pixel states it has
  seen are cached in `%APPDATA%\BMX-XXX-Recomp\pipelines.json`.
- Memory card slot A is emulated and saved to `memcard/slotA.bin` (override with `BMX_MEMCARD`).

### Diagnostics
- `bmx_trace.log` (written every 5 s): hottest functions, thread states, audio and frame pacing.
- `BMX_PROBE=1` writes the performance probe report to `bmx_probe.txt`.
- If the game hangs, the log names the stuck function. Fix it by adding the SDK function to
  `sheets/hle.csv` with a handler in `gc-runtime/src/hle.rs`, then rerun `gen_rust.py`. That
  rewrites only changed files, so the 24 generated crates are not rebuilt.

### Ghidra fix-ups
- `FixNoReturn.java`: auto-analysis marked `OSReport` as non-returning, so the code after all 20 calls to
  it (including the second half of `OSInit`) was never disassembled. The script clears wrong no-return
  flags, disassembles the fall-through and repairs the callers. The 3 remaining `bl OSReport` with no code
  after them are followed by a deliberate illegal instruction (fatal-error crash).

### Known gaps
- HVQM4 movies (`.h4m`) are decoded by the game's own recompiled CPU code (no native fast path).
- The DSP CARD-unlock ucode isn't emulated; the CARD API is emulated at the function level instead.
- No rumble.

"""Build the Spreadsheet-Method tables from the raw Ghidra export.

Inputs  (../BMX-XXX/export/):  functions_raw.csv calls.csv strings.csv globals.csv sections.csv raw_c/*.c
Outputs (sheets/):  meta.csv functions.csv units.csv calls.csv strings.csv globals.csv sections.csv
                    source_files.csv disc_files.csv
        (src/units/, src/raw_c/): one C file per unit, projected from the sheet (regenerable)

Re-running preserves hand-edited columns (name, unit, status, owner, notes) in
sheets/functions.csv, so the sheet stays the source of truth across re-exports.
"""
import csv
import os
import hashlib
import struct
import re
import shutil
from collections import Counter, defaultdict
from pathlib import Path

OUT = Path(__file__).resolve().parent.parent          # BMX-XXX-Decomp/
WORK = Path(os.environ.get("BMX_WORK", OUT.parent / "BMX-XXX"))  # Ghidra project, export/, extracted/ (override: BMX_WORK)
EXP, SHEETS, SRC = WORK / "export", OUT / "sheets", OUT / "src" / "units"
BLOCK = 40  # functions per anonymous work unit
SRC_RE = re.compile(r"([A-Za-z0-9_\-]+\.(?:cpp|cxx|cc|c|h|hpp))\b")
STATUSES = ["raw", "named", "typed", "clean", "verified"]
KEEP = ["name", "unit", "status", "owner", "notes"]


def read(p):
    with open(p, newline="", encoding="utf-8") as fh:
        return list(csv.DictReader(fh))


def write(p, rows, cols):
    with open(p, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, cols, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)


def main():
    SHEETS.mkdir(parents=True, exist_ok=True)
    funcs = read(EXP / "functions_raw.csv")
    strings = read(EXP / "strings.csv")
    prev = {r["addr"]: r for r in read(SHEETS / "functions.csv")} if (SHEETS / "functions.csv").exists() else {}

    # Seed units from source-file names in strings (asserts / __FILE__).
    seeds, src_rows = defaultdict(Counter), []
    for s in strings:
        m = SRC_RE.search(s["text"])
        if not m:
            continue
        src_rows.append({"source_file": m.group(1), "string_addr": s["addr"], "text": s["text"], "ref_funcs": s["ref_funcs"]})
        for f in filter(None, s["ref_funcs"].split(";")):
            seeds[f][m.group(1)] += 1
    seed = {f: c.most_common(1)[0][0] for f, c in seeds.items()}

    funcs.sort(key=lambda r: int(r["addr"], 16))
    unit = [seed.get(f["addr"]) for f in funcs]
    # Fill gaps bracketed by the same source file (CodeWarrior keeps TUs contiguous).
    last_i = None
    for i, u in enumerate(unit):
        if u is None:
            continue
        if last_i is not None and unit[last_i] == u and funcs[last_i]["section"] == funcs[i]["section"]:
            for j in range(last_i + 1, i):
                unit[j] = u
        last_i = i
    # Remaining runs become fixed-size anonymous blocks so work can be split.
    run = []
    def flush():
        for k in range(0, len(run), BLOCK):
            chunk = run[k:k + BLOCK]
            name = f"blk_{funcs[chunk[0]]['addr'][2:]}"
            for j in chunk:
                unit[j] = name
        run.clear()
    for i, u in enumerate(unit):
        if u is None and (not run or funcs[run[-1]]["section"] == funcs[i]["section"]):
            run.append(i)
        else:
            flush()
            if u is None:
                run.append(i)
    flush()

    rows = []
    for f, u in zip(funcs, unit):
        old = prev.get(f["addr"], {})
        default_status = "raw" if f["name"].startswith("FUN_") else "named"
        rows.append({
            **f,
            "name": old.get("name") or f["name"],
            "unit": old.get("unit") or u.replace(".", "_"),
            "status": old.get("status") or default_status,
            "owner": old.get("owner", ""),
            "notes": old.get("notes", ""),
            "c_path": f"src/raw_c/{f['addr'][2:]}.c",
        })
    fcols = ["addr", "name", "size", "section", "unit", "status", "owner", "calls_out", "called_by",
             "decomp_lines", "decomp_ok", "signature", "c_path", "notes"]
    write(SHEETS / "functions.csv", rows, fcols)

    by_unit = defaultdict(list)
    for r in rows:
        by_unit[r["unit"]].append(r)
    units = []
    for u, rs in by_unit.items():
        st = Counter(r["status"] for r in rs)
        units.append({
            "unit": u, "kind": "anon" if u.startswith("blk_") else "source",
            "section": rs[0]["section"], "first_addr": rs[0]["addr"], "last_addr": rs[-1]["addr"],
            "n_funcs": len(rs), "total_size": sum(int(r["size"]) for r in rs),
            **{s: st.get(s, 0) for s in STATUSES},
            "owner": "", "src_path": f"src/units/{u}.c",
        })
    units.sort(key=lambda r: int(r["first_addr"], 16))
    write(SHEETS / "units.csv", units, ["unit", "kind", "section", "first_addr", "last_addr", "n_funcs",
                                        "total_size", *STATUSES, "owner", "src_path"])

    write(SHEETS / "source_files.csv", src_rows, ["source_file", "string_addr", "text", "ref_funcs"])
    for name in ["calls.csv", "strings.csv", "globals.csv", "sections.csv", "instructions.csv", "memory.csv", "registers.csv"]:
        shutil.copy(EXP / name, SHEETS / name)
    disc = WORK / "extracted" / "disc_files.csv"
    if disc.exists():
        shutil.copy(disc, SHEETS / "disc_files.csv")

    raw = OUT / "src" / "raw_c"
    if raw.exists():
        shutil.rmtree(raw)
    shutil.copytree(EXP / "raw_c", raw)

    # Projection: concatenate each unit's functions into one C file.
    if SRC.exists():
        shutil.rmtree(SRC)
    SRC.mkdir(parents=True)
    ident = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")
    names = {r["addr"][2:].lower(): r["name"] for r in rows
             if not r["name"].startswith("FUN_") and ident.match(r["name"])}
    fun_re = re.compile(r"\bFUN_([0-9a-f]{8})\b")
    def rename(code):
        return fun_re.sub(lambda m: names.get(m.group(1), m.group(0)), code)
    for u, rs in by_unit.items():
        with open(SRC / f"{u}.c", "w", encoding="utf-8") as fh:
            fh.write(f"// unit {u}: {len(rs)} functions {rs[0]['addr']}..{rs[-1]['addr']}\n"
                     f"// GENERATED from sheets/functions.csv - edit the sheet, not this file.\n\n")
            for r in rs:
                fh.write(rename((OUT / r["c_path"]).read_text(encoding="utf-8")))
                fh.write("\n")

    dol = (WORK / "extracted" / "main.dol").read_bytes()
    bss, bss_size, entry = struct.unpack_from(">3I", dol, 0xD8)
    write(SHEETS / "meta.csv", [
        {"key": "game_id", "value": "GB3E51"},
        {"key": "title", "value": "BMX XXX (USA)"},
        {"key": "engine", "value": "Z-Axis (zobj/zgui/zfile ...), Dolphin SDK, Metrowerks runtime"},
        {"key": "entry_point", "value": f"0x{entry:08X}"},
        {"key": "bss", "value": f"0x{bss:08X}+0x{bss_size:X}"},
        {"key": "dol_sha1", "value": hashlib.sha1(dol).hexdigest()},
        {"key": "dol_size", "value": str(len(dol))},
        {"key": "toolchain", "value": "Ghidra 12.1.4 + GameCubeLoader via ghidra-cli 0.2.2"},
    ], ["key", "value"])

    tot = Counter(r["status"] for r in rows)
    print(f"{len(rows)} functions, {len(units)} units "
          f"({sum(u['kind'] == 'source' for u in units)} from source names), status {dict(tot)}")


if __name__ == "__main__":
    main()

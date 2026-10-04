"""Name Dolphin SDK / MSL / runtime functions by matching their masked machine code
against an already-named GameCube build (MKDD, whose names come from doldecomp/mkdd).

BMX XXX ships no symbols, but it links the same Metrowerks runtime and a Dolphin SDK of
the same era. A function whose instruction words are identical once relocated fields are
masked (branch targets, @ha/@l immediates, SDA offsets) is the same library function.
Only hashes unique on both sides are accepted. Run build_sheets.py again afterwards to
roll the new names/units into units.csv and src/.

Inputs : $BMX_WORK/extracted/main.dol (default ../BMX-XXX), sheets/functions.csv
         $MKDD_WORK/extracted/main.dol, $MKDD_DECOMP/sheets/functions.csv (reference; defaults
         ../MKDD, ../MKDD-Decomp; optional: without it only sheets/manual_names.csv is applied)
Output : sheets/functions.csv (name/unit/status/notes updated in place), sheets/symbol_matches.csv
"""
import csv
import os
from difflib import SequenceMatcher
import hashlib
import struct
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SHEETS = ROOT / "sheets"
DOL = Path(os.environ.get("BMX_WORK", ROOT.parent / "BMX-XXX")) / "extracted" / "main.dol"
REF_DOL = Path(os.environ.get("MKDD_WORK", ROOT.parent / "MKDD")) / "extracted" / "main.dol"
REF_FUNCS = Path(os.environ.get("MKDD_DECOMP", ROOT.parent / "MKDD-Decomp")) / "sheets" / "functions.csv"
MIN_INSNS = 4  # shorter functions are too generic to identify by shape


class Dol:
    def __init__(self, path):
        self.d = Path(path).read_bytes()
        offs = struct.unpack_from(">18I", self.d, 0)
        addrs = struct.unpack_from(">18I", self.d, 0x48)
        sizes = struct.unpack_from(">18I", self.d, 0x90)
        self.secs = [(a, s, o) for o, a, s in zip(offs, addrs, sizes) if s]

    def read(self, addr, size):
        for a, s, o in self.secs:
            if a <= addr and addr + size <= a + s:
                return self.d[o + addr - a:o + addr - a + size]
        return None


def mask(word):
    op = word >> 26
    if op == 18:                      # b/bl: target
        return word & 0xFC000003
    if op == 16:                      # bc: target
        return word & 0xFFFF0003
    if op in (14, 15, 24, 25):        # addi/addis/ori/oris: @l/@ha/@h halves
        return word & 0xFFFF0000
    if 32 <= op <= 57 or op in (60, 61):     # incl. paired-single psq_l/psq_st
        ra = (word >> 16) & 31        # D-form load/store: keep stack-frame offsets only
        return word if ra == 1 else word & 0xFFFF0000
    return word


def shape(dol, addr, size):
    code = dol.read(addr, size)
    if code is None or size < 4 * MIN_INSNS:
        return None
    words = struct.unpack(f">{size // 4}I", code[:size - size % 4])
    return hashlib.sha1(struct.pack(f">{len(words)}I", *map(mask, words))).hexdigest()


def read(p):
    with open(p, newline="", encoding="utf-8") as fh:
        r = csv.DictReader(fh)
        return r.fieldnames, list(r)


def words(dol, addr, size):
    code = dol.read(addr, size - size % 4)
    return list(struct.unpack(f">{len(code) // 4}I", code)) if code else []


def callees(dol, addr, size):
    """Ordered call targets: every `bl`, plus `b` tail calls that leave the function."""
    out = []
    for i, w in enumerate(words(dol, addr, size)):
        if w >> 26 != 18 or w & 2:
            continue
        off = w & 0x03FFFFFC
        if off & 0x02000000:
            off -= 0x04000000
        t = (addr + 4 * i + off) & 0xFFFFFFFF
        if w & 1 or not addr <= t < addr + size:
            out.append(t)
    return out


def similarity(a, b):
    if len(a) != len(b) or not a:
        return 0.0
    return sum(mask(x) == mask(y) for x, y in zip(a, b)) / len(a)


def sim_seq(a, b):
    ma, mb = [mask(x) for x in a], [mask(x) for x in b]
    return SequenceMatcher(None, ma, mb, autojunk=False).ratio() if ma and mb else 0.0


def align(dol, ref_dol, by_addr, ref_size, gm, gr):
    """Global alignment of two address lists; returns matched pairs scoring >= 0.6."""
    score = {}
    for i, t in enumerate(gm):
        s1 = int(by_addr[t]["size"])
        for j, rt in enumerate(gr):
            s2 = ref_size[rt]
            if abs(s1 - s2) > max(16, s2 * 2 // 5):
                continue
            v = sim_seq(words(dol, t, s1), words(ref_dol, rt, s2))
            if v >= 0.6:
                score[i, j] = v
    if not score:
        return []
    n, m = len(gm), len(gr)
    best = [[0.0] * (m + 1) for _ in range(n + 1)]
    for i in range(n - 1, -1, -1):
        for j in range(m - 1, -1, -1):
            best[i][j] = max(best[i + 1][j], best[i][j + 1], best[i + 1][j + 1] + score.get((i, j), -1.0))
    out, i, j = [], 0, 0
    while i < n and j < m:
        if (i, j) in score and best[i][j] == best[i + 1][j + 1] + score[i, j]:
            out.append((gm[i], gr[j]))
            i, j = i + 1, j + 1
        elif best[i][j] == best[i + 1][j]:
            i += 1
        else:
            j += 1
    return out


def main():
    have_ref = REF_FUNCS.exists() and REF_DOL.exists()
    if not have_ref:
        print(f"MKDD reference not found ({REF_FUNCS}); applying sheets/manual_names.csv only")
    _, ref_rows = read(REF_FUNCS) if have_ref else (None, [])
    ref_dol, dol = (Dol(REF_DOL) if have_ref else None), Dol(DOL)
    ref_named = {int(r["addr"], 16): r for r in ref_rows
                 if not r["name"].startswith(("FUN_", "LAB_")) and r["status"] != "raw"}
    ref_all = sorted(int(r["addr"], 16) for r in ref_rows)
    ref_size = {int(r["addr"], 16): int(r["size"]) for r in ref_rows}

    cols, rows = read(SHEETS / "functions.csv")
    by_addr = {int(r["addr"], 16): r for r in rows}
    mine_all = sorted(by_addr)
    # Idempotent: names this tool gave earlier are recomputed from scratch.
    for r in rows:
        if r["notes"].startswith("name from MKDD"):  # incl. manual: re-applied below
            r["name"], r["status"], r["notes"] = f"FUN_{int(r['addr'], 16):08x}", "raw", ""
    free = {a for a, r in by_addr.items() if r["name"].startswith("FUN_")}
    pair = {}          # bmx addr -> (ref addr, method)
    used = set()       # ref addrs already paired

    def take(a, ra, method):
        if a in free and a not in pair and ra in ref_named and ra not in used:
            pair[a] = (ra, method)
            used.add(ra)
            return True
        return False

    # 1. Identical masked code, unique on both sides.
    ref_h, mine_h = defaultdict(list), defaultdict(list)
    for ra, r in ref_named.items():
        h = shape(ref_dol, ra, ref_size[ra])
        if h:
            ref_h[h].append(ra)
    for a in free:
        h = shape(dol, a, int(by_addr[a]["size"]))
        if h:
            mine_h[h].append(a)
    for h, as_ in mine_h.items():
        rs = ref_h.get(h, [])
        if len(as_) == 1 and len({ref_named[x]["name"] for x in rs}) == 1:
            take(as_[0], rs[0], "masked-code")

    changed = True
    while changed:
        changed = False
        # 2. Call-graph propagation: paired functions with the same number of calls name their
        #    callees position by position (sizes within 25%).
        for a, (ra, _) in list(pair.items()):
            mc, rc = callees(dol, a, int(by_addr[a]["size"])), callees(ref_dol, ra, ref_size[ra])
            if len(mc) != len(rc):
                continue
            for t, rt in zip(mc, rc):
                if t in by_addr and rt in ref_size:
                    s1, s2 = int(by_addr[t]["size"]), ref_size[rt]
                    if abs(s1 - s2) <= max(8, s2 // 4) and take(t, rt, "call-graph"):
                        changed = True
        # 3. Link order: between two consecutive anchors with equally many functions on both
        #    sides, pair positionally when sizes agree within 25%.
        anchors = sorted((a, ra) for a, (ra, _) in pair.items())
        mi = {a: i for i, a in enumerate(mine_all)}
        ri = {a: i for i, a in enumerate(ref_all)}
        for (a1, r1), (a2, r2) in zip(anchors, anchors[1:]):
            gm = mine_all[mi[a1] + 1:mi[a2]]
            gr = ref_all[ri[r1] + 1:ri[r2]] if r2 > r1 else None
            if not gm or gr is None or len(gm) != len(gr) or len(gm) > 24:
                continue
            for t, rt in zip(gm, gr):
                s1, s2 = int(by_addr[t]["size"]), ref_size[rt]
                if abs(s1 - s2) <= max(8, s2 // 4) and take(t, rt, "link-order"):
                    changed = True
        # 4. Alignment: inside each gap between consecutive anchors (up to 48 functions a side),
        #    align the two function lists by masked-code similarity (Needleman-Wunsch) and keep
        #    pairs that are at least 60% similar with sizes within 40%.
        if not changed:
            anchors = sorted((a, ra) for a, (ra, _) in pair.items())
            for (a1, r1), (a2, r2) in zip(anchors, anchors[1:]):
                if r2 <= r1:
                    continue
                gm = [t for t in mine_all[mi[a1] + 1:mi[a2]] if t in free and t not in pair]
                gr = [t for t in ref_all[ri[r1] + 1:ri[r2]] if t in ref_named and t not in used]
                if not gm or not gr or len(gm) > 48 or len(gr) > 48:
                    continue
                for t, rt in align(dol, ref_dol, by_addr, ref_size, gm, gr):
                    if take(t, rt, "aligned"):
                        changed = True
        # 5. Near-identical code: same length, >= 85% of masked words equal, unique best.
        if not changed:
            by_len = defaultdict(list)
            for ra in ref_named:
                if ra not in used and ref_size[ra] >= 4 * MIN_INSNS * 2:
                    by_len[ref_size[ra]].append(ra)
            for a in sorted(free - set(pair)):
                size = int(by_addr[a]["size"])
                cands = by_len.get(size, [])
                if not cands or size < 4 * MIN_INSNS * 2:
                    continue
                mw = words(dol, a, size)
                scored = sorted(((similarity(mw, words(ref_dol, ra, size)), ra) for ra in cands), reverse=True)
                best = scored[0]
                if best[0] >= 0.85 and (len(scored) == 1 or scored[1][0] < best[0] - 0.05):
                    if take(a, best[1], f"near-code {best[0]:.2f}"):
                        changed = True
                        by_len[size].remove(best[1])

    matches = []
    for a, (ra, method) in sorted(pair.items()):
        r, c = by_addr[a], ref_named[ra]
        r["name"], r["status"] = c["name"], "named"
        if not c["unit"].startswith("blk_"):
            r["unit"] = c["unit"]
        r["notes"] = f"name from MKDD {c['addr']} ({method})"
        matches.append({"addr": r["addr"], "name": c["name"], "size": r["size"],
                        "ref_addr": c["addr"], "ref_unit": c["unit"], "method": method})

    # Hand-identified functions (sheets/manual_names.csv) win over every automatic pass.
    manual = SHEETS / "manual_names.csv"
    if manual.exists():
        for mrow in read(manual)[1]:
            r = by_addr[int(mrow["addr"], 16)]
            r["name"], r["status"], r["unit"] = mrow["name"], "named", mrow["unit"]
            r["notes"] = f"name from MKDD-style evidence (manual): {mrow['evidence']}"
            matches = [m for m in matches if m["addr"] != r["addr"]]
            matches.append({"addr": r["addr"], "name": r["name"], "size": r["size"],
                            "ref_addr": "", "ref_unit": r["unit"], "method": "manual"})
        matches.sort(key=lambda m: int(m["addr"], 16))

    with open(SHEETS / "functions.csv", "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, cols)
        w.writeheader()
        w.writerows(rows)
    with open(SHEETS / "symbol_matches.csv", "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, ["addr", "name", "size", "ref_addr", "ref_unit", "method"])
        w.writeheader()
        w.writerows(matches)
    by_method = defaultdict(int)
    for m in matches:
        by_method[m["method"].split()[0]] += 1
    print(f"{len(matches)} functions named from MKDD reference: {dict(by_method)}")


if __name__ == "__main__":
    main()

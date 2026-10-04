"""Extract main.dol and the disc filesystem table from a GameCube .ciso image.

CISO layout: "CISO" magic, u32le block_size, then a 0x7FF8-byte block map
(1 = block present, stored sequentially after the 0x8000-byte header).
"""
import csv
import struct
import sys
from pathlib import Path


class Ciso:
    HEADER = 0x8000

    def __init__(self, path):
        self.f = open(path, "rb")
        hdr = self.f.read(self.HEADER)
        assert hdr[:4] == b"CISO", "not a CISO image"
        self.bs = struct.unpack_from("<I", hdr, 4)[0]
        self.slot = {}
        n = 0
        for i, present in enumerate(hdr[8:]):
            if present:
                self.slot[i] = n
                n += 1

    def read(self, off, size):
        out = bytearray()
        while size:
            blk, inner = divmod(off, self.bs)
            take = min(size, self.bs - inner)
            if blk in self.slot:
                self.f.seek(self.HEADER + self.slot[blk] * self.bs + inner)
                out += self.f.read(take)
            else:
                out += b"\0" * take
            off += take
            size -= take
        return bytes(out)


def dol_size(dol_hdr):
    offs = struct.unpack_from(">18I", dol_hdr, 0)
    sizes = struct.unpack_from(">18I", dol_hdr, 0x90)
    return max(o + s for o, s in zip(offs, sizes))


def walk_fst(fst):
    n = struct.unpack_from(">I", fst, 8)[0]
    names = 12 * n
    def name(i):
        p = names + (struct.unpack_from(">I", fst, i * 12)[0] & 0xFFFFFF)
        return fst[p:fst.index(b"\0", p)].decode("shift_jis", "replace")
    rows, stack, i = [], [(n, "")], 1
    while i < n:
        while stack and i >= stack[-1][0]:
            stack.pop()
        flags, off, size = fst[i * 12], *struct.unpack_from(">II", fst, i * 12 + 4)
        path = stack[-1][1] + "/" + name(i)
        if flags:
            stack.append((size, path))
        else:
            rows.append((path, off, size))
        i += 1
    return rows


def main(iso, outdir):
    out = Path(outdir)
    out.mkdir(parents=True, exist_ok=True)
    disc = Ciso(iso)
    boot = disc.read(0, 0x440)
    game_id, title = boot[:6].decode(), boot[0x20:0x60].split(b"\0")[0].decode()
    dol_off, fst_off, fst_size = struct.unpack_from(">III", boot, 0x420)
    dol = disc.read(dol_off, dol_size(disc.read(dol_off, 0x100)))
    (out / "main.dol").write_bytes(dol)
    rows = walk_fst(disc.read(fst_off, fst_size))
    with open(out / "disc_files.csv", "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["path", "disc_offset", "size"])
        for p, o, s in rows:
            w.writerow([p, f"0x{o:08X}", s])
    print(f"{game_id} {title}: main.dol {len(dol)} bytes, {len(rows)} files")


if __name__ == "__main__":
    main(*sys.argv[1:3])

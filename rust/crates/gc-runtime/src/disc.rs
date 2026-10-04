//! Read-only access to the game disc image (.iso/.gcm or .ciso).

use std::fs::File;
use std::io::{Read, Seek, SeekFrom};
use std::path::Path;
use std::sync::Mutex;

pub struct Disc {
    file: Mutex<File>,
    /// CISO: (block size, block -> stored slot). None for plain ISO.
    ciso: Option<(u64, Vec<Option<u64>>)>,
}

const CISO_HEADER: u64 = 0x8000;

impl Disc {
    pub fn open(path: &Path) -> std::io::Result<Disc> {
        let mut file = File::open(path)?;
        let mut hdr = vec![0u8; CISO_HEADER as usize];
        file.read_exact(&mut hdr[..8])?;
        let ciso = if &hdr[..4] == b"CISO" {
            file.read_exact(&mut hdr[8..])?;
            let bs = u32::from_le_bytes(hdr[4..8].try_into().unwrap()) as u64;
            let mut n = 0;
            let map = hdr[8..]
                .iter()
                .map(|&p| (p != 0).then(|| { n += 1; n - 1 }))
                .collect();
            Some((bs, map))
        } else {
            None
        };
        Ok(Disc { file: Mutex::new(file), ciso })
    }

    /// Read `out.len()` bytes at disc offset `off`. Missing CISO blocks read as zero.
    pub fn read(&self, mut off: u64, out: &mut [u8]) {
        let mut f = self.file.lock().unwrap();
        let mut done = 0;
        while done < out.len() {
            let (pos, take) = match &self.ciso {
                None => (Some(off), out.len() - done),
                Some((bs, map)) => {
                    let blk = (off / bs) as usize;
                    let inner = off % bs;
                    let take = ((bs - inner) as usize).min(out.len() - done);
                    (map.get(blk).copied().flatten().map(|s| CISO_HEADER + s * bs + inner), take)
                }
            };
            let dst = &mut out[done..done + take];
            match pos {
                Some(p) if f.seek(SeekFrom::Start(p)).is_ok() => {
                    if f.read_exact(dst).is_err() {
                        dst.fill(0);
                    }
                }
                _ => dst.fill(0),
            }
            done += take;
            off += take as u64;
        }
    }

    pub fn read_u32(&self, off: u64) -> u32 {
        let mut b = [0u8; 4];
        self.read(off, &mut b);
        u32::from_be_bytes(b)
    }
}

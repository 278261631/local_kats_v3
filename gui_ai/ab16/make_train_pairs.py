#!/usr/bin/env python3
"""Build the final training set by recombining + rotating labeled A/B pairs.

Reads `ai_labeled_16pix/<class>/*.fits` (each FITS = 16x16 A/B pair in two
HDUs) and writes a NEW tree `ai_train_final_16pix/<class>/*.fits`.  The source
tree is never modified.

For every output pair (same class):
  * A is taken from file i, B from file j (i may equal j) -- class-only
    random pairing, no SNR matching;
  * A and B are rotated by the SAME transform;
  * A (signed difference) and B (raw) are normalised to comparable units
    (zero-median / robust scale) so they can be fed as 2 channels.

Output FITS: HDU0 = A float32, HDU1 = B float32 (EXTNAME 'B'), with header
keys CLASS / SRC_A / SRC_B / ROT.  Also writes `manifest.csv`.

Usage:
    python train_AB16pix/make_train_pairs.py --per-class 2000 --rot any
    python train_AB16pix/make_train_pairs.py            # auto-balance to the largest class
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np
from astropy.io import fits
from scipy import ndimage

HERE = Path(__file__).resolve().parent


def load_pair(path: Path):
    with fits.open(path) as h:
        a = np.asarray(h[0].data, dtype=np.float32)
        b = np.asarray(h[1].data, dtype=np.float32) if len(h) > 1 and h[1].data is not None \
            else np.zeros_like(a)
    return a, b


def robust_scale(x: np.ndarray) -> float:
    x = np.nan_to_num(x, nan=0.0)
    s = float(np.percentile(np.abs(x), 99.0))
    if not np.isfinite(s) or s <= 1e-6:
        s = float(np.max(np.abs(x))) if np.max(np.abs(x)) > 0 else 1.0
    return s


def normalise(a: np.ndarray, b: np.ndarray):
    """Signed difference A and raw B -> comparable float32 units."""
    a = np.nan_to_num(a, nan=0.0)
    a = np.clip(a / robust_scale(a), -8.0, 8.0)
    bb = np.nan_to_num(b, nan=float(np.median(b)) if np.isfinite(np.median(b)) else 0.0)
    bb = bb - float(np.median(bb))
    bb = np.clip(bb / robust_scale(bb), -8.0, 8.0)
    return a.astype(np.float32), bb.astype(np.float32)


D4 = [
    lambda x: x,
    lambda x: np.rot90(x, 1),
    lambda x: np.rot90(x, 2),
    lambda x: np.rot90(x, 3),
    lambda x: x[:, ::-1],
    lambda x: x[::-1, :],
    lambda x: x.T,
    lambda x: np.rot90(x.T, 2),
]


def apply_transform(a: np.ndarray, b: np.ndarray, rot: str, rng: np.random.RandomState):
    if rot == "none":
        return a, b, 0.0
    if rot == "d4":
        k = int(rng.randint(len(D4)))
        f = D4[k]
        return np.ascontiguousarray(f(a)), np.ascontiguousarray(f(b)), float(k * 90)
    # arbitrary angle (high-SNR assumption: bilinear blur acceptable)
    ang = float(rng.uniform(0.0, 360.0))
    ra = ndimage.rotate(a, ang, reshape=False, order=1, mode="constant", cval=0.0)
    rb = ndimage.rotate(b, ang, reshape=False, order=1, mode="nearest")
    return ra.astype(np.float32), rb.astype(np.float32), ang


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--src", type=Path, default=HERE.parent / "ai_labeled_16pix")
    p.add_argument("--dst", type=Path, default=HERE.parent / "ai_train_final_16pix")
    p.add_argument("--per-class", type=int, default=0,
                   help="output pairs per class (0 = auto-balance to the largest class)")
    p.add_argument("--rot", choices=["d4", "any", "none"], default="any",
                   help="rotation: d4 (90/180/270 + flips), any (arbitrary angle), none")
    p.add_argument("--same-file-prob", type=float, default=0.5,
                   help="probability that A and B come from the same source file (i==j)")
    p.add_argument("--seed", type=int, default=0)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    rng = np.random.RandomState(args.seed)
    classes = sorted(d.name for d in args.src.iterdir() if d.is_dir())
    if not classes:
        raise SystemExit(f"no class folders in {args.src}")
    files = {c: sorted((args.src / c).glob("*.fits")) for c in classes}
    for c in classes:
        print(f"  {c}: {len(files[c])} labeled files")
    auto = max(len(v) for v in files.values())
    per = args.per_class if args.per_class > 0 else auto
    print(f"generating {per} pairs/class into {args.dst}  (rot={args.rot})")

    # cache the (normalised) source A/B per class to avoid re-reading
    cache = {}
    for c in classes:
        arrs = []
        for f in files[c]:
            a, b = load_pair(f)
            arrs.append(normalise(a, b))
        cache[c] = arrs

    man = []
    for c in classes:
        outdir = args.dst / c
        outdir.mkdir(parents=True, exist_ok=True)
        n = len(cache[c])
        for k in range(per):
            i = int(rng.randint(n))
            j = i if rng.rand() < args.same_file_prob else int(rng.randint(n))
            a, b = cache[c][i][0], cache[c][j][1]
            a, b, ang = apply_transform(a, b, args.rot, rng)
            ph = fits.PrimaryHDU(a.astype(np.float32))
            ph.header["CLASS"] = c
            ph.header["SRC_A"] = files[c][i].name
            ph.header["SRC_B"] = files[c][j].name
            ph.header["ROT"] = float(ang)
            bh = fits.ImageHDU(b.astype(np.float32), name="B")
            name = f"{c}_{k:05d}.fits"
            fits.HDUList([ph, bh]).writeto(outdir / name, overwrite=True)
            man.append([c, name, files[c][i].name, files[c][j].name, round(ang, 3)])

    with (args.dst / "manifest.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["class", "out_file", "src_a", "src_b", "rot_deg"])
        w.writerows(man)
    print(f"done: {len(man)} pairs written; manifest at {args.dst / 'manifest.csv'}")


if __name__ == "__main__":
    main()

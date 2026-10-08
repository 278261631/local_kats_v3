#!/usr/bin/env python3
"""Build a leakage-free train/val/test split for the A/B classifier.

Splits the labeled SOURCE files (`ai_labeled_16pix/<class>/*.fits`) by class
and builds:

  train : augmented pairs  (A from src i, B from src j, same class; rotation)
  val   : original pairs   (i == j, no rotation)  from held-out sources
  test  : original pairs   (i == j, no rotation)  from held-out sources

Because val/test sources never appear in train (and are not rotated /
cross-mixed), the metrics are honest.  A and B are normalised to comparable
units (A signed difference, B raw).  Output FITS: HDU0 = A float32,
HDU1 = B float32 (EXTNAME 'B'); header CLASS / SRC_A / SRC_B / ROT / SNR /
MPCC / VARC.

Usage:
    python train_AB16pix/train_ab/make_split_dataset.py --train-per-class 2000
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np
from astropy.io import fits
from scipy import ndimage

HERE = Path(__file__).resolve().parent
SRC_DEFAULT = HERE.parent / "ai_labeled_16pix"
DST_DEFAULT = HERE.parent / "ai_split_16pix"

D4 = [
    lambda x: x, lambda x: np.rot90(x, 1), lambda x: np.rot90(x, 2),
    lambda x: np.rot90(x, 3), lambda x: x[:, ::-1], lambda x: x[::-1, :],
    lambda x: x.T, lambda x: np.rot90(x.T, 2),
]


def load_pair(path: Path):
    with fits.open(path) as h:
        a = np.asarray(h[0].data, dtype=np.float32)
        b = np.asarray(h[1].data, dtype=np.float32) if len(h) > 1 and h[1].data is not None \
            else np.zeros_like(a)
        hdr = h[0].header
        meta = {k: hdr.get(k, None) for k in ("SNR", "MPCC", "VARC")}
    return a, b, meta


def robust_scale(x: np.ndarray) -> float:
    x = np.nan_to_num(x, nan=0.0)
    s = float(np.percentile(np.abs(x), 99.0))
    if not np.isfinite(s) or s <= 1e-6:
        s = float(np.max(np.abs(x))) if np.max(np.abs(x)) > 0 else 1.0
    return s


def normalise(a: np.ndarray, b: np.ndarray):
    a = np.nan_to_num(a, nan=0.0)
    a = np.clip(a / robust_scale(a), -8.0, 8.0)
    bb = np.nan_to_num(b, nan=float(np.median(b)) if np.isfinite(np.median(b)) else 0.0)
    bb = bb - float(np.median(bb))
    bb = np.clip(bb / robust_scale(bb), -8.0, 8.0)
    return a.astype(np.float32), bb.astype(np.float32)


def transform(a, b, rot, rng):
    if rot == "none":
        return a, b, 0.0
    if rot == "d4":
        k = int(rng.randint(len(D4)))
        f = D4[k]
        return np.ascontiguousarray(f(a)), np.ascontiguousarray(f(b)), float(k * 90)
    ang = float(rng.uniform(0.0, 360.0))
    ra = ndimage.rotate(a, ang, reshape=False, order=1, mode="constant", cval=0.0)
    rb = ndimage.rotate(b, ang, reshape=False, order=1, mode="nearest")
    return ra.astype(np.float32), rb.astype(np.float32), ang


def write_pair(out: Path, a, b, cls, src_a, src_b, rot, meta):
    ph = fits.PrimaryHDU(a.astype(np.float32))
    ph.header["CLASS"] = cls
    ph.header["SRC_A"] = src_a
    ph.header["SRC_B"] = src_b
    ph.header["ROT"] = float(rot)
    for k in ("SNR", "MPCC", "VARC"):
        v = meta.get(k)
        if v is not None:
            ph.header[k] = v
    fits.HDUList([ph, fits.ImageHDU(b.astype(np.float32), name="B")]).writeto(
        out, overwrite=True)


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--src", type=Path, default=SRC_DEFAULT)
    p.add_argument("--dst", type=Path, default=DST_DEFAULT)
    p.add_argument("--train-per-class", type=int, default=2000)
    p.add_argument("--val-frac", type=float, default=0.15)
    p.add_argument("--test-frac", type=float, default=0.15)
    p.add_argument("--rot", choices=["d4", "any", "none"], default="any")
    p.add_argument("--same-file-prob", type=float, default=0.5)
    p.add_argument("--seed", type=int, default=0)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    rng = np.random.RandomState(args.seed)
    classes = sorted(d.name for d in args.src.iterdir() if d.is_dir())
    splits = ("train", "val", "test")
    for s in splits:
        for c in classes:
            (args.dst / s / c).mkdir(parents=True, exist_ok=True)

    stats = {}
    manifests = {s: [] for s in splits}
    for c in classes:
        files = sorted((args.src / c).glob("*.fits"))
        order = rng.permutation(len(files))
        n_val = int(round(args.val_frac * len(files)))
        n_test = int(round(args.test_frac * len(files)))
        val_i = order[:n_val]
        test_i = order[n_val:n_val + n_test]
        train_i = order[n_val + n_test:]
        val_files = [files[i] for i in val_i]
        test_files = [files[i] for i in test_i]
        train_files = [files[i] for i in train_i]

        # cache normalised A/B for the train pool
        cache = []
        for f in train_files:
            a, b, meta = load_pair(f)
            cache.append((*normalise(a, b), meta, f.name))

        # train: augmented combinations
        for k in range(args.train_per_class):
            i = int(rng.randint(len(cache)))
            j = i if rng.rand() < args.same_file_prob else int(rng.randint(len(cache)))
            ai, bi, meta_i, name_i = cache[i]
            aj, bj, meta_j, name_j = cache[j]
            a, b, ang = transform(ai, bj, args.rot, rng)
            out = args.dst / "train" / c / f"{c}_{k:05d}.fits"
            write_pair(out, a, b, c, name_i, name_j, ang, meta_i)
            manifests["train"].append([c, out.name, name_i, name_j, round(ang, 3)])

        # val/test: original pairs (i == j), no rotation
        for split, srcs in (("val", val_files), ("test", test_files)):
            for k, f in enumerate(srcs):
                a, b, meta = load_pair(f)
                a, b = normalise(a, b)
                out = args.dst / split / c / f"{c}_{k:04d}.fits"
                write_pair(out, a, b, c, f.name, f.name, 0.0, meta)
                manifests[split].append([c, out.name, f.name, f.name, 0.0])

        stats[c] = {"labeled": len(files), "train_src": len(train_files),
                    "val_src": len(val_files), "test_src": len(test_files),
                    "train_pairs": args.train_per_class}
        print(f"  {c}: {len(files)} labeled -> {len(train_files)}/{len(val_files)}/"
              f"{len(test_files)} (train/val/test sources), {args.train_per_class} train pairs")

    for s in splits:
        with (args.dst / f"manifest_{s}.csv").open("w", newline="", encoding="utf-8") as f:
            w = csv.writer(f); w.writerow(["class", "out_file", "src_a", "src_b", "rot_deg"])
            w.writerows(manifests[s])
    with (args.dst / "stats.json").open("w", encoding="utf-8") as f:
        json.dump(stats, f, indent=2)
    print(f"done -> {args.dst}")


if __name__ == "__main__":
    main()

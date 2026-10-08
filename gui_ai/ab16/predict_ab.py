#!/usr/bin/env python3
"""Run the trained A/B classifier on one FITS pair (or a folder of them).

Usage:
    python predict_ab.py models_ab/best.pt some_pair.fits
    python predict_ab.py models_ab/best.pt ai_split_16pix/test/target
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch
from astropy.io import fits

HERE = Path(__file__).resolve().parent
import sys
sys.path.insert(0, str(HERE))
from model_ab import ABClassifier  # noqa: E402


def load_input(path: Path, use_snr: bool):
    with fits.open(path) as h:
        a = np.asarray(h[0].data, dtype=np.float32)
        b = np.asarray(h[1].data, dtype=np.float32) if len(h) > 1 and h[1].data is not None \
            else np.zeros_like(a)
        snr = float(h[0].header.get("SNR", 0.0) or 0.0)
    x = torch.from_numpy(np.stack([a, b]))[None]
    ex = torch.tensor([[np.log1p(max(snr, 0.0))]], dtype=torch.float32) if use_snr else None
    return x, ex


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("checkpoint", type=Path)
    ap.add_argument("target", type=Path, help="a .fits file or a folder of them")
    ap.add_argument("--device", default="auto")
    args = ap.parse_args()
    dev = args.device if args.device != "auto" else ("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(dev)

    ck = torch.load(args.checkpoint, map_location=device)
    classes = ck.get("classes", ["noise", "pixelshift", "target"])
    use_snr = bool(ck.get("use_snr", False))
    net = ABClassifier(n_cls=len(classes), base=int(ck.get("base", 32)),
                       n_extra=1 if use_snr else 0).to(device).eval()
    net.load_state_dict(ck["model"])

    files = sorted(args.target.glob("*.fits")) if args.target.is_dir() else [args.target]
    with torch.no_grad():
        for f in files:
            x, ex = load_input(f, use_snr)
            logits = net(x.to(device), ex.to(device) if use_snr else None)
            p = torch.softmax(logits, 1)[0].cpu().numpy()
            order = np.argsort(-p)
            top = ", ".join(f"{classes[i]} {p[i]:.2f}" for i in order)
            print(f"{f.name}:  -> {classes[order[0]]}   [{top}]")


if __name__ == "__main__":
    main()

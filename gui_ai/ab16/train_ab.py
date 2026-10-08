#!/usr/bin/env python3
"""Train the small dual-channel A/B classifier (noise / pixelshift / target).

Reads the leakage-free split from make_split_dataset.py:
    <data>/{train,val,test}/<class>/*.fits
Each FITS = 16x16 A (HDU0) + B (HDU1); optional header SNR fed as a scalar.

Usage:
    python train_AB16pix/train_ab/train_ab.py --data ../ai_split_16pix --use-snr
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
from astropy.io import fits
from sklearn.metrics import (
    average_precision_score, confusion_matrix,
    precision_recall_fscore_support,
)
from sklearn.preprocessing import label_binarize
from torch import nn
from torch.utils.data import DataLoader, Dataset

HERE = Path(__file__).resolve().parent
import sys
sys.path.insert(0, str(HERE))
from model_ab import ABClassifier  # noqa: E402


class ABDataset(Dataset):
    def __init__(self, root: Path, classes: list[str], use_snr: bool):
        self.items = []
        self.use_snr = use_snr
        for li, c in enumerate(classes):
            for f in sorted((root / c).glob("*.fits")):
                self.items.append((f, li))

    def __len__(self):
        return len(self.items)

    def __getitem__(self, i):
        path, label = self.items[i]
        with fits.open(path) as h:
            a = np.asarray(h[0].data, dtype=np.float32)
            b = np.asarray(h[1].data, dtype=np.float32) if len(h) > 1 and h[1].data is not None \
                else np.zeros_like(a)
            snr = float(h[0].header.get("SNR", 0.0) or 0.0)
        x = torch.from_numpy(np.stack([a, b]))          # (2,16,16)
        extra = torch.tensor([np.log1p(max(snr, 0.0))], dtype=torch.float32) \
            if self.use_snr else torch.zeros(0)
        return x, extra, label


def evaluate(model, loader, device, n_cls, use_snr):
    model.eval()
    probs, gts = [], []
    with torch.no_grad():
        for x, extra, y in loader:
            logits = model(x.to(device), extra.to(device) if use_snr else None)
            probs.append(torch.softmax(logits, 1).cpu().numpy())
            gts.append(y.numpy())
    probs = np.concatenate(probs); gts = np.concatenate(gts)
    pred = probs.argmax(1)
    cm = confusion_matrix(gts, pred, labels=list(range(n_cls)))
    P, R, F1, _ = precision_recall_fscore_support(
        gts, pred, labels=list(range(n_cls)), zero_division=0)
    try:
        yb = label_binarize(gts, classes=list(range(n_cls)))
        prap = float(average_precision_score(yb, probs, average="macro"))
    except Exception:
        prap = float("nan")
    return {"acc": float((pred == gts).mean()), "macro_f1": float(F1.mean()),
            "per_class_p": P.tolist(), "per_class_r": R.tolist(),
            "per_class_f1": F1.tolist(), "pr_auc_macro": prap, "cm": cm.tolist()}


def fmt_m(m, classes):
    out = [f"acc {m['acc']:.3f}  macroF1 {m['macro_f1']:.3f}  PR-AUC {m['pr_auc_macro']:.3f}"]
    for i, c in enumerate(classes):
        out.append(f"   {c:>10}: P {m['per_class_p'][i]:.3f}  R {m['per_class_r'][i]:.3f}"
                   f"  F1 {m['per_class_f1'][i]:.3f}")
    out.append("   confusion (rows=true): " + str(m["cm"]))
    return "\n".join(out)


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data", type=Path, default=HERE.parent / "ai_split_16pix")
    p.add_argument("--out", type=Path, default=HERE / "models_ab")
    p.add_argument("--epochs", type=int, default=30)
    p.add_argument("--batch", type=int, default=128)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--base", type=int, default=32)
    p.add_argument("--use-snr", action="store_true")
    p.add_argument("--max-train", type=int, default=0)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", type=str, default="auto")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    dev = args.device if args.device != "auto" else ("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(dev)
    torch.manual_seed(args.seed); np.random.seed(args.seed)
    print("device:", device)

    classes = sorted(d.name for d in (args.data / "train").iterdir() if d.is_dir())
    n_cls = len(classes)
    tr = ABDataset(args.data / "train", classes, args.use_snr)
    va = ABDataset(args.data / "val", classes, args.use_snr)
    te = ABDataset(args.data / "test", classes, args.use_snr)
    if args.max_train > 0:
        tr.items = tr.items[:args.max_train]
    print(f"classes {classes}  train {len(tr)} val {len(va)} test {len(te)}")

    counts = np.bincount([l for _, l in tr.items], minlength=n_cls).astype(float)
    w = counts.sum() / (n_cls * np.maximum(counts, 1))
    weights = torch.tensor(w, dtype=torch.float32, device=device)
    print("class weights:", np.round(w, 3).tolist())

    tr_ld = DataLoader(tr, batch_size=args.batch, shuffle=True, num_workers=0)
    va_ld = DataLoader(va, batch_size=args.batch, shuffle=False, num_workers=0)
    te_ld = DataLoader(te, batch_size=args.batch, shuffle=False, num_workers=0)

    n_extra = 1 if args.use_snr else 0
    model = ABClassifier(n_cls=n_cls, base=args.base, n_extra=n_extra).to(device)
    print(f"params: {sum(p.numel() for p in model.parameters())/1e3:.1f} K")
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs)
    lossf = nn.CrossEntropyLoss(weight=weights)
    best = -1.0

    for ep in range(args.epochs):
        model.train(); t0 = time.perf_counter(); tot = 0.0; nb = 0
        for x, extra, y in tr_ld:
            x = x.to(device); y = y.to(device)
            ex = extra.to(device) if args.use_snr else None
            opt.zero_grad()
            loss = lossf(model(x, ex), y)
            loss.backward(); opt.step()
            tot += float(loss) * len(y); nb += len(y)
        sched.step()
        vm = evaluate(model, va_ld, device, n_cls, args.use_snr)
        print(f"ep {ep+1:02d}/{args.epochs}  loss {tot/nb:.4f}  "
              f"val acc {vm['acc']:.3f} macroF1 {vm['macro_f1']:.3f}  ({time.perf_counter()-t0:.0f}s)")
        if vm["macro_f1"] > best:
            best = vm["macro_f1"]
            args.out.mkdir(parents=True, exist_ok=True)
            torch.save({"model": model.state_dict(), "classes": classes,
                        "base": args.base, "use_snr": args.use_snr, "val": vm},
                       args.out / "best.pt")

    ck = torch.load(args.out / "best.pt", map_location=device)
    model.load_state_dict(ck["model"])
    tm = evaluate(model, te_ld, device, n_cls, args.use_snr)
    print("\n=== TEST ===\n" + fmt_m(tm, classes))
    with (args.out / "test_metrics.json").open("w", encoding="utf-8") as f:
        json.dump({"classes": classes, "test": tm, "val_best": ck["val"]}, f, indent=2)
    print(f"saved -> {args.out/'best.pt'}")


if __name__ == "__main__":
    main()

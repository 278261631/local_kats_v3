#!/usr/bin/env python3
"""Inspect the A/B classifier's predictions on the test split.

Shows, per test pair: A signed, A|diff| gray, B raw; the true vs predicted
class with per-class probabilities; a file list coloured by correctness with a
filter (all / wrong only / by true class); and the overall summary.

Run:
    python train_AB16pix/train_ab/validate_ab_gui.py
    python train_AB16pix/train_ab/validate_ab_gui.py --ckpt models_ab/best.pt --data ../ai_split_16pix/test
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch
from astropy.io import fits
from PySide6 import QtGui, QtWidgets
from PySide6.QtCore import Qt
from PySide6.QtGui import QColor, QImage, QPixmap
from PySide6.QtWidgets import (
    QComboBox, QFileDialog, QGridLayout, QHBoxLayout, QLabel, QListWidget,
    QListWidgetItem, QMainWindow, QPushButton, QVBoxLayout, QWidget,
)

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from model_ab import ABClassifier  # noqa: E402

PANEL = 240


def signed_rgb(a):
    a = np.nan_to_num(a, nan=0.0)
    s = float(np.percentile(np.abs(a), 99.0))
    if not np.isfinite(s) or s <= 1e-6:
        s = max(1.0, float(np.max(np.abs(a))))
    v = np.clip(a / s, -1, 1)
    rgb = np.zeros(a.shape + (3,), np.float32)
    rgb[..., 0] = np.clip(-v, 0, 1)
    rgb[..., 1] = np.clip(v, 0, 1)
    return rgb


def gray_rgb(x):
    g = np.nan_to_num(x, nan=0.0)
    lo, hi = np.percentile(g, (1.0, 99.0))
    g = np.clip((g - lo) / max(1e-6, hi - lo), 0, 1)
    return np.stack([g, g, g], -1).astype(np.float32)


def qimg(rgb):
    u8 = np.ascontiguousarray((np.clip(rgb, 0, 1) * 255).astype(np.uint8))
    return QImage(u8.data, u8.shape[1], u8.shape[0], u8.strides[0],
                  QImage.Format.Format_RGB888)


def load_pair(path: Path):
    with fits.open(path) as h:
        a = np.asarray(h[0].data, dtype=np.float32)
        b = np.asarray(h[1].data, dtype=np.float32) if len(h) > 1 and h[1].data is not None \
            else np.zeros_like(a)
        snr = float(h[0].header.get("SNR", 0.0) or 0.0)
    return a, b, snr


class ABGui(QMainWindow):
    def __init__(self, ckpt: Path, data: Path) -> None:
        super().__init__()
        self.setWindowTitle("A/B classifier - test inspection")
        self.data = data
        ck = torch.load(ckpt, map_location="cpu")
        self.classes = ck.get("classes", ["noise", "pixelshift", "target"])
        self.use_snr = bool(ck.get("use_snr", False))
        self.net = ABClassifier(n_cls=len(self.classes), base=int(ck.get("base", 32)),
                                n_extra=1 if self.use_snr else 0).eval()
        self.net.load_state_dict(ck["model"])
        self.records = self._predict_all()
        self.view = list(range(len(self.records)))
        self._build_ui()
        self._refresh_filter()
        self._show()

    def _predict_all(self):
        recs = []
        for ti, c in enumerate(self.classes):
            for f in sorted((self.data / c).glob("*.fits")):
                a, b, snr = load_pair(f)
                x = torch.from_numpy(np.stack([a, b]))[None]
                ex = torch.tensor([[np.log1p(max(snr, 0.0))]], dtype=torch.float32) \
                    if self.use_snr else None
                with torch.no_grad():
                    p = torch.softmax(self.net(x, ex), 1)[0].numpy()
                recs.append({"path": f, "true": ti, "pred": int(p.argmax()), "prob": p, "img": (a, b)})
        return recs

    def _build_ui(self):
        c = QWidget(); self.setCentralWidget(c)
        root = QVBoxLayout(c)
        bar = QHBoxLayout()
        bar.addWidget(QLabel("filter"))
        self.cb = QComboBox()
        self.cb.addItems(["all", "wrong only", "correct only"] + [f"true: {c}" for c in self.classes])
        self.cb.currentIndexChanged.connect(self._refresh_filter)
        bar.addWidget(self.cb)
        self.btn_prev = QPushButton("< prev"); self.btn_prev.clicked.connect(lambda: self._step(-1))
        self.btn_next = QPushButton("next >"); self.btn_next.clicked.connect(lambda: self._step(1))
        self.btn_rand = QPushButton("random"); self.btn_rand.clicked.connect(self._random)
        for b in (self.btn_prev, self.btn_next, self.btn_rand):
            bar.addWidget(b)
        bar.addStretch(1)
        self.lbl_sum = QLabel(); self.lbl_sum.setStyleSheet(
            "font-family:Consolas,monospace; color:#9fb4d8;")
        bar.addWidget(self.lbl_sum)
        root.addLayout(bar)

        mid = QHBoxLayout()
        self.list = QListWidget(); self.list.setMinimumWidth(320)
        self.list.setStyleSheet("font-family:Consolas,monospace; font-size:12px;")
        self.list.currentRowChanged.connect(self._on_row)
        mid.addWidget(self.list)

        grid = QGridLayout()
        self.pA = self._panel("A signed (red-/green+)")
        self.pAg = self._panel("A |diff| gray")
        self.pB = self._panel("B raw gray")
        grid.addWidget(self.pA[1], 0, 0); grid.addWidget(self.pAg[1], 0, 1)
        grid.addWidget(self.pB[1], 1, 0)
        self.lbl_info = QLabel(); self.lbl_info.setAlignment(Qt.AlignmentFlag.AlignTop)
        self.lbl_info.setStyleSheet("font-family:Consolas,monospace; color:#cdd6e6;")
        box = QWidget(); v = QVBoxLayout(box); v.addWidget(self.lbl_info); v.addStretch(1)
        grid.addWidget(box, 1, 1)
        mid.addLayout(grid, 1)
        root.addLayout(mid, 1)

        self.status = self.statusBar()
        self._s = QLabel(); self.status.addPermanentWidget(self._s)
        QtGui.QShortcut(QtGui.QKeySequence(Qt.Key.Key_Right), self, lambda: self._step(1))
        QtGui.QShortcut(QtGui.QKeySequence(Qt.Key.Key_Left), self, lambda: self._step(-1))
        QtGui.QShortcut(QtGui.QKeySequence("R"), self, self._random)

    def _panel(self, title):
        box = QWidget(); v = QVBoxLayout(box); v.setContentsMargins(4, 4, 4, 4)
        t = QLabel(title); t.setAlignment(Qt.AlignmentFlag.AlignCenter)
        img = QLabel(); img.setFixedSize(PANEL, PANEL)
        img.setStyleSheet("background:#0b0d12;")
        v.addWidget(t); v.addWidget(img)
        return t, box, img

    # -- filter/nav ---------------------------------------------------------
    def _refresh_filter(self):
        mode = self.cb.currentIndex()
        if mode == 0:
            self.view = list(range(len(self.records)))
        elif mode == 1:
            self.view = [i for i, r in enumerate(self.records) if r["pred"] != r["true"]]
        elif mode == 2:
            self.view = [i for i, r in enumerate(self.records) if r["pred"] == r["true"]]
        else:
            t = mode - 3
            self.view = [i for i, r in enumerate(self.records) if r["true"] == t]
        self.pos = 0
        self._populate_list()
        self._show()

    def _populate_list(self):
        self.list.blockSignals(True); self.list.clear(); self.rows = {}
        for i in self.view:
            r = self.records[i]
            ok = r["pred"] == r["true"]
            txt = f"[{'OK ' if ok else 'ERR'}] {self.classes[r['true']][:5]:>5}->{self.classes[r['pred']][:5]:<5} {r['path'].name}"
            it = QListWidgetItem(txt)
            it.setForeground(QColor("#7dff9b" if ok else "#ff6b6b"))
            self.list.addItem(it); self.rows[i] = it
        self.list.blockSignals(False)
        self._summary()

    def _summary(self):
        n = len(self.records)
        acc = np.mean([r["pred"] == r["true"] for r in self.records])
        cm = np.zeros((len(self.classes), len(self.classes)), int)
        for r in self.records:
            cm[r["true"], r["pred"]] += 1
        self.lbl_sum.setText(f"test n={n}  acc={acc:.3f}   view {len(self.view)}")
        self._s.setText("confusion (rows=true): " + str(cm.tolist()))

    def _cur(self):
        return self.records[self.view[self.pos]] if self.view else None

    def _step(self, d):
        if self.view:
            self.pos = (self.pos + d) % len(self.view); self._show()

    def _random(self):
        if self.view:
            self.pos = int(np.random.randint(len(self.view))); self._show()

    def _on_row(self, row):
        if 0 <= row < len(self.view) and self.view[row] in self.rows:
            self.pos = row; self._show()

    # -- render -------------------------------------------------------------
    def _show(self):
        r = self._cur()
        if r is None:
            self.lbl_info.setText("(no items)"); return
        a, b = r["img"]
        self.pA[2].setPixmap(QPixmap.fromImage(qimg(signed_rgb(a))).scaled(
            PANEL, PANEL, Qt.AspectRatioMode.KeepAspectRatio, Qt.TransformationMode.FastTransformation))
        self.pAg[2].setPixmap(QPixmap.fromImage(qimg(gray_rgb(np.abs(a)))).scaled(
            PANEL, PANEL, Qt.AspectRatioMode.KeepAspectRatio, Qt.TransformationMode.FastTransformation))
        self.pB[2].setPixmap(QPixmap.fromImage(qimg(gray_rgb(b))).scaled(
            PANEL, PANEL, Qt.AspectRatioMode.KeepAspectRatio, Qt.TransformationMode.FastTransformation))
        ok = r["pred"] == r["true"]
        lines = [f"file   {r['path'].name}",
                 f"true   {self.classes[r['true']]}",
                 f"pred   {self.classes[r['pred']]}   {'(OK)' if ok else '(WRONG)'}", ""]
        for i, c in enumerate(self.classes):
            lines.append(f"  P({c:<10}) = {r['prob'][i]:.3f}")
        self.lbl_info.setText("\n".join(lines))
        self.lbl_info.setStyleSheet(
            "font-family:Consolas,monospace; color:" + ("#7dff9b" if ok else "#ff6b6b") + ";")
        it = self.rows.get(self.view[self.pos])
        if it is not None:
            self.list.blockSignals(True); self.list.setCurrentItem(it)
            self.list.scrollToItem(it); self.list.blockSignals(False)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--ckpt", type=Path, default=HERE / "models_ab" / "best.pt")
    ap.add_argument("--data", type=Path, default=HERE.parent / "ai_split_16pix" / "test")
    args = ap.parse_args()
    if not args.ckpt.exists():
        raise SystemExit(f"checkpoint not found: {args.ckpt}")
    if not args.data.exists():
        raise SystemExit(f"test data not found: {args.data}")
    app = QtWidgets.QApplication(sys.argv)
    win = ABGui(args.ckpt, args.data)
    win.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()

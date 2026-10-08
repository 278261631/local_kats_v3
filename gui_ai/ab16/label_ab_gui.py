#!/usr/bin/env python3
"""Manual vetting GUI for A/B candidate cutout pairs.

Each source FITS holds a 16x16 A/B pair in two HDUs (HDU0 = A, processed
float32; HDU1 = B, raw uint16).  The tool shows the current folder / file /
class status, a scrollable file list with per-file status, and FOUR comparison
views (A signed, A gray, B gray, B+A overlay).  Classifying copies the FITS
pair into a per-class sub-folder under the destination root; the source tree
(`ai_train_16pix`) is never modified.

Run:
    python train_AB16pix/label_ab_gui.py
    python train_AB16pix/label_ab_gui.py --src ai_train_16pix --dst ai_labeled_16pix

Keys:  1/2/3 classify   Space/-> next   <- prev   R random   U undo
"""

from __future__ import annotations

import argparse
import csv
import shutil
import sys
from pathlib import Path

import numpy as np
from astropy.io import fits
from PySide6 import QtGui, QtWidgets
from PySide6.QtCore import Qt
from PySide6.QtGui import QColor, QImage, QPixmap
from PySide6.QtWidgets import (
    QCheckBox, QFileDialog, QGridLayout, QHBoxLayout, QLabel, QListWidget,
    QListWidgetItem, QMainWindow, QPushButton, QVBoxLayout, QWidget,
)

HERE = Path(__file__).resolve().parent

CLASSES = [
    ("noise", "1 noise"),
    ("pixelshift", "2 pixel-shift"),
    ("target", "3 target"),
]
CLASS_COLORS = {"noise": "#7fb0ff", "pixelshift": "#ffc04d", "target": "#7dff9b"}
PANEL = 260
CSV_NAME = "labels.csv"


def load_pair(path: Path):
    with fits.open(path) as h:
        a = np.asarray(h[0].data, dtype=np.float32)
        b = np.asarray(h[1].data, dtype=np.float32) if len(h) > 1 and h[1].data is not None \
            else np.zeros_like(a)
        hdr = {k: h[0].header[k] for k in h[0].header if k not in ("COMMENT", "HISTORY")}
    return a, b, hdr


def _safe_scale(a: np.ndarray) -> float:
    s = float(np.nanpercentile(np.abs(a), 99.0))
    if not np.isfinite(s) or s <= 1e-6:
        m = np.nanmax(np.abs(a)) if np.isfinite(np.nanmax(np.abs(a))) else 1.0
        s = max(1.0, float(m))
    return s


def signed_rgb(a: np.ndarray) -> np.ndarray:
    """red = negative, green = positive (difference image polarity)."""
    v = np.nan_to_num(a, nan=0.0) / _safe_scale(a)
    rgb = np.zeros(a.shape + (3,), dtype=np.float32)
    rgb[..., 0] = np.clip(-v, 0, 1)
    rgb[..., 1] = np.clip(v, 0, 1)
    return rgb


def gray_of(x: np.ndarray) -> np.ndarray:
    g = np.nan_to_num(x, nan=0.0)
    lo, hi = np.percentile(g, (1.0, 99.0))
    g = np.clip((g - lo) / max(1e-6, hi - lo), 0, 1)
    return np.stack([g, g, g], axis=-1).astype(np.float32)


def gray_abs(a: np.ndarray) -> np.ndarray:
    return gray_of(np.abs(a))


def overlay_rgb(b: np.ndarray, a: np.ndarray) -> np.ndarray:
    """B in gray with the A difference tinted on top (red neg / green pos)."""
    rgb = gray_of(b)
    v = np.nan_to_num(a, nan=0.0) / _safe_scale(a)
    pos = np.clip(v, 0, 1); neg = np.clip(-v, 0, 1)
    rgb[..., 0] = np.clip(rgb[..., 0] + 0.85 * neg, 0, 1)
    rgb[..., 1] = np.clip(rgb[..., 1] + 0.85 * pos, 0, 1)
    rgb[..., 2] = np.clip(rgb[..., 2] + 0.85 * neg, 0, 1)
    return rgb


def rgb_qimage(rgb: np.ndarray) -> QImage:
    u8 = np.ascontiguousarray((np.clip(rgb, 0, 1) * 255).astype(np.uint8))
    return QImage(u8.data, u8.shape[1], u8.shape[0], u8.strides[0],
                  QImage.Format.Format_RGB888)


class LabelGui(QMainWindow):
    def __init__(self, src: Path, dst: Path) -> None:
        super().__init__()
        self.setWindowTitle("A/B candidate vetting")
        self.src, self.dst = src, dst
        self.dst.mkdir(parents=True, exist_ok=True)
        for cls, _ in CLASSES:
            (self.dst / cls).mkdir(exist_ok=True)
        self.labels = self._read_csv()
        self.undo_stack = []
        self.files = sorted(self.src.rglob("*.fits"))
        self.view = list(range(len(self.files)))
        self.pos = 0
        self._build_ui()
        self._populate_list()
        self._refresh_only()
        self._show()

    # -- persistence --------------------------------------------------------
    def _csv_path(self) -> Path:
        return self.dst / CSV_NAME

    def _read_csv(self) -> dict:
        out = {}
        p = self._csv_path()
        if p.exists():
            with p.open(newline="", encoding="utf-8") as f:
                for row in csv.reader(f):
                    if len(row) == 2:
                        out[row[0]] = row[1]
        return out

    def _rewrite_csv(self) -> None:
        with self._csv_path().open("w", newline="", encoding="utf-8") as f:
            w = csv.writer(f); w.writerow(["src_relpath", "class"])
            for k, v in self.labels.items():
                w.writerow([k, v])

    # -- ui -----------------------------------------------------------------
    def _build_ui(self) -> None:
        c = QWidget(); self.setCentralWidget(c)
        root = QVBoxLayout(c)

        bar = QHBoxLayout()
        b_src = QPushButton("open data folder..."); b_src.clicked.connect(self._pick_src)
        b_dst = QPushButton("output folder..."); b_dst.clicked.connect(self._pick_dst)
        self.chk_only = QCheckBox("only unlabeled"); self.chk_only.setChecked(True)
        self.chk_only.toggled.connect(self._refresh_only)
        self.btn_prev = QPushButton("< prev"); self.btn_prev.clicked.connect(lambda: self._step(-1))
        self.btn_next = QPushButton("next >"); self.btn_next.clicked.connect(lambda: self._step(1))
        self.btn_rand = QPushButton("random"); self.btn_rand.clicked.connect(self._random)
        self.btn_undo = QPushButton("undo"); self.btn_undo.clicked.connect(self._undo)
        for w in (b_src, b_dst, self.chk_only, self.btn_prev, self.btn_next, self.btn_rand, self.btn_undo):
            bar.addWidget(w)
        bar.addStretch(1)
        root.addLayout(bar)

        self.lbl_path = QLabel(); self.lbl_path.setStyleSheet(
            "font-family:Consolas,monospace; font-size:12px; color:#9fb4d8;")
        root.addWidget(self.lbl_path)

        mid = QHBoxLayout()

        self.list = QListWidget(); self.list.setMinimumWidth(300)
        self.list.setStyleSheet("font-family:Consolas,monospace; font-size:12px;")
        self.list.currentRowChanged.connect(self._on_list_row)
        mid.addWidget(self.list)

        grid = QGridLayout()
        self.pA = self._panel("A signed (red neg / green pos)")
        self.pAg = self._panel("A |diff| gray")
        self.pB = self._panel("B raw gray")
        self.pOv = self._panel("overlay: B + A sign")
        grid.addWidget(self.pA[1], 0, 0); grid.addWidget(self.pAg[1], 0, 1)
        grid.addWidget(self.pB[1], 1, 0); grid.addWidget(self.pOv[1], 1, 1)
        mid.addLayout(grid, 1)
        root.addLayout(mid, 1)

        info_bar = QHBoxLayout()
        self.lbl_info = QLabel(); self.lbl_info.setStyleSheet(
            "font-family:Consolas,monospace; color:#cdd6e6;")
        self.lbl_status = QLabel(); self.lbl_status.setStyleSheet("font-size:13px;")
        info_bar.addWidget(self.lbl_info, 1); info_bar.addWidget(self.lbl_status)
        root.addLayout(info_bar)

        cls_bar = QHBoxLayout()
        for folder, label in CLASSES:
            b = QPushButton(label); b.setMinimumHeight(36)
            b.clicked.connect(lambda _=False, cl=folder: self._classify(cl))
            b.setStyleSheet(f"font-size:14px; color:{CLASS_COLORS.get(folder, '#fff')};")
            cls_bar.addWidget(b)
        root.addLayout(cls_bar)

        self.status = self.statusBar()
        self._s = QLabel(); self.status.addPermanentWidget(self._s)

        for i in range(1, 4):
            QtGui.QShortcut(QtGui.QKeySequence(str(i)), self,
                            lambda n=i: self._classify(CLASSES[n - 1][0]))
        QtGui.QShortcut(QtGui.QKeySequence(Qt.Key.Key_Space), self, lambda: self._step(1))
        QtGui.QShortcut(QtGui.QKeySequence(Qt.Key.Key_Right), self, lambda: self._step(1))
        QtGui.QShortcut(QtGui.QKeySequence(Qt.Key.Key_Left), self, lambda: self._step(-1))
        QtGui.QShortcut(QtGui.QKeySequence("R"), self, self._random)
        QtGui.QShortcut(QtGui.QKeySequence("U"), self, self._undo)

    def _panel(self, title: str):
        box = QWidget(); v = QVBoxLayout(box); v.setContentsMargins(4, 4, 4, 4)
        t = QLabel(title); t.setAlignment(Qt.AlignmentFlag.AlignCenter)
        img = QLabel(); img.setFixedSize(PANEL, PANEL)
        img.setAlignment(Qt.AlignmentFlag.AlignCenter)
        img.setStyleSheet("background:#0b0d12;")
        v.addWidget(t); v.addWidget(img)
        return t, box, img

    # -- file list ----------------------------------------------------------
    def _rel(self, p: Path) -> str:
        return str(p.relative_to(self.src)).replace("\\", "/")

    def _list_text(self, path: Path) -> str:
        cls = self.labels.get(self._rel(path))
        tag = f"[{cls}]" if cls else "[ - ]"
        return f"{tag:>13}  {self._rel(path)}"

    def _populate_list(self) -> None:
        self.list.blockSignals(True)
        self.list.clear()
        self.list_items = {}
        for p in self.files:
            it = QListWidgetItem(self._list_text(p))
            cls = self.labels.get(self._rel(p))
            if cls and cls in CLASS_COLORS:
                it.setForeground(QColor(CLASS_COLORS[cls]))
            self.list.addItem(it)
            self.list_items[self._rel(p)] = it
        self.list.blockSignals(False)

    def _update_list_item(self, path: Path) -> None:
        it = getattr(self, "list_items", {}).get(self._rel(path))
        if it is None:
            return
        it.setText(self._list_text(path))
        cls = self.labels.get(self._rel(path))
        it.setForeground(QColor(CLASS_COLORS[cls]) if cls in CLASS_COLORS else QColor("#c8c8c8"))

    def _on_list_row(self, row: int) -> None:
        if row < 0 or row >= len(self.files):
            return
        if self.files[row] not in [self.files[i] for i in self.view]:
            return  # hidden by the "only unlabeled" filter
        self.pos = self.view.index(row)
        self._show()

    # -- navigation ---------------------------------------------------------
    def _refresh_only(self) -> None:
        only = self.chk_only.isChecked()
        self.view = [i for i in range(len(self.files))
                     if (not only) or (self._rel(self.files[i]) not in self.labels)]
        self.pos = min(self.pos, max(0, len(self.view) - 1))
        self._show()

    def _cur(self) -> Path | None:
        return self.files[self.view[self.pos]] if self.view else None

    def _step(self, d: int) -> None:
        if not self.view:
            return
        self.pos = (self.pos + d) % len(self.view)
        self._show()

    def _random(self) -> None:
        if self.view:
            self.pos = int(np.random.randint(len(self.view))); self._show()

    def _pick_src(self) -> None:
        d = QFileDialog.getExistingDirectory(self, "select data folder", str(self.src))
        if d:
            self.src = Path(d)
            self.files = sorted(self.src.rglob("*.fits"))
            self.pos = 0; self._populate_list(); self._refresh_only()

    def _pick_dst(self) -> None:
        d = QFileDialog.getExistingDirectory(self, "select output folder", str(self.dst))
        if d:
            self.dst = Path(d)
            for cls, _ in CLASSES:
                (self.dst / cls).mkdir(parents=True, exist_ok=True)
            self.labels = self._read_csv(); self._populate_list(); self._refresh_only()

    # -- classify -----------------------------------------------------------
    def _classify(self, cls: str) -> None:
        src = self._cur()
        if src is None:
            return
        outdir = self.dst / cls
        outdir.mkdir(exist_ok=True)
        dst_file = outdir / src.name
        i = 1
        while dst_file.exists():
            dst_file = outdir / f"{src.stem}_{i}{src.suffix}"; i += 1
        shutil.copy2(src, dst_file)
        self.undo_stack.append((dst_file, src, cls))
        self.labels[self._rel(src)] = cls
        self._rewrite_csv()
        self._update_list_item(src)
        if self.chk_only.isChecked():
            self._refresh_only()
        else:
            self._step(1)

    def _undo(self) -> None:
        if not self.undo_stack:
            return
        dst_file, src, cls = self.undo_stack.pop()
        try:
            dst_file.unlink()
        except FileNotFoundError:
            pass
        self.labels.pop(self._rel(src), None)
        self._rewrite_csv()
        self._update_list_item(src)
        self._refresh_only()
        if self.files.index(src) in self.view:
            self.pos = self.view.index(self.files.index(src))
        self._show()

    # -- render -------------------------------------------------------------
    def _show(self) -> None:
        src = self._cur()
        if src is None:
            for p in (self.pA, self.pAg, self.pB, self.pOv):
                p[2].clear()
            self.lbl_path.setText(f"folder: {self.src}")
            self.lbl_info.setText(""); self.lbl_status.setText("all labeled")
            self._s.setText(f"total {len(self.files)}, labeled {len(self.labels)}")
            return
        try:
            a, b, hdr = load_pair(src)
        except Exception as exc:  # noqa: BLE001
            self.lbl_info.setText(f"read failed: {exc}"); return
        nan = int(np.isnan(a).sum())

        def _set(panel, rgb):
            panel[2].setPixmap(QPixmap.fromImage(rgb_qimage(rgb)).scaled(
                PANEL, PANEL, Qt.AspectRatioMode.KeepAspectRatio,
                Qt.TransformationMode.FastTransformation))
        _set(self.pA, signed_rgb(a))
        _set(self.pAg, gray_abs(a))
        _set(self.pB, gray_of(b))
        _set(self.pOv, overlay_rgb(b, a))

        self.lbl_path.setText(
            f"src folder: {self.src}\n"
            f"out folder: {self.dst}\n"
            f"file:       {self._rel(src)}")
        self.lbl_info.setText(
            f"A mpcc={hdr.get('MPCC','?')} varc={hdr.get('VARC','?')}  "
            f"score={hdr.get('SCORE','?')}  snr={hdr.get('SNR','?')}  "
            f"pix=({hdr.get('PIXX','?')},{hdr.get('PIXY','?')})  NaN(A)={nan}")
        cls = self.labels.get(self._rel(src))
        if cls:
            self.lbl_status.setText(f"labeled: {cls}")
            self.lbl_status.setStyleSheet(f"font-size:13px; color:{CLASS_COLORS.get(cls,'#fff')};")
        else:
            self.lbl_status.setText("unlabeled")
            self.lbl_status.setStyleSheet("font-size:13px; color:#ff6b6b;")

        # sync the list selection to the current file
        it = getattr(self, "list_items", {}).get(self._rel(src))
        if it is not None:
            self.list.blockSignals(True)
            self.list.setCurrentItem(it)
            self.list.scrollToItem(it)
            self.list.blockSignals(False)
        self._s.setText(
            f"{self.pos+1}/{len(self.view)}  (total {len(self.files)}, "
            f"labeled {len(self.labels)})")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--src", type=Path, default=HERE.parent / "ai_train_16pix")
    ap.add_argument("--dst", type=Path, default=HERE.parent / "ai_labeled_16pix")
    args = ap.parse_args()
    if not args.src.exists():
        raise SystemExit(f"source not found: {args.src}")
    app = QtWidgets.QApplication(sys.argv)
    win = LabelGui(args.src, args.dst)
    win.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()

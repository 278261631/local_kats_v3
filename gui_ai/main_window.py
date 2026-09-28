#!/usr/bin/env python3
"""gui_ai 主界面（PySide6）。

功能：
    * 左侧选择根目录、文件/文件夹节点；
    * 选择文件节点 -> 处理单个 B；选择文件夹节点 -> 处理其下所有 FITS；
    * A（参考）按原版逻辑从模板根目录自动定位；
    * 重新做 WCS 重投影，把 B 映射到 A 网格后切 256 瓦片（默认 10% overlap）推理；
    * 右侧显示检测列表与 A/B 叠加预览。
"""

from __future__ import annotations

import csv
import os
import sys
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
from PySide6.QtCore import QDir, QModelIndex, Qt, QThread, Signal
from PySide6.QtGui import QColor, QImage, QPainter, QPen, QPixmap
from PySide6.QtWidgets import (
    QAbstractItemView,
    QApplication,
    QCheckBox,
    QComboBox,
    QDoubleSpinBox,
    QFileDialog,
    QFileSystemModel,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QLineEdit,
    QMainWindow,
    QMessageBox,
    QPlainTextEdit,
    QProgressBar,
    QPushButton,
    QSpinBox,
    QSplitter,
    QTableWidget,
    QTableWidgetItem,
    QTabWidget,
    QTreeView,
    QVBoxLayout,
    QWidget,
)

from config import DEFAULTS
from pair_infer import PairModel
from pipeline import collect_fits_files, process_b_file

_COLUMNS = ["文件", "x", "y", "score", "dx", "dy", "roll", "tile_x", "tile_y"]
# 掩码/峰值颜色
_DET_COLOR = QColor(255, 60, 60)
_SEL_COLOR = QColor(60, 255, 120)


def numpy_to_qimage(arr: np.ndarray) -> QImage:
    arr = np.ascontiguousarray(arr)
    h, w = arr.shape
    img = QImage(arr.data, w, h, w, QImage.Format_Grayscale8).copy()
    # 转 RGB 以便绘制彩色检测圈
    return img.convertToFormat(QImage.Format_RGB888)


class ProcessWorker(QThread):
    """后台处理线程：加载模型并对选中的文件/文件夹逐一处理。"""

    log = Signal(str)
    progress = Signal(int, int)  # current, total
    file_done = Signal(dict)
    finished_all = Signal()
    failed = Signal(str)

    def __init__(self, target: str, params: Dict) -> None:
        super().__init__()
        self.target = target
        self.params = params
        self._stop = False

    def stop(self) -> None:
        self._stop = True

    def run(self) -> None:
        try:
            files = collect_fits_files(self.target)
            if not files:
                self.failed.emit("选中的节点下没有 FITS 文件")
                self.finished_all.emit()
                return
            self.log.emit(f"待处理 B 文件: {len(files)} 个")
            self.log.emit(f"模型目录: {self.params['model_dir']}")
            model = PairModel(
                self.params["model_dir"],
                device=self.params["device"],
                det_threshold=self.params["det_threshold"],
                batch_size=self.params["batch_size"],
            )
            self.log.emit(f"模型已加载 (device={model.device}, size={model.model_size})")

            total = len(files)
            for i, f in enumerate(files, 1):
                if self._stop:
                    self.log.emit("已停止")
                    break
                self.progress.emit(i - 1, total)
                self.log.emit(f"[{i}/{total}] {os.path.basename(f)}")
                try:
                    res = process_b_file(
                        f,
                        model,
                        template_root=self.params["template_root"],
                        tile_size=self.params["tile_size"],
                        overlap=self.params["overlap"],
                        dedup_radius=self.params["dedup_radius"],
                        reproject_chunk_rows=self.params["reproject_chunk_rows"],
                        fill_invalid_with_a=self.params["fill_invalid_with_a"],
                        log_cb=self.log.emit,
                    )
                except Exception as ex:  # noqa: BLE001
                    self.log.emit(f"处理失败: {ex}")
                    continue
                self.file_done.emit(res)
            self.progress.emit(total, total)
        except Exception as ex:  # noqa: BLE001
            self.failed.emit(str(ex))
        finally:
            self.finished_all.emit()


class MainWindow(QMainWindow):
    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle("gui_ai - WCS 重投影 + PairRegNet 瞬变检测")
        self.resize(1500, 950)

        self.worker: Optional[ProcessWorker] = None
        self._results: List[Dict] = []          # 每个文件的结果
        self._row_map: List[Optional[Dict]] = []  # 表格行 -> 检测项（含 _result 引用）

        self._build_ui()

    # ------------------------------------------------------------------ UI
    def _build_ui(self) -> None:
        central = QWidget()
        self.setCentralWidget(central)
        root = QHBoxLayout(central)

        root.addWidget(self._build_left_panel(), 0)
        root.addWidget(self._build_right_panel(), 1)

    def _build_left_panel(self) -> QWidget:
        panel = QWidget()
        panel.setFixedWidth(420)
        lay = QVBoxLayout(panel)

        src = QGroupBox("数据源")
        src_lay = QVBoxLayout(src)
        row = QHBoxLayout()
        self.root_edit = QLineEdit(DEFAULTS["root"])
        browse = QPushButton("选择根目录…")
        browse.clicked.connect(self._choose_root)
        row.addWidget(self.root_edit, 1)
        row.addWidget(browse)
        src_lay.addLayout(row)

        self.fs_model = QFileSystemModel(self)
        self.fs_model.setNameFilters(["*.fit", "*.fits", "*.fts"])
        self.fs_model.setNameFilterDisables(False)
        self.fs_model.setFilter(QDir.AllDirs | QDir.NoDotAndDotDot | QDir.Files)
        self.tree = QTreeView()
        self.tree.setModel(self.fs_model)
        for c in range(1, 4):
            self.tree.hideColumn(c)
        self.tree.setSelectionMode(QAbstractItemView.SingleSelection)
        self.tree.selectionModel().currentChanged.connect(self._on_select)
        src_lay.addWidget(self.tree, 1)
        self.sel_label = QLabel("未选择节点")
        self.sel_label.setWordWrap(True)
        src_lay.addWidget(self.sel_label)
        lay.addWidget(src, 1)

        params = QGroupBox("参数")
        form = QFormLayout(params)
        self.template_edit = QLineEdit(DEFAULTS["template_root"])
        form.addRow("模板根目录", self.template_edit)
        self.model_edit = QLineEdit(DEFAULTS["model_dir"])
        form.addRow("模型目录", self.model_edit)
        self.device_combo = QComboBox()
        self.device_combo.addItems(["auto", "cpu", "cuda"])
        self.device_combo.setCurrentText(DEFAULTS["device"])
        form.addRow("设备", self.device_combo)
        self.tile_spin = QSpinBox()
        self.tile_spin.setRange(64, 1024)
        self.tile_spin.setValue(DEFAULTS["tile_size"])
        form.addRow("瓦片边长", self.tile_spin)
        self.overlap_spin = QDoubleSpinBox()
        self.overlap_spin.setRange(0.0, 0.9)
        self.overlap_spin.setSingleStep(0.05)
        self.overlap_spin.setDecimals(2)
        self.overlap_spin.setValue(DEFAULTS["overlap"])
        form.addRow("overlap 比例", self.overlap_spin)
        self.thresh_spin = QDoubleSpinBox()
        self.thresh_spin.setRange(0.0, 1.0)
        self.thresh_spin.setSingleStep(0.05)
        self.thresh_spin.setDecimals(2)
        self.thresh_spin.setValue(DEFAULTS["det_threshold"])
        form.addRow("检测阈值", self.thresh_spin)
        self.batch_spin = QSpinBox()
        self.batch_spin.setRange(1, 128)
        self.batch_spin.setValue(DEFAULTS["batch_size"])
        form.addRow("批大小", self.batch_spin)
        self.fill_check = QCheckBox("B 未覆盖区域用 A 填充")
        self.fill_check.setChecked(DEFAULTS["fill_invalid_with_a"])
        form.addRow("", self.fill_check)
        lay.addWidget(params)

        btns = QHBoxLayout()
        self.run_btn = QPushButton("开始处理")
        self.run_btn.clicked.connect(self._start)
        self.stop_btn = QPushButton("停止")
        self.stop_btn.setEnabled(False)
        self.stop_btn.clicked.connect(self._stop)
        self.export_btn = QPushButton("导出检测 CSV")
        self.export_btn.clicked.connect(self._export_csv)
        btns.addWidget(self.run_btn)
        btns.addWidget(self.stop_btn)
        btns.addWidget(self.export_btn)
        lay.addLayout(btns)

        self.progress = QProgressBar()
        self.progress.setValue(0)
        lay.addWidget(self.progress)

        return panel

    def _build_right_panel(self) -> QWidget:
        splitter = QSplitter(Qt.Vertical)

        preview_box = QGroupBox("预览（左=A 参考，右=B 重投影，红圈=检测，绿=当前）")
        pl = QHBoxLayout(preview_box)
        self.preview_a = QLabel("A")
        self.preview_b = QLabel("B")
        for lbl in (self.preview_a, self.preview_b):
            lbl.setAlignment(Qt.AlignCenter)
            lbl.setMinimumHeight(300)
            lbl.setStyleSheet("background:#111; color:#888;")
            pl.addWidget(lbl, 1)

        preview_wrap = QWidget()
        pw = QVBoxLayout(preview_wrap)
        pw.setContentsMargins(0, 0, 0, 0)
        pw.addWidget(preview_box)

        self.view_combo = QComboBox()
        self.view_combo.addItems(["全图叠加", "选中目标裁切"])
        self.view_combo.currentIndexChanged.connect(lambda _=0: self._refresh_preview())
        vrow = QHBoxLayout()
        vrow.addWidget(QLabel("预览模式:"))
        vrow.addWidget(self.view_combo)
        vrow.addStretch(1)
        pw.addLayout(vrow)

        tabs = QTabWidget()
        self.table = QTableWidget(0, len(_COLUMNS))
        self.table.setHorizontalHeaderLabels(_COLUMNS)
        self.table.horizontalHeader().setSectionResizeMode(QHeaderView.Interactive)
        self.table.horizontalHeader().setStretchLastSection(True)
        self.table.setSelectionBehavior(QAbstractItemView.SelectRows)
        self.table.setEditTriggers(QAbstractItemView.NoEditTriggers)
        self.table.itemSelectionChanged.connect(self._on_row_selected)
        tabs.addTab(self.table, "检测结果")

        self.log_view = QPlainTextEdit()
        self.log_view.setReadOnly(True)
        self.log_view.setMaximumBlockCount(5000)
        tabs.addTab(self.log_view, "运行日志")

        splitter.addWidget(preview_wrap)
        splitter.addWidget(tabs)
        splitter.setStretchFactor(0, 3)
        splitter.setStretchFactor(1, 2)
        return splitter

    # --------------------------------------------------------------- events
    def _choose_root(self) -> None:
        d = QFileDialog.getExistingDirectory(self, "选择根目录", self.root_edit.text())
        if d:
            self.root_edit.setText(d)
            self._set_root(d)

    def _set_root(self, root: str) -> None:
        if not os.path.isdir(root):
            return
        self.fs_model.setRootPath(root)
        self.tree.setRootIndex(self.fs_model.index(root))
        self.tree.setColumnWidth(0, 260)

    def _on_select(self, current: QModelIndex, _prev: QModelIndex) -> None:
        if not current.isValid():
            return
        path = self.fs_model.filePath(current)
        kind = "文件夹" if os.path.isdir(path) else "文件"
        self.sel_label.setText(f"{kind}: {path}")

    def _log(self, msg: str) -> None:
        self.log_view.appendPlainText(msg)

    def _params(self) -> Dict:
        return {
            "template_root": self.template_edit.text().strip(),
            "model_dir": self.model_edit.text().strip(),
            "device": self.device_combo.currentText(),
            "tile_size": int(self.tile_spin.value()),
            "overlap": float(self.overlap_spin.value()),
            "det_threshold": float(self.thresh_spin.value()),
            "batch_size": int(self.batch_spin.value()),
            "dedup_radius": float(DEFAULTS["dedup_radius"]),
            "reproject_chunk_rows": int(DEFAULTS["reproject_chunk_rows"]),
            "fill_invalid_with_a": bool(self.fill_check.isChecked()),
        }

    # -------------------------------------------------------------- process
    def _start(self) -> None:
        idx = self.tree.currentIndex()
        if not idx.isValid():
            QMessageBox.warning(self, "提示", "请先在左侧选择文件或文件夹节点")
            return
        target = self.fs_model.filePath(idx)
        params = self._params()
        if not os.path.isdir(params["template_root"]):
            QMessageBox.warning(self, "提示", f"模板根目录不存在: {params['template_root']}")
            return
        if not os.path.isdir(params["model_dir"]):
            QMessageBox.warning(self, "提示", f"模型目录不存在: {params['model_dir']}")
            return

        self._results.clear()
        self._row_map.clear()
        self.table.setRowCount(0)
        self.run_btn.setEnabled(False)
        self.stop_btn.setEnabled(True)
        self._log(f"==== 开始处理: {target} ====")

        self.worker = ProcessWorker(target, params)
        self.worker.log.connect(self._log)
        self.worker.progress.connect(self._on_progress)
        self.worker.file_done.connect(self._on_file_done)
        self.worker.failed.connect(lambda m: QMessageBox.critical(self, "错误", m))
        self.worker.finished_all.connect(self._on_finished)
        self.worker.start()

    def _stop(self) -> None:
        if self.worker:
            self.worker.stop()
            self._log("请求停止…")

    def _on_progress(self, cur: int, total: int) -> None:
        self.progress.setMaximum(max(1, total))
        self.progress.setValue(cur)

    def _on_finished(self) -> None:
        self.run_btn.setEnabled(True)
        self.stop_btn.setEnabled(False)
        self._log(f"==== 处理结束，命中检测 {self.table.rowCount()} 条 ====")

    def _on_file_done(self, res: Dict) -> None:
        self._results.append(res)
        for d in res["detections"]:
            row = self.table.rowCount()
            self.table.insertRow(row)
            name = os.path.basename(res["b_path"])
            vals = [
                name, f"{d['x']:.1f}", f"{d['y']:.1f}", f"{d['score']:.3f}",
                f"{d['dx']:+.2f}", f"{d['dy']:+.2f}", f"{d['roll']:+.2f}",
                str(d["tile_x"]), str(d["tile_y"]),
            ]
            for c, v in enumerate(vals):
                self.table.setItem(row, c, QTableWidgetItem(v))
            entry = dict(d)
            entry["_result"] = res
            self._row_map.append(entry)
        # 每个文件处理完都刷新一次预览（无选中行时显示该文件全图叠加）
        self._refresh_preview()

    # --------------------------------------------------------------- preview
    def _selected_entry(self) -> Optional[Dict]:
        rows = self.table.selectionModel().selectedRows() if self.table.selectionModel() else []
        if not rows:
            return None
        r = rows[0].row()
        if 0 <= r < len(self._row_map):
            return self._row_map[r]
        return None

    def _on_row_selected(self) -> None:
        self._refresh_preview()

    def _current_result(self) -> Optional[Dict]:
        entry = self._selected_entry()
        if entry is not None:
            return entry["_result"]
        return self._results[-1] if self._results else None

    @staticmethod
    def _draw_marker(img: QImage, x: float, y: float, color: QColor, r: int = 8) -> None:
        p = QPainter(img)
        p.setPen(QPen(color, 2))
        p.drawEllipse(int(x) - r, int(y) - r, 2 * r, 2 * r)
        p.end()

    def _refresh_preview(self) -> None:
        res = self._current_result()
        if not res or res.get("a_u8") is None:
            return
        a, b, s = res["a_u8"], res["b_u8"], res["preview_scale"]

        a_img = numpy_to_qimage(a)
        b_img = numpy_to_qimage(b)

        entry = self._selected_entry()
        mode_crop = self.view_combo.currentIndex() == 1 and entry is not None

        if mode_crop:
            cx, cy = entry["x"] * s, entry["y"] * s
            rad = 140
            x0 = int(max(0, min(a.shape[1] - 1, cx - rad)))
            y0 = int(max(0, min(a.shape[0] - 1, cy - rad)))
            x1 = int(min(a.shape[1], x0 + 2 * rad))
            y1 = int(min(a.shape[0], y0 + 2 * rad))
            a_img = a_img.copy(x0, y0, x1 - x0, y1 - y0)
            b_img = b_img.copy(x0, y0, x1 - x0, y1 - y0)
            self._draw_marker(b_img, cx - x0, cy - y0, _SEL_COLOR)
        else:
            for d in res["detections"]:
                self._draw_marker(b_img, d["x"] * s, d["y"] * s, _DET_COLOR)
            if entry is not None and entry["_result"] is res:
                self._draw_marker(b_img, entry["x"] * s, entry["y"] * s, _SEL_COLOR)

        self.preview_a.setPixmap(QPixmap.fromImage(a_img).scaled(
            self.preview_a.width(), self.preview_a.height(),
            Qt.KeepAspectRatio, Qt.SmoothTransformation))
        self.preview_b.setPixmap(QPixmap.fromImage(b_img).scaled(
            self.preview_b.width(), self.preview_b.height(),
            Qt.KeepAspectRatio, Qt.SmoothTransformation))

    # ---------------------------------------------------------------- export
    def _export_csv(self) -> None:
        if self.table.rowCount() == 0:
            QMessageBox.information(self, "提示", "没有可导出的检测")
            return
        path, _ = QFileDialog.getSaveFileName(
            self, "导出检测 CSV", "gui_ai_detections.csv", "CSV (*.csv)"
        )
        if not path:
            return
        with open(path, "w", newline="", encoding="utf-8-sig") as f:
            w = csv.writer(f)
            w.writerow(["file", "x", "y", "score", "dx", "dy", "roll", "tile_x", "tile_y"])
            for e in self._row_map:
                w.writerow([
                    os.path.basename(e.get("_result", {}).get("b_path", "")),
                    f"{e['x']:.2f}", f"{e['y']:.2f}", f"{e['score']:.4f}",
                    f"{e['dx']:.3f}", f"{e['dy']:.3f}", f"{e['roll']:.3f}",
                    e["tile_x"], e["tile_y"],
                ])
        self._log(f"已导出 CSV: {path}")


def main() -> None:
    app = QApplication(sys.argv)
    win = MainWindow()
    default_root = DEFAULTS["root"]
    if not os.path.isdir(default_root):
        default_root = str(Path.home())
    win._set_root(default_root)
    win.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()

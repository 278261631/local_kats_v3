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
import json
import os
import sys
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
from PySide6.QtCore import QDir, QModelIndex, QPoint, Qt, QThread, Signal
from PySide6.QtGui import (
    QColor,
    QImage,
    QKeySequence,
    QMouseEvent,
    QPainter,
    QPen,
    QPixmap,
    QPolygon,
    QShortcut,
)
from PySide6.QtWidgets import (
    QAbstractItemView,
    QApplication,
    QCheckBox,
    QComboBox,
    QDoubleSpinBox,
    QFileDialog,
    QFileSystemModel,
    QFormLayout,
    QFrame,
    QGridLayout,
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
    QScrollArea,
    QSpinBox,
    QSplitter,
    QTableWidget,
    QTableWidgetItem,
    QTabWidget,
    QTreeView,
    QVBoxLayout,
    QWidget,
)

from config import DEFAULTS, load_settings, save_settings
from native_crop import FitsCache, load_native_pair_crops
from pair_infer import PairModel
from pipeline import collect_fits_files, process_b_file
import results_io

_COLUMNS = ["文件", "状态", "x", "y", "score", "dx", "dy", "roll", "tile_x", "tile_y"]
_DAILY_COLUMNS = ["日期", "系统", "天区", "文件", "状态", "x", "y", "score", "dx", "dy", "roll"]
# 掩码/峰值颜色
_DET_COLOR = QColor(255, 60, 60)
_SEL_COLOR = QColor(60, 255, 120)

_STATUS_TEXT = {
    "keep": "命中",
    "satellite": "卫星",
    "b_uncovered": "B未覆盖",
    "a_invalid": "A无效区",
    "dedup": "重复",
}
_STATUS_COLOR = {
    "satellite": QColor(255, 170, 80),
    "b_uncovered": QColor(150, 150, 200),
    "a_invalid": QColor(150, 150, 200),
    "dedup": QColor(160, 160, 160),
}


def _det_status_color(status: str) -> QColor:
    return _DET_COLOR if status == "keep" else _STATUS_COLOR.get(status, _DET_COLOR)


# 卫星掩码叠加颜色
_SAT_RGB = (255, 170, 0)
# B 覆盖区（WCS 对齐后）边框颜色
_COVER_EDGE_RGB = (255, 230, 0)
# A 模板有效区（星空/空白）边界
_A_VALID_COLOR = QColor(255, 255, 255)
# B（重投影后）有效区边界
_B_VALID_COLOR = QColor(0, 220, 180)


def _swatch(color) -> str:
    c = QColor(*color) if isinstance(color, tuple) else color
    return f'<span style="color:{c.name()}">\u25a0</span>'


def numpy_to_qimage(arr: np.ndarray) -> QImage:
    arr = np.ascontiguousarray(arr)
    h, w = arr.shape
    img = QImage(arr.data, w, h, w, QImage.Format_Grayscale8).copy()
    # 转 RGB 以便绘制彩色检测圈
    return img.convertToFormat(QImage.Format_RGB888)


def rgb_to_qimage(rgb: np.ndarray) -> QImage:
    rgb = np.ascontiguousarray(rgb)
    h, w, _ = rgb.shape
    return QImage(rgb.data, w, h, rgb.strides[0], QImage.Format_RGB888).copy()


def gray_to_rgb(gray: np.ndarray) -> np.ndarray:
    return np.repeat(gray[:, :, None], 3, axis=2)


def overlay_ob(rgb: np.ndarray, ob: np.ndarray, channels: Dict[int, tuple]) -> np.ndarray:
    """把 ob 掩码以半透明色叠加到 RGB 图上。channels: {通道: (r,g,b)}。"""
    out = rgb.astype(np.float32)
    for c, col in channels.items():
        if c >= ob.shape[2]:
            continue
        a = (ob[:, :, c].astype(np.float32) / 255.0 * 0.5)[:, :, None]
        colv = np.array(col, dtype=np.float32)[None, None, :]
        out = out * (1.0 - a) + colv * a
    return np.clip(out, 0.0, 255.0).astype(np.uint8)


def _mask_edge(mask: np.ndarray, thicken: bool = True) -> np.ndarray:
    """布尔掩码的边界（内边缘 + 可选加粗 1px）。"""
    m = mask.astype(bool)
    er = m.copy()
    er[1:, :] &= m[:-1, :]
    er[:-1, :] &= m[1:, :]
    er[:, 1:] &= m[:, :-1]
    er[:, :-1] &= m[:, 1:]
    edge = m & ~er
    if thicken:
        e = edge.copy()
        e[1:, :] |= edge[:-1, :]
        e[:-1, :] |= edge[1:, :]
        e[:, 1:] |= edge[:, :-1]
        e[:, :-1] |= edge[:, 1:]
        edge = e
    return edge


def paint_mask(rgb: np.ndarray, mask: np.ndarray, color: tuple) -> np.ndarray:
    out = rgb.copy()
    out[mask.astype(bool)] = np.array(color, dtype=np.uint8)
    return out


def _mtf(m: float, x: np.ndarray) -> np.ndarray:
    """中间调传递函数 (Midtone Transfer Function)。m=0.5 时为恒等。"""
    denom = (2.0 * m - 1.0) * x - m
    denom = np.where(np.abs(denom) < 1e-8, 1e-8, denom)
    return np.clip(((m - 1.0) * x) / denom, 0.0, 1.0)


def _mtf_solve(x: float, target: float) -> float:
    """求 m 使 _mtf(m, x) == target（x、target 均已归一化到 (0,1)）。"""
    if abs(x - target) < 1e-6:
        return 0.5
    m = x * (1.0 - target) / (x - target * (2.0 * x - 1.0))
    return float(min(1.0 - 1e-4, max(1e-4, m)))


def _linear_u8(arr: np.ndarray) -> np.ndarray:
    """局部线性百分位(1/99.5)拉伸；NaN 视为背景。用于裁切预览。"""
    x = np.asarray(arr, dtype=np.float32)
    finite = np.isfinite(x)
    if not finite.any():
        return np.zeros(x.shape, dtype=np.uint8)
    v = x[finite]
    lo, hi = np.percentile(v, (1.0, 99.5))
    if hi - lo <= 1e-6:
        lo, hi = float(v.min()), float(v.max())
    if hi - lo <= 1e-6:
        return np.zeros(x.shape, dtype=np.uint8)
    n = (np.nan_to_num(x, nan=lo) - lo) / (hi - lo)
    return np.clip(n * 255.0, 0.0, 255.0).astype(np.uint8)


def _stretch_u8(arr: np.ndarray, target_bg: float = 0.25,
                shadow_clip: float = -2.8, highlight_pct: float = 99.8) -> np.ndarray:
    """STF 式自动拉伸（背景锚定 + 中间调传递），NaN 视为背景。

    - 黑点：median - k*MAD（背景锚定，抗亮星/热点影响）
    - 白点：高百分位
    - 中间调：把背景映射到 target_bg，非线性提升暗弱细节
    """
    x = np.asarray(arr, dtype=np.float32)
    finite = np.isfinite(x)
    if not finite.any():
        return np.zeros(x.shape, dtype=np.uint8)
    v = x[finite]
    med = float(np.median(v))
    mad = float(np.median(np.abs(v - med)))
    sigma = 1.4826 * mad
    if sigma > 0:
        lo = med + shadow_clip * sigma
    else:
        lo = float(np.percentile(v, 0.1))
    hi = float(np.percentile(v, highlight_pct))
    if hi - lo <= 1e-6:
        lo, hi = float(v.min()), float(v.max())
    if hi - lo <= 1e-6:
        return np.zeros(x.shape, dtype=np.uint8)
    n = np.clip((np.nan_to_num(x, nan=lo) - lo) / (hi - lo), 0.0, 1.0)
    bg = float(np.clip((med - lo) / (hi - lo), 1e-4, 1.0 - 1e-4))
    m = _mtf_solve(bg, target_bg)
    return np.clip(_mtf(m, n) * 255.0, 0.0, 255.0).astype(np.uint8)


class _ThumbLabel(QLabel):
    """小图墙单元格：双击发出信号。"""

    double_clicked = Signal()

    def mouseDoubleClickEvent(self, event: QMouseEvent) -> None:  # noqa: N802
        self.double_clicked.emit()
        super().mouseDoubleClickEvent(event)


class ClickableLabel(QLabel):
    """可点击的图片标签，发出点击处在图像像素坐标系中的坐标。"""

    clicked = Signal(float, float)
    double_clicked = Signal(float, float)

    def __init__(self, text: str = "") -> None:
        super().__init__(text)
        self._img_w = 0
        self._img_h = 0

    def set_image_size(self, w: int, h: int) -> None:
        self._img_w = int(w)
        self._img_h = int(h)

    def _map_pos(self, event: QMouseEvent):
        pm = self.pixmap()
        if pm is None or pm.isNull() or self._img_w <= 0 or self._img_h <= 0:
            return None
        pw, ph = pm.width(), pm.height()
        ox = (self.width() - pw) / 2.0
        oy = (self.height() - ph) / 2.0
        x = event.position().x() - ox
        y = event.position().y() - oy
        if 0 <= x < pw and 0 <= y < ph:
            return (x * self._img_w / pw, y * self._img_h / ph)
        return None

    def mousePressEvent(self, event: QMouseEvent) -> None:  # noqa: N802
        p = self._map_pos(event)
        if p is not None:
            self.clicked.emit(*p)
        super().mousePressEvent(event)

    def mouseDoubleClickEvent(self, event: QMouseEvent) -> None:  # noqa: N802
        p = self._map_pos(event)
        if p is not None:
            self.double_clicked.emit(*p)
        super().mouseDoubleClickEvent(event)


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
                        valid_overlap_filter=self.params.get("valid_overlap_filter", True),
                        log_cb=self.log.emit,
                    )
                except Exception as ex:  # noqa: BLE001
                    self.log.emit(f"处理失败: {ex}")
                    continue
                try:
                    results_io.save_result(res, self.params)
                    self.log.emit(f"  已保存结果: {os.path.basename(res['b_path'])}{results_io.SUFFIX_JSON}")
                except Exception as ex:  # noqa: BLE001
                    self.log.emit(f"  结果保存失败: {ex}")
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
        self._fits_cache = FitsCache(max_items=2)  # 原生裁块用（A/B 各一）
        self._manual_center: Optional[tuple] = None  # 双击指定的裁切中心(原生坐标)
        self._manual_result = None
        self._daily_rows: List[list] = []

        self._build_ui()
        self._settings = load_settings()
        self._apply_settings()

        # Tab 在 全图叠加 / 选中目标裁切 之间切换
        self.toggle_shortcut = QShortcut(QKeySequence(Qt.Key_Tab), self)
        self.toggle_shortcut.activated.connect(self._toggle_view)

    def _apply_settings(self) -> None:
        s = self._settings
        self.root_edit.setText(s.get("root", DEFAULTS["root"]))
        self.template_edit.setText(s.get("template_root", DEFAULTS["template_root"]))
        self.model_edit.setText(s.get("model_dir", DEFAULTS["model_dir"]))
        self.device_combo.setCurrentText(s.get("device", DEFAULTS["device"]))
        self.tile_spin.setValue(int(s.get("tile_size", DEFAULTS["tile_size"])))
        self.overlap_spin.setValue(float(s.get("overlap", DEFAULTS["overlap"])))
        self.thresh_spin.setValue(float(s.get("det_threshold", DEFAULTS["det_threshold"])))
        self.batch_spin.setValue(int(s.get("batch_size", DEFAULTS["batch_size"])))
        self.crop_spin.setValue(int(s.get("crop_size", DEFAULTS["crop_size"])))
        self.fill_check.setChecked(
            bool(s.get("fill_invalid_with_a", DEFAULTS["fill_invalid_with_a"])))
        self.valid_overlap_check.setChecked(
            bool(s.get("valid_overlap_filter", DEFAULTS["valid_overlap_filter"])))
        self.show_filtered_check.setChecked(bool(s.get("show_filtered", False)))
        self.sat_check.setChecked(bool(s.get("show_sat", True)))
        self.cover_check.setChecked(bool(s.get("show_cover", True)))
        self.validpoly_check.setChecked(bool(s.get("show_a_validpoly", True)))
        self.validpoly_b_check.setChecked(bool(s.get("show_b_validpoly", True)))
        self.view_combo.setCurrentIndex(int(s.get("view_mode", 1)))

    def _collect_settings(self) -> dict:
        return {
            "root": self.root_edit.text().strip(),
            "template_root": self.template_edit.text().strip(),
            "model_dir": self.model_edit.text().strip(),
            "device": self.device_combo.currentText(),
            "tile_size": int(self.tile_spin.value()),
            "overlap": float(self.overlap_spin.value()),
            "det_threshold": float(self.thresh_spin.value()),
            "batch_size": int(self.batch_spin.value()),
            "crop_size": int(self.crop_spin.value()),
            "fill_invalid_with_a": bool(self.fill_check.isChecked()),
            "valid_overlap_filter": bool(self.valid_overlap_check.isChecked()),
            "show_filtered": bool(self.show_filtered_check.isChecked()),
            "show_sat": bool(self.sat_check.isChecked()),
            "show_cover": bool(self.cover_check.isChecked()),
            "show_a_validpoly": bool(self.validpoly_check.isChecked()),
            "show_b_validpoly": bool(self.validpoly_b_check.isChecked()),
            "view_mode": int(self.view_combo.currentIndex()),
        }

    def closeEvent(self, event) -> None:  # noqa: N802
        save_settings(self._collect_settings())
        super().closeEvent(event)

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
        self.valid_overlap_check = QCheckBox("按A∩B有效区过滤检测")
        self.valid_overlap_check.setChecked(DEFAULTS["valid_overlap_filter"])
        form.addRow("", self.valid_overlap_check)
        lay.addWidget(params)

        btns = QHBoxLayout()
        self.run_btn = QPushButton("开始处理")
        self.run_btn.clicked.connect(self._start)
        self.stop_btn = QPushButton("停止")
        self.stop_btn.setEnabled(False)
        self.stop_btn.clicked.connect(self._stop)
        self.export_btn = QPushButton("导出检测 CSV")
        self.export_btn.clicked.connect(self._export_csv)
        self.load_btn = QPushButton("加载已有结果")
        self.load_btn.clicked.connect(self._load_results_from_node)
        btns.addWidget(self.run_btn)
        btns.addWidget(self.stop_btn)
        btns.addWidget(self.load_btn)
        btns.addWidget(self.export_btn)
        lay.addLayout(btns)

        self.progress = QProgressBar()
        self.progress.setValue(0)
        lay.addWidget(self.progress)

        return panel

    def _build_right_panel(self) -> QWidget:
        splitter = QSplitter(Qt.Vertical)

        preview_box = QGroupBox("预览（左=A 参考，右=B 重投影；红=检测，绿=当前；空心十字标注）")
        pl = QHBoxLayout(preview_box)
        self.preview_a = ClickableLabel("A")
        self.preview_b = ClickableLabel("B")
        for lbl in (self.preview_a, self.preview_b):
            lbl.setAlignment(Qt.AlignCenter)
            lbl.setMinimumHeight(300)
            lbl.setStyleSheet("background:#111; color:#888;")
            lbl.setCursor(Qt.PointingHandCursor)
            lbl.clicked.connect(self._on_preview_click)
            lbl.double_clicked.connect(self._on_preview_double_click)
            pl.addWidget(lbl, 1)

        preview_wrap = QWidget()
        pw = QVBoxLayout(preview_wrap)
        pw.setContentsMargins(0, 0, 0, 0)
        pw.addWidget(preview_box)

        self.view_combo = QComboBox()
        self.view_combo.addItems(["全图叠加", "选中目标裁切"])
        self.view_combo.setCurrentIndex(1)
        self.view_combo.currentIndexChanged.connect(lambda _=0: self._refresh_preview())
        vrow = QHBoxLayout()
        vrow.addWidget(QLabel("预览模式:"))
        vrow.addWidget(self.view_combo)
        vrow.addSpacing(12)
        vrow.addWidget(QLabel("裁切像素:"))
        self.crop_spin = QSpinBox()
        self.crop_spin.setRange(32, 4096)
        self.crop_spin.setSingleStep(64)
        self.crop_spin.setValue(DEFAULTS["crop_size"])
        self.crop_spin.valueChanged.connect(lambda _=0: self._refresh_preview())
        vrow.addWidget(self.crop_spin)
        hint = QLabel("（Tab 切换；全图点击十字→裁切；全图双击任意处→看该处局部对比）")
        hint.setStyleSheet("color:#666;")
        vrow.addWidget(hint)
        vrow.addSpacing(12)
        self.show_filtered_check = QCheckBox("显示被过滤结果")
        self.show_filtered_check.setChecked(False)
        self.show_filtered_check.stateChanged.connect(lambda _=0: self._rebuild_table())
        vrow.addWidget(self.show_filtered_check)
        self.sat_check = QCheckBox("叠加卫星掩码")
        self.sat_check.setChecked(True)
        self.sat_check.stateChanged.connect(lambda _=0: self._refresh_preview())
        vrow.addWidget(self.sat_check)
        self.cover_check = QCheckBox("显示 B 覆盖边框")
        self.cover_check.setChecked(True)
        self.cover_check.stateChanged.connect(lambda _=0: self._refresh_preview())
        vrow.addWidget(self.cover_check)
        self.validpoly_check = QCheckBox("显示A有效区边界")
        self.validpoly_check.setChecked(True)
        self.validpoly_check.stateChanged.connect(lambda _=0: self._refresh_preview())
        vrow.addWidget(self.validpoly_check)
        self.validpoly_b_check = QCheckBox("显示B有效区边界")
        self.validpoly_b_check.setChecked(True)
        self.validpoly_b_check.stateChanged.connect(lambda _=0: self._refresh_preview())
        vrow.addWidget(self.validpoly_b_check)
        vrow.addStretch(1)
        pw.addLayout(vrow)

        legend = QLabel()
        legend.setTextFormat(Qt.RichText)
        legend.setWordWrap(True)
        legend.setStyleSheet("font-size:11px; color:#333; padding:2px;")
        legend.setText(
            "<b>标注颜色：</b>"
            + _swatch(_DET_COLOR) + " 命中(keep)　"
            + _swatch(_STATUS_COLOR["satellite"]) + " 卫星(satellite)　"
            + _swatch(_STATUS_COLOR["b_uncovered"]) + " B未覆盖　"
            + _swatch(_STATUS_COLOR["dedup"]) + " 重复(dedup)　"
            + _swatch(_SEL_COLOR) + " 当前选中"
            + "　　|　　<b>叠加：</b>"
            + _swatch(_SAT_RGB) + " 卫星掩码　"
            + _swatch(_COVER_EDGE_RGB) + " B覆盖边框"
            + "　　|　　<b>有效区边界：</b>"
            + _swatch(_A_VALID_COLOR) + " A有效区　"
            + _swatch(_B_VALID_COLOR) + " B有效区"
        )
        pw.addWidget(legend)

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

        montage_wrap = QWidget()
        mv = QVBoxLayout(montage_wrap)
        mrow = QHBoxLayout()
        self.montage_btn = QPushButton("生成小图墙")
        self.montage_btn.clicked.connect(lambda: self._build_montage())
        self.montage_info = QLabel("（每个检测一张 A|B 裁切；双击单元格可跳转到该行）")
        self.montage_info.setStyleSheet("color:#666;")
        mrow.addWidget(self.montage_btn)
        mrow.addWidget(self.montage_info)
        mrow.addStretch(1)
        mv.addLayout(mrow)
        self.montage_area = QScrollArea()
        self.montage_area.setWidgetResizable(True)
        self.montage_container = QWidget()
        self.montage_grid = QGridLayout(self.montage_container)
        self.montage_area.setWidget(self.montage_container)
        mv.addWidget(self.montage_area)
        tabs.addTab(montage_wrap, "小图墙")

        # 每日总表
        daily_wrap = QWidget()
        dv = QVBoxLayout(daily_wrap)
        drow = QHBoxLayout()
        self.daily_btn = QPushButton("汇总选中目录(按天)")
        self.daily_btn.clicked.connect(self._build_daily_summary)
        self.daily_export_btn = QPushButton("导出总表 CSV")
        self.daily_export_btn.clicked.connect(self._export_daily_csv)
        self.daily_info = QLabel("（扫描选中目录下所有 *.gui_ai.json 汇总）")
        self.daily_info.setStyleSheet("color:#666;")
        drow.addWidget(self.daily_btn)
        drow.addWidget(self.daily_export_btn)
        drow.addWidget(self.daily_info)
        drow.addStretch(1)
        dv.addLayout(drow)
        self.daily_table = QTableWidget(0, len(_DAILY_COLUMNS))
        self.daily_table.setHorizontalHeaderLabels(_DAILY_COLUMNS)
        self.daily_table.horizontalHeader().setSectionResizeMode(QHeaderView.Interactive)
        self.daily_table.horizontalHeader().setStretchLastSection(True)
        self.daily_table.setEditTriggers(QAbstractItemView.NoEditTriggers)
        dv.addWidget(self.daily_table)
        tabs.addTab(daily_wrap, "每日总表")

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
            save_settings(self._collect_settings())

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
            "valid_overlap_filter": bool(self.valid_overlap_check.isChecked()),
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
        self._manual_center = None
        self._manual_result = None
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
        n_keep = sum(int(r.get("n_keep", 0)) for r in self._results)
        self._log(
            f"==== 处理结束，命中 {n_keep} 条（表格显示 {self.table.rowCount()} 行）===="
        )

    def _append_detection_row(self, res: Dict, d: Dict) -> None:
        row = self.table.rowCount()
        self.table.insertRow(row)
        status = d.get("status", "keep")
        vals = [
            os.path.basename(res.get("b_path") or ""),
            _STATUS_TEXT.get(status, status),
            f"{d['x']:.1f}", f"{d['y']:.1f}", f"{d['score']:.3f}",
            f"{d['dx']:+.2f}", f"{d['dy']:+.2f}", f"{d['roll']:+.2f}",
            str(d["tile_x"]), str(d["tile_y"]),
        ]
        for c, v in enumerate(vals):
            item = QTableWidgetItem(v)
            if c == 1 and status in _STATUS_COLOR:
                item.setForeground(_STATUS_COLOR[status])
            self.table.setItem(row, c, item)
        entry = dict(d)
        entry["_result"] = res
        entry["_det"] = d
        self._row_map.append(entry)

    def _rebuild_table(self) -> None:
        self.table.setRowCount(0)
        self._row_map.clear()
        show_all = self.show_filtered_check.isChecked()
        for res in self._results:
            for d in res["detections"]:
                if not show_all and d.get("status", "keep") != "keep":
                    continue
                self._append_detection_row(res, d)
        self._refresh_preview()

    def _on_file_done(self, res: Dict) -> None:
        self._results.append(res)
        show_all = self.show_filtered_check.isChecked()
        for d in res["detections"]:
            if not show_all and d.get("status", "keep") != "keep":
                continue
            self._append_detection_row(res, d)
        # 每个文件处理完都刷新一次预览（无选中行时显示该文件全图叠加）
        self._refresh_preview()

    # ---------------------------------------------------------- 结果持久化
    def _load_results_from_node(self) -> None:
        idx = self.tree.currentIndex()
        if not idx.isValid():
            QMessageBox.warning(self, "提示", "请先在左侧选择文件或文件夹节点")
            return
        target = self.fs_model.filePath(idx)
        jsons = results_io.scan_results(target)
        if not jsons:
            QMessageBox.information(
                self, "提示", f"未找到结果文件 (*{results_io.SUFFIX_JSON})")
            return
        self._results.clear()
        self._row_map.clear()
        self._manual_center = None
        self._manual_result = None
        self.table.setRowCount(0)
        loaded = 0
        for j in jsons:
            try:
                res = results_io.load_result(j)
            except Exception as ex:  # noqa: BLE001
                self._log(f"加载失败 {j}: {ex}")
                continue
            self._on_file_done(res)
            loaded += 1
        self._log(f"已加载 {loaded} 个结果 (来自 {target})")

    def _build_daily_summary(self) -> None:
        idx = self.tree.currentIndex()
        target = (self.fs_model.filePath(idx) if idx.isValid()
                  else self.root_edit.text().strip())
        jsons = results_io.scan_results(target)
        self._daily_rows = []
        for j in jsons:
            info = results_io.parse_path_info(j)
            try:
                d = json.loads(Path(j).read_text(encoding="utf-8"))
            except Exception:
                continue
            fname = os.path.basename(d.get("b_path") or j)
            for det in d.get("detections", []):
                st = det.get("status", "keep")
                self._daily_rows.append([
                    info["date"], info["tel"], info["region"], fname,
                    _STATUS_TEXT.get(st, st),
                    f"{det.get('x', 0):.1f}", f"{det.get('y', 0):.1f}",
                    f"{det.get('score', 0):.3f}",
                    f"{det.get('dx', 0):+.2f}", f"{det.get('dy', 0):+.2f}",
                    f"{det.get('roll', 0):+.2f}",
                ])
        self.daily_table.setRowCount(len(self._daily_rows))
        for r, row in enumerate(self._daily_rows):
            for c, v in enumerate(row):
                self.daily_table.setItem(r, c, QTableWidgetItem(str(v)))
        self.daily_info.setText(
            f"（{len(jsons)} 个结果文件, {len(self._daily_rows)} 条检测）")
        self._log(f"每日总表: {len(jsons)} 个结果文件, {len(self._daily_rows)} 条检测")

    def _export_daily_csv(self) -> None:
        if not self._daily_rows:
            QMessageBox.information(self, "提示", "总表为空，请先汇总")
            return
        path, _ = QFileDialog.getSaveFileName(
            self, "导出总表 CSV", "gui_ai_daily.csv", "CSV (*.csv)")
        if not path:
            return
        with open(path, "w", newline="", encoding="utf-8-sig") as f:
            w = csv.writer(f)
            w.writerow(_DAILY_COLUMNS)
            w.writerows(self._daily_rows)
        self._log(f"已导出总表: {path}")

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
        self._manual_center = None
        self._manual_result = None
        self._refresh_preview()

    def _toggle_view(self) -> None:
        self.view_combo.setCurrentIndex(0 if self.view_combo.currentIndex() == 1 else 1)

    def _on_preview_click(self, ix: float, iy: float) -> None:
        """全图模式下点击到十字标记附近时，选中该目标并切到裁切模式。"""
        if self.view_combo.currentIndex() != 0:
            return
        res = self._current_result()
        if not res or res.get("a_u8") is None:
            return
        s = res["preview_scale"]
        r2 = 30.0 ** 2  # 预览像素命中半径
        best_row: Optional[int] = None
        best_d2: Optional[float] = None
        for row, e in enumerate(self._row_map):
            if e.get("_result") is not res:
                continue
            d2 = (e["x"] * s - ix) ** 2 + (e["y"] * s - iy) ** 2
            if best_d2 is None or d2 < best_d2:
                best_d2, best_row = d2, row
        if best_row is not None and best_d2 is not None and best_d2 <= r2:
            self.table.selectRow(best_row)
            self.view_combo.setCurrentIndex(1)

    def _on_preview_double_click(self, ix: float, iy: float) -> None:
        """全图模式下双击任意位置，切到该点的局部 A/B 对比（无需命中检测）。"""
        if self.view_combo.currentIndex() != 0:
            return
        res = self._current_result()
        if not res or res.get("a_u8") is None:
            return
        s = res["preview_scale"]
        self._manual_center = (ix / s, iy / s)  # 预览坐标 -> 原生坐标
        self._manual_result = res
        self.view_combo.setCurrentIndex(1)  # 触发刷新

    def _current_result(self) -> Optional[Dict]:
        entry = self._selected_entry()
        if entry is not None:
            return entry["_result"]
        return self._results[-1] if self._results else None

    @staticmethod
    def _draw_crosshair(
        img: QImage, x: float, y: float, color: QColor,
        size: int = 12, gap: int = 4, width: int = 2,
    ) -> None:
        """空心十字标注（中心留空）。"""
        p = QPainter(img)
        p.setPen(QPen(color, width))
        xi, yi = int(round(x)), int(round(y))
        p.drawLine(xi - size, yi, xi - gap, yi)
        p.drawLine(xi + gap, yi, xi + size, yi)
        p.drawLine(xi, yi - size, xi, yi - gap)
        p.drawLine(xi, yi + gap, xi, yi + size)
        p.end()

    @staticmethod
    def _draw_polyline(img: QImage, pts, color: QColor, width: int = 1) -> None:
        p = QPainter(img)
        p.setPen(QPen(color, width))
        p.drawPolygon(QPolygon(pts))
        p.end()

    def _refresh_preview(self) -> None:
        res = self._current_result()
        if not res or res.get("a_u8") is None or res.get("b_raw") is None:
            return
        a = res["a_u8"]
        b_raw = res["b_raw"]
        s = res["preview_scale"]
        sat = res.get("sat_prev")
        cov = res.get("cov_prev")
        show_sat = self.sat_check.isChecked() and sat is not None
        show_cov = self.cover_check.isChecked() and cov is not None
        show_validpoly = self.validpoly_check.isChecked()
        show_validpoly_b = self.validpoly_b_check.isChecked()
        show_all = self.show_filtered_check.isChecked()

        entry = self._selected_entry()
        manual = self._manual_center if self._manual_result is res else None
        mode_crop = (
            self.view_combo.currentIndex() == 1 and (entry is not None or manual is not None)
        )

        if mode_crop:
            center = manual if manual is not None else (entry["x"], entry["y"])
            size = int(self.crop_spin.value())
            a_gray = None
            try:
                crops = load_native_pair_crops(
                    self._fits_cache, res["a_path"], res["b_path"],
                    [center], size,
                    fill_invalid_with_a=res.get("fill_invalid_with_a", True),
                )
                a_crop, b_crop = crops[0]
                # 原生分辨率裁块（无降采样），A 线性、B 局部线性
                a_gray = _linear_u8(a_crop)
                b_gray = _linear_u8(b_crop)
            except Exception as ex:  # noqa: BLE001
                self._log(f"原生裁切失败，回退预览: {ex}")
            if a_gray is None:
                cx, cy = center[0] * s, center[1] * s
                half = max(8.0, size * s / 2.0)
                x0 = int(max(0, round(cx - half)))
                y0 = int(max(0, round(cy - half)))
                x1 = int(max(x0 + 1, min(a.shape[1], round(cx + half))))
                y1 = int(max(y0 + 1, min(a.shape[0], round(cy + half))))
                a_gray = _linear_u8(a[y0:y1, x0:x1].astype(np.float32))
                b_gray = _linear_u8(b_raw[y0:y1, x0:x1])
            # 预览尺度窗口（供 ob/cov/对比掩码对齐到裁块）
            cx, cy = center[0] * s, center[1] * s
            halfp = max(1.0, size * s / 2.0)
            mpx0 = int(max(0, round(cx - halfp)))
            mpy0 = int(max(0, round(cy - halfp)))
            mpx1 = int(max(mpx0 + 1, min(a.shape[1], round(cx + halfp))))
            mpy1 = int(max(mpy0 + 1, min(a.shape[0], round(cy + halfp))))
            px = py = size // 2
            vx0 = int(round(center[0])) - size // 2
            vy0 = int(round(center[1])) - size // 2
            vscale = 1.0
        else:
            a_gray = a
            # B：按全图做自适应拉伸（STF）
            b_gray = _stretch_u8(b_raw)
            mpx0 = mpy0 = 0
            mpx1, mpy1 = a.shape[1], a.shape[0]
            px = py = 0.0
            vx0 = vy0 = 0.0
            vscale = s

        import cv2

        img_h, img_w = a_gray.shape[:2]

        def align2d(prevmask):
            if prevmask is None:
                return None
            sub = prevmask[mpy0:mpy1, mpx0:mpx1]
            if mode_crop:
                sub = cv2.resize(sub, (img_w, img_h), interpolation=cv2.INTER_NEAREST)
            return np.asarray(sub)

        a_rgb = gray_to_rgb(a_gray)
        b_rgb = gray_to_rgb(b_gray)
        if show_sat and sat is not None:
            # 卫星掩码只作用于 B（PairRegNet ob 通道 2）
            b_rgb = overlay_ob(b_rgb, align2d(sat)[:, :, None], {0: _SAT_RGB})
        if show_cov and cov is not None:
            edge = _mask_edge(align2d(cov) > 0)
            a_rgb = paint_mask(a_rgb, edge, _COVER_EDGE_RGB)
            b_rgb = paint_mask(b_rgb, edge, _COVER_EDGE_RGB)

        a_img = rgb_to_qimage(a_rgb)
        b_img = rgb_to_qimage(b_rgb)

        if mode_crop:
            csz = max(9, int(min(a_img.width(), a_img.height()) * 0.18))
            # 裁切模式：十字用该目标的状态(类别)颜色；手动双击点用选中色
            if entry is not None and manual is None:
                col = _det_status_color(entry.get("status", "keep"))
            else:
                col = _SEL_COLOR
            self._draw_crosshair(a_img, px, py, col, size=csz, gap=max(3, csz // 3))
            self._draw_crosshair(b_img, px, py, col, size=csz, gap=max(3, csz // 3))
        else:
            for d in res["detections"]:
                st = d.get("status", "keep")
                if st != "keep" and not show_all:
                    continue
                dx, dy = d["x"] * s, d["y"] * s
                col = _det_status_color(st)
                self._draw_crosshair(a_img, dx, dy, col)
                self._draw_crosshair(b_img, dx, dy, col)
            if entry is not None and entry["_result"] is res:
                sx, sy = entry["x"] * s, entry["y"] * s
                self._draw_crosshair(a_img, sx, sy, _SEL_COLOR)
                self._draw_crosshair(b_img, sx, sy, _SEL_COLOR)

        if show_validpoly or show_validpoly_b:
            def _draw_polys(polys, color):
                for poly in polys:
                    pts = [
                        QPoint(int(round((x - vx0) * vscale)),
                               int(round((y - vy0) * vscale)))
                        for x, y in poly
                    ]
                    if len(pts) >= 3:
                        self._draw_polyline(a_img, pts, color)
                        self._draw_polyline(b_img, pts, color)

            if show_validpoly and res.get("a_valid_polys"):
                _draw_polys(res["a_valid_polys"], _A_VALID_COLOR)
            if show_validpoly_b and res.get("b_valid_polys"):
                _draw_polys(res["b_valid_polys"], _B_VALID_COLOR)

        self.preview_a.set_image_size(a_img.width(), a_img.height())
        self.preview_b.set_image_size(b_img.width(), b_img.height())
        self.preview_a.setPixmap(QPixmap.fromImage(a_img).scaled(
            self.preview_a.width(), self.preview_a.height(),
            Qt.KeepAspectRatio, Qt.SmoothTransformation))
        self.preview_b.setPixmap(QPixmap.fromImage(b_img).scaled(
            self.preview_b.width(), self.preview_b.height(),
            Qt.KeepAspectRatio, Qt.SmoothTransformation))

    def _thumb_from_crops(self, a_crop: np.ndarray, b_crop: np.ndarray,
                          side: int = 128) -> QPixmap:
        """由原生裁块拼出 A|B 缩略图（带十字）。"""
        a_img = numpy_to_qimage(_linear_u8(a_crop))
        b_img = numpy_to_qimage(_linear_u8(b_crop))
        cx, cy = a_img.width() // 2, a_img.height() // 2
        self._draw_crosshair(a_img, cx, cy, _SEL_COLOR)
        self._draw_crosshair(b_img, cx, cy, _SEL_COLOR)
        w = a_img.width() + b_img.width()
        h = max(a_img.height(), b_img.height())
        combined = QImage(w, h, QImage.Format_RGB888)
        combined.fill(QColor(0, 0, 0))
        p = QPainter(combined)
        p.drawImage(0, 0, a_img)
        p.drawImage(a_img.width(), 0, b_img)
        p.end()
        return QPixmap.fromImage(combined).scaled(
            side * 2, side, Qt.KeepAspectRatio, Qt.SmoothTransformation)

    def _build_montage(self, limit: int = 300) -> None:
        """把当前所有检测做成小图墙（每格 A|B 原生裁切 + 信息）。"""
        limit = int(limit) or 300
        while self.montage_grid.count():
            item = self.montage_grid.takeAt(0)
            w = item.widget()
            if w is not None:
                w.deleteLater()
        show_all = self.show_filtered_check.isChecked()
        size = int(self.crop_spin.value())
        cols = 4
        n = 0
        for res in self._results:
            if n >= limit:
                break
            dets = [
                d for d in res["detections"]
                if show_all or d.get("status", "keep") == "keep"
            ]
            dets = dets[: max(0, limit - n)]
            if not dets:
                continue
            # 每个文件一次性读取原生裁块
            crops: List = []
            try:
                crops = load_native_pair_crops(
                    self._fits_cache, res["a_path"], res["b_path"],
                    [(d["x"], d["y"]) for d in dets], size,
                    fill_invalid_with_a=res.get("fill_invalid_with_a", True),
                )
            except Exception as ex:  # noqa: BLE001
                self._log(f"小图墙原生裁切失败: {ex}")
                crops = [(None, None)] * len(dets)
            for det, cr in zip(dets, crops):
                if n >= limit or cr is None or cr[0] is None:
                    continue
                st = det.get("status", "keep")
                pm = self._thumb_from_crops(cr[0], cr[1])
                cell = QFrame()
                cell.setFrameShape(QFrame.Box)
                cl = QVBoxLayout(cell)
                cl.setContentsMargins(2, 2, 2, 2)
                img_lbl = _ThumbLabel()
                img_lbl.setPixmap(pm)
                img_lbl.setAlignment(Qt.AlignCenter)
                img_lbl.setCursor(Qt.PointingHandCursor)
                img_lbl.double_clicked.connect(
                    lambda r=res, d=det: self._jump_to_detection(r, d)
                )
                cap = QLabel(
                    f"{_STATUS_TEXT.get(st, st)}  {det['score']:.2f}\n"
                    f"({det['x']:.0f},{det['y']:.0f})"
                )
                cap.setAlignment(Qt.AlignCenter)
                cap.setStyleSheet("color:#444; font-size:10px;")
                cl.addWidget(img_lbl)
                cl.addWidget(cap)
                r, c = divmod(n, cols)
                self.montage_grid.addWidget(cell, r, c)
                n += 1
            if n >= limit:
                break
        self.montage_info.setText(
            f"（已生成 {n} 个检测的 A|B 裁切"
            + ("，已截断" if n >= limit else "")
            + "）"
        )

    def _jump_to_detection(self, res: Dict, det: Dict) -> None:
        """小图墙双击：在表格中选中对应行并切到裁切预览。"""
        for row, e in enumerate(self._row_map):
            if e.get("_det") is det:
                self.table.selectRow(row)
                self.view_combo.setCurrentIndex(1)
                return
        self._log("该检测当前未显示在表格中（可能被“显示被过滤结果”隐藏）")

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
            w.writerow(["file", "status", "x", "y", "score", "dx", "dy", "roll", "tile_x", "tile_y"])
            for e in self._row_map:
                w.writerow([
                    os.path.basename(e.get("_result", {}).get("b_path") or ""),
                    e.get("status", "keep"),
                    f"{e['x']:.2f}", f"{e['y']:.2f}", f"{e['score']:.4f}",
                    f"{e['dx']:.3f}", f"{e['dy']:.3f}", f"{e['roll']:.3f}",
                    e["tile_x"], e["tile_y"],
                ])
        self._log(f"已导出 CSV: {path}")


def main() -> None:
    app = QApplication(sys.argv)
    win = MainWindow()
    default_root = win.root_edit.text().strip() or DEFAULTS["root"]
    if not os.path.isdir(default_root):
        default_root = str(Path.home())
    win._set_root(default_root)
    win.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()

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
import threading
from concurrent.futures import (
    FIRST_COMPLETED,
    ThreadPoolExecutor,
    as_completed,
    wait,
)
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
    QDialog,
    QDoubleSpinBox,
    QFileDialog,
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
    QSlider,
    QSpinBox,
    QSplitter,
    QTableWidget,
    QTableWidgetItem,
    QTabWidget,
    QTreeView,
    QTreeWidget,
    QTreeWidgetItem,
    QVBoxLayout,
    QWidget,
)

from config import DEFAULTS, load_settings, save_settings
from native_crop import FitsCache, load_native_pair_crops
from pair_infer import PairModel
from pipeline import collect_fits_files, process_b_file
from pipeline import _det_metrics
import query_servers
import results_io
import web_export

class NoWheelSpinBox(QSpinBox):
    """数值框：忽略鼠标滚轮，避免鼠标经过时误改数值。"""

    def wheelEvent(self, event):  # noqa: N802
        event.ignore()


class NoWheelDoubleSpinBox(QDoubleSpinBox):
    """小数数值框：忽略鼠标滚轮，避免鼠标经过时误改数值。"""

    def wheelEvent(self, event):  # noqa: N802
        event.ignore()


_AB_CLASSES = list(DEFAULTS.get("ab_classes", ["noise", "pixelshift", "target"]))
_AB_HEADERS = ["分类"] + [f"P({c})" for c in _AB_CLASSES]


def _ab_cells(d: dict) -> list:
    """检测的 A/B 分类单元格：类别 + 各类别概率（缺失显示 -）。"""
    probs = d.get("ab_probs") or {}
    label = d.get("ab_class")
    cells = [label if label else "-"]
    for c in _AB_CLASSES:
        v = probs.get(c)
        cells.append("-" if v is None else f"{float(v):.2f}")
    return cells


_COLUMNS = ["文件", "状态", "变星", "MPC", "x", "y", "score", "SNR", "conc", "dx", "dy", "roll", "tile_x", "tile_y"] + _AB_HEADERS
_DAILY_COLUMNS = ["日期", "系统", "天区", "文件", "状态", "变星", "MPC", "x", "y", "score", "SNR", "conc", "dx", "dy", "roll"] + _AB_HEADERS + ["手动分类"]
_MANUAL_COL = len(_DAILY_COLUMNS) - 1
# 掩码/峰值颜色
_DET_COLOR = QColor(255, 60, 60)
_SEL_COLOR = QColor(60, 255, 120)

_STATUS_TEXT = {
    "keep": "命中",
    "satellite": "卫星",
    "b_uncovered": "B未覆盖",
    "a_invalid": "A无效区",
    "low_snr": "低信噪",
    "edge": "边缘",
    "isolated": "孤立点",
    "spike": "亮尖峰",
    "dedup": "重复",
    "ab_reject": "非目标",
}
_STATUS_COLOR = {
    "satellite": QColor(255, 170, 80),
    "b_uncovered": QColor(150, 150, 200),
    "a_invalid": QColor(150, 150, 200),
    "low_snr": QColor(120, 120, 120),
    "edge": QColor(110, 110, 110),
    "isolated": QColor(200, 120, 120),
    "spike": QColor(220, 90, 90),
    "dedup": QColor(160, 160, 160),
    "ab_reject": QColor(180, 140, 255),
}


def _det_status_color(status: str) -> QColor:
    return _DET_COLOR if status == "keep" else _STATUS_COLOR.get(status, _DET_COLOR)


# 卫星掩码叠加颜色
_SAT_RGB = (255, 170, 0)
# 边缘过滤带颜色
_EDGE_BAND_RGB = (255, 60, 160)
# 最终计算边框颜色
_FINAL_RGB = (0, 255, 120)
# B 覆盖区（WCS 对齐后）边框颜色
_COVER_EDGE_RGB = (255, 230, 0)
# A 模板有效区（星空/空白）边界
_A_VALID_COLOR = QColor(255, 255, 255)
# B（重投影后）有效区边界
_B_VALID_COLOR = QColor(0, 220, 180)
# B 二次有效区(剔除暗边)边界
_B2_VALID_COLOR = QColor(255, 120, 0)
# 变星(VSX) / MPC 命中点
_VAR_COLOR = QColor(0, 220, 255)
_MPC_COLOR = QColor(255, 80, 200)


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
            model = None
            ab_model = None

            total = len(files)
            for i, f in enumerate(files, 1):
                if self._stop:
                    self.log.emit("已停止")
                    break
                self.progress.emit(i - 1, total)
                self.log.emit(f"[{i}/{total}] {os.path.basename(f)}")
                if self.params.get("skip_existing", True):
                    j = results_io.result_json_path(f)
                    if os.path.exists(j):
                        try:
                            res = results_io.load_result(j)
                            self.log.emit("  跳过(已有结果)，直接加载")
                            self.file_done.emit(res)
                            continue
                        except Exception as ex:  # noqa: BLE001
                            self.log.emit(f"  已有结果读取失败，改为重新处理: {ex}")
                if model is None:
                    self.log.emit(f"模型目录: {self.params['model_dir']}")
                    model = PairModel(
                        self.params["model_dir"],
                        device=self.params["device"],
                        det_threshold=self.params["det_threshold"],
                        batch_size=self.params["batch_size"],
                        amp=self.params.get("amp", True),
                    )
                    self.log.emit(
                        f"模型已加载 (device={model.device}, size={model.model_size})")
                if self.params.get("ab_filter") and ab_model is None:
                    from ab_classify import ABModel
                    ab_model = ABModel(
                        self.params.get("ab_model_dir"),
                        device=self.params["device"],
                    )
                    self.log.emit(
                        f"A/B分类器已加载 (类别={ab_model.classes}, "
                        f"noise<={self.params.get('ab_noise_max')}, "
                        f"pixelshift<={self.params.get('ab_pixelshift_max')})")
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
                        snr_filter=self.params.get("snr_filter", True),
                        shading_k=self.params.get("shading_k", 3.0),
                        b2_ksize=self.params.get("b2_ksize", 21),
                        noise_k=self.params.get("noise_k", 3.0),
                        aperture_radius=self.params.get("aperture_radius", 3),
                        aperture_annulus=self.params.get("aperture_annulus", 6),
                        aperture_snr_min=self.params.get("aperture_snr_min", 4.0),
                        det_center_search=self.params.get("det_center_search", 2),
                        edge_band=self.params.get("edge_band", 5),
                        boundary_scale=self.params.get("boundary_scale", 4),
                        isolated_filter=self.params.get("isolated_filter", True),
                        isolated_win=self.params.get("isolated_win", 7),
                        isolated_k=self.params.get("isolated_k", 3.0),
                        isolated_min_px=self.params.get("isolated_min_px", 2),
                        shape_conc_max=self.params.get("shape_conc_max", 0.5),
                        shape_fwhm_min=self.params.get("shape_fwhm_min", 0.8),
                        shape_fwhm_max=self.params.get("shape_fwhm_max", 0.0),
                        median_filter=self.params.get("median_filter", False),
                        median_ksize=self.params.get("median_ksize", 3),
                        ab_model=ab_model,
                        ab_filter=bool(self.params.get("ab_filter", False)),
                        ab_keep_classes=self.params.get("ab_keep_classes", ()),
                        ab_noise_max=float(self.params.get("ab_noise_max", 0.8)),
                        ab_pixelshift_max=float(self.params.get("ab_pixelshift_max", 0.6)),
                        ab_patch=int(self.params.get("ab_patch", 16)),
                        anomaly_min_keep=int(self.params.get("anomaly_min_keep", 40)),
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


class QueryWorker(QThread):
    """后台执行变星(VSX)/MPC 查询：先变星全部查完，再查 MPC。"""

    log = Signal(str)
    progress = Signal(str, int, int)  # phase("var"/"mpc"), done, total
    finished_all = Signal()

    def __init__(self, tasks: list, cfg: dict) -> None:
        super().__init__()
        self.tasks = tasks
        self.cfg = cfg
        self._resume = threading.Event()
        self._resume.set()
        self._stop = threading.Event()

    def pause(self) -> None:
        self._resume.clear()

    def resume(self) -> None:
        self._resume.set()

    def is_paused(self) -> bool:
        return not self._resume.is_set()

    def stop(self) -> None:
        self._stop.set()
        self._resume.set()

    def run(self) -> None:
        cfg = self.cfg
        lock = threading.Lock()
        vstate = {"down": False, "logged": False}
        mstate = {"down": False, "logged": False}

        def hits_to_pix(hits, wcs):
            out = []
            for h in hits:
                try:
                    px, py = wcs.all_world2pix(float(h["ra"]), float(h["dec"]), 0)
                    out.append([float(px), float(py)])
                except Exception:
                    continue
            return out

        def do_vsx(task):
            det, wcs, epoch, ra, dec = task
            if vstate["down"]:
                det["var_count"] = -1
                return []
            try:
                hits = query_servers.query_vsx(
                    ra, dec, cfg["radius"], mag_limit=cfg["mag_limit"],
                    host=cfg["vsx_host"], port=cfg["vsx_port"],
                    timeout=cfg["vsx_timeout"])
                det["var_count"] = len(hits)
                det["var_hits"] = hits_to_pix(hits, wcs)
            except Exception as ex:  # noqa: BLE001
                det["var_count"] = -1
                det["var_hits"] = []
                with lock:
                    vstate["down"] = True
                    if not vstate["logged"]:
                        vstate["logged"] = True
                        return [f"变星服务({cfg['vsx_url']})不可用: {ex}"]
            return []

        def do_mpc(task):
            det, wcs, epoch, ra, dec = task
            if mstate["down"] or epoch is None:
                det["mpc_count"] = -1
                return []
            try:
                hits = query_servers.query_mpc(
                    ra, dec, epoch, cfg["radius"],
                    host=cfg["mpc_host"], port=cfg["mpc_port"],
                    timeout=cfg["mpc_timeout"])
                det["mpc_count"] = len(hits)
                det["mpc_hits"] = hits_to_pix(hits, wcs)
            except Exception as ex:  # noqa: BLE001
                det["mpc_count"] = -1
                det["mpc_hits"] = []
                with lock:
                    mstate["down"] = True
                    if not mstate["logged"]:
                        mstate["logged"] = True
                        return [f"MPC服务({cfg['mpc_url']})不可用: {ex}"]
            return []

        def run_phase(work, phase_tasks, phase, label):
            total = len(phase_tasks)
            self.progress.emit(phase, 0, total)
            self.log.emit(f"查询{label}: {total} 个目标（并发 {cfg['threads']}）…")
            if total == 0:
                return
            done = 0
            idx = 0
            pending = {}
            with ThreadPoolExecutor(max_workers=cfg["threads"]) as pool:
                while idx < total or pending:
                    # 暂停：不提交新任务，等待恢复
                    while not self._resume.wait(0.2):
                        if self._stop.is_set():
                            break
                    if self._stop.is_set():
                        idx = total
                    while idx < total and len(pending) < cfg["threads"]:
                        fut = pool.submit(work, phase_tasks[idx])
                        pending[fut] = True
                        idx += 1
                    if not pending:
                        break
                    done_futs, _ = wait(list(pending), timeout=0.2,
                                        return_when=FIRST_COMPLETED)
                    for fut in done_futs:
                        pending.pop(fut, None)
                        try:
                            for m in fut.result():
                                self.log.emit(m)
                        except Exception as ex:  # noqa: BLE001
                            self.log.emit(f"查询任务异常: {ex}")
                        done += 1
                        self.progress.emit(phase, done, total)

        vsx_tasks = [t for t in self.tasks
                     if not (cfg["skip_done"] and t[0].get("var_count", -1) >= 0)]
        mpc_tasks = [t for t in self.tasks
                     if not (cfg["skip_done"] and (t[0].get("mpc_count", -1) >= 0 or t[2] is None))]
        if cfg["skip_done"]:
            self.log.emit(f"跳过已查询: 变星 {len(self.tasks)-len(vsx_tasks)} 个, "
                          f"MPC {len(self.tasks)-len(mpc_tasks)} 个")
        run_phase(do_vsx, vsx_tasks, "var", "变星")
        if not self._stop.is_set():
            run_phase(do_mpc, mpc_tasks, "mpc", "MPC")
        self.log.emit(
            f"查询完成（变星服务不可用={vstate['down']}, MPC服务不可用={mstate['down']}）")
        if cfg.get("n_no_epoch"):
            self.log.emit(
                f"MPC 跳过: {cfg['n_no_epoch']} 个目标因 B 头缺少观测时间(MJD-OBS/JD/DATE-OBS)")
        self.finished_all.emit()


class ConcDialog(QDialog):
    """验证 conc：A|B 原始局部块，分别手动拉伸；B 可 180° 翻转。"""

    def __init__(self, parent, patches: Dict, cx: int, cy: int,
                 r: int, R: int, search: int) -> None:
        super().__init__(parent)
        self.setWindowTitle("验证 conc（A|B 原始数据 + 分别拉伸）")
        self.patches = {k: np.asarray(v, dtype=np.float32) for k, v in patches.items()}
        self.cx, self.cy = int(cx), int(cy)
        self.r, self.R, self.search = int(r), int(R), int(search)
        self.patch = self.patches.get("B", next(iter(self.patches.values())))
        self._loading = False
        self.st: Dict[str, Dict] = {}
        for k, p in self.patches.items():
            fin = p[np.isfinite(p)]
            vmin = float(np.nanmin(fin)) if fin.size else 0.0
            vmax = float(np.nanmax(fin)) if fin.size else 1.0
            if vmax <= vmin:
                vmax = vmin + 1.0
            lo = float(np.percentile(fin, 1.0)) if fin.size else vmin
            hi = float(np.percentile(fin, 99.5)) if fin.size else vmax
            if hi <= lo:
                hi = lo + 1.0
            self.st[k] = {"lo": lo, "hi": hi, "gamma": 1.0, "asinh": False,
                          "vmin": vmin, "vmax": vmax}

        self.flip_b = QCheckBox("B翻转(180°)")
        self.flip_b.setChecked(True)
        self.flip_b.stateChanged.connect(lambda _=0: self.update_img())
        self.overlay = QCheckBox("显示标注")
        self.overlay.setChecked(False)
        self.overlay.stateChanged.connect(lambda _=0: self.update_img())
        top = QHBoxLayout()
        top.addWidget(self.flip_b)
        top.addWidget(self.overlay)
        top.addStretch(1)

        row = QHBoxLayout()
        for k, title in (("A", "A (参考)"), ("B", "B (检测)")):
            if k in self.patches:
                row.addWidget(self._panel(title, k), 1)

        self.info = QLabel("")
        self.info.setStyleSheet("font-family: Consolas, monospace;")
        lay = QVBoxLayout(self)
        lay.addLayout(top)
        lay.addLayout(row, 1)
        lay.addWidget(self.info)
        self.update_img()

    def _panel(self, title: str, key: str) -> QWidget:
        box = QGroupBox(title)
        v = QVBoxLayout(box)
        img = QLabel()
        img.setMinimumSize(360, 360)
        img.setAlignment(Qt.AlignCenter)
        img.setStyleSheet("background:#000;")
        setattr(self, "img" + key, img)
        s = self.st[key]
        black = NoWheelDoubleSpinBox(); black.setRange(-1e12, 1e12); black.setDecimals(1)
        black.setValue(s["lo"])
        white = NoWheelDoubleSpinBox(); white.setRange(-1e12, 1e12); white.setDecimals(1)
        white.setValue(s["hi"])
        gamma = NoWheelDoubleSpinBox(); gamma.setRange(0.1, 5.0); gamma.setSingleStep(0.1)
        gamma.setValue(1.0)
        asinh = QCheckBox("asinh")
        bsl = QSlider(Qt.Horizontal); bsl.setRange(0, 1000)
        bsl.setValue(self._to_slider(key, s["lo"]))
        wsl = QSlider(Qt.Horizontal); wsl.setRange(0, 1000)
        wsl.setValue(self._to_slider(key, s["hi"]))

        def on_black(val):
            if self._loading:
                return
            s["lo"] = float(val)
            self._loading = True
            bsl.setValue(self._to_slider(key, float(val)))
            self._loading = False
            self.update_img()

        def on_white(val):
            if self._loading:
                return
            s["hi"] = float(val)
            self._loading = True
            wsl.setValue(self._to_slider(key, float(val)))
            self._loading = False
            self.update_img()

        def on_gamma(val):
            if self._loading:
                return
            s["gamma"] = float(val)
            self.update_img()

        def on_asinh(_=0):
            if self._loading:
                return
            s["asinh"] = bool(asinh.isChecked())
            self.update_img()

        black.valueChanged.connect(on_black)
        white.valueChanged.connect(on_white)
        gamma.valueChanged.connect(on_gamma)
        asinh.stateChanged.connect(on_asinh)
        bsl.valueChanged.connect(
            lambda val: (not self._loading) and black.setValue(self._from_slider(key, val)))
        wsl.valueChanged.connect(
            lambda val: (not self._loading) and white.setValue(self._from_slider(key, val)))

        r1 = QHBoxLayout()
        r1.addWidget(QLabel("黑点")); r1.addWidget(black); r1.addWidget(bsl, 1)
        r2 = QHBoxLayout()
        r2.addWidget(QLabel("白点")); r2.addWidget(white); r2.addWidget(wsl, 1)
        r3 = QHBoxLayout()
        r3.addWidget(QLabel("gamma")); r3.addWidget(gamma); r3.addWidget(asinh)
        r3.addStretch(1)
        v.addWidget(img, 1)
        v.addLayout(r1)
        v.addLayout(r2)
        v.addLayout(r3)
        return box

    def _to_slider(self, key: str, v: float) -> int:
        s = self.st[key]
        return int(round(1000.0 * (v - s["vmin"]) / (s["vmax"] - s["vmin"])))

    def _from_slider(self, key: str, sl: int) -> float:
        s = self.st[key]
        return s["vmin"] + (s["vmax"] - s["vmin"]) * float(sl) / 1000.0

    def _stretch(self, patch: np.ndarray, key: str) -> np.ndarray:
        s = self.st[key]
        lo, hi, g = s["lo"], s["hi"], s["gamma"]
        if hi <= lo:
            hi = lo + 1.0
        n = np.clip((np.nan_to_num(patch, nan=lo) - lo) / (hi - lo), 0.0, 1.0)
        if abs(g - 1.0) > 1e-6:
            n = np.power(n, g)
        if s["asinh"]:
            n = np.arcsinh(n * 8.0) / np.arcsinh(8.0)
        return (np.clip(n, 0.0, 1.0) * 255.0).astype(np.uint8)

    @staticmethod
    def _qimg(u8: np.ndarray) -> QImage:
        h, w = u8.shape
        return QImage(np.ascontiguousarray(u8).data, w, h, w,
                      QImage.Format_Grayscale8).copy().convertToFormat(QImage.Format_RGB888)

    def _peak_xy(self):
        h, w = self.patch.shape
        yy, xx = np.mgrid[0:h, 0:w]
        d2 = (xx - self.cx) ** 2 + (yy - self.cy) ** 2
        m = (d2 <= float(self.search) ** 2) & np.isfinite(self.patch)
        if not m.any():
            return self.cx, self.cy
        py, px = np.unravel_index(
            int(np.argmax(np.where(m, self.patch, -np.inf))), self.patch.shape)
        return int(px), int(py)

    def update_img(self):
        gpx, gpy = self._peak_xy()
        show = self.overlay.isChecked()
        if "A" in self.patches:
            qA = self._qimg(self._stretch(self.patches["A"], "A"))
            if show:
                pa = QPainter(qA)
                pa.setPen(QPen(QColor(0, 200, 255), 1))
                pa.drawLine(self.cx - 6, self.cy, self.cx + 6, self.cy)
                pa.drawLine(self.cx, self.cy - 6, self.cx, self.cy + 6)
                pa.end()
            self.imgA.setPixmap(QPixmap.fromImage(qA).scaled(
                self.imgA.width(), self.imgA.height(),
                Qt.KeepAspectRatio, Qt.FastTransformation))
        if "B" in self.patches:
            if self.flip_b.isChecked():
                bsrc = self.patches["B"][::-1, ::-1]
                hB, wB = bsrc.shape
                fcx, fcy = wB - 1 - self.cx, hB - 1 - self.cy
                fgpx, fgpy = wB - 1 - gpx, hB - 1 - gpy
            else:
                bsrc = self.patches["B"]
                hB, wB = bsrc.shape
                fcx, fcy = self.cx, self.cy
                fgpx, fgpy = gpx, gpy
            qB = self._qimg(self._stretch(bsrc, "B"))
            if show:
                pb = QPainter(qB)
                pb.setPen(QPen(QColor(0, 255, 0), 1))
                pb.drawEllipse(fcx - self.r, fcy - self.r, 2 * self.r, 2 * self.r)
                pb.setPen(QPen(QColor(255, 200, 0), 1))
                pb.drawEllipse(fcx - self.R, fcy - self.R, 2 * self.R, 2 * self.R)
                pb.setPen(QPen(QColor(0, 200, 255), 1))
                pb.drawLine(fcx - 6, fcy, fcx + 6, fcy)
                pb.drawLine(fcx, fcy - 6, fcx, fcy + 6)
                pb.setPen(QPen(QColor(255, 60, 60), 1))
                pb.drawLine(fgpx - 7, fgpy, fgpx + 7, fgpy)
                pb.drawLine(fgpx, fgpy - 7, fgpx, fgpy + 7)
                pb.end()
            self.imgB.setPixmap(QPixmap.fromImage(qB).scaled(
                self.imgB.width(), self.imgB.height(),
                Qt.KeepAspectRatio, Qt.FastTransformation))
        snr, conc, sig = _det_metrics(
            self.patch, self.cx, self.cy, self.r, self.R, self.search)
        h, w = self.patch.shape
        pv = float(self.patch[gpy, gpx]) if 0 <= gpy < h and 0 <= gpx < w else 0.0
        sb = self.st["B"] if "B" in self.st else {"lo": 0, "hi": 0}
        self.info.setText(
            f"conc={conc:.3f}  SNR={snr:.2f}  FWHM={2.355*sig:.2f}px  峰值B={pv:.0f}  "
            f"中心B=({self.cx},{self.cy}) 峰值点=({gpx},{gpy})  B黑白={sb['lo']:.0f}/{sb['hi']:.0f}")


class MainWindow(QMainWindow):
    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle("gui_ai - WCS 重投影 + PairRegNet 瞬变检测")
        self.resize(1500, 860)

        self.worker: Optional[ProcessWorker] = None
        self.query_worker: Optional[QueryWorker] = None
        self._results: List[Dict] = []          # 每个文件的结果
        self._row_map: List[Optional[Dict]] = []  # 表格行 -> 检测项（含 _result 引用）
        self._fits_cache = FitsCache(max_items=2)  # 原生裁块用（A/B 各一）
        self._manual_center: Optional[tuple] = None  # 双击指定的裁切中心(原生坐标)
        self._manual_result = None
        self._daily_rows: List[list] = []
        self._daily_meta: List[dict] = []
        self._loaded_path: Optional[str] = None

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
        self.ab_filter_check.setChecked(bool(s.get("ab_filter", DEFAULTS["ab_filter"])))
        self.ab_noise_spin.setValue(float(s.get("ab_noise_max", DEFAULTS["ab_noise_max"])))
        self.ab_shift_spin.setValue(
            float(s.get("ab_pixelshift_max", DEFAULTS["ab_pixelshift_max"])))
        self.file_anomaly_check.setChecked(
            bool(s.get("file_anomaly_filter", DEFAULTS["file_anomaly_filter"])))
        self.file_anomaly_spin.setValue(
            int(s.get("file_anomaly_min_keep", DEFAULTS["file_anomaly_min_keep"])))
        self.hide_noise_check.setChecked(
            bool(s.get("daily_hide_noise", DEFAULTS["daily_hide_noise"])))
        self.hide_shift_check.setChecked(
            bool(s.get("daily_hide_shift", DEFAULTS["daily_hide_shift"])))
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
            "ab_filter": bool(self.ab_filter_check.isChecked()),
            "ab_noise_max": float(self.ab_noise_spin.value()),
            "ab_pixelshift_max": float(self.ab_shift_spin.value()),
            "file_anomaly_filter": bool(self.file_anomaly_check.isChecked()),
            "file_anomaly_min_keep": int(self.file_anomaly_spin.value()),
            "daily_hide_noise": bool(self.hide_noise_check.isChecked()),
            "daily_hide_shift": bool(self.hide_shift_check.isChecked()),
            "show_filtered": bool(self.show_filtered_check.isChecked()),
            "show_sat": bool(self.sat_check.isChecked()),
            "show_cover": bool(self.cover_check.isChecked()),
            "show_a_validpoly": bool(self.validpoly_check.isChecked()),
            "show_b_validpoly": bool(self.validpoly_b_check.isChecked()),
            "view_mode": int(self.view_combo.currentIndex()),
        }

    def closeEvent(self, event) -> None:  # noqa: N802
        save_settings(self._collect_settings())
        for wk in (self.worker, self.query_worker):
            try:
                if wk is not None and wk.isRunning():
                    wk.wait(3000)
            except Exception:
                pass
        super().closeEvent(event)

    # ------------------------------------------------------------------ UI
    def _build_ui(self) -> None:
        central = QWidget()
        self.setCentralWidget(central)
        root = QHBoxLayout(central)
        root.setContentsMargins(4, 4, 4, 4)
        root.setSpacing(6)

        # 左侧参数较多，放入滚动区，避免撑高窗口；高度可自由压缩
        left_scroll = QScrollArea()
        left_scroll.setWidgetResizable(True)
        left_scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        left_scroll.setFixedWidth(440)
        left_scroll.setWidget(self._build_left_panel())
        root.addWidget(left_scroll, 0)
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

        self.tree = QTreeWidget()
        self.tree.setHeaderHidden(True)
        self.tree.setSelectionMode(QAbstractItemView.SingleSelection)
        self.tree.itemExpanded.connect(self._on_tree_expand)
        self.tree.currentItemChanged.connect(self._on_select)
        self.tree.setColumnCount(1)
        src_lay.addWidget(self.tree, 1)
        self.sel_label = QLabel("未选择节点")
        self.sel_label.setWordWrap(True)
        src_lay.addWidget(self.sel_label)
        lay.addWidget(src, 1)

        params = QGroupBox("参数（点击展开）")
        params.setCheckable(True)
        form = QFormLayout(params)
        self.template_edit = QLineEdit(DEFAULTS["template_root"])
        form.addRow("模板根目录", self.template_edit)
        self.model_edit = QLineEdit(DEFAULTS["model_dir"])
        form.addRow("模型目录", self.model_edit)
        self.device_combo = QComboBox()
        self.device_combo.addItems(["auto", "cpu", "cuda"])
        self.device_combo.setCurrentText(DEFAULTS["device"])
        form.addRow("设备", self.device_combo)
        self.tile_spin = NoWheelSpinBox()
        self.tile_spin.setRange(64, 1024)
        self.tile_spin.setValue(DEFAULTS["tile_size"])
        form.addRow("瓦片边长", self.tile_spin)
        self.overlap_spin = NoWheelDoubleSpinBox()
        self.overlap_spin.setRange(0.0, 0.9)
        self.overlap_spin.setSingleStep(0.05)
        self.overlap_spin.setDecimals(2)
        self.overlap_spin.setValue(DEFAULTS["overlap"])
        form.addRow("overlap 比例", self.overlap_spin)
        self.thresh_spin = NoWheelDoubleSpinBox()
        self.thresh_spin.setRange(0.0, 1.0)
        self.thresh_spin.setSingleStep(0.05)
        self.thresh_spin.setDecimals(2)
        self.thresh_spin.setValue(DEFAULTS["det_threshold"])
        form.addRow("检测阈值", self.thresh_spin)
        self.batch_spin = NoWheelSpinBox()
        self.batch_spin.setRange(1, 128)
        self.batch_spin.setValue(DEFAULTS["batch_size"])
        form.addRow("批大小", self.batch_spin)
        self.fill_check = QCheckBox("B 未覆盖区域用 A 填充")
        self.fill_check.setChecked(DEFAULTS["fill_invalid_with_a"])
        form.addRow("", self.fill_check)
        self.valid_overlap_check = QCheckBox("按A∩B有效区过滤检测")
        self.valid_overlap_check.setChecked(DEFAULTS["valid_overlap_filter"])
        form.addRow("", self.valid_overlap_check)
        self.snr_check = QCheckBox("按局部SNR/形状过滤(含暗边、亮尖峰)")
        self.snr_check.setChecked(DEFAULTS["snr_filter"])
        form.addRow("", self.snr_check)
        self.snr_min_spin = NoWheelDoubleSpinBox()
        self.snr_min_spin.setRange(0.0, 100000.0)
        self.snr_min_spin.setDecimals(1)
        self.snr_min_spin.setSingleStep(1.0)
        self.snr_min_spin.setValue(float(DEFAULTS["aperture_snr_min"]))
        form.addRow("SNR下限", self.snr_min_spin)
        self.isolated_check = QCheckBox("排除孤立点(宇宙线/热像素)")
        self.isolated_check.setChecked(DEFAULTS["isolated_filter"])
        form.addRow("", self.isolated_check)
        self.median_check = QCheckBox("B中值滤波(对比用)")
        self.median_check.setChecked(DEFAULTS["median_filter"])
        form.addRow("", self.median_check)
        self.ab_filter_check = QCheckBox("A/B分类过滤")
        self.ab_filter_check.setChecked(bool(DEFAULTS["ab_filter"]))
        self.ab_filter_check.setToolTip(
            "用 ab16/models_ab 的 A/B 16x16 分类器过滤检测："
            "剔除噪声/像移概率超限的检测")
        form.addRow("", self.ab_filter_check)
        self.ab_noise_spin = NoWheelDoubleSpinBox()
        self.ab_noise_spin.setRange(0.0, 1.0)
        self.ab_noise_spin.setSingleStep(0.05)
        self.ab_noise_spin.setDecimals(2)
        self.ab_noise_spin.setValue(float(DEFAULTS["ab_noise_max"]))
        form.addRow("AB噪声概率上限", self.ab_noise_spin)
        self.ab_shift_spin = NoWheelDoubleSpinBox()
        self.ab_shift_spin.setRange(0.0, 1.0)
        self.ab_shift_spin.setSingleStep(0.05)
        self.ab_shift_spin.setDecimals(2)
        self.ab_shift_spin.setValue(float(DEFAULTS["ab_pixelshift_max"]))
        form.addRow("AB像移概率上限", self.ab_shift_spin)
        self.file_anomaly_check = QCheckBox("整文件异常过滤(命中数≥阈值)")
        self.file_anomaly_check.setChecked(bool(DEFAULTS["file_anomaly_filter"]))
        self.file_anomaly_check.setToolTip(
            "某文件命中的检测数 ≥ 阈值时判为图像异常，整文件不显示/不导出")
        self.file_anomaly_check.stateChanged.connect(lambda _=0: self._rebuild_table())
        form.addRow("", self.file_anomaly_check)
        self.file_anomaly_spin = NoWheelSpinBox()
        self.file_anomaly_spin.setRange(1, 100000)
        self.file_anomaly_spin.setValue(int(DEFAULTS["file_anomaly_min_keep"]))
        self.file_anomaly_spin.valueChanged.connect(lambda _=0: self._rebuild_table())
        form.addRow("异常命中数阈值", self.file_anomaly_spin)
        self.edge_spin = NoWheelSpinBox()
        self.edge_spin.setRange(0, 200)
        self.edge_spin.setValue(int(DEFAULTS["edge_band"]))
        form.addRow("边缘过滤(px)", self.edge_spin)
        self.skip_existing_check = QCheckBox("跳过已有结果")
        self.skip_existing_check.setChecked(DEFAULTS["skip_existing"])
        form.addRow("", self.skip_existing_check)
        self.query_skip_check = QCheckBox("查询跳过已完成")
        self.query_skip_check.setChecked(DEFAULTS["query_skip_done"])
        form.addRow("", self.query_skip_check)

        def _toggle_params(checked: bool) -> None:
            for i in range(form.count()):
                w = form.itemAt(i).widget()
                if w is not None:
                    w.setVisible(checked)

        params.toggled.connect(_toggle_params)
        params.setChecked(False)   # 默认折叠
        _toggle_params(False)
        lay.addWidget(params)

        self.run_btn = QPushButton("开始处理")
        self.run_btn.clicked.connect(self._start)
        self.stop_btn = QPushButton("停止")
        self.stop_btn.setEnabled(False)
        self.stop_btn.clicked.connect(self._stop)
        self.export_btn = QPushButton("导出检测 CSV")
        self.export_btn.clicked.connect(self._export_csv)
        self.export_web_btn = QPushButton("导出网页ZIP(V4)")
        self.export_web_btn.clicked.connect(self._export_web_zip)
        self.train_btn = QPushButton("导出训练FITS")
        self.train_btn.clicked.connect(self._export_train_fits)
        self.load_btn = QPushButton("加载已有结果")
        self.load_btn.clicked.connect(self._load_results_from_node)
        self.refilter_btn = QPushButton("重算SNR/形状过滤")
        self.refilter_btn.clicked.connect(self._refilter_snr)

        row1 = QHBoxLayout()
        row1.addWidget(self.run_btn)
        row1.addWidget(self.stop_btn)
        lay.addLayout(row1)
        row2 = QHBoxLayout()
        row2.addWidget(self.load_btn)
        row2.addWidget(self.refilter_btn)
        lay.addLayout(row2)
        row4 = QHBoxLayout()
        row4.addWidget(self.export_btn)
        row4.addWidget(self.export_web_btn)
        row4.addWidget(self.train_btn)
        self.anomaly_btn = QPushButton("导出异常文件清单")
        self.anomaly_btn.clicked.connect(self._export_anomaly_list)
        row4.addWidget(self.anomaly_btn)
        row4.addStretch(1)
        lay.addLayout(row4)

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
            lbl.setMinimumHeight(200)
            lbl.setStyleSheet("background:#111; color:#888;")
            lbl.setCursor(Qt.PointingHandCursor)
            lbl.clicked.connect(self._on_preview_click)
            lbl.double_clicked.connect(self._on_preview_double_click)
            pl.addWidget(lbl, 1)

        # 中心 32px 原生对比小图（A|B，最近邻放大）
        zoom_box = QWidget()
        zv = QVBoxLayout(zoom_box)
        zv.setContentsMargins(0, 0, 0, 0)
        zt = QLabel("中心 %dpx A|B" % int(DEFAULTS.get("zoom_patch", 32)))
        zt.setAlignment(Qt.AlignCenter)
        self.preview_zoom = QLabel("（选中目标后显示）")
        self.preview_zoom.setAlignment(Qt.AlignCenter)
        self.preview_zoom.setMinimumHeight(180)
        self.preview_zoom.setFixedWidth(320)
        self.preview_zoom.setStyleSheet("background:#111; color:#777;")
        zv.addWidget(zt)
        zv.addWidget(self.preview_zoom, 1)
        pl.addWidget(zoom_box, 0)

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
        self.crop_spin = NoWheelSpinBox()
        self.crop_spin.setRange(32, 4096)
        self.crop_spin.setSingleStep(64)
        self.crop_spin.setValue(DEFAULTS["crop_size"])
        self.crop_spin.valueChanged.connect(lambda _=0: self._refresh_preview())
        vrow.addWidget(self.crop_spin)
        vrow.addSpacing(12)
        self.conc_btn = QPushButton("验证conc")
        self.conc_btn.clicked.connect(self._open_conc_viewer)
        vrow.addWidget(self.conc_btn)
        hint = QLabel("（Tab切换；全图点击十字→裁切；双击任意处→看该处局部对比）")
        hint.setStyleSheet("color:#666;")
        vrow.addWidget(hint)
        vrow.addStretch(1)
        pw.addLayout(vrow)

        # 手动分类按钮（作用于当前选中的检测；结果写回 .gui_ai.json）
        mbar = QHBoxLayout()
        mbar.addWidget(QLabel("手动分类:"))
        self.m_target_btn = QPushButton("m-target")
        self.m_target_btn.clicked.connect(lambda: self._set_manual_class("m-target"))
        self.m_noise_btn = QPushButton("m-noise")
        self.m_noise_btn.clicked.connect(lambda: self._set_manual_class("m-noise"))
        self.m_shift_btn = QPushButton("m-pix-shift")
        self.m_shift_btn.clicked.connect(lambda: self._set_manual_class("m-pix-shift"))
        for b in (self.m_target_btn, self.m_noise_btn, self.m_shift_btn):
            mbar.addWidget(b)
        self.manual_status = QLabel("（未选中检测）")
        self.manual_status.setStyleSheet("color:#666;")
        mbar.addWidget(self.manual_status)
        mbar.addStretch(1)
        self.remove_det_btn = QPushButton("移除当前检测")
        self.remove_det_btn.setToolTip("从当前结果中删除选中的检测（写回 .gui_ai.json）")
        self.remove_det_btn.clicked.connect(self._remove_current_detection)
        mbar.addWidget(self.remove_det_btn)
        pw.addLayout(mbar)

        self.show_filtered_check = QCheckBox("显示被过滤结果")
        self.show_filtered_check.setChecked(False)
        self.show_filtered_check.stateChanged.connect(lambda _=0: self._rebuild_table())
        self.sat_check = QCheckBox("叠加卫星掩码")
        self.sat_check.setChecked(True)
        self.sat_check.stateChanged.connect(lambda _=0: self._refresh_preview())
        self.cover_check = QCheckBox("显示B覆盖边框")
        self.cover_check.setChecked(True)
        self.cover_check.stateChanged.connect(lambda _=0: self._refresh_preview())
        self.validpoly_check = QCheckBox("显示A有效区边界")
        self.validpoly_check.setChecked(True)
        self.validpoly_check.stateChanged.connect(lambda _=0: self._refresh_preview())
        self.validpoly_b_check = QCheckBox("显示B有效区边界")
        self.validpoly_b_check.setChecked(True)
        self.validpoly_b_check.stateChanged.connect(lambda _=0: self._refresh_preview())
        self.validpoly_b2_check = QCheckBox("显示B二级有效区边界")
        self.validpoly_b2_check.setChecked(True)
        self.validpoly_b2_check.stateChanged.connect(lambda _=0: self._refresh_preview())
        self.edge_check = QCheckBox("显示边缘过滤带")
        self.edge_check.setChecked(True)
        self.edge_check.stateChanged.connect(lambda _=0: self._refresh_preview())
        self.final_check = QCheckBox("显示最终边框")
        self.final_check.setChecked(True)
        self.final_check.stateChanged.connect(lambda _=0: self._refresh_preview())

        crow1 = QHBoxLayout()
        crow1.addWidget(self.show_filtered_check)
        crow1.addWidget(self.sat_check)
        crow1.addWidget(self.cover_check)
        crow1.addStretch(1)
        pw.addLayout(crow1)

        crow2 = QHBoxLayout()
        crow2.addWidget(self.validpoly_check)
        crow2.addWidget(self.validpoly_b_check)
        crow2.addWidget(self.validpoly_b2_check)
        crow2.addWidget(self.edge_check)
        crow2.addWidget(self.final_check)
        crow2.addStretch(1)
        pw.addLayout(crow2)

        legend = QLabel()
        legend.setTextFormat(Qt.RichText)
        legend.setWordWrap(True)
        legend.setStyleSheet("font-size:11px; color:#333; padding:2px;")
        legend.setText(
            "<b>标注颜色：</b>"
            + _swatch(_DET_COLOR) + " 命中(keep)　"
            + _swatch(_STATUS_COLOR["satellite"]) + " 卫星(satellite)　"
            + _swatch(_STATUS_COLOR["b_uncovered"]) + " B未覆盖　"
            + _swatch(_STATUS_COLOR["a_invalid"]) + " A无效区　"
            + _swatch(_STATUS_COLOR["low_snr"]) + " 低信噪　"
            + _swatch(_STATUS_COLOR["edge"]) + " 边缘　"
            + _swatch(_STATUS_COLOR["isolated"]) + " 孤立点　"
            + _swatch(_STATUS_COLOR["spike"]) + " 亮尖峰　"
            + _swatch(_STATUS_COLOR["dedup"]) + " 重复(dedup)　"
            + _swatch(_SEL_COLOR) + " 当前选中"
            + "　　|　　<b>叠加：</b>"
            + _swatch(_SAT_RGB) + " 卫星掩码　"
            + _swatch(_COVER_EDGE_RGB) + " B覆盖边框　"
            + _swatch(_EDGE_BAND_RGB) + " 边缘过滤带　"
            + _swatch(_FINAL_RGB) + " 最终边框"
            + "　　|　　<b>有效区边界/查询：</b>"
            + _swatch(_A_VALID_COLOR) + " A有效区　"
            + _swatch(_B_VALID_COLOR) + " B有效区　"
            + _swatch(_B2_VALID_COLOR) + " B二级有效区　"
            + _swatch(_VAR_COLOR) + " 变星　"
            + _swatch(_MPC_COLOR) + " MPC"
        )
        pw.addWidget(legend)

        # 变星/MPC 查询：进度(当前/总数) + 按钮（移到右侧）
        qbar = QHBoxLayout()
        qbar.addWidget(QLabel("变星 VSX:"))
        self.var_progress = QProgressBar()
        self.var_progress.setFormat("%v/%m")
        self.var_progress.setMinimumWidth(140)
        qbar.addWidget(self.var_progress, 1)
        qbar.addSpacing(12)
        qbar.addWidget(QLabel("MPC:"))
        self.mpc_progress = QProgressBar()
        self.mpc_progress.setFormat("%v/%m")
        self.mpc_progress.setMinimumWidth(140)
        qbar.addWidget(self.mpc_progress, 1)
        qbar.addSpacing(12)
        self.query_btn = QPushButton("查询变星/MPC")
        self.query_btn.clicked.connect(self._run_queries)
        self.query_pause_btn = QPushButton("暂停查询")
        self.query_pause_btn.setEnabled(False)
        self.query_pause_btn.clicked.connect(self._toggle_query_pause)
        qbar.addWidget(self.query_btn)
        qbar.addWidget(self.query_pause_btn)
        pw.addLayout(qbar)

        tabs = QTabWidget()
        self.tabs = tabs
        self.table = QTableWidget(0, len(_COLUMNS))
        self.table.setHorizontalHeaderLabels(_COLUMNS)
        self.table.horizontalHeader().setSectionResizeMode(QHeaderView.Interactive)
        self.table.horizontalHeader().setStretchLastSection(True)
        self.table.setSelectionBehavior(QAbstractItemView.SelectRows)
        self.table.setEditTriggers(QAbstractItemView.NoEditTriggers)
        self.table.itemSelectionChanged.connect(self._on_row_selected)
        self.table.cellDoubleClicked.connect(self._jump_table_crop)
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
        self.daily_zero_check = QCheckBox("仅 var=0 & mpc=0")
        self.daily_zero_check.setChecked(False)
        self.daily_zero_check.stateChanged.connect(lambda _=0: self._build_daily_summary())
        self.hide_noise_check = QCheckBox("隐藏 m-noise")
        self.hide_noise_check.setChecked(bool(DEFAULTS.get("daily_hide_noise", True)))
        self.hide_noise_check.stateChanged.connect(
            lambda _=0: self._apply_daily_manual_filter())
        self.hide_shift_check = QCheckBox("隐藏 m-pix-shift")
        self.hide_shift_check.setChecked(bool(DEFAULTS.get("daily_hide_shift", True)))
        self.hide_shift_check.stateChanged.connect(
            lambda _=0: self._apply_daily_manual_filter())
        self.daily_info = QLabel("（扫描选中目录下所有 *.gui_ai.json 汇总；仅命中）")
        self.daily_info.setStyleSheet("color:#666;")
        drow.addWidget(self.daily_btn)
        drow.addWidget(self.daily_export_btn)
        drow.addWidget(self.daily_zero_check)
        drow.addWidget(self.hide_noise_check)
        drow.addWidget(self.hide_shift_check)
        drow.addWidget(self.daily_info)
        drow.addStretch(1)
        dv.addLayout(drow)
        self.daily_table = QTableWidget(0, len(_DAILY_COLUMNS))
        self.daily_table.setHorizontalHeaderLabels(_DAILY_COLUMNS)
        self.daily_table.horizontalHeader().setSectionResizeMode(QHeaderView.Interactive)
        self.daily_table.horizontalHeader().setStretchLastSection(True)
        self.daily_table.setEditTriggers(QAbstractItemView.NoEditTriggers)
        self.daily_table.cellDoubleClicked.connect(self._jump_daily_row)
        self.daily_table.itemSelectionChanged.connect(self._on_daily_selected)
        dv.addWidget(self.daily_table)
        self._daily_tab_index = tabs.addTab(daily_wrap, "每日总表")

        splitter.addWidget(preview_wrap)
        splitter.addWidget(tabs)
        # 下方面板（检测结果/日志/小图墙/总表）给更大空间
        splitter.setStretchFactor(0, 2)
        splitter.setStretchFactor(1, 3)
        splitter.setSizes([320, 480])
        splitter.setChildrenCollapsible(False)
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
        self.tree.clear()
        top = QTreeWidgetItem([f"数据源: {root}"])
        top.setData(0, Qt.UserRole, root)
        top.setData(0, Qt.UserRole + 1, "dir")
        self.tree.addTopLevelItem(top)
        self._populate_tree(top)
        top.setExpanded(True)
        self.tree.setCurrentItem(top)

    def _populate_tree(self, item: QTreeWidgetItem) -> None:
        path = item.data(0, Qt.UserRole)
        try:
            entries = sorted(os.listdir(path))
        except Exception:
            return
        for name in entries:
            full = os.path.join(path, name)
            if os.path.isdir(full):
                child = QTreeWidgetItem([name])
                child.setData(0, Qt.UserRole, full)
                child.setData(0, Qt.UserRole + 1, "dir")
                child.addChild(QTreeWidgetItem(["(展开加载)"]))  # 占位
                item.addChild(child)
            elif name.lower().endswith((".fit", ".fits", ".fts")):
                child = QTreeWidgetItem([name])
                child.setData(0, Qt.UserRole, full)
                child.setData(0, Qt.UserRole + 1, "file")
                item.addChild(child)

    def _on_tree_expand(self, item: QTreeWidgetItem) -> None:
        if item.childCount() == 1 and item.child(0).data(0, Qt.UserRole) is None:
            item.takeChild(0)
            self._populate_tree(item)

    def _current_path(self) -> Optional[str]:
        it = self.tree.currentItem()
        return it.data(0, Qt.UserRole) if it is not None else None

    def _on_select(self, current: QTreeWidgetItem, _prev: QTreeWidgetItem) -> None:
        if current is None:
            return
        path = current.data(0, Qt.UserRole)
        if current.data(0, Qt.UserRole + 1) == "dir":
            self.sel_label.setText(f"文件夹: {path}")
            self._loaded_path = None
            return
        self.sel_label.setText(f"文件: {path}")
        # 选中文件时若有已保存结果，直接加载
        if self.worker is None or not self.worker.isRunning():
            self._maybe_load_result_for_file(path)

    def _maybe_load_result_for_file(self, path: str) -> None:
        key = os.path.normcase(os.path.abspath(path))
        if getattr(self, "_loaded_path", None) == key:
            return
        j = results_io.result_json_path(path)
        if not os.path.exists(j):
            return
        try:
            res = results_io.load_result(j)
        except Exception as ex:  # noqa: BLE001
            self._log(f"加载结果失败 {j}: {ex}")
            return
        self._show_result(res)
        self._loaded_path = key
        self._log(f"已自动加载结果: {os.path.basename(path)}{results_io.SUFFIX_JSON}")

    def _show_result(self, res: Dict, target_det: Optional[Dict] = None) -> None:
        """用单个结果替换当前显示；可跳到指定检测的裁切。"""
        self._results.clear()
        self._row_map.clear()
        self.table.setRowCount(0)
        self._manual_center = None
        self._manual_result = None
        self._on_file_done(res)
        if target_det is None:
            return
        row = None
        for r, e in enumerate(self._row_map):
            d = e["_det"]
            if (abs(d.get("x", 0) - target_det.get("x", 0)) < 1e-3
                    and abs(d.get("y", 0) - target_det.get("y", 0)) < 1e-3
                    and abs(d.get("score", 0) - target_det.get("score", 0)) < 1e-4
                    and d.get("status") == target_det.get("status")):
                row = r
                break
        if row is not None:
            self.table.selectRow(row)
            self._manual_center = None
            self._manual_result = None
        else:
            self._manual_center = (target_det.get("x"), target_det.get("y"))
            self._manual_result = res
        self.view_combo.setCurrentIndex(1)

    def _jump_table_crop(self, row: int, _col: int = 0) -> None:
        """检测结果表双击 -> 跳到该目标的裁切。"""
        if 0 <= row < len(self._row_map):
            self.table.selectRow(row)
            self._manual_center = None
            self._manual_result = None
            self.view_combo.setCurrentIndex(1)

    def _jump_daily_row(self, row: int, _col: int = 0) -> None:
        """每日总表双击 -> 载入该文件并跳到该目标的裁切。"""
        if not (0 <= row < len(self._daily_meta)):
            return
        meta = self._daily_meta[row]
        try:
            res = results_io.load_result(meta["json"])
        except Exception as ex:  # noqa: BLE001
            self._log(f"加载结果失败 {meta['json']}: {ex}")
            return
        self._show_result(res, meta.get("det"))

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
            "skip_existing": bool(self.skip_existing_check.isChecked()),
            "query_skip_done": bool(self.query_skip_check.isChecked()),
            "snr_filter": bool(self.snr_check.isChecked()),
            "shading_k": float(DEFAULTS["shading_k"]),
            "b2_ksize": int(DEFAULTS["b2_ksize"]),
            "noise_k": float(DEFAULTS["noise_k"]),
            "aperture_radius": int(DEFAULTS["aperture_radius"]),
            "aperture_annulus": int(DEFAULTS["aperture_annulus"]),
            "aperture_snr_min": float(self.snr_min_spin.value()),
            "det_center_search": int(DEFAULTS["det_center_search"]),
            "edge_band": int(self.edge_spin.value()),
            "isolated_filter": bool(self.isolated_check.isChecked()),
            "isolated_win": int(DEFAULTS["isolated_win"]),
            "isolated_k": float(DEFAULTS["isolated_k"]),
            "isolated_min_px": int(DEFAULTS["isolated_min_px"]),
            "shape_conc_max": float(DEFAULTS["shape_conc_max"]),
            "shape_fwhm_min": float(DEFAULTS["shape_fwhm_min"]),
            "shape_fwhm_max": float(DEFAULTS["shape_fwhm_max"]),
            "median_filter": bool(self.median_check.isChecked()),
            "median_ksize": int(DEFAULTS["median_ksize"]),
            "boundary_scale": int(DEFAULTS["boundary_scale"]),
            "ab_filter": bool(self.ab_filter_check.isChecked()),
            "ab_model_dir": str(DEFAULTS["ab_model_dir"]),
            "ab_keep_classes": list(DEFAULTS["ab_keep_classes"]),
            "ab_noise_max": float(self.ab_noise_spin.value()),
            "ab_pixelshift_max": float(self.ab_shift_spin.value()),
            "ab_patch": int(DEFAULTS["ab_patch"]),
            "anomaly_min_keep": int(self.file_anomaly_spin.value()),
            "amp": bool(DEFAULTS["amp"]),
        }

    # -------------------------------------------------------------- process
    def _start(self) -> None:
        target = self._current_path()
        if not target:
            QMessageBox.warning(self, "提示", "请先在左侧选择文件或文件夹节点")
            return
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
        self._loaded_path = None
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
        snr = d.get("snr")
        vals = [
            os.path.basename(res.get("b_path") or ""),
            _STATUS_TEXT.get(status, status),
            str(d.get("var_count", -1)),
            str(d.get("mpc_count", -1)),
            f"{d['x']:.1f}", f"{d['y']:.1f}", f"{d['score']:.3f}",
            "-" if snr is None else f"{snr:.1f}",
            "-" if d.get("conc") is None else f"{d.get('conc'):.2f}",
            f"{d['dx']:+.2f}", f"{d['dy']:+.2f}", f"{d['roll']:+.2f}",
            str(d["tile_x"]), str(d["tile_y"]),
        ] + _ab_cells(d)
        for c, v in enumerate(vals):
            item = QTableWidgetItem(v)
            if c == 1 and status in _STATUS_COLOR:
                item.setForeground(_STATUS_COLOR[status])
            self.table.setItem(row, c, item)
        entry = dict(d)
        entry["_result"] = res
        entry["_det"] = d
        self._row_map.append(entry)

    def _file_is_anomaly(self, res: Dict) -> bool:
        """整文件异常判定：命中的检测数 >= 阈值。"""
        try:
            thr = int(self.file_anomaly_spin.value())
        except Exception:
            thr = int(DEFAULTS["file_anomaly_min_keep"])
        return int(res.get("n_keep", 0) or 0) >= thr

    def _rebuild_table(self) -> None:
        self.table.setRowCount(0)
        self._row_map.clear()
        show_all = self.show_filtered_check.isChecked()
        file_filter = self.file_anomaly_check.isChecked()
        for res in self._results:
            if file_filter and self._file_is_anomaly(res):
                continue
            for d in res["detections"]:
                if not show_all and d.get("status", "keep") != "keep":
                    continue
                self._append_detection_row(res, d)
        self._refresh_preview()

    def _on_file_done(self, res: Dict) -> None:
        self._results.append(res)
        show_all = self.show_filtered_check.isChecked()
        file_filter = self.file_anomaly_check.isChecked()
        if file_filter and self._file_is_anomaly(res):
            self._log(
                f"  [整文件异常] 命中 {res.get('n_keep')} ≥ "
                f"{int(self.file_anomaly_spin.value())}，已跳过该文件全部检测")
            self._refresh_preview()
            return
        for d in res["detections"]:
            if not show_all and d.get("status", "keep") != "keep":
                continue
            self._append_detection_row(res, d)
        # 每个文件处理完都刷新一次预览（无选中行时显示该文件全图叠加）
        self._refresh_preview()

    # ---------------------------------------------------------- 结果持久化
    def _load_jsons(self, jsons: list) -> int:
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
        return loaded

    def _load_results_from_node(self) -> None:
        target = self._current_path()
        if not target:
            QMessageBox.warning(self, "提示", "请先在左侧选择文件或文件夹节点")
            return
        jsons = results_io.scan_results(target)
        if not jsons:
            QMessageBox.information(
                self, "提示", f"未找到结果文件 (*{results_io.SUFFIX_JSON})")
            return
        loaded = self._load_jsons(jsons)
        self._log(f"已加载 {loaded} 个结果 (来自 {target})")

    def _build_daily_summary(self) -> None:
        target = self._current_path() or self.root_edit.text().strip()
        jsons = results_io.scan_results(target)
        zero_only = self.daily_zero_check.isChecked()
        self._daily_rows = []
        self._daily_meta = []
        self._manual_result = None
        self._manual_center = None
        for j in jsons:
            info = results_io.parse_path_info(j)
            try:
                d = json.loads(Path(j).read_text(encoding="utf-8"))
            except Exception:
                continue
            fname = os.path.basename(d.get("b_path") or j)
            if (self.file_anomaly_check.isChecked()
                    and int(d.get("n_keep", 0) or 0) >= int(self.file_anomaly_spin.value())):
                continue  # 整文件异常，跳过
            for det in d.get("detections", []):
                if det.get("status", "keep") != "keep":
                    continue  # 总表只汇总命中
                if zero_only and not (det.get("var_count") == 0 and det.get("mpc_count") == 0):
                    continue
                st = det.get("status", "keep")
                self._daily_rows.append([
                    info["date"], info["tel"], info["region"], fname,
                    _STATUS_TEXT.get(st, st),
                    str(det.get("var_count", -1)), str(det.get("mpc_count", -1)),
                    f"{det.get('x', 0):.1f}", f"{det.get('y', 0):.1f}",
                    f"{det.get('score', 0):.3f}",
                    "-" if det.get("snr") is None else f"{det.get('snr'):.1f}",
                    "-" if det.get("conc") is None else f"{det.get('conc'):.2f}",
                    f"{det.get('dx', 0):+.2f}", f"{det.get('dy', 0):+.2f}",
                    f"{det.get('roll', 0):+.2f}",
                ] + _ab_cells(det) + [det.get("manual_class", "")])
                self._daily_meta.append({"json": j, "det": det})
        self.daily_table.setRowCount(len(self._daily_rows))
        for r, row in enumerate(self._daily_rows):
            for c, v in enumerate(row):
                self.daily_table.setItem(r, c, QTableWidgetItem(str(v)))
        self.daily_info.setText(
            f"（{len(jsons)} 个结果文件, 命中 {len(self._daily_rows)} 条"
            + ("，仅 var=0 & mpc=0" if zero_only else "") + "）")
        self._apply_daily_manual_filter()
        self._log(f"每日总表: {len(jsons)} 个结果文件, 命中 {len(self._daily_rows)} 条")

    def _daily_result(self, json_path: str) -> Dict:
        """按需加载并缓存某结果（含预览），供每日总表跳转显示 AB 裁切。"""
        cache = getattr(self, "_daily_res_cache", None)
        if cache is None:
            cache = {}
            self._daily_res_cache = cache
        res = cache.get(json_path)
        if res is None:
            res = results_io.load_result(json_path)  # 带预览
            if len(cache) >= 8:
                cache.clear()
            cache[json_path] = res
        return res

    def _show_daily_row_preview(self, r: int) -> None:
        """把 AB 预览切到每日总表第 r 行对应的检测。"""
        if not (0 <= r < len(self._daily_meta)):
            return
        meta = self._daily_meta[r]
        det = meta.get("det") or {}
        self.manual_status.setText(f"当前: {det.get('manual_class') or '未分类'}")
        try:
            res = self._daily_result(meta.get("json"))
        except Exception:
            return
        if res.get("a_u8") is None or res.get("b_raw") is None:
            return
        self._manual_result = res
        self._manual_center = (float(det.get("x", 0)), float(det.get("y", 0)))
        self._refresh_preview()

    def _on_daily_selected(self) -> None:
        """每日总表选中行 -> 更新状态标签并切换 AB 预览到该检测。"""
        r = self.daily_table.currentRow()
        if 0 <= r < len(self._daily_meta):
            self._show_daily_row_preview(r)

    def _update_manual_status(self, entry: Dict) -> None:
        det = (entry or {}).get("_det") or {}
        self.manual_status.setText(f"当前: {det.get('manual_class') or '未分类'}")

    def _sync_daily_row(self, det: Dict, value: str) -> None:
        """把某检测的手动分类同步到每日总表对应行。"""
        x = det.get("x", 0)
        y = det.get("y", 0)
        changed = False
        for r, meta in enumerate(self._daily_meta):
            d2 = meta.get("det") or {}
            try:
                if (abs(float(d2.get("x", 0)) - float(x)) < 1e-3
                        and abs(float(d2.get("y", 0)) - float(y)) < 1e-3):
                    d2["manual_class"] = value
                    if r < len(self._daily_rows):
                        self._daily_rows[r][_MANUAL_COL] = value
                    it = self.daily_table.item(r, _MANUAL_COL)
                    if it is not None:
                        it.setText(value)
                    changed = True
            except Exception:
                continue
        if changed:
            self._apply_daily_manual_filter()

    def _set_manual_class(self, value: str) -> None:
        """对当前选中的检测写入手动分类。

        目标按**当前所在标签页**选择：在“每日总表”页作用于总表当前行；
        其它页作用于“检测结果”表当前行。避免在总表（如“仅 var=0 & mpc=0”）
        操作时误改检测结果表里残留选中的另一行，导致跳到错误的行。
        """
        use_daily = (getattr(self, "_daily_tab_index", -1) >= 0
                     and self.tabs.currentIndex() == self._daily_tab_index
                     and self.daily_table.rowCount() > 0)
        entry = None if use_daily else self._selected_entry()
        if entry is not None:
            res = entry.get("_result") or {}
            det = entry.get("_det") or {}
            b = res.get("b_path")
            det["manual_class"] = value
            ok = False
            if b:
                jp = results_io.result_json_path(b)
                if os.path.exists(jp):
                    try:
                        ok = results_io.update_detection_field(
                            jp, det.get("x", 0), det.get("y", 0),
                            "manual_class", value)
                    except Exception as ex:  # noqa: BLE001
                        self._log(f"手动分类写回失败: {ex}")
            self._log(
                f"手动分类 {os.path.basename(b or '')} "
                f"({det.get('x')},{det.get('y')}) -> {value}"
                + ("" if ok else "  [写回失败/无结果文件]"))
            self._update_manual_status(entry)
            self._sync_daily_row(det, value)
            self._select_next_detection()
            return
        r = self.daily_table.currentRow()
        if r < 0 and self.daily_table.rowCount() > 0:
            r = 0
        if 0 <= r < len(self._daily_meta):
            meta = self._daily_meta[r]
            det = meta.get("det") or {}
            det["manual_class"] = value
            ok = False
            try:
                ok = results_io.update_detection_field(
                    meta["json"], det.get("x", 0), det.get("y", 0),
                    "manual_class", value)
            except Exception as ex:  # noqa: BLE001
                self._log(f"手动分类写回失败: {ex}")
            if r < len(self._daily_rows):
                self._daily_rows[r][_MANUAL_COL] = value
                it = self.daily_table.item(r, _MANUAL_COL)
                if it is not None:
                    it.setText(value)
            self._log(
                f"手动分类 {os.path.basename(meta.get('json', ''))} "
                f"({det.get('x')},{det.get('y')}) -> {value}"
                + ("" if ok else "  [写回失败]"))
            self._apply_daily_manual_filter()
            self._select_next_daily_row()
            self._show_daily_row_preview(self.daily_table.currentRow())
            return
        self._log("手动分类: 请先在检测结果表或每日总表中选中一条")

    def _select_next_detection(self) -> None:
        """移到检测结果表的下一条。"""
        sm = self.table.selectionModel()
        rows = sm.selectedRows() if sm else []
        nxt = (rows[0].row() + 1) if rows else 0
        if 0 <= nxt < self.table.rowCount():
            self.table.selectRow(nxt)
            it = self.table.item(nxt, 0)
            if it is not None:
                self.table.scrollToItem(it)

    def _select_next_daily_row(self) -> None:
        """移到每日总表的下一条可见行。"""
        n = self.daily_table.rowCount()
        r = self.daily_table.currentRow() + 1
        while r < n and self.daily_table.isRowHidden(r):
            r += 1
        if r < n:
            self.daily_table.selectRow(r)

    def _remove_current_detection(self) -> None:
        """移除当前选中的检测（内存 + 写回 .gui_ai.json）。"""
        entry = self._selected_entry()
        if entry is None:
            self._log("移除检测: 请先在检测结果表选中一条")
            return
        res = entry.get("_result")
        det = entry.get("_det")
        if res is None or det is None:
            return
        x, y = det.get("x", 0), det.get("y", 0)
        try:
            res["detections"].remove(det)
        except ValueError:
            pass
        res["n_total"] = len(res.get("detections", []))
        res["n_keep"] = sum(
            1 for d in res.get("detections", []) if d.get("status") == "keep")
        b = res.get("b_path")
        ok = False
        if b:
            jp = results_io.result_json_path(b)
            if os.path.exists(jp):
                try:
                    ok = results_io.remove_detection(jp, x, y)
                except Exception as ex:  # noqa: BLE001
                    self._log(f"移除检测写回失败: {ex}")
        self._manual_center = None
        self._manual_result = None
        self._rebuild_table()
        self._log(
            f"已移除检测 {os.path.basename(b or '')} ({x},{y})"
            + ("" if ok else "  [结果文件未更新]"))

    def _apply_daily_manual_filter(self) -> None:
        """按勾选隐藏手动分类为 m-noise / m-pix-shift 的行。"""
        reject = set()
        if self.hide_noise_check.isChecked():
            reject.add("m-noise")
        if self.hide_shift_check.isChecked():
            reject.add("m-pix-shift")
        n = 0
        for r in range(self.daily_table.rowCount()):
            mc = self._daily_rows[r][_MANUAL_COL] if r < len(self._daily_rows) else ""
            hidden = bool(reject and mc in reject)
            self.daily_table.setRowHidden(r, hidden)
            if hidden:
                n += 1
        if hasattr(self, "daily_info") and self.daily_table.rowCount() > 0:
            cur = self.daily_info.text().split("  |隐藏")[0]
            self.daily_info.setText(cur + (f"  |隐藏 {n} 条" if reject else ""))

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

    def _refilter_snr(self) -> None:
        """对已加载/已保存的结果重算孔径SNR并更新状态（无需重投影）。"""
        # 按选中节点（文件/文件夹）扫描已有结果并加载后再重算
        target = self._current_path()
        if target:
            jsons = results_io.scan_results(target)
            if jsons:
                loaded = self._load_jsons(jsons)
                self._log(f"重算前加载 {loaded} 个结果 (来自 {target})")
        if not self._results:
            QMessageBox.information(self, "提示", "没有结果可过滤")
            return
        r = int(DEFAULTS["aperture_radius"])
        R = int(DEFAULTS["aperture_annulus"])
        smin = float(self.snr_min_spin.value())
        conc_max = float(DEFAULTS["shape_conc_max"])
        fmin = float(DEFAULTS["shape_fwhm_min"])
        fmax = float(DEFAULTS["shape_fwhm_max"])
        search = int(DEFAULTS["det_center_search"])
        use_snr = self.snr_check.isChecked()
        n = 0
        for res in self._results:
            a_path = res.get("a_path")
            b_path = res.get("b_path")
            if not a_path or not b_path:
                continue
            try:
                wcs_a = self._fits_cache.get(a_path)[2]
                b_data = self._fits_cache.get(b_path)[1]
                wcs_b = self._fits_cache.get(b_path)[2]
            except Exception as ex:  # noqa: BLE001
                self._log(f"重算SNR失败(读取文件): {ex}")
                continue
            for det in res.get("detections", []):
                if det.get("status", "keep") not in ("keep", "low_snr", "spike"):
                    continue
                try:
                    w = wcs_a.all_pix2world([[float(det["x"]), float(det["y"])]], 0)[0]
                    bxy = wcs_b.all_world2pix([[float(w[0]), float(w[1])]], 0)[0]
                    bxi = int(round(float(bxy[0])))
                    byi = int(round(float(bxy[1])))
                    snr, conc, sig = _det_metrics(b_data, bxi, byi, r, R, search)
                except Exception:
                    continue
                det["snr"] = float(snr)
                det["conc"] = float(conc)
                det["fwhm"] = float(2.355 * sig)
                if use_snr:
                    fwhm = det["fwhm"]
                    if snr < smin:
                        det["status"] = "low_snr"
                    elif (conc > conc_max
                          or (fmin > 0 and fwhm < fmin)
                          or (fmax > 0 and fwhm > fmax)):
                        det["status"] = "spike"
                    else:
                        det["status"] = "keep"
                n += 1
        self._log(f"重算SNR/形状过滤: 处理 {n} 个检测（SNR下限 {smin}, conc上限 {conc_max}）")
        self._rebuild_table()
        self._refresh_preview()
        for res in self._results:
            try:
                results_io.save_result(res, self._params(), with_preview=False)
            except Exception:
                pass

    # --------------------------------------------------- 变星 / MPC 查询
    def _safe_wcs(self, a_path):
        if not a_path:
            return None
        try:
            return self._fits_cache.get(a_path)[2]
        except Exception:
            return None

    def _safe_epoch_mjd(self, b_path):
        if not b_path:
            return None
        try:
            header = self._fits_cache.get(b_path)[0][0].header
        except Exception:
            return None
        from astropy.time import Time
        for k in ("MJD-OBS", "MJD_OBS", "MJD"):
            if k in header:
                try:
                    return float(header[k])
                except Exception:
                    pass
        for k in ("JD", "JD-OBS", "JD_OBS", "JDAVG", "JD-AVG"):
            if k in header:
                try:
                    return float(header[k]) - 2400000.5
                except Exception:
                    pass
        for k in ("DATE-OBS", "DATEOBS"):
            if k in header:
                try:
                    return float(Time(str(header[k]), scale="utc").mjd)
                except Exception:
                    pass
        return None

    @staticmethod
    def _hits_to_pix(hits, wcs):
        out = []
        for h in hits:
            try:
                px, py = wcs.all_world2pix(float(h["ra"]), float(h["dec"]), 0)
                out.append([float(px), float(py)])
            except Exception:
                continue
        return out

    def _run_queries(self) -> None:
        # 已加载结果（含预览）则直接复用；否则按选中节点轻量读取（不加载预览 npz）
        if not self._results:
            target = self._current_path()
            if target:
                jsons = results_io.scan_results(target)
                for j in jsons:
                    try:
                        self._results.append(
                            results_io.load_result(j, with_preview=False))
                    except Exception:
                        continue
                if jsons:
                    self._log(f"查询前轻量加载 {len(self._results)} 个结果 (来自 {target})")
        if not self._results:
            QMessageBox.information(self, "提示", "没有可查询的结果（请先选中已处理的文件/文件夹）")
            return
        # 组织候选任务（仅命中的检测，需能定位 WCS）
        skip_done = bool(self.query_skip_check.isChecked())
        file_filter = self.file_anomaly_check.isChecked()
        try:
            anomaly_thr = int(self.file_anomaly_spin.value())
        except Exception:
            anomaly_thr = int(DEFAULTS["file_anomaly_min_keep"])
        tasks = []
        n_anom_skip = 0
        for res in self._results:
            if file_filter and int(res.get("n_keep", 0) or 0) >= anomaly_thr:
                for det in res.get("detections", []):
                    if det.get("status", "keep") == "keep":
                        det["var_count"] = det["mpc_count"] = -1
                n_anom_skip += 1
                continue
            wcs = self._safe_wcs(res.get("a_path"))
            epoch = self._safe_epoch_mjd(res.get("b_path"))
            for det in res.get("detections", []):
                if det.get("status", "keep") != "keep":
                    det.setdefault("var_count", -1)
                    det.setdefault("mpc_count", -1)
                    continue
                if wcs is None:
                    det["var_count"] = det["mpc_count"] = -1
                    continue
                try:
                    ra, dec = wcs.all_pix2world(float(det["x"]), float(det["y"]), 0)
                    tasks.append((det, wcs, epoch, float(ra), float(dec)))
                    if epoch is None and det.get("mpc_count") is None:
                        det["mpc_count"] = -1
                except Exception:
                    det["var_count"] = det["mpc_count"] = -1
        if n_anom_skip:
            self._log(f"整文件异常跳过查询: {n_anom_skip} 个文件（命中 ≥ {anomaly_thr}）")

        cfg = {
            "radius": float(DEFAULTS["query_radius_arcsec"]),
            "mag_limit": float(DEFAULTS["query_mag_limit"]),
            "vsx_timeout": float(DEFAULTS["vsx_timeout"]),
            "mpc_timeout": float(DEFAULTS["mpc_timeout"]),
            "threads": max(1, int(DEFAULTS.get("query_threads", 3))),
            "skip_done": skip_done,
            "vsx_host": DEFAULTS["vsx_host"], "vsx_port": int(DEFAULTS["vsx_port"]),
            "mpc_host": DEFAULTS["mpc_host"], "mpc_port": int(DEFAULTS["mpc_port"]),
            "vsx_url": f"{DEFAULTS['vsx_host']}:{DEFAULTS['vsx_port']}",
            "mpc_url": f"{DEFAULTS['mpc_host']}:{DEFAULTS['mpc_port']}",
            "n_no_epoch": 0,
        }
        cfg["n_no_epoch"] = sum(
            1 for t in tasks
            if t[2] is None and not (skip_done and t[0].get("mpc_count", -1) >= 0))

        # 后台线程执行，不阻塞界面
        self.query_btn.setEnabled(False)
        self.query_pause_btn.setEnabled(True)
        self.query_pause_btn.setText("暂停查询")
        self.query_worker = QueryWorker(tasks, cfg)
        self.query_worker.log.connect(self._log)
        self.query_worker.progress.connect(self._on_query_progress)
        self.query_worker.finished_all.connect(self._on_query_finished)
        self.query_worker.start()

    def _on_query_progress(self, phase: str, done: int, total: int) -> None:
        bar = self.var_progress if phase == "var" else self.mpc_progress
        bar.setRange(0, max(1, total))
        bar.setValue(done)

    def _toggle_query_pause(self) -> None:
        wk = self.query_worker
        if wk is None or not wk.isRunning():
            return
        if wk.is_paused():
            wk.resume()
            self.query_pause_btn.setText("暂停查询")
        else:
            wk.pause()
            self.query_pause_btn.setText("继续查询")

    def _on_query_finished(self) -> None:
        self.query_btn.setEnabled(True)
        self.query_pause_btn.setEnabled(False)
        self.query_pause_btn.setText("暂停查询")
        self._rebuild_table()
        self._refresh_preview()
        # 只改检测字段，不重写预览 npz（否则会重新压缩数百 MB，极慢）
        for res in self._results:
            try:
                results_io.save_result(res, self._params(), with_preview=False)
            except Exception:
                pass

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
        entry = self._selected_entry()
        if entry is not None:
            self._update_manual_status(entry)
        elif hasattr(self, "manual_status"):
            self.manual_status.setText("（未选中检测）")
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
        # 手动指定的结果（如每日总表跳转）优先，其次检测结果表选中项
        if self._manual_result is not None:
            return self._manual_result
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

    @staticmethod
    def _draw_ring(img: QImage, x: float, y: float, color: QColor,
                   r: int = 6, width: int = 2) -> None:
        p = QPainter(img)
        p.setPen(QPen(color, width))
        p.drawEllipse(int(round(x - r)), int(round(y - r)), 2 * r, 2 * r)
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
        show_validpoly_b2 = self.validpoly_b2_check.isChecked()
        show_edge = self.edge_check.isChecked()
        show_final = self.final_check.isChecked()
        show_all = self.show_filtered_check.isChecked()

        entry = self._selected_entry()
        manual = self._manual_center if self._manual_result is res else None
        mode_crop = (
            self.view_combo.currentIndex() == 1 and (entry is not None or manual is not None)
        )
        zoom_pm = None

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
                zoom_pm = self._zoom_pixmap(a_crop, b_crop)
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
        if show_edge and res.get("edge_prev") is not None:
            e = align2d(res["edge_prev"])
            a_rgb = overlay_ob(a_rgb, e[:, :, None], {0: _EDGE_BAND_RGB})
            b_rgb = overlay_ob(b_rgb, e[:, :, None], {0: _EDGE_BAND_RGB})
        if show_final and res.get("final_prev") is not None:
            fe = _mask_edge(align2d(res["final_prev"]) > 0)
            a_rgb = paint_mask(a_rgb, fe, _FINAL_RGB)
            b_rgb = paint_mask(b_rgb, fe, _FINAL_RGB)
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

        if show_validpoly or show_validpoly_b or show_validpoly_b2:
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
            if show_validpoly_b2 and res.get("b2_valid_polys"):
                _draw_polys(res["b2_valid_polys"], _B2_VALID_COLOR)

        # 变星/MPC 命中点（投影回像素）
        for det in res.get("detections", []):
            for hx, hy in det.get("var_hits", []) or []:
                xx, yy = (hx - vx0) * vscale, (hy - vy0) * vscale
                self._draw_ring(a_img, xx, yy, _VAR_COLOR)
                self._draw_ring(b_img, xx, yy, _VAR_COLOR)
            for hx, hy in det.get("mpc_hits", []) or []:
                xx, yy = (hx - vx0) * vscale, (hy - vy0) * vscale
                self._draw_ring(a_img, xx, yy, _MPC_COLOR)
                self._draw_ring(b_img, xx, yy, _MPC_COLOR)

        self.preview_a.set_image_size(a_img.width(), a_img.height())
        self.preview_b.set_image_size(b_img.width(), b_img.height())
        self.preview_a.setPixmap(QPixmap.fromImage(a_img).scaled(
            self.preview_a.width(), self.preview_a.height(),
            Qt.KeepAspectRatio, Qt.SmoothTransformation))
        self.preview_b.setPixmap(QPixmap.fromImage(b_img).scaled(
            self.preview_b.width(), self.preview_b.height(),
            Qt.KeepAspectRatio, Qt.SmoothTransformation))

        if zoom_pm is not None:
            self.preview_zoom.setPixmap(zoom_pm.scaled(
                self.preview_zoom.width(), self.preview_zoom.height(),
                Qt.KeepAspectRatio, Qt.FastTransformation))
        else:
            self.preview_zoom.setText("（选中目标后显示）")

    def _zoom_pixmap(self, a_crop: np.ndarray,
                     b_crop: np.ndarray) -> Optional[QPixmap]:
        """中心 zoom_patch×zoom_patch 原生裁块，A|B 最近邻放大对比。"""
        if a_crop is None or b_crop is None:
            return None
        try:
            import cv2
            zsz = int(DEFAULTS.get("zoom_patch", 32))
            disp = int(DEFAULTS.get("zoom_disp", 150))
            ay, ax = a_crop.shape[0] // 2, a_crop.shape[1] // 2
            half = zsz // 2

            def cut(x):
                h, w = x.shape[:2]
                y0 = max(0, min(h - zsz, ay - half)) if h >= zsz else 0
                x0 = max(0, min(w - zsz, ax - half)) if w >= zsz else 0
                sub = np.asarray(x[y0:y0 + zsz, x0:x0 + zsz], dtype=np.float32)
                return _linear_u8(sub)

            a_u8 = cv2.resize(cut(a_crop), (disp, disp),
                              interpolation=cv2.INTER_NEAREST)
            b_u8 = cv2.resize(cut(b_crop), (disp, disp),
                              interpolation=cv2.INTER_NEAREST)
            ai = numpy_to_qimage(a_u8)
            bi = numpy_to_qimage(b_u8)
            c = disp // 2
            cs, gp = max(6, disp // 8), max(2, disp // 20)
            self._draw_crosshair(ai, c, c, _SEL_COLOR, size=cs, gap=gp)
            self._draw_crosshair(bi, c, c, _SEL_COLOR, size=cs, gap=gp)
            w = ai.width() + bi.width() + 2
            comp = QImage(w, disp, QImage.Format_RGB888)
            comp.fill(QColor(0, 0, 0))
            p = QPainter(comp)
            p.drawImage(0, 0, ai)
            p.drawImage(ai.width() + 2, 0, bi)
            p.end()
            return QPixmap.fromImage(comp)
        except Exception:
            return None

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
            w.writerow(["file", "status", "x", "y", "score", "dx", "dy", "roll", "tile_x", "tile_y"]
                       + ["ab_class"] + [f"P({c})" for c in _AB_CLASSES])
            for e in self._row_map:
                w.writerow([
                    os.path.basename(e.get("_result", {}).get("b_path") or ""),
                    e.get("status", "keep"),
                    f"{e['x']:.2f}", f"{e['y']:.2f}", f"{e['score']:.4f}",
                    f"{e['dx']:.3f}", f"{e['dy']:.3f}", f"{e['roll']:.3f}",
                    e["tile_x"], e["tile_y"],
                ] + _ab_cells(e))
        self._log(f"已导出 CSV: {path}")

    def _export_anomaly_list(self) -> None:
        """导出整文件异常清单（命中数 >= 阈值）。"""
        thr = int(self.file_anomaly_spin.value())
        rows = []
        for res in self._results:
            if not self._file_is_anomaly(res):
                continue
            fs = res.get("file_stats") or {}
            rows.append([
                os.path.basename(res.get("b_path") or ""),
                res.get("n_keep"), res.get("n_total"),
                f"{fs.get('density', 0):.1f}" if fs.get('density') is not None else "-",
                "-" if fs.get("keep_conc_med") is None else f"{fs['keep_conc_med']:.3f}",
                "-" if fs.get("keep_fwhm_med") is None else f"{fs['keep_fwhm_med']:.2f}",
                "-" if fs.get("keep_snr_med") is None else f"{fs['keep_snr_med']:.1f}",
                fs.get("ab_reject", "-"),
                res.get("a_path") or "", res.get("b_path") or "",
            ])
        if not rows:
            QMessageBox.information(self, "提示", f"没有命中数 ≥ {thr} 的异常文件")
            return
        path, _ = QFileDialog.getSaveFileName(
            self, "导出异常文件清单", "gui_ai_anomaly_files.csv", "CSV (*.csv)")
        if not path:
            return
        with open(path, "w", newline="", encoding="utf-8-sig") as f:
            w = csv.writer(f)
            w.writerow(["file", "n_keep", "n_total", "density", "keep_conc_med",
                        "keep_fwhm_med", "keep_snr_med", "ab_reject", "a_path", "b_path"])
            w.writerows(rows)
        self._log(f"已导出异常文件清单 {len(rows)} 个: {path}")
        QMessageBox.information(self, "完成", f"异常文件 {len(rows)} 个，阈值 n_keep ≥ {thr}\n{path}")

    def _export_web_zip(self) -> None:
        """导出网页 ZIP（参考原版；输出根目录同原版，文件名带 V4）。"""
        # 只需 JSON（导出读原生裁块），不加载预览 npz；优先反映磁盘最新手动分类
        export_results = None
        target = self._current_path()
        if target:
            jsons = results_io.scan_results(target)
            if jsons:
                export_results = []
                for j in jsons:
                    try:
                        export_results.append(
                            results_io.load_result(j, with_preview=False))
                    except Exception:
                        continue
                self._log(f"导出前加载 {len(export_results)} 个结果 (来自 {target})")
        if not export_results:
            export_results = self._results
        if not export_results:
            QMessageBox.information(self, "提示", "没有可导出的结果")
            return
        keep_only = not self.show_filtered_check.isChecked()
        try:
            n, zip_path, out_dir = web_export.export_results_web(
                export_results,
                out_root=DEFAULTS["web_zip_root"],
                patch_size=int(self.crop_spin.value()),
                keep_only=keep_only,
                tag=str(DEFAULTS["web_zip_tag"]),
                group_radius_px=float(DEFAULTS["web_group_radius_px"]),
                manual_exclude=tuple(DEFAULTS.get(
                    "manual_reject_classes", ["m-noise", "m-pix-shift"])),
                log=self._log,
            )
        except Exception as ex:  # noqa: BLE001
            QMessageBox.critical(self, "导出失败", str(ex))
            return
        self._log(f"已导出网页ZIP: {zip_path} ({n} 个目标)")
        QMessageBox.information(
            self, "完成", f"已导出 {n} 个目标:\n{zip_path}")

    def _current_detection(self):
        """返回当前视图对应的 (res, det)。

        优先使用“每日总表”跳转/预览指定的目标（`_manual_result/_manual_center`，
        若该坐标附近有检测），否则用“检测结果”表当前选中行；都没有则 (None,None)。
        """
        entry = self._selected_entry()
        if (self._manual_result is not None and self._manual_center is not None
                and (entry is None or entry.get("_result") is not self._manual_result)):
            res = self._manual_result
            cx, cy = self._manual_center
            best, bd = None, None
            for d in res.get("detections", []):
                dd = (d.get("x", 0) - cx) ** 2 + (d.get("y", 0) - cy) ** 2
                if bd is None or dd < bd:
                    bd, best = dd, d
            if best is not None and bd is not None and bd <= 4.0:
                return res, best
        if entry is not None:
            return entry["_result"], entry["_det"]
        return None, None

    def _open_conc_viewer(self) -> None:
        """打开 conc 验证窗口：读取当前检测处的原始 B 局部块。"""
        res, det = self._current_detection()
        if res is None or det is None:
            QMessageBox.information(self, "提示", "请先选中一条检测")
            return
        a_path = res.get("a_path")
        b_path = res.get("b_path")
        if not a_path or not b_path:
            QMessageBox.information(self, "提示", "该结果缺少 A/B 文件路径")
            return
        try:
            wcs_a = self._fits_cache.get(a_path)[2]
            a_data = self._fits_cache.get(a_path)[1]
            b_data = self._fits_cache.get(b_path)[1]
            wcs_b = self._fits_cache.get(b_path)[2]
            wpt = wcs_a.all_pix2world([[float(det["x"]), float(det["y"])]], 0)[0]
            bxy = wcs_b.all_world2pix([[float(wpt[0]), float(wpt[1])]], 0)[0]
        except Exception as ex:  # noqa: BLE001
            QMessageBox.critical(self, "错误", f"读取/WCS 失败: {ex}")
            return
        size = 64

        def crop(data, cx, cy):
            half = size // 2
            h, w = data.shape
            x0, y0 = int(round(cx)) - half, int(round(cy)) - half
            x1, y1 = x0 + size, y0 + size
            xs, ys, xe, ye = max(0, x0), max(0, y0), min(w, x1), min(h, y1)
            out = np.asarray(data[ys:ye, xs:xe], dtype=np.float32)
            if out.size == 0:
                return None
            pad = ((max(0, -y0), max(0, y1 - h)), (max(0, -x0), max(0, x1 - w)))
            if any(pad[0]) or any(pad[1]):
                out = np.pad(out, pad, mode="edge")
            return out

        a_patch = crop(a_data, float(det["x"]), float(det["y"]))
        b_patch = crop(b_data, float(bxy[0]), float(bxy[1]))
        if a_patch is None or b_patch is None:
            QMessageBox.information(self, "提示", "该位置超出数据范围")
            return
        dlg = ConcDialog(
            self, {"A": a_patch, "B": b_patch}, size // 2, size // 2,
            int(DEFAULTS["aperture_radius"]), int(DEFAULTS["aperture_annulus"]),
            int(DEFAULTS["det_center_search"]))
        dlg.exec()

    def _export_train_fits(self) -> None:
        """导出中心16×16 A/B 训练数据到 ai_train_16pix_m：
        - var>0 或 mpc>0   -> var_or_mpc/（原方法）
        - var=0 且 mpc=0   -> 按人工标注 manual_class 分目录（未标注 -> unlabeled/）
        只读 JSON（不加载预览 npz），并跳过整文件异常图像。
        """
        export_results = None
        target = self._current_path()
        if target:
            jsons = results_io.scan_results(target)
            if jsons:
                export_results = []
                for j in jsons:
                    try:
                        export_results.append(
                            results_io.load_result(j, with_preview=False))
                    except Exception:
                        continue
                self._log(f"导出训练FITS前加载 {len(export_results)} 个结果 (来自 {target})")
        if not export_results:
            export_results = self._results
        if not export_results:
            QMessageBox.information(self, "提示", "没有可导出的结果")
            return
        from astropy.io import fits

        size = int(DEFAULTS["train_patch"])
        half = size // 2
        base = Path(DEFAULTS["train_export_dir_m"])
        (base / "var_or_mpc").mkdir(parents=True, exist_ok=True)

        def crop16(data, cx, cy):
            h, w = data.shape
            x0, y0 = int(round(cx)) - half, int(round(cy)) - half
            x1, y1 = x0 + size, y0 + size
            xs, ys, xe, ye = max(0, x0), max(0, y0), min(w, x1), min(h, y1)
            out = np.asarray(data[ys:ye, xs:xe])
            if out.size == 0:
                return None
            pad = ((max(0, -y0), max(0, y1 - h)), (max(0, -x0), max(0, x1 - w)))
            if any(pad[0]) or any(pad[1]):
                out = np.pad(out, pad, mode="edge")
            return out

        n_or = 0
        n_by_manual: Dict[str, int] = {}
        n_anomaly = n_skip = 0
        for res in export_results:
            if res.get("anomaly"):
                n_anomaly += 1
                continue
            a_path = res.get("a_path")
            b_path = res.get("b_path")
            if not a_path or not b_path:
                continue
            try:
                wcs_a = self._fits_cache.get(a_path)[2]
                a_data = self._fits_cache.get(a_path)[1]
                wcs_b = self._fits_cache.get(b_path)[2]
                b_data = self._fits_cache.get(b_path)[1]
            except Exception:
                continue
            stem = web_export.sanitize_name(Path(b_path).stem) if b_path else "unknown"
            for i, d in enumerate(res.get("detections", [])):
                vc = d.get("var_count", -1)
                mc = d.get("mpc_count", -1)
                manual = str(d.get("manual_class", "") or "").strip()
                if vc > 0 or mc > 0:
                    grp = "var_or_mpc"
                elif vc == 0 and mc == 0:
                    grp = manual if manual else "unlabeled"
                else:
                    n_skip += 1
                    continue
                try:
                    wpt = wcs_a.all_pix2world([[float(d["x"]), float(d["y"])]], 0)[0]
                    bxy = wcs_b.all_world2pix([[float(wpt[0]), float(wpt[1])]], 0)[0]
                except Exception:
                    n_skip += 1
                    continue
                a_crop = crop16(a_data, d["x"], d["y"])
                b_crop = crop16(b_data, float(bxy[0]), float(bxy[1]))
                if a_crop is None or b_crop is None:
                    n_skip += 1
                    continue
                hdu0 = fits.PrimaryHDU(np.asarray(a_crop))
                hdu1 = fits.ImageHDU(np.asarray(b_crop), name="B")
                hdr = hdu0.header
                hdr["VARC"] = int(vc)
                hdr["MPCC"] = int(mc)
                hdr["MANUAL"] = manual
                hdr["STATUS"] = str(d.get("status", ""))
                hdr["SCORE"] = float(d.get("score", 0.0))
                snr = d.get("snr")
                if snr is not None:
                    hdr["SNR"] = float(snr)
                hdr["PIXX"] = float(d["x"])
                hdr["PIXY"] = float(d["y"])
                grp_dir = base / web_export.sanitize_name(grp)
                grp_dir.mkdir(parents=True, exist_ok=True)
                out = grp_dir / f"{stem}_{i:04d}_16.fits"
                try:
                    fits.HDUList([hdu0, hdu1]).writeto(str(out), overwrite=True)
                except Exception:
                    n_skip += 1
                    continue
                if grp == "var_or_mpc":
                    n_or += 1
                else:
                    n_by_manual[grp] = n_by_manual.get(grp, 0) + 1
        detail = "  ".join(f"{k}={v}" for k, v in sorted(n_by_manual.items()))
        self._log(
            f"训练FITS导出 -> {base}: var>0/mpc>0={n_or}；按标注 {detail}；"
            f"异常文件跳过={n_anomaly}，检测跳过={n_skip}")
        QMessageBox.information(
            self, "完成",
            f"已导出到: {base}\n"
            f"var>0或mpc>0: {n_or}\n"
            f"var=0&mpc=0(按人工标注): {detail or '无'}\n"
            f"异常文件跳过: {n_anomaly}\n检测跳过: {n_skip}")


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

#!/usr/bin/env python3
"""gui_ai 默认配置。

新程序只依赖本目录内的模型（models_256_final），不修改项目其它目录。
"""

from __future__ import annotations

import json
from pathlib import Path

#: 本模块所在目录（gui_ai/）
BASE_DIR = Path(__file__).resolve().parent

#: 界面状态持久化文件
SETTINGS_PATH = BASE_DIR / "settings.json"

DEFAULTS = {
    # 界面文件树默认根目录
    "root": "E:/fix_data/download",
    # 参考模板根目录：<root>/<tel小写>/<天区>.fits
    "template_root": "E:/fix_data/template",
    # 模型目录：内含 best.pt / best_info.json（model.py 在本目录内共用）
    "model_dir": str(BASE_DIR / "models_256_final"),
    # 瓦片边长（= 模型输入尺寸）
    "tile_size": 256,
    # 相邻瓦片重叠比例（占瓦片宽度的百分比），默认 10%
    "overlap": 0.10,
    # 预览"选中目标裁切"的裁切边长（原生像素）
    "crop_size": 256,
    # 检测阈值（sigmoid 分数）
    "det_threshold": 0.45,
    # 跨瓦片去重半径（原生像素）
    "dedup_radius": 8.0,
    # 批推理 batch size
    "batch_size": 32,
    # 推理设备：auto / cpu / cuda
    "device": "auto",
    # 重投影分块行数
    "reproject_chunk_rows": 256,
    # 边界检测降采样倍率（有效区/二级/黑帽在半分辨率上算）
    "boundary_scale": 4,
    # GPU 混合精度推理
    "amp": True,
    # 处理 B 帧时，reproject 后 B 的无效区（nan）是否沿用 A 的灰度
    "fill_invalid_with_a": False,
    # 是否按 A、B 有效区的重叠部分过滤检测
    "valid_overlap_filter": True,
    # 处理时跳过已有结果(<B>.gui_ai.json)的文件（直接加载）
    "skip_existing": True,
    # 局部低信噪/暗边过滤（孔径信噪比）
    "snr_filter": True,
    "shading_k": 3.0,   # 暗边判定：黑帽响应 > k*sigma
    "b2_ksize": 21,     # 黑帽结构元尺寸(px)，约大于黑边宽度
    "noise_k": 3.0,     # 瓦片噪声超过 k*全局sigma 视为噪声区
    "aperture_radius": 3,     # 孔径半径(px)，约等于PSF半径
    "aperture_annulus": 6,    # 背景环外半径(px)
    "aperture_snr_min": 4.0,  # 孔径信噪比下限
    "det_center_search": 5,   # 在检测点周围重新定位峰值的搜索半径(px)
    # 形状判据（并入 SNR 过滤：剔除亮孤立尖峰，保留 PSF 星点）
    "shape_conc_max": 0.5,   # 集中度上限，超过判为尖峰
    "shape_fwhm_min": 0.8,   # FWHM 下限(px)，低于判为尖峰/过小
    "shape_fwhm_max": 0.0,   # FWHM 上限(px)，0=不限
    "edge_band": 5,     # A/B/B二级 有效区边界内边带宽度(px)，命中落在带内也过滤
    # 孤立点(宇宙线/热像素)过滤（默认关闭）
    "isolated_filter": False,
    "isolated_win": 7,      # 检测点周围窗口(px)
    "isolated_k": 3.0,      # 阈值 = 局部bg + k*sigma
    "isolated_min_px": 2,   # 连通块小于该像素数判为孤立点
    # 预处理：对 B 做中值滤波（默认开）
    "median_filter": True,
    "median_ksize": 3,
    # 变星(VSX)本地服务
    "vsx_host": "localhost",
    "vsx_port": 5000,
    # MPC(小行星)本地服务
    "mpc_host": "localhost",
    "mpc_port": 5001,
    # 查询参数
    "query_radius_arcsec": 36.0,
    "query_mag_limit": 16.0,
    "vsx_timeout": 30.0,   # 变星服务超时(秒)
    "mpc_timeout": 180.0,  # MPC 服务较慢(星历计算), 超时放长
    "query_threads": 3,    # 变星/MPC 查询并发线程数
    "query_skip_done": True,  # 跳过已查询(变星与MPC均已完成)的检测
    # 输出根目录（导出 CSV/PNG 时使用；留空表示写到 B 同目录）
    "output_root": "",
    # 网页 ZIP 导出（默认与原版一致: zip_output_directory）
    "web_zip_root": "E:/kats_sync",
    "web_zip_tag": "V4",
    "web_patch_size": 512,
}


def load_settings() -> dict:
    """读取界面状态（不存在/损坏时返回空 dict）。"""
    try:
        return json.loads(SETTINGS_PATH.read_text(encoding="utf-8"))
    except Exception:
        return {}


def save_settings(data: dict) -> None:
    """写入界面状态。"""
    try:
        SETTINGS_PATH.write_text(
            json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8"
        )
    except Exception:
        pass

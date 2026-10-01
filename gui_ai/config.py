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
    "batch_size": 16,
    # 推理设备：auto / cpu / cuda
    "device": "auto",
    # 重投影分块行数
    "reproject_chunk_rows": 256,
    # 处理 B 帧时，reproject 后 B 的无效区（nan）是否沿用 A 的灰度
    "fill_invalid_with_a": False,
    # 是否按 A、B 有效区的重叠部分过滤检测
    "valid_overlap_filter": True,
    # 处理时跳过已有结果(<B>.gui_ai.json)的文件（直接加载）
    "skip_existing": True,
    # 局部低信噪/暗边过滤
    "snr_filter": True,
    "shading_k": 3.0,   # 暗边判定：黑帽响应 > k*sigma
    "b2_ksize": 21,     # 黑帽结构元尺寸(px)，约大于黑边宽度
    "noise_k": 3.0,     # 瓦片噪声超过 k*全局sigma 视为噪声区
    "snr_min": 3.0,     # 峰值局部信噪比下限
    # 变星(VSX)本地服务
    "vsx_host": "localhost",
    "vsx_port": 5000,
    # MPC(小行星)本地服务
    "mpc_host": "localhost",
    "mpc_port": 5001,
    # 查询参数
    "query_radius_arcsec": 36.0,
    "query_mag_limit": 16.0,
    "query_timeout": 8.0,
    # 输出根目录（导出 CSV/PNG 时使用；留空表示写到 B 同目录）
    "output_root": "",
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

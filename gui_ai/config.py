#!/usr/bin/env python3
"""gui_ai 默认配置。

新程序只依赖本目录内的模型（models_256_final），不修改项目其它目录。
"""

from __future__ import annotations

from pathlib import Path

#: 本模块所在目录（gui_ai/）
BASE_DIR = Path(__file__).resolve().parent

DEFAULTS = {
    # 界面文件树默认根目录
    "root": "E:/fix_data/download",
    # 参考模板根目录：<root>/<tel小写>/<天区>.fits
    "template_root": "E:/fix_data/template",
    # 模型目录：内含 best.pt / best_info.json（model.py 在本目录内共用）
    "model_dir": str(BASE_DIR / "models_256_final"),
    # 单帧 OB/不可用区分割器（OBNet）目录
    "ob_model_dir": str(BASE_DIR / "models_ob"),
    # 单帧 OBNet 的不可用区判定阈值
    "ob_threshold": 0.5,
    # 是否默认用单帧 OBNet 的 B 不可用区过滤检测
    "use_obnet_filter": True,
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
    # 输出根目录（导出 CSV/PNG 时使用；留空表示写到 B 同目录）
    "output_root": "",
}

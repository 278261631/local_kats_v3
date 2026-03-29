#!/usr/bin/env python3
"""
Diff 集成占位模块

原 process_diff 对齐+差分流水线已移除，便于后续接入新实现。
当前保留：模板查找、文件名解析、可扩展的 process_diff 桩函数。
"""

from __future__ import annotations

import logging
import os
from typing import Any, Dict, Optional, Tuple

from filename_parser import FITSFilenameParser


class DiffOrbIntegration:
    """Diff 集成（占位）：仅模板解析；process_diff 待实现。"""

    def __init__(self, gui_callback=None):
        self.logger = logging.getLogger(__name__)
        self.filename_parser = FITSFilenameParser()
        self.gui_callback = gui_callback
    
    def is_available(self) -> bool:
        """模板查找等基础能力是否可用（不表示 Diff 流水线已实现）。"""
        return True

    def is_diff_pipeline_implemented(self) -> bool:
        """是否已实现完整 Diff 流水线（对齐+差分+检测）。当前为 False。"""
        return False

    def can_process_file(self, file_path: str, template_dir: str) -> Tuple[bool, str]:
        if not os.path.exists(file_path):
            return False, "文件不存在"
        
        if not os.path.exists(template_dir):
            return False, "模板目录不存在"
        
        parsed_info = self.filename_parser.parse_filename(file_path)
        if not parsed_info:
            return False, "无法解析文件名"
        
        if "tel_name" not in parsed_info:
            return False, "文件名中缺少望远镜信息"
        
        tel_name = parsed_info["tel_name"]
        k_number = parsed_info.get("k_full", parsed_info.get("k_number", ""))

        template_file = self.filename_parser.find_template_file(template_dir, tel_name, k_number)
        if not template_file:
            return False, f"未找到匹配的模板文件 (tel_name: {tel_name}, k_number: {k_number})"
        
        return True, f"找到模板文件: {os.path.basename(template_file)}"
    
    def find_template_file(self, download_file: str, template_dir: str) -> Optional[str]:
        try:
            parsed_info = self.filename_parser.parse_filename(download_file)
            if not parsed_info or "tel_name" not in parsed_info:
                self.logger.error("无法从文件名中提取信息: %s", download_file)
                return None
            
            tel_name = parsed_info["tel_name"]
            k_number = parsed_info.get("k_full", parsed_info.get("k_number", ""))

            template_file = self.filename_parser.find_template_file(
                template_dir, tel_name, k_number
            )
            
            if template_file:
                self.logger.info(
                    "为 %s 找到模板文件: %s",
                    os.path.basename(download_file),
                    os.path.basename(template_file),
                )
            else:
                self.logger.warning(
                    "未找到匹配的模板文件: tel_name=%s, k_number=%s", tel_name, k_number
                )
            
            return template_file
            
        except Exception as e:
            self.logger.error("查找模板文件时出错: %s", e)
            return None
    
    def process_diff(self, download_file: str, template_file: str, output_dir: Optional[str] = None, **kwargs: Any) -> Optional[Dict[str, Any]]:
        """
        Diff 流水线占位：尚未实现，始终返回 None。

        保留 **kwargs 以兼容旧调用方；新实现可在此收敛参数。
        """
        self.logger.warning(
            "process_diff 尚未实现（已移除旧 diff_orb 流水线），跳过: %s",
            os.path.basename(download_file) if download_file else "",
        )
        return None

    def get_diff_summary(self, result: Optional[Dict[str, Any]]) -> str:
        if not result or not result.get("success"):
            return "Diff 未执行或失败（流水线待接入新实现）"
        lines = [
            "diff操作完成",
            f"对齐状态: {'成功' if result.get('alignment_success') else '失败'}",
            f"检测到新亮点: {result.get('new_bright_spots', 0)} 个",
        ]
        return "\n".join(lines)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    integration = DiffOrbIntegration()
    print("Diff 流水线已实现:", integration.is_diff_pipeline_implemented())
    print("基础能力可用:", integration.is_available())

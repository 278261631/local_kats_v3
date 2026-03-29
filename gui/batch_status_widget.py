#!/usr/bin/env python3
"""
批量处理状态显示组件
用于显示批量下载和diff操作的实时状态
"""

import tkinter as tk
from tkinter import ttk
from typing import Dict, List


class BatchStatusWidget:
    """批量处理状态显示组件"""

    # 状态定义
    STATUS_PENDING = "pending"           # 等待处理
    STATUS_DOWNLOADING = "downloading"   # 下载中
    STATUS_DOWNLOAD_SUCCESS = "download_success"  # 下载成功
    STATUS_DOWNLOAD_FAILED = "download_failed"    # 下载失败
    STATUS_DOWNLOAD_SKIPPED = "download_skipped"  # 跳过下载
    STATUS_WCS_CHECKING = "wcs_checking"          # 检查WCS
    STATUS_WCS_FOUND = "wcs_found"                # 有WCS信息
    STATUS_WCS_MISSING = "wcs_missing"            # 缺少WCS信息
    STATUS_DIFF_PROCESSING = "diff_processing"    # Diff处理中
    STATUS_DIFF_SUCCESS = "diff_success"          # Diff成功
    STATUS_DIFF_FAILED = "diff_failed"            # Diff失败
    STATUS_DIFF_SKIPPED = "diff_skipped"          # 跳过Diff

    # 状态图标和颜色
    STATUS_CONFIG = {
        STATUS_PENDING: {"icon": "⏳", "color": "gray", "text": "等待"},
        STATUS_DOWNLOADING: {"icon": "⬇️", "color": "blue", "text": "下载中"},
        STATUS_DOWNLOAD_SUCCESS: {"icon": "✓", "color": "green", "text": "下载成功"},
        STATUS_DOWNLOAD_FAILED: {"icon": "✗", "color": "red", "text": "下载失败"},
        STATUS_DOWNLOAD_SKIPPED: {"icon": "⊙", "color": "orange", "text": "跳过下载"},
        STATUS_WCS_CHECKING: {"icon": "🔍", "color": "blue", "text": "检查WCS"},
        STATUS_WCS_FOUND: {"icon": "✓", "color": "green", "text": "有WCS"},
        STATUS_WCS_MISSING: {"icon": "⚠", "color": "orange", "text": "缺WCS"},
        STATUS_DIFF_PROCESSING: {"icon": "⚙", "color": "blue", "text": "Diff处理"},
        STATUS_DIFF_SUCCESS: {"icon": "✓", "color": "green", "text": "Diff成功"},
        STATUS_DIFF_FAILED: {"icon": "✗", "color": "red", "text": "Diff失败"},
        STATUS_DIFF_SKIPPED: {"icon": "⊙", "color": "orange", "text": "跳过Diff"},
    }

    def __init__(self, parent_frame):
        """
        初始化批量状态显示组件

        Args:
            parent_frame: 父框架
        """
        self.parent_frame = parent_frame
        self.file_items = {}  # 文件名 -> 显示项的映射
        self.container = None
        self.canvas = None
        self.scrollbar = None
        self.inner_frame = None

        # 为了在上千条记录时保持可见性与性能，限制可见条目数量（超过后优先清理已完成项）
        self.max_visible_items = 1200

    def create_widget(self):
        """创建状态显示组件"""
        # 创建容器框架
        self.container = ttk.LabelFrame(self.parent_frame, text="批量处理状态", padding=10)
        self.container.pack(fill=tk.BOTH, expand=True, pady=(10, 0))

        # 创建Canvas和滚动条（移除固定高度，随父容器自适应）
        self.canvas = tk.Canvas(self.container, bg="white")
        self.scrollbar = ttk.Scrollbar(self.container, orient=tk.VERTICAL, command=self.canvas.yview)

        # 创建内部框架
        self.inner_frame = ttk.Frame(self.canvas)

        # 配置Canvas
        self.canvas.configure(yscrollcommand=self.scrollbar.set)

        # 布局
        self.canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        self.scrollbar.pack(side=tk.RIGHT, fill=tk.Y)

        # 将内部框架添加到Canvas
        self.canvas_window = self.canvas.create_window((0, 0), window=self.inner_frame, anchor=tk.NW)

        # 绑定事件
        self.inner_frame.bind("<Configure>", self._on_frame_configure)
        self.canvas.bind("<Configure>", self._on_canvas_configure)

        return self.container

    def _on_frame_configure(self, event=None):
        """内部框架大小改变时更新滚动区域"""
        self.canvas.configure(scrollregion=self.canvas.bbox("all"))

    def _on_canvas_configure(self, event):
        """Canvas大小改变时调整内部框架宽度"""
        self.canvas.itemconfig(self.canvas_window, width=event.width)

    def show(self):
        """显示状态组件"""
        if self.container:
            self.container.pack(fill=tk.BOTH, expand=True, pady=(10, 0))

    def hide(self):
        """隐藏状态组件"""
        if self.container:
            self.container.pack_forget()

    def clear(self):
        """清空所有状态项"""
        for widget in self.inner_frame.winfo_children():
            widget.destroy()
        self.file_items.clear()

    def add_file(self, filename: str):
        """
        添加文件到状态列表（惰性创建UI：先只记录，首次状态更新时再创建行）

        Args:
            filename: 文件名
        """
        if filename in self.file_items:
            return

        # 仅记录数据，不立即创建UI，避免一次性成千上万条目撑爆滚动区域
        self.file_items[filename] = {
            "frame": None,
            "status_label": None,
            "name_label": None,
            "text_label": None,
            "status": self.STATUS_PENDING
        }

    def update_status(self, filename: str, status: str, extra_info: str = ""):
        """
        更新文件状态

        Args:
            filename: 文件名
            status: 状态代码
            extra_info: 额外信息（如错误消息、亮点数量等）
        """
        if filename not in self.file_items:
            return

        item = self.file_items[filename]
        config = self.STATUS_CONFIG.get(status, {"icon": "?", "color": "black", "text": "未知"})

        # 如尚未创建UI行，则在首次更新时创建
        if not item.get("frame"):
            try:
                # 先清理旧的完成项，尽量腾出显示空间
                self._prune_if_needed()
                item_frame = ttk.Frame(self.inner_frame)
                item_frame.pack(fill=tk.X, pady=2)
                status_label = tk.Label(item_frame, text="⏳", font=("Arial", 12), width=3)
                status_label.pack(side=tk.LEFT)
                name_label = tk.Label(item_frame, text=filename, anchor=tk.W, width=40)
                name_label.pack(side=tk.LEFT, padx=(5, 10))
                text_label = tk.Label(item_frame, text="等待", anchor=tk.W, width=15, fg="gray")
                text_label.pack(side=tk.LEFT)
                item["frame"] = item_frame
                item["status_label"] = status_label
                item["name_label"] = name_label
                item["text_label"] = text_label
                self._on_frame_configure()
            except Exception:
                pass

        # 更新图标
        if item.get("status_label"):
            item["status_label"].config(text=config["icon"])

        # 更新状态文本
        status_text = config["text"]
        if extra_info:
            status_text += f" - {extra_info}"
        if item.get("text_label"):
            item["text_label"].config(text=status_text, fg=config["color"])

        # 保存当前状态
        item["status"] = status
        # 将该项滚动入视野，并尝试清理过旧的已完成项
        try:
            self.scroll_to_file(filename)
        except Exception:
            pass
        self._prune_if_needed()

    def get_status(self, filename: str) -> str:
        """
        获取文件当前状态

        Args:
            filename: 文件名

        Returns:
            str: 状态代码
        """
        if filename in self.file_items:
            return self.file_items[filename]["status"]
        return None

    def get_statistics(self) -> Dict[str, int]:
        """
        获取状态统计

        Returns:
            dict: 各状态的数量统计
        """
        stats = {
            "total": len(self.file_items),
            "pending": 0,
            "downloading": 0,
            "download_success": 0,
            "download_failed": 0,
            "download_skipped": 0,
            "diff_processing": 0,
            "diff_success": 0,
            "diff_failed": 0,
            "diff_skipped": 0,
        }

        for item in self.file_items.values():
            status = item["status"]
            if status in stats:
                stats[status] += 1

        return stats

    def scroll_to_bottom(self):
        """滚动到底部"""
        self.canvas.update_idletasks()
        self.canvas.yview_moveto(1.0)

    def scroll_to_file(self, filename: str):
        """
        滚动到指定文件

        Args:
            filename: 文件名
        """
        if filename not in self.file_items:
            return

        item = self.file_items[filename]
        frame = item.get("frame")
        if not frame:
            return

        # 获取框架在Canvas中的位置
        self.canvas.update_idletasks()
        bbox = self.canvas.bbox(self.canvas_window)
        if bbox:
            frame_y = frame.winfo_y()
            canvas_height = self.canvas.winfo_height()
            scroll_region = self.canvas.cget("scrollregion").split()
            if scroll_region:
                total_height = float(scroll_region[3])
                if total_height > 0:
                    # 计算滚动位置（将文件项显示在中间）
                    scroll_pos = (frame_y - canvas_height / 2) / total_height
                    scroll_pos = max(0, min(1, scroll_pos))
                    self.canvas.yview_moveto(scroll_pos)


    def _prune_if_needed(self):
        """
        当可见UI行数超过上限时，优先清理最早的“已完成”项的UI（仅销毁UI，不删除数据），
        以保持画布滚动区域可达最新项。
        """
        try:
            visible_count = len(self.inner_frame.winfo_children())
            if visible_count <= self.max_visible_items:
                return
            terminal_statuses = {
                self.STATUS_DOWNLOAD_FAILED,
                self.STATUS_DOWNLOAD_SKIPPED,
                self.STATUS_DOWNLOAD_SUCCESS,
                self.STATUS_DIFF_SUCCESS,
                self.STATUS_DIFF_FAILED,
                self.STATUS_DIFF_SKIPPED,
            }
            # 依插入顺序清理最早的已完成项
            for k, v in self.file_items.items():
                if visible_count <= self.max_visible_items:
                    break
                if v.get("frame") is not None and v.get("status") in terminal_statuses:
                    try:
                        v.get("frame").destroy()
                    except Exception:
                        pass
                    # 仅移除UI引用，保留数据
                    v["frame"] = None
                    v["status_label"] = None
                    v["name_label"] = None
                    v["text_label"] = None
                    visible_count -= 1
            # 更新滚动区域
            self._on_frame_configure()
        except Exception:
            # 保守处理，出现异常不影响主流程
            pass


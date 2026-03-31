#!/usr/bin/env python3
"""
单实例 IPC 工具：
- 基于项目路径生成稳定端口
- 使用 localhost TCP 发送简短 JSON 命令
"""

import hashlib
import json
import os
import socket
from typing import Dict, Optional, Tuple


def get_endpoint() -> Tuple[str, int]:
    """根据项目根路径生成稳定的本地 IPC 端口。"""
    project_root = os.path.normcase(
        os.path.normpath(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
    )
    digest = hashlib.sha1(project_root.encode("utf-8")).hexdigest()
    # 选择非保留端口范围，降低冲突概率
    port = 43000 + (int(digest[:8], 16) % 10000)
    return "127.0.0.1", port


def send_command(payload: Dict, timeout: float = 0.8) -> bool:
    """向已运行实例发送一条 JSON 命令。发送成功返回 True。"""
    host, port = get_endpoint()
    data = (json.dumps(payload, ensure_ascii=False) + "\n").encode("utf-8")
    try:
        with socket.create_connection((host, port), timeout=timeout) as conn:
            conn.sendall(data)
        return True
    except Exception:
        return False


def recv_json_line(conn: socket.socket, max_bytes: int = 65536) -> Optional[Dict]:
    """从连接中读取一行 JSON 并解析。"""
    chunks = []
    total = 0
    while total < max_bytes:
        part = conn.recv(4096)
        if not part:
            break
        chunks.append(part)
        total += len(part)
        if b"\n" in part:
            break

    if not chunks:
        return None

    raw = b"".join(chunks).split(b"\n", 1)[0].strip()
    if not raw:
        return None
    try:
        obj = json.loads(raw.decode("utf-8", errors="ignore"))
        return obj if isinstance(obj, dict) else None
    except Exception:
        return None

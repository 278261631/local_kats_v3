#!/usr/bin/env python3
"""变星(VSX) / MPC(小行星) 本地服务客户端。

服务接口（原版 GUI 约定）：
    变星:  GET http://<host>:5000/search?ra=&dec=&radius=(角秒)
           -> {"results":[{"name","ra","dec","mag_max","mag_min","period"}]}
    MPC :  GET http://<host>:5001/search?ra=&dec=&epoch=(MJD)&radius=(角秒)
           -> {"success":bool,"results":[{"name","ra","dec","mag"}],"degraded":bool}

服务未开/失败时抛异常，由调用方记 -1。
"""

from __future__ import annotations

import json
from typing import Dict, List
from urllib.parse import urlencode
from urllib.request import urlopen


def query_vsx(ra: float, dec: float, radius_arcsec: float,
              mag_limit: float = 16.0, host: str = "localhost",
              port: int = 5000, timeout: float = 8.0) -> List[Dict]:
    params = {"ra": ra, "dec": dec, "radius": radius_arcsec}
    url = f"http://{host}:{port}/search?{urlencode(params)}"
    with urlopen(url, timeout=timeout) as resp:
        payload = json.loads(resp.read().decode("utf-8", errors="replace"))
    rows = payload.get("results") or []
    out: List[Dict] = []
    for row in rows:
        if not isinstance(row, dict):
            continue
        mm = row.get("mag_max")
        try:
            if mm is not None and float(mm) > float(mag_limit):
                continue
        except Exception:
            pass
        out.append({
            "name": str(row.get("name") or row.get("label") or ""),
            "ra": row.get("ra"), "dec": row.get("dec"),
            "mag_max": mm, "mag_min": row.get("mag_min"),
            "period": row.get("period"),
        })
    return out


def query_mpc(ra: float, dec: float, epoch_mjd: float, radius_arcsec: float,
              host: str = "localhost", port: int = 5001,
              timeout: float = 8.0) -> List[Dict]:
    params = {"ra": ra, "dec": dec, "epoch": epoch_mjd, "radius": radius_arcsec}
    url = f"http://{host}:{port}/search?{urlencode(params)}"
    with urlopen(url, timeout=timeout) as resp:
        payload = json.loads(resp.read().decode("utf-8", errors="replace"))
    if not payload.get("success", False):
        raise RuntimeError(str(payload.get("error") or "pympc 服务返回失败"))
    rows = payload.get("results") or []
    return [
        {"name": str(r.get("name", "")), "ra": r.get("ra"),
         "dec": r.get("dec"), "mag": r.get("mag")}
        for r in rows if isinstance(r, dict)
    ]

#!/usr/bin/env python3
"""控制台查询核心（无 Qt）：组织任务 + 两阶段并发查询变星/MPC。"""

from __future__ import annotations

import threading
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait

import query_servers
import results_io
from native_crop import FitsCache


def load_all_results(root: str) -> list:
    jsons = results_io.scan_results(root)
    out = []
    for j in jsons:
        try:
            out.append(results_io.load_result(j))
        except Exception:
            continue
    return out


def epoch_mjd_from_header(header) -> float | None:
    if header is None:
        return None
    try:
        from astropy.time import Time
    except Exception:
        Time = None
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
    if Time is not None:
        for k in ("DATE-OBS", "DATEOBS"):
            if k in header:
                try:
                    return float(Time(str(header[k]), scale="utc").mjd)
                except Exception:
                    pass
    return None


def _hits_to_pix(hits, wcs) -> list:
    out = []
    for h in hits:
        try:
            px, py = wcs.all_world2pix(float(h["ra"]), float(h["dec"]), 0)
            out.append([float(px), float(py)])
        except Exception:
            continue
    return out


def build_tasks(results: list, cache: FitsCache | None = None,
                anomaly_min_keep: int = 0) -> tuple:
    """返回 (tasks, n_no_epoch)。tasks: (det, wcs, epoch, ra, dec)。

    anomaly_min_keep>0 时，命中的检测数 ≥ 该值的文件视为整文件异常，跳过查询
    （其命中检测的 var_count/mpc_count 置 -1）。
    """
    cache = cache or FitsCache(max_items=2)
    tasks = []
    n_no_epoch = 0
    for res in results:
        if anomaly_min_keep > 0 and int(res.get("n_keep", 0) or 0) >= int(anomaly_min_keep):
            for det in res.get("detections", []):
                if det.get("status", "keep") == "keep":
                    det["var_count"] = det["mpc_count"] = -1
            continue
        a_path = res.get("a_path")
        b_path = res.get("b_path")
        wcs = None
        if a_path:
            try:
                wcs = cache.get(a_path)[2]
            except Exception:
                wcs = None
        epoch = None
        if b_path:
            try:
                epoch = epoch_mjd_from_header(cache.get(b_path)[0][0].header)
            except Exception:
                epoch = None
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
    n_no_epoch = sum(1 for t in tasks if t[2] is None)
    return tasks, n_no_epoch


def run_queries(tasks: list, cfg: dict, log, progress) -> dict:
    """两阶段查询：先变星全部结束，再 MPC。log(str), progress(phase,done,total)。"""
    lock = threading.Lock()
    vstate = {"down": False, "logged": False}
    mstate = {"down": False, "logged": False}

    def do_vsx(task):
        det, wcs, epoch, ra, dec = task
        if vstate["down"]:
            det["var_count"] = -1
            return []
        try:
            hits = query_servers.query_vsx(
                ra, dec, cfg["radius"], mag_limit=cfg["mag_limit"],
                host=cfg["vsx_host"], port=cfg["vsx_port"], timeout=cfg["vsx_timeout"])
            det["var_count"] = len(hits)
            det["var_hits"] = _hits_to_pix(hits, wcs)
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
                host=cfg["mpc_host"], port=cfg["mpc_port"], timeout=cfg["mpc_timeout"])
            det["mpc_count"] = len(hits)
            det["mpc_hits"] = _hits_to_pix(hits, wcs)
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
        progress(phase, 0, total)
        log(f"查询{label}: {total} 个目标（并发 {cfg['threads']}）")
        done = 0
        idx = 0
        pending = {}
        if total == 0:
            return
        with ThreadPoolExecutor(max_workers=cfg["threads"]) as pool:
            while idx < total or pending:
                while idx < total and len(pending) < cfg["threads"]:
                    pending[pool.submit(work, phase_tasks[idx])] = True
                    idx += 1
                if not pending:
                    break
                done_futs, _ = wait(list(pending), timeout=0.2,
                                    return_when=FIRST_COMPLETED)
                for fut in done_futs:
                    pending.pop(fut, None)
                    try:
                        for m in fut.result():
                            log(m)
                    except Exception as ex:  # noqa: BLE001
                        log(f"查询任务异常: {ex}")
                    done += 1
                    progress(phase, done, total)

    skip = bool(cfg.get("skip_done", True))
    vsx_tasks = [t for t in tasks if not (skip and t[0].get("var_count", -1) >= 0)]
    mpc_tasks = [t for t in tasks
                 if not (skip and (t[0].get("mpc_count", -1) >= 0 or t[2] is None))]
    if skip:
        log(f"跳过已查询: 变星 {len(tasks)-len(vsx_tasks)} 个, MPC {len(tasks)-len(mpc_tasks)} 个")
    run_phase(do_vsx, vsx_tasks, "var", "变星")
    run_phase(do_mpc, mpc_tasks, "mpc", "MPC")
    log(f"查询完成（变星服务不可用={vstate['down']}, MPC服务不可用={mstate['down']}）")
    return {"vsx_down": vstate["down"], "mpc_down": mstate["down"]}

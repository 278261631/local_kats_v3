#!/usr/bin/env python3
"""网页 ZIP 导出（参考原版 export_filtered_web_zip_runner）。

对每个检测裁 A|B 局部图、拉伸成 PNG，生成 index.html + meta.json 并打包 ZIP。
输出根目录默认与原版一致(zip_output_directory)，ZIP 文件名带 V4 以区分。
"""

from __future__ import annotations

import html
import json
import re
import shutil
import zipfile
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
from PIL import Image

from native_crop import FitsCache, load_native_pair_crops

#: 拉伸强度参数 (p_low, p_high, gamma, asinh_scale)
_LEVELS = {
    "linear": (1.0, 99.5, 1.0, 0.0),   # 与预览裁切一致
    "low": (5.0, 99.5, 1.0, 3.0),
    "medium": (2.0, 99.8, 0.9, 5.0),
    "high": (0.5, 99.95, 0.75, 8.0),
}


def sanitize_name(value) -> str:
    text = str(value or "").strip()
    text = re.sub(r'[<>:"/\\|?*\s%#&]+', "_", text)
    text = re.sub(r"_+", "_", text).strip("._")
    return text or "item"


def stretch_patch(patch: np.ndarray, level: str = "high") -> np.ndarray:
    arr = np.asarray(patch, dtype=np.float64)
    finite = np.isfinite(arr)
    if arr.size == 0 or not finite.any():
        return np.zeros(arr.shape if arr.size else (1, 1), dtype=np.uint8)
    valid = arr[finite]
    p_low, p_high, gamma, asinh_scale = _LEVELS.get(str(level).lower(), _LEVELS["high"])
    lo = float(np.percentile(valid, p_low))
    hi = float(np.percentile(valid, p_high))
    if not np.isfinite(lo):
        lo = float(np.min(valid))
    if not np.isfinite(hi):
        hi = float(np.max(valid))
    if hi <= lo:
        hi = lo + 1e-8
    norm = np.clip((np.clip(arr, lo, hi) - lo) / (hi - lo), 0.0, 1.0)
    if gamma != 1.0:
        norm = np.power(norm, gamma)
    if asinh_scale > 0:
        norm = np.arcsinh(norm * asinh_scale) / np.arcsinh(asinh_scale)
    out = np.zeros_like(arr, dtype=np.float64)
    out[finite] = np.clip(norm[finite], 0.0, 1.0)
    return np.round(out * 255.0).astype(np.uint8)


def _with_cross(u8: np.ndarray, size: int = 6, gap: int = 3) -> np.ndarray:
    """灰度转 RGB 并在中心画**空心十字**（中心留空）。"""
    img = np.repeat(u8[:, :, None], 3, axis=2)
    h, w, _ = img.shape
    cx, cy = w // 2, h // 2
    img[cy, max(0, cx - size):max(0, cx - gap)] = (255, 0, 0)
    img[cy, min(w, cx + gap + 1):min(w, cx + size + 1)] = (255, 0, 0)
    img[max(0, cy - size):max(0, cy - gap), cx] = (255, 0, 0)
    img[min(h, cy + gap + 1):min(h, cy + size + 1), cx] = (255, 0, 0)
    return img


def _side_by_side(a_u8: np.ndarray, b_u8: np.ndarray) -> np.ndarray:
    a = _with_cross(a_u8)
    b = _with_cross(b_u8)
    h = max(a.shape[0], b.shape[0])

    def pad(x):
        if x.shape[0] == h:
            return x
        return np.pad(x, ((0, h - x.shape[0]), (0, 0), (0, 0)), mode="edge")

    sep = np.zeros((h, 2, 3), dtype=np.uint8)
    return np.concatenate([pad(a), sep, pad(b)], axis=1)


def _radec(wcs, x: float, y: float):
    if wcs is None:
        return None, None
    try:
        w = wcs.all_pix2world([[float(x), float(y)]], 0)[0]
        return float(w[0]), float(w[1])
    except Exception:
        return None, None


def _card(item: Dict) -> str:
    return (
        '<div class="card">'
        f'<img src="{html.escape(item["img_rel"])}" alt="patch">'
        '<div class="meta">'
        f'<div>status: {html.escape(str(item.get("status", "")))}</div>'
        f'<div>score: {html.escape(str(item.get("score", "")))}</div>'
        f'<div>SNR: {html.escape(str(item.get("snr", "")))}</div>'
        f'<div>var/mpc: {html.escape(str(item.get("var_count", "")))} / '
        f'{html.escape(str(item.get("mpc_count", "")))}</div>'
        f'<div>ra/dec: {html.escape(str(item.get("radec", "")))}</div>'
        f'<div>xy: {html.escape(str(item.get("xy", "")))}</div>'
        f'<div>file: {html.escape(str(item.get("file", "")))}</div>'
        "</div></div>"
    )


def _group_blocks(items: List[Dict]) -> str:
    grouped: Dict[str, List[Dict]] = {}
    for it in items:
        grouped.setdefault(it.get("group", "UNGROUPED"), []).append(it)
    blocks = []
    for key in sorted(grouped.keys()):
        cards = "".join(_card(it) for it in grouped[key])
        blocks.append(
            '<section class="group">'
            f'<h3>分组: {html.escape(key)} <span class="count">({len(grouped[key])})</span></h3>'
            f'<div class="grid">{cards}</div></section>'
        )
    return "\n".join(blocks)


def _build_html(items: List[Dict], summary: str, patch_size: int, hist_level: str,
                snr_split: float = 10.0) -> str:
    hi, lo = [], []
    for it in items:
        s = it.get("_snr")
        (hi if (s is not None and s >= snr_split) else lo).append(it)
    hi_html = _group_blocks(hi) or "<p>无</p>"
    if lo:
        lo_html = (
            f'<details><summary>SNR&lt;{snr_split:g}（{len(lo)} 个，默认折叠，点击展开）'
            f'</summary>{_group_blocks(lo)}</details>'
        )
    else:
        lo_html = ""
    return f"""<!doctype html>
<html lang="zh-CN"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>gui_ai 检测导出 (V4)</title>
<style>
 body {{ font-family: Arial, sans-serif; margin: 16px; }}
 .summary {{ background:#f6f8fa; padding:10px 12px; border-radius:8px; margin-bottom:12px; }}
 .grid {{ display:grid; grid-template-columns: repeat(auto-fill, minmax(320px,1fr)); gap:10px; }}
 .card {{ border:1px solid #ddd; border-radius:8px; overflow:hidden; background:#fff; }}
 .card img {{ width:100%; height:auto; display:block; background:#000; }}
 .meta {{ font-size:12px; line-height:1.45; padding:8px; }}
 .count {{ color:#666; font-weight:normal; font-size:12px; }}
 .card img {{ cursor: zoom-in; }}
 details {{ margin-top:14px; border-top:1px solid #eee; padding-top:8px; }}
 summary {{ cursor:pointer; font-weight:bold; }}
 #lb {{ display:none; position:fixed; inset:0; background:rgba(0,0,0,.85);
        z-index:1000; align-items:center; justify-content:center; cursor: zoom-out; }}
 #lb img {{ max-width:96%; max-height:96%; transform-origin:center center;
            transition: transform .06s ease-out; cursor: default; }}
</style></head><body>
<h2>gui_ai 检测导出网页 (V4)</h2>
<div class="summary">
 <div>导出时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</div>
 <div>命中导出: {len(items)}  （SNR≥{snr_split:g}: {len(hi)}，SNR&lt;{snr_split:g}: {len(lo)}）</div>
 <div>patch尺寸: {patch_size}px，拉伸: {html.escape(hist_level)}</div>
 <pre>条件: {html.escape(summary)}</pre>
</div>
<h3>SNR≥{snr_split:g}（{len(hi)}）</h3>
{hi_html}
{lo_html}
<div id="lb"><img id="lbimg" alt="zoom"></div>
<script>
var lb = document.getElementById('lb');
var lbimg = document.getElementById('lbimg');
var scale = 1;
document.querySelectorAll('.card img').forEach(function(im){{
  im.addEventListener('click', function(ev){{
    ev.stopPropagation();
    scale = 1; lbimg.style.transform = 'scale(1)';
    lbimg.src = im.src;
    lb.style.display = 'flex';
  }});
}});
lb.addEventListener('click', function(){{ lb.style.display = 'none'; }});
lbimg.addEventListener('click', function(ev){{ ev.stopPropagation(); }});
lb.addEventListener('wheel', function(ev){{
  ev.preventDefault();
  scale *= (ev.deltaY < 0 ? 1.15 : 1 / 1.15);
  scale = Math.min(12, Math.max(0.2, scale));
  lbimg.style.transform = 'scale(' + scale + ')';
}}, {{ passive: false }});
document.addEventListener('keydown', function(ev){{ if (ev.key === 'Escape') lb.style.display = 'none'; }});
</script>
</body></html>"""


def export_results_web(
    results: List[Dict],
    out_root: str,
    patch_size: int = 256,
    hist_level: str = "linear",
    keep_only: bool = True,
    tag: str = "V4",
    snr_split: float = 10.0,
    group_radius_px: float = 100.0,
    manual_exclude: Tuple[str, ...] = ("m-noise", "m-pix-shift"),
    log=print,
) -> Tuple[int, Path, Path]:
    out_dir = Path(out_root) / f"out_zip_{datetime.now().strftime('%Y%m%d')}"
    out_dir.mkdir(parents=True, exist_ok=True)
    run_tag = datetime.now().strftime("%Y%m%d_%H%M%S")
    stage = out_dir / f"web_export_{run_tag}"
    assets = stage / "assets"
    assets.mkdir(parents=True, exist_ok=True)

    cache = FitsCache(max_items=2)
    items: List[Dict] = []
    n_anomaly = 0
    n_manual = 0
    excl = set(manual_exclude or ())
    try:
        for res in results:
            if res.get("anomaly"):
                n_anomaly += 1
                log(f"跳过异常图像(整文件): {Path(res.get('b_path') or '').name}")
                continue
            a_path = res.get("a_path")
            b_path = res.get("b_path")
            dets = []
            for d in res.get("detections", []):
                if d.get("manual_class") in excl:  # 人工判为假阳性，排除
                    n_manual += 1
                    continue
                if ((not keep_only or d.get("status") == "keep")
                        and d.get("var_count") == 0 and d.get("mpc_count") == 0):
                    dets.append(d)
            if not dets:
                continue
            wcs_a = None
            if a_path:
                try:
                    wcs_a = cache.get(a_path)[2]
                except Exception:
                    wcs_a = None
            wcs_b = None
            if b_path:
                try:
                    wcs_b = cache.get(b_path)[2]
                except Exception:
                    wcs_b = None
            centers = [(float(d["x"]), float(d["y"])) for d in dets]
            try:
                crops = load_native_pair_crops(
                    cache, a_path, b_path, centers, patch_size, True)
            except Exception as ex:  # noqa: BLE001
                log(f"读取裁块失败 {b_path}: {ex}")
                continue
            stem = sanitize_name(Path(b_path).stem) if b_path else "unknown"
            region = parse_region(b_path)
            for d, cr in zip(dets, crops):
                if cr is None:
                    continue
                a_c, b_c = cr
                img = _side_by_side(stretch_patch(a_c, hist_level),
                                    stretch_patch(b_c, hist_level))
                fname = f"{stem}_{d.get('status','x')}_{len(items):04d}.png"
                try:
                    Image.fromarray(img).save(str(assets / fname))
                except Exception as ex:  # noqa: BLE001
                    log(f"写 PNG 失败: {ex}")
                    continue
                ra, dec = _radec(wcs_a, d["x"], d["y"])
                bx = by = None
                if wcs_b is not None and ra is not None:
                    try:
                        bxy = wcs_b.all_world2pix([[float(ra), float(dec)]], 0)[0]
                        bx, by = float(bxy[0]), float(bxy[1])
                    except Exception:
                        bx = by = None
                items.append({
                    "img_rel": f"assets/{fname}",
                    "group": region,
                    "region": region,
                    "_bx": bx,
                    "_by": by,
                    "file": stem,
                    "status": d.get("status", ""),
                    "score": f"{d.get('score', 0):.3f}",
                    "snr": "-" if d.get("snr") is None else f"{d.get('snr'):.1f}",
                    "_snr": d.get("snr"),
                    "var_count": d.get("var_count", -1),
                    "mpc_count": d.get("mpc_count", -1),
                    "radec": "" if ra is None else f"{ra:.6f}, {dec:.6f}",
                    "xy": f"{d.get('x', 0):.0f},{d.get('y', 0):.0f}",
                    "ab_class": d.get("ab_class", ""),
                    "ab_probs": d.get("ab_probs", {}),
                })

        _cluster_by_pixel(items, group_radius_px)
        n_hi = sum(1 for it in items
                   if it.get("_snr") is not None and it["_snr"] >= snr_split)
        if n_anomaly:
            log(f"已跳过 {n_anomaly} 个整文件异常图像")
        if n_manual:
            log(f"已排除 {n_manual} 个手动判为 {sorted(excl)} 的检测")
        summary = (f"同天区(GYx+Ky)按B图像素聚类(半径{group_radius_px:g}px)；"
                   f"仅 var=0 且 mpc=0；状态={'仅命中' if keep_only else '全部'}；"
                   + (f"已排除手动 {sorted(excl)}；" if excl else "")
                   + f"SNR≥{snr_split:g} 优先，其余折叠")
        (stage / "index.html").write_text(
            _build_html(items, summary, patch_size, hist_level, snr_split),
            encoding="utf-8")
        (stage / "meta.json").write_text(json.dumps({
            "generated_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "count": len(items),
            "snr_split": snr_split,
            "n_snr_high": n_hi,
            "n_snr_low": len(items) - n_hi,
            "patch_size_px": patch_size,
            "hist_level": hist_level,
            "keep_only": keep_only,
            "tag": tag,
        }, ensure_ascii=False, indent=2), encoding="utf-8")

        zip_path = out_dir / f"gui_ai_web_{tag}_{run_tag}.zip"
        with zipfile.ZipFile(str(zip_path), "w", compression=zipfile.ZIP_DEFLATED) as zf:
            for p in sorted(stage.rglob("*")):
                if p.is_file():
                    zf.write(str(p), arcname=str(p.relative_to(stage)))
        return len(items), zip_path.resolve(), out_dir.resolve()
    finally:
        shutil.rmtree(stage, ignore_errors=True)


def _parse_group(b_path) -> str:
    if not b_path:
        return "UNGROUPED"
    for part in Path(b_path).parts:
        if re.fullmatch(r"\d{8}", part):
            return part
    return "UNGROUPED"


def parse_region(b_path) -> str:
    """天区 = 望远镜(GYx) + K编号(Ky)，忽略 -1/-2 后缀。"""
    base = Path(b_path).name if b_path else ""
    m = re.search(r"(GY[1-6])[_\-\s]?(K\d{3})", base, re.IGNORECASE)
    if m:
        return f"{m.group(1).upper()}-{m.group(2).upper()}"
    m2 = re.search(r"(K\d{3})", base, re.IGNORECASE)
    if m2:
        return m2.group(1).upper()
    return "UNGROUPED"


def _cluster_by_pixel(items: List[Dict], radius_px: float) -> None:
    """同一天区内按 B 像素位置贪心聚类，写回 item['group']。"""
    if radius_px <= 0:
        for it in items:
            it.setdefault("group", it.get("region", "UNGROUPED"))
        return
    regions: Dict[str, List[Dict]] = {}
    for it in items:
        regions.setdefault(it.get("region", "UNGROUPED"), []).append(it)
    r2 = float(radius_px) ** 2
    for region, its in regions.items():
        clusters: List = []  # [(bx,by) or None, [items]]
        for it in sorted(its, key=lambda x: -(x.get("_snr") or -1)):
            bx, by = it.get("_bx"), it.get("_by")
            if bx is None or by is None:
                clusters.append([None, [it]])
                continue
            placed = False
            for c in clusters:
                if c[0] is None:
                    continue
                cx, cy = c[0]
                if (bx - cx) ** 2 + (by - cy) ** 2 <= r2:
                    c[1].append(it)
                    placed = True
                    break
            if not placed:
                clusters.append([(bx, by), [it]])
        for i, c in enumerate(clusters, 1):
            for it in c[1]:
                it["group"] = f"{region} #{i}"

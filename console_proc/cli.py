#!/usr/bin/env python3
"""
console_proc 独立版：FITS 扫描与下载

设计目标：
1) 不依赖 gui/ 或 data_collect/ 的任何代码
2) 仅通过 console_proc/config.json 配置运行
3) 支持三种模式：
   - 全天全系统：--date
   - 单系统全天：--date --telescope GY1
   - 单天区：    --date --telescope GY1 --region K019
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import re
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from urllib.parse import urljoin

import requests
import urllib3
from bs4 import BeautifulSoup


DEFAULT_USER_AGENT = "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"


@dataclass
class RuntimeConfig:
    url_template: str
    telescopes: List[str]
    download_root: str
    max_workers: int
    retry_times: int
    timeout: int
    verify_ssl: bool
    disable_proxy: bool
    user_agent: str
    include_exts: Tuple[str, ...]
    exclude_keywords: Tuple[str, ...]


def prepare_files_data_dir(date_text: str) -> Path:
    """
    在当前工作目录创建 files_data/YYYYMMDD 目录。
    """
    date_dir = Path.cwd() / "files_data" / date_text
    date_dir.mkdir(parents=True, exist_ok=True)
    return date_dir


def dump_region_files_data(
    date_dir: Path,
    tel_name: str,
    date_text: str,
    region: str,
    region_url: str,
    target_dir: Path,
    files: List[Tuple[str, str]],
) -> Dict[str, object]:
    """
    为每个天区写出文件清单与元信息，供后续流程发现文件使用。
    """
    region_dir = date_dir / tel_name / region
    region_dir.mkdir(parents=True, exist_ok=True)

    files_txt = region_dir / "files.txt"
    urls_txt = region_dir / "urls.txt"
    meta_json = region_dir / "meta.json"

    with files_txt.open("w", encoding="utf-8") as f:
        for name, _url in files:
            f.write(f"{name}\n")

    with urls_txt.open("w", encoding="utf-8") as f:
        for _name, url in files:
            f.write(f"{url}\n")

    meta = {
        "date": date_text,
        "telescope": tel_name,
        "region": region,
        "region_url": region_url,
        "download_dir": str(target_dir),
        "file_count": len(files),
        "files_txt": str(files_txt),
        "urls_txt": str(urls_txt),
    }
    with meta_json.open("w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)

    return meta


def dump_files_data_index(date_dir: Path, region_items: List[Dict[str, object]]) -> None:
    """
    写出当日总索引，便于后续批处理直接读取。
    """
    index_json = date_dir / "index.json"
    index_tsv = date_dir / "index.tsv"

    index_payload = {
        "date": date_dir.name,
        "region_count": len(region_items),
        "items": region_items,
    }
    with index_json.open("w", encoding="utf-8") as f:
        json.dump(index_payload, f, ensure_ascii=False, indent=2)

    with index_tsv.open("w", encoding="utf-8") as f:
        f.write("date\ttelescope\tregion\tfile_count\tdownload_dir\tregion_url\n")
        for item in region_items:
            f.write(
                f"{item['date']}\t{item['telescope']}\t{item['region']}\t"
                f"{item['file_count']}\t{item['download_dir']}\t{item['region_url']}\n"
            )


def setup_logging(verbose: bool) -> None:
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s - %(levelname)s - %(message)s",
    )


def _is_process_alive(pid: int) -> bool:
    if pid <= 0:
        return False
    try:
        os.kill(pid, 0)
    except PermissionError:
        # 进程存在但无权限发送信号
        return True
    except OSError:
        return False
    return True


def _read_lock_meta(lock_path: Path) -> Dict[str, object]:
    try:
        with lock_path.open("r", encoding="utf-8") as f:
            data = json.load(f)
        if isinstance(data, dict):
            return data
    except Exception:
        pass
    return {}


def acquire_date_process_lock(date_text: str, date_dir: Path) -> Path:
    """
    基于日期目录获取进程锁，避免同一天重复启动导致并发叠加。
    """
    lock_path = date_dir / ".download.lock"
    lock_meta = {
        "date": date_text,
        "pid": os.getpid(),
        "created_at": int(time.time()),
        "argv": sys.argv,
    }

    for attempt in range(2):
        try:
            with lock_path.open("x", encoding="utf-8") as f:
                json.dump(lock_meta, f, ensure_ascii=False, indent=2)
            return lock_path
        except FileExistsError:
            existing = _read_lock_meta(lock_path)
            existing_pid = int(existing.get("pid", 0) or 0)
            if attempt == 0 and existing_pid > 0 and not _is_process_alive(existing_pid):
                logging.warning(
                    "发现陈旧日期锁，准备清理后重试: %s (pid=%s)",
                    str(lock_path),
                    existing_pid,
                )
                try:
                    lock_path.unlink(missing_ok=True)
                    continue
                except Exception:
                    # 删除失败则走统一报错
                    pass

            created_at = existing.get("created_at", "unknown")
            raise RuntimeError(
                "检测到同日期任务正在运行，已阻止重复启动。"
                f" date={date_text}, lock={lock_path}, pid={existing_pid}, created_at={created_at}"
            )

    raise RuntimeError(f"无法创建日期锁: {lock_path}")


def release_date_process_lock(lock_path: Optional[Path]) -> None:
    if not lock_path:
        return
    try:
        lock_path.unlink(missing_ok=True)
    except Exception as ex:
        logging.warning("清理日期锁失败: %s (%s)", str(lock_path), ex)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="console_proc 独立版 FITS 扫描下载",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "示例:\n"
            "  python console_proc/cli.py --date 20260408\n"
            "  python console_proc/cli.py --date 20260408 --telescope GY1\n"
            "  python console_proc/cli.py --date 20260408 --telescope GY1 --region K019\n"
        ),
    )
    parser.add_argument("--config", default="console_proc/config.json", help="配置文件路径")
    parser.add_argument("--date", required=True, help="日期，格式 YYYYMMDD")
    parser.add_argument("--telescope", help="系统名，例如 GY1")
    parser.add_argument("--region", help="天区名，例如 K019")
    parser.add_argument("--max-workers", type=int, help="下载并发线程数，覆盖配置")
    parser.add_argument("--retry-times", type=int, help="下载重试次数，覆盖配置")
    parser.add_argument("--timeout", type=int, help="网络超时秒数，覆盖配置")
    parser.add_argument("--download-root", help="下载根目录，覆盖配置")
    parser.add_argument("--scan-only", action="store_true", help="仅扫描不下载")
    parser.add_argument("--verbose", action="store_true", help="输出调试日志")
    return parser.parse_args()


def validate_date(date_text: str) -> bool:
    return bool(re.fullmatch(r"\d{8}", date_text))


def normalize_region(region: str) -> str:
    r = region.strip().upper()
    if not r.startswith("K"):
        r = f"K{r}"
    if re.fullmatch(r"K\d{1,3}", r):
        number = int(r[1:])
        return f"K{number:03d}"
    if not re.fullmatch(r"K\d{3}", r):
        raise ValueError(f"非法天区格式: {region}")
    return r


def load_runtime_config(args: argparse.Namespace) -> RuntimeConfig:
    config_path = Path(args.config)
    if not config_path.exists():
        raise FileNotFoundError(
            f"配置文件不存在: {config_path}，请先复制 console_proc/config.example.json 为 config.json"
        )

    with config_path.open("r", encoding="utf-8") as f:
        raw = json.load(f)

    url_template = raw.get("url_template", "").strip()
    if not url_template:
        raise ValueError("配置项 url_template 不能为空")

    telescopes = raw.get("telescopes", ["GY1", "GY2", "GY3", "GY4", "GY5", "GY6"])
    if not telescopes:
        raise ValueError("配置项 telescopes 不能为空")

    network = raw.get("network", {})
    download = raw.get("download", {})
    path_cfg = raw.get("paths", {})
    file_filter = raw.get("file_filter", {})

    download_root = args.download_root or path_cfg.get("download_root", "downloads")
    max_workers = args.max_workers or int(download.get("max_workers", 4))
    retry_times = args.retry_times or int(download.get("retry_times", 3))
    timeout = args.timeout or int(network.get("timeout", 30))

    include_exts = tuple(file_filter.get("include_extensions", [".fits", ".fit", ".fts"]))
    exclude_keywords = tuple(file_filter.get("exclude_keywords", ["calibration", "_fz", "flat", "fiel"]))

    return RuntimeConfig(
        url_template=url_template,
        telescopes=telescopes,
        download_root=download_root,
        max_workers=max_workers,
        retry_times=retry_times,
        timeout=timeout,
        verify_ssl=bool(network.get("verify_ssl", False)),
        disable_proxy=bool(network.get("disable_proxy", True)),
        user_agent=network.get("user_agent", DEFAULT_USER_AGENT),
        include_exts=include_exts,
        exclude_keywords=exclude_keywords,
    )


def build_session(cfg: RuntimeConfig) -> requests.Session:
    session = requests.Session()
    session.headers.update({"User-Agent": cfg.user_agent})
    if cfg.disable_proxy:
        # requests 默认会读取环境变量代理；关闭它以确保彻底禁用代理。
        session.trust_env = False
        session.proxies = {"http": None, "https": None}
        logging.info("网络配置: 已禁用代理（忽略环境变量代理设置）")
    else:
        logging.info("网络配置: 允许使用系统/环境代理")
    if not cfg.verify_ssl:
        urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)
    return session


def render_url(url_template: str, tel_name: str, date: str, region: str) -> str:
    params: Dict[str, str] = {"tel_name": tel_name, "date": date, "k_number": region}
    if "{year_of_date}" in url_template:
        params["year_of_date"] = date[:4]
    return url_template.format(**params).rstrip("/")


def should_include_file(name: str, cfg: RuntimeConfig) -> bool:
    lower = name.lower()
    if not any(lower.endswith(ext.lower()) for ext in cfg.include_exts):
        return False
    return not any(keyword.lower() in lower for keyword in cfg.exclude_keywords)


def scan_regions(session: requests.Session, cfg: RuntimeConfig, tel_name: str, date: str) -> List[str]:
    base_url = render_url(cfg.url_template, tel_name, date, "")
    logging.info("扫描天区: %s", base_url)
    resp = session.get(base_url, timeout=cfg.timeout, verify=cfg.verify_ssl)
    resp.raise_for_status()
    content = resp.text

    soup = BeautifulSoup(content, "html.parser")
    regions: List[str] = []
    for link in soup.find_all("a", href=True):
        href = link["href"].split("?")[0].split("#")[0]
        item = href.strip("/").split("/")[-1].upper()
        if re.fullmatch(r"K\d{3}", item):
            if item not in regions:
                regions.append(item)

    regions.sort()
    return regions


def scan_files_in_region(
    session: requests.Session,
    cfg: RuntimeConfig,
    region_url: str,
) -> List[Tuple[str, str]]:
    resp = session.get(region_url, timeout=cfg.timeout, verify=cfg.verify_ssl)
    resp.raise_for_status()
    content = resp.text

    # 优先正则匹配目录页
    pattern = r'<a\s+href="([^"]+)"[^>]*>([^<]*)</a>'
    pairs = re.findall(pattern, content, re.IGNORECASE)

    results: List[Tuple[str, str]] = []
    seen: set[str] = set()
    for href, _display in pairs:
        filename = href.split("?")[0].split("/")[-1]
        if not filename:
            continue
        if not should_include_file(filename, cfg):
            continue
        full_url = urljoin(f"{region_url}/", href)
        key = full_url.lower()
        if key in seen:
            continue
        seen.add(key)
        results.append((filename, full_url))

    # 正则没扫到时，回退 bs4
    if not results:
        soup = BeautifulSoup(content, "html.parser")
        for link in soup.find_all("a", href=True):
            href = link["href"]
            filename = href.split("?")[0].split("/")[-1]
            if not filename or not should_include_file(filename, cfg):
                continue
            full_url = urljoin(f"{region_url}/", href)
            key = full_url.lower()
            if key in seen:
                continue
            seen.add(key)
            results.append((filename, full_url))

    return results


def download_one_file(
    session: requests.Session,
    cfg: RuntimeConfig,
    url: str,
    output_path: Path,
) -> str:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = output_path.with_name(f"{output_path.name}.part")
    if output_path.exists() and output_path.stat().st_size > 0:
        return f"跳过: {output_path.name} (已存在)"

    for attempt in range(cfg.retry_times):
        try:
            with session.get(url, timeout=cfg.timeout, stream=True, verify=cfg.verify_ssl) as resp:
                resp.raise_for_status()
                with temp_path.open("wb") as f:
                    for chunk in resp.iter_content(chunk_size=1024 * 128):
                        if chunk:
                            f.write(chunk)

            if temp_path.exists() and temp_path.stat().st_size > 0:
                temp_path.replace(output_path)
                return f"成功: {output_path.name}"
            raise RuntimeError("下载后文件为空")
        except Exception as ex:
            if temp_path.exists():
                try:
                    temp_path.unlink()
                except Exception:
                    pass
            is_last = attempt >= cfg.retry_times - 1
            if is_last:
                return f"失败: {output_path.name} - {ex}"
            sleep_seconds = 2 ** attempt
            time.sleep(sleep_seconds)

    return f"失败: {output_path.name} - 未知错误"


def download_files(
    session: requests.Session,
    cfg: RuntimeConfig,
    files: List[Tuple[str, str]],
    target_dir: Path,
) -> None:
    if not files:
        return

    target_dir.mkdir(parents=True, exist_ok=True)
    logging.info("开始下载 %d 个文件 -> %s", len(files), str(target_dir))

    completed = 0
    with ThreadPoolExecutor(max_workers=max(1, cfg.max_workers)) as pool:
        future_map = {
            pool.submit(download_one_file, session, cfg, url, target_dir / name): (name, url)
            for name, url in files
        }
        for future in as_completed(future_map):
            completed += 1
            message = future.result()
            logging.info("[%d/%d] %s", completed, len(files), message)


def run(args: argparse.Namespace) -> int:
    if not validate_date(args.date):
        logging.error("日期格式错误，应为 YYYYMMDD: %s", args.date)
        return 2

    cfg = load_runtime_config(args)
    session = build_session(cfg)
    files_data_date_dir = prepare_files_data_dir(args.date)
    lock_path: Optional[Path] = None
    try:
        lock_path = acquire_date_process_lock(args.date, files_data_date_dir)
        logging.info("已获取日期进程锁: %s", str(lock_path))
    except RuntimeError as ex:
        logging.error("%s", ex)
        return 3

    files_data_items: List[Dict[str, object]] = []

    try:
        if args.region and not args.telescope:
            logging.error("使用 --region 时必须同时提供 --telescope")
            return 2

        telescopes = [args.telescope] if args.telescope else list(cfg.telescopes)
        if args.telescope and args.telescope not in cfg.telescopes:
            logging.warning("配置 telescopes 中未包含 %s，仍将继续处理", args.telescope)

        total_regions = 0
        total_files = 0
        for tel_name in telescopes:
            try:
                if args.region:
                    regions = [normalize_region(args.region)]
                else:
                    regions = scan_regions(session, cfg, tel_name, args.date)
                    if not regions:
                        logging.warning("[%s %s] 未扫描到天区", tel_name, args.date)
                        continue
            except Exception as ex:
                logging.error("[%s %s] 扫描天区失败: %s", tel_name, args.date, ex)
                continue

            for region in regions:
                total_regions += 1
                region_url = render_url(cfg.url_template, tel_name, args.date, region)
                logging.info("[%s/%s/%s] 扫描文件: %s", tel_name, args.date, region, region_url)
                try:
                    files = scan_files_in_region(session, cfg, region_url)
                except Exception as ex:
                    logging.error("[%s/%s/%s] 扫描失败: %s", tel_name, args.date, region, ex)
                    continue

                logging.info("[%s/%s/%s] 找到 FITS: %d", tel_name, args.date, region, len(files))
                total_files += len(files)
                target_dir = Path(cfg.download_root) / tel_name / args.date / region

                item = dump_region_files_data(
                    date_dir=files_data_date_dir,
                    tel_name=tel_name,
                    date_text=args.date,
                    region=region,
                    region_url=region_url,
                    target_dir=target_dir,
                    files=files,
                )
                files_data_items.append(item)

                if args.scan_only:
                    continue

                download_files(session, cfg, files, target_dir)

        dump_files_data_index(files_data_date_dir, files_data_items)
        logging.info("files_data 索引已生成: %s", str(files_data_date_dir / "index.json"))
        logging.info("完成: 天区=%d, 文件=%d", total_regions, total_files)
        return 0
    finally:
        release_date_process_lock(lock_path)


def main() -> None:
    args = parse_args()
    setup_logging(args.verbose)
    try:
        code = run(args)
    except KeyboardInterrupt:
        logging.error("用户中断")
        code = 130
    except Exception as ex:
        logging.exception("运行异常: %s", ex)
        code = 1
    raise SystemExit(code)


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
配置文件管理器
用于保存和加载GUI应用程序的配置信息
"""

import json
import os
import sys
from datetime import datetime
from typing import Dict, List, Any
import logging

# 添加config目录到路径
sys.path.append(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'config'))
from url_config_manager import url_config_manager


class ConfigManager:
    """配置文件管理器"""

    def __init__(self, config_file="gui_config.json"):
        self.config_file = config_file
        self.logger = logging.getLogger(__name__)

        # 预计算本地库默认路径（gui/mpc_variables）
        current_dir = os.path.dirname(os.path.abspath(__file__))
        mpc_variables_dir = os.path.join(current_dir, 'mpc_variables')
        default_asteroid_path = os.path.join(mpc_variables_dir, 'MPCORB.DAT')
        if not os.path.exists(default_asteroid_path):
            default_asteroid_path = ""
        # 变星库候选（优先顺序）：catalog_gaia_variables.dat -> .dat.gz -> .csv -> vclassre.dat.gz
        vsx_candidates = [
            os.path.join(mpc_variables_dir, 'catalog_gaia_variables.dat'),
            os.path.join(mpc_variables_dir, 'catalog_gaia_variables.dat.gz'),
            os.path.join(mpc_variables_dir, 'catalog_gaia_variables.csv'),
            os.path.join(mpc_variables_dir, 'vclassre.dat.gz'),
        ]
        default_vsx_path = ""
        for p in vsx_candidates:
            if os.path.exists(p):
                default_vsx_path = p
                break
        self._default_asteroid_catalog_path = default_asteroid_path
        self._default_vsx_catalog_path = default_vsx_path

        # 默认配置
        self.default_config = {
            "telescope_names": ["GY1", "GY2", "GY3", "GY4", "GY5", "GY6"],
            "k_numbers": [f"K{i:03d}" for i in range(1, 100)],  # K001 - K099
            "last_selected": {
                "telescope_name": "GY5",
                "date": datetime.now().strftime("%Y%m%d"),
                "k_number": "K096",
                "download_directory": "E:/fix_data/download",
                "template_directory": "E:/fix_data/template",
                "diff_output_directory": "E:/fix_data/output",
                "detected_directory": "E:/fix_data/output\\detected",
                "zip_output_directory": ""
            },
            "download_settings": {
                "max_workers": 1,
                "max_workers_limit": 1,  # 锁定为1，不允许修改
                "max_workers_enabled": False,  # 禁用并发数设置
                "retry_times": 3,
                "timeout": 30
            },
            "batch_process_settings": {
                "thread_count": 4,  # 批量处理线程数（GUI默认值：4）
                "fast_mode": True,  # 快速模式（GUI默认值：True）
            },
            "diff_pipeline_settings": {
                "script_paths": {
                    "export_fits_stars": "E:/github/misaligned_fits/export_fits_stars.py",
                    "recommended_pipeline_console": "E:/github/fits_data_view_process_3d/std_process/recommended_pipeline_console.py",
                    "reproject_wcs_and_export_stars": "E:/github/misaligned_fits/reproject_wcs_and_export_stars.py",
                    "solve_alignment_from_stars": "E:/github/misaligned_fits/solve_alignment_from_stars.py",
                    "render_alignment_outputs": "E:/github/misaligned_fits/render_alignment_outputs.py",
                    "rank_variable_candidates": "E:/github/misaligned_fits/rank_variable_candidates.py",
                    "crossmatch_nonref_candidates": "E:/github/misaligned_fits/crossmatch_nonref_candidates.py",
                    "export_nonref_candidate_ab_cutouts": "E:/github/misaligned_fits/export_nonref_candidate_ab_cutouts.py",
                },
                "export_uniform_grid_x": 7,
                "export_uniform_grid_y": 7,
                "export_uniform_per_cell": 100,
                "preprocess_box": 48,
                "preprocess_clip_sigma": 3.0,
                "preprocess_median_ksize": 3,
                "preprocess_denoise_sigma": 2.0,
                "preprocess_mix_alpha": 0.7,
                "reproject_max_stars": 5000,
                "reproject_uniform_grid_x": 7,
                "reproject_uniform_grid_y": 7,
                "reproject_uniform_per_cell": 100,
                "solve_radii": [24, 32, 40],
                "rank_min_observations": 2,
                "enable_crossmatch_nonref_candidates": True,
                "enable_export_nonref_candidate_ab_cutouts": False,
                "nonref_candidate_cutout_size": 128
            },
            "dss_flip_settings": {
                "flip_vertical": True,  # 上下翻转DSS（默认值：True）
                "flip_horizontal": False  # 左右翻转DSS（默认值：False）
            },
            "rank_flip_settings": {
                "flip_vertical": False  # 上下翻转 Rank 图（默认值：False）
            },
            "gps_settings": {
                "latitude": 43.4,  # 纬度（默认值：43.4°N）
                "longitude": 87.1  # 经度（默认值：87.1°E）
            },
            "mpc_settings": {
                "mpc_code": "N87"  # MPC观测站代码（默认值：N87）
            },
            "query_settings": {
                "search_radius": 0.01,                  # 搜索半径（度）（默认值：0.01）
                "batch_query_interval_seconds": 5.0,    # 批量查询之间的间隔时间（秒）（默认值：5秒）
                "batch_query_threads": 5,               # pympc批量查询线程数（默认值：5）
                "batch_pympc_server_threads": 3,        # pympc server批量查询线程数（默认值：3）
                "batch_vsx_server_threads": 3           # 变星server批量查询线程数（默认值：3）
            },
            "display_settings": {
                "auto_select_from_download_dir": True,
                # CSV候选浏览默认尺寸（像素）
                "csv_candidate_patch_size": "512",
                # CSV候选局部拉伸档位
                "csv_local_hist_level": "high",
                # CSV条件搜索默认参数
                "csv_search_median_flux_min": "30",
                "csv_search_median_flux_max": "800",
                "csv_search_variable_count_mode": "=0",
                "csv_search_mpc_count_mode": "=0",
                "csv_search_ai_class_mode": "=0",
                "csv_search_skip_mode": "=0",
                "crossmatch_rerun_parallel_workers": "3",
                "ai_classifier_model_path": "gui/classifier_model.joblib",
                "csv_filter_skip_large_rows_enabled": False,
                "csv_filter_max_rows": "200",
            },
            "local_catalog_settings": {
                "asteroid_catalog_path": self._default_asteroid_catalog_path,
                "vsx_catalog_path": self._default_vsx_catalog_path,
                "last_asteroid_update": "",
                "last_vsx_update": "",
                "mpc_h_limit": 20,
                "ephemeris_file_path": "",
                "last_ephemeris_update": "",
                "pympc_catalog_path": "",
                "last_pympc_update": "",
                "pympc_use_observatory": False  # 使用pympc时是否使用观测站代码，默认不使用
            },
            "url_template_type": "standard",  # "standard" 或 "with_year"
            # URL模板现在从独立的URL配置文件中读取
            "url_templates": url_config_manager.get_available_templates(),
            # 依赖检查状态
            "dependencies_checked": False,  # 是否已通过依赖检查
            "dependencies_check_date": "",  # 依赖检查通过日期
            "missing_dependencies": []  # 缺失的依赖包列表
        }

        # 加载配置
        self.config = self.load_config()

    def load_config(self) -> Dict[str, Any]:
        """
        加载配置文件

        Returns:
            Dict[str, Any]: 配置字典
        """
        try:
            if os.path.exists(self.config_file):
                with open(self.config_file, 'r', encoding='utf-8') as f:
                    loaded_config = json.load(f)

                # 合并默认配置和加载的配置
                config = self.default_config.copy()
                self._deep_update(config, loaded_config)

                self.logger.info(f"配置文件加载成功: {self.config_file}")
                return config
            else:
                self.logger.info("配置文件不存在，使用默认配置")
                return self.default_config.copy()

        except Exception as e:
            self.logger.error(f"加载配置文件失败: {str(e)}")
            return self.default_config.copy()

    def save_config(self) -> bool:
        """
        保存配置文件

        Returns:
            bool: 是否保存成功
        """
        try:
            with open(self.config_file, 'w', encoding='utf-8') as f:
                json.dump(self.config, f, indent=2, ensure_ascii=False)

            self.logger.info(f"配置文件保存成功: {self.config_file}")
            return True

        except Exception as e:
            self.logger.error(f"保存配置文件失败: {str(e)}")
            return False

    def _deep_update(self, base_dict: Dict, update_dict: Dict):
        """深度更新字典"""
        for key, value in update_dict.items():
            if key in base_dict and isinstance(base_dict[key], dict) and isinstance(value, dict):
                self._deep_update(base_dict[key], value)
            else:
                base_dict[key] = value

    def get_telescope_names(self) -> List[str]:
        """获取望远镜名称列表"""
        return self.config["telescope_names"]

    def get_k_numbers(self) -> List[str]:
        """获取K序号列表"""
        return self.config["k_numbers"]

    def get_last_selected(self) -> Dict[str, str]:
        """获取上次选择的值"""
        return self.config["last_selected"]

    def update_last_selected(self, **kwargs):
        """更新上次选择的值"""
        # 确保last_selected存在
        if "last_selected" not in self.config:
            self.config["last_selected"] = self.default_config["last_selected"].copy()

        for key, value in kwargs.items():
            # 允许添加新字段，不仅限于已存在的字段
            self.config["last_selected"][key] = value
        self.save_config()

    def get_download_settings(self) -> Dict[str, int]:
        """获取下载设置"""
        return self.config["download_settings"]

    def update_download_settings(self, **kwargs):
        """更新下载设置"""
        for key, value in kwargs.items():
            if key in self.config["download_settings"]:
                self.config["download_settings"][key] = value
        self.save_config()

    def get_display_settings(self) -> Dict[str, Any]:
        """获取显示设置"""
        if "display_settings" not in self.config:
            self.config["display_settings"] = self.default_config.get("display_settings", {}).copy()
            self.save_config()
            return self.config["display_settings"]

        # 兼容旧配置：补齐缺失键
        changed = False
        default_display = self.default_config.get("display_settings", {})
        for key, default_value in default_display.items():
            if key not in self.config["display_settings"]:
                self.config["display_settings"][key] = default_value
                changed = True
        if changed:
            self.save_config()
        return self.config["display_settings"]

    def update_display_settings(self, **kwargs):
        """更新显示设置"""
        if "display_settings" not in self.config:
            self.config["display_settings"] = self.default_config.get("display_settings", {}).copy()
        for key, value in kwargs.items():
            if key in self.default_config.get("display_settings", {}):
                self.config["display_settings"][key] = value
        self.save_config()

    def get_local_catalog_settings(self) -> Dict[str, Any]:
        """获取本地目录查询设置"""
        if "local_catalog_settings" not in self.config:
            self.config["local_catalog_settings"] = self.default_config.get("local_catalog_settings", {}).copy()
            self.save_config()
        settings = self.config["local_catalog_settings"]
        # 自动填充默认路径（如果未设置或路径不存在）
        changed = False
        try:
            asteroid_path = settings.get("asteroid_catalog_path", "")
            if (not asteroid_path or not os.path.exists(asteroid_path)) and getattr(self, "_default_asteroid_catalog_path", ""):
                if os.path.exists(self._default_asteroid_catalog_path):
                    settings["asteroid_catalog_path"] = self._default_asteroid_catalog_path
                    changed = True
            vsx_path = settings.get("vsx_catalog_path", "")
            if (not vsx_path or not os.path.exists(vsx_path)) and getattr(self, "_default_vsx_catalog_path", ""):
                if os.path.exists(self._default_vsx_catalog_path):
                    settings["vsx_catalog_path"] = self._default_vsx_catalog_path
                    changed = True
        except Exception:
            pass

        # 兼容旧版本: 填充pympc相关字段

        if "pympc_catalog_path" not in settings:
            settings["pympc_catalog_path"] = ""
            changed = True
        if "last_pympc_update" not in settings:
            settings["last_pympc_update"] = ""
            changed = True
        if "pympc_use_observatory" not in settings:
            settings["pympc_use_observatory"] = False
            changed = True

        if changed:
            self.save_config()
        return settings

    def update_local_catalog_settings(self, **kwargs):
        """更新本地目录查询设置"""
        if "local_catalog_settings" not in self.config:
            self.config["local_catalog_settings"] = self.default_config.get("local_catalog_settings", {}).copy()
        for key, value in kwargs.items():
            self.config["local_catalog_settings"][key] = value
        self.save_config()
    def get_batch_process_settings(self) -> Dict[str, Any]:
        """获取批量处理设置"""
        # 如果配置中没有批量处理设置，使用默认值
        if "batch_process_settings" not in self.config:
            self.config["batch_process_settings"] = self.default_config["batch_process_settings"].copy()
            self.save_config()
            return self.config["batch_process_settings"]

        # 兼容旧配置：补齐缺失键
        changed = False
        default_batch = self.default_config.get("batch_process_settings", {})
        for key, default_value in default_batch.items():
            if key not in self.config["batch_process_settings"]:
                self.config["batch_process_settings"][key] = default_value
                changed = True
        if changed:
            self.save_config()
        return self.config["batch_process_settings"]

    def update_batch_process_settings(self, **kwargs):
        """更新批量处理设置"""
        if "batch_process_settings" not in self.config:
            self.config["batch_process_settings"] = self.default_config["batch_process_settings"].copy()

        for key, value in kwargs.items():
            if key in self.config["batch_process_settings"]:
                self.config["batch_process_settings"][key] = value
        self.save_config()

    def get_diff_pipeline_settings(self) -> Dict[str, Any]:
        """获取Diff替代流程命令配置（脚本路径与关键参数）。"""
        if "diff_pipeline_settings" not in self.config:
            self.config["diff_pipeline_settings"] = self.default_config["diff_pipeline_settings"].copy()
            self.save_config()
            return self.config["diff_pipeline_settings"]

        changed = False
        defaults = self.default_config.get("diff_pipeline_settings", {})
        current = self.config["diff_pipeline_settings"]
        for key, default_value in defaults.items():
            if key not in current:
                current[key] = default_value
                changed = True
            elif isinstance(default_value, dict) and isinstance(current.get(key), dict):
                for sub_key, sub_default in default_value.items():
                    if sub_key not in current[key]:
                        current[key][sub_key] = sub_default
                        changed = True
        if changed:
            self.save_config()
        return current

    def update_diff_pipeline_settings(self, **kwargs):
        """更新Diff替代流程命令配置。"""
        if "diff_pipeline_settings" not in self.config:
            self.config["diff_pipeline_settings"] = self.default_config["diff_pipeline_settings"].copy()
        for key, value in kwargs.items():
            self.config["diff_pipeline_settings"][key] = value
        self.save_config()

    def get_dss_flip_settings(self) -> Dict[str, Any]:
        """获取DSS翻转设置"""
        # 如果配置中没有DSS翻转设置，使用默认值
        if "dss_flip_settings" not in self.config:
            self.config["dss_flip_settings"] = self.default_config["dss_flip_settings"].copy()
            self.save_config()
        return self.config["dss_flip_settings"]

    def update_dss_flip_settings(self, **kwargs):
        """更新DSS翻转设置"""
        if "dss_flip_settings" not in self.config:
            self.config["dss_flip_settings"] = self.default_config["dss_flip_settings"].copy()

        for key, value in kwargs.items():
            if key in ["flip_vertical", "flip_horizontal"]:
                self.config["dss_flip_settings"][key] = value
        self.save_config()

    def get_rank_flip_settings(self) -> Dict[str, Any]:
        """获取Rank图翻转设置"""
        if "rank_flip_settings" not in self.config:
            self.config["rank_flip_settings"] = self.default_config["rank_flip_settings"].copy()
            self.save_config()
        return self.config["rank_flip_settings"]

    def update_rank_flip_settings(self, **kwargs):
        """更新Rank图翻转设置"""
        if "rank_flip_settings" not in self.config:
            self.config["rank_flip_settings"] = self.default_config["rank_flip_settings"].copy()

        for key, value in kwargs.items():
            if key in ["flip_vertical"]:
                self.config["rank_flip_settings"][key] = bool(value)
        self.save_config()

    def get_gps_settings(self) -> Dict[str, Any]:
        """获取GPS设置"""
        # 如果配置中没有GPS设置，使用默认值
        if "gps_settings" not in self.config:
            self.config["gps_settings"] = self.default_config["gps_settings"].copy()
            self.save_config()
        return self.config["gps_settings"]

    def update_gps_settings(self, **kwargs):
        """更新GPS设置"""
        if "gps_settings" not in self.config:
            self.config["gps_settings"] = self.default_config["gps_settings"].copy()

        for key, value in kwargs.items():
            if key in ["latitude", "longitude"]:
                self.config["gps_settings"][key] = value
        self.save_config()

    def get_mpc_settings(self) -> Dict[str, Any]:
        """获取MPC代码设置"""
        # 如果配置中没有MPC设置，使用默认值
        if "mpc_settings" not in self.config:
            self.config["mpc_settings"] = self.default_config["mpc_settings"].copy()
            self.save_config()
        return self.config["mpc_settings"]

    def update_mpc_settings(self, **kwargs):
        """更新MPC代码设置"""
        if "mpc_settings" not in self.config:
            self.config["mpc_settings"] = self.default_config["mpc_settings"].copy()

        for key, value in kwargs.items():
            if key == "mpc_code":
                self.config["mpc_settings"][key] = value
        self.save_config()

    def get_query_settings(self) -> Dict[str, Any]:
        """获取查询设置"""
        # 如果配置中没有查询设置，使用默认值
        if "query_settings" not in self.config:
            self.config["query_settings"] = self.default_config["query_settings"].copy()
            self.save_config()

        # 确保所有默认键都存在（兼容旧配置文件）
        changed = False
        for key, default_value in self.default_config["query_settings"].items():
            if key not in self.config["query_settings"]:
                self.config["query_settings"][key] = default_value
                changed = True
        if changed:
            self.save_config()

        return self.config["query_settings"]

    def update_query_settings(self, **kwargs):
        """更新查询设置"""
        if "query_settings" not in self.config:
            self.config["query_settings"] = self.default_config["query_settings"].copy()

        for key, value in kwargs.items():
            # 允许更新所有已知的查询设置键（包括搜索半径和批量查询间隔）
            self.config["query_settings"][key] = value
        self.save_config()

    def get_url_template_type(self) -> str:
        """获取URL模板类型"""
        return self.config.get("url_template_type", "standard")

    def get_url_template(self) -> str:
        """获取当前URL模板"""
        template_type = self.get_url_template_type()

        # 从URL配置管理器获取模板，并添加基础URL
        base_url = url_config_manager.get_base_url()
        template = url_config_manager.get_url_template(template_type)

        # 将模板中的{base_url}替换为实际的基础URL
        return template.replace("{base_url}", base_url)

    def get_url_templates(self) -> Dict[str, str]:
        """获取所有URL模板"""
        # 从URL配置管理器获取模板
        templates = url_config_manager.get_available_templates()
        base_url = url_config_manager.get_base_url()

        # 将所有模板中的{base_url}替换为实际的基础URL
        result = {}
        for key, template in templates.items():
            result[key] = template.replace("{base_url}", base_url)

        return result

    def update_url_template_type(self, template_type: str):
        """更新URL模板类型"""
        if template_type in ["standard", "with_year"]:
            self.config["url_template_type"] = template_type
            self.save_config()
        else:
            raise ValueError(f"无效的URL模板类型: {template_type}")

    def get_url_template_options(self) -> Dict[str, str]:
        """获取URL模板选项的显示名称"""
        return {
            "standard": "标准格式 (/{date}/)",
            "with_year": "包含年份 (/{year}/{date}/)"
        }

    def build_url(self, tel_name: str = None, date: str = None, k_number: str = None) -> str:
        """
        构建URL

        Args:
            tel_name (str): 望远镜名称，如果为None则使用上次选择的值
            date (str): 日期，如果为None则使用上次选择的值
            k_number (str): K序号，如果为None则使用上次选择的值

        Returns:
            str: 构建的URL
        """
        last_selected = self.get_last_selected()

        tel_name = tel_name or last_selected["telescope_name"]
        date = date or last_selected["date"]
        k_number = k_number or last_selected["k_number"]

        # 使用URL配置管理器构建URL
        template_type = self.get_url_template_type()
        return url_config_manager.build_url(tel_name, date, k_number, template_type)

    def validate_date(self, date_str: str) -> bool:
        """
        验证日期格式

        Args:
            date_str (str): 日期字符串 (YYYYMMDD)

        Returns:
            bool: 是否有效
        """
        try:
            datetime.strptime(date_str, '%Y%m%d')
            return True
        except ValueError:
            return False

    def validate_k_number(self, k_number: str) -> bool:
        """
        验证K序号格式

        Args:
            k_number (str): K序号字符串

        Returns:
            bool: 是否有效
        """
        return k_number in self.get_k_numbers()

    def validate_telescope_name(self, tel_name: str) -> bool:
        """
        验证望远镜名称

        Args:
            tel_name (str): 望远镜名称

        Returns:
            bool: 是否有效
        """
        return tel_name in self.get_telescope_names()

    def get_recent_dates(self, days: int = 7) -> List[str]:
        """
        获取最近几天的日期列表

        Args:
            days (int): 天数

        Returns:
            List[str]: 日期列表 (YYYYMMDD格式)
        """
        from datetime import timedelta

        dates = []
        base_date = datetime.now()

        for i in range(days):
            date = base_date - timedelta(days=i)
            dates.append(date.strftime('%Y%m%d'))

        return dates

    def export_config(self, export_file: str) -> bool:
        """
        导出配置到指定文件

        Args:
            export_file (str): 导出文件路径

        Returns:
            bool: 是否导出成功
        """
        try:
            with open(export_file, 'w', encoding='utf-8') as f:
                json.dump(self.config, f, indent=2, ensure_ascii=False)

            self.logger.info(f"配置导出成功: {export_file}")
            return True

        except Exception as e:
            self.logger.error(f"导出配置失败: {str(e)}")
            return False

    def import_config(self, import_file: str) -> bool:
        """
        从指定文件导入配置

        Args:
            import_file (str): 导入文件路径

        Returns:
            bool: 是否导入成功
        """
        try:
            if not os.path.exists(import_file):
                raise FileNotFoundError(f"配置文件不存在: {import_file}")

            with open(import_file, 'r', encoding='utf-8') as f:
                imported_config = json.load(f)

            # 验证配置格式
            if not isinstance(imported_config, dict):
                raise ValueError("配置文件格式无效")

            # 合并配置
            self._deep_update(self.config, imported_config)
            self.save_config()

            self.logger.info(f"配置导入成功: {import_file}")
            return True

        except Exception as e:
            self.logger.error(f"导入配置失败: {str(e)}")
            return False
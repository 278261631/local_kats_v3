#!/usr/bin/env python3
"""
对比已对齐的FITS文件差异检测脚本
专门用于处理 E:\fix_data\align-diff 文件夹中的已对齐FITS文件
输出FITS和JPG格式的差异结果
"""

import os
import sys
import json
import csv
import shutil
import numpy as np
import cv2
import matplotlib
# 使用非交互后端，避免在子线程/无主循环环境触发 Tk 错误
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from astropy.io import fits
from scipy.ndimage import gaussian_filter, median_filter, zoom
from pathlib import Path
import logging
from datetime import datetime
import warnings
import glob
import subprocess
import time
import shlex

# 忽略警告
warnings.filterwarnings('ignore', category=RuntimeWarning)
warnings.filterwarnings('ignore', category=UserWarning)

# 设置matplotlib中文字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False


class AlignedFITSComparator:
    """已对齐FITS文件差异比较器"""
    
    def __init__(self):
        """初始化比较器"""
        self.setup_logging()
        
        # 差异检测参数
        self.diff_params = {
            'gaussian_sigma': 1.0,
            'diff_threshold': 0.1,
            'min_spot_area': 5,
            'max_spot_area': 1000,
            'overlap_edge_exclusion_px': 40,
            'apply_rpca_bg_subtraction': True,
            'rpca_target_side': 384,
            'rpca_max_iter': 30
        }
    
    def setup_logging(self):
        """设置日志"""
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler('aligned_fits_comparison.log'),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger(__name__)
    
    def load_fits_data(self, fits_path):
        """
        加载FITS文件数据
        
        Args:
            fits_path (str): FITS文件路径
            
        Returns:
            numpy.ndarray: 图像数据，如果失败返回None
        """
        try:
            with fits.open(fits_path) as hdul:
                data = hdul[0].data.astype(np.float32)  # 优化：使用float32减少内存50%，提升速度24%

                # 处理可能的3D数据（取第一个通道）
                if len(data.shape) == 3:
                    data = data[0]

                self.logger.info(f"成功加载FITS文件: {os.path.basename(fits_path)}, 形状: {data.shape}")
                return data
                
        except Exception as e:
            self.logger.error(f"加载FITS文件失败 {fits_path}: {str(e)}")
            return None
    
    def normalize_image(self, image):
        """
        标准化图像数据到0-1范围

        Args:
            image (numpy.ndarray): 输入图像

        Returns:
            numpy.ndarray: 标准化后的图像
        """
        # 使用百分位数进行鲁棒标准化
        p1, p99 = np.percentile(image, [1, 99])
        normalized = np.clip((image - p1) / (p99 - p1), 0, 1)
        return normalized

    def create_overlap_mask(self, ref_image, aligned_image, threshold=1e-6):
        """
        创建重叠区域掩码，识别两个图像的有效重叠区域

        Args:
            ref_image (numpy.ndarray): 参考图像
            aligned_image (numpy.ndarray): 对齐后的图像
            threshold (float): 判断有效像素的阈值

        Returns:
            numpy.ndarray: 重叠区域掩码（1表示重叠，0表示非重叠）
        """
        # 创建有效像素掩码
        ref_valid = np.abs(ref_image) > threshold
        aligned_valid = np.abs(aligned_image) > threshold

        # 重叠区域是两个图像都有有效像素的区域
        overlap_mask = (ref_valid & aligned_valid).astype(np.uint8)

        self.logger.info(f"重叠区域像素数: {np.sum(overlap_mask)}, "
                        f"总像素数: {overlap_mask.size}, "
                        f"重叠比例: {np.sum(overlap_mask)/overlap_mask.size:.2%}")

        return overlap_mask

    def trim_overlap_mask_edge(self, overlap_mask, edge_width_px):
        """
        去除重叠区域边界附近指定宽度的像素。

        Args:
            overlap_mask (numpy.ndarray): 原始重叠掩码
            edge_width_px (int): 要剔除的边界宽度（像素）

        Returns:
            numpy.ndarray: 剔除边界后的重叠掩码
        """
        if edge_width_px <= 0:
            return overlap_mask

        # 距离变换：每个重叠像素到最近边界(非重叠像素)的距离
        dist = cv2.distanceTransform(overlap_mask.astype(np.uint8), cv2.DIST_L2, 3)
        trimmed_mask = (dist > float(edge_width_px)).astype(np.uint8)

        original_pixels = int(np.sum(overlap_mask > 0))
        trimmed_pixels = int(np.sum(trimmed_mask > 0))

        if trimmed_pixels == 0 and original_pixels > 0:
            self.logger.warning(
                f"边界剔除后无有效重叠区域（边界宽度={edge_width_px}px），回退为原始重叠掩码"
            )
            return overlap_mask

        self.logger.info(
            f"已剔除重叠边界 {edge_width_px}px：有效像素 {original_pixels} -> {trimmed_pixels}"
        )
        return trimmed_mask

    def _rpca_decompose(self, matrix, lam=None, max_iter=30, tol=1e-5):
        """Inexact ALM RPCA: M = L + S。"""
        m = matrix.astype(np.float64)
        norm_m = np.linalg.norm(m, ord='fro')
        if norm_m == 0:
            return np.zeros_like(m), np.zeros_like(m)

        if lam is None:
            lam = 1.0 / np.sqrt(max(m.shape))

        sparse = np.zeros_like(m)
        lagrange = np.zeros_like(m)

        spectral_norm = np.linalg.norm(m, ord=2)
        mu = 1.25 / (spectral_norm + 1e-8)
        mu_bar = mu * 1e7
        rho = 1.5

        low_rank = np.zeros_like(m)
        for _ in range(max_iter):
            u, s, vt = np.linalg.svd(m - sparse + lagrange / mu, full_matrices=False)
            s_threshold = np.maximum(s - 1.0 / mu, 0)
            rank = int(np.sum(s_threshold > 0))
            if rank > 0:
                low_rank = (u[:, :rank] * s_threshold[:rank]) @ vt[:rank, :]
            else:
                low_rank = np.zeros_like(m)

            residual_for_sparse = m - low_rank + lagrange / mu
            sparse = np.sign(residual_for_sparse) * np.maximum(np.abs(residual_for_sparse) - lam / mu, 0)

            residual = m - low_rank - sparse
            err = np.linalg.norm(residual, ord='fro') / (norm_m + 1e-8)
            lagrange = lagrange + mu * residual
            mu = min(mu * rho, mu_bar)
            if err < tol:
                break

        return low_rank, sparse

    def rpca_background_subtract_difference(self, diff_image, overlap_mask, diff_calc_mode='abs'):
        """
        对 difference 图做一次 RPCA 背景去除，仅在重叠区域生效。

        Args:
            diff_image (numpy.ndarray): 差异图
            overlap_mask (numpy.ndarray): 重叠区域掩码
            diff_calc_mode (str): 'abs' 或 'signed'

        Returns:
            numpy.ndarray: 去背景后的差异图
        """
        if np.sum(overlap_mask > 0) == 0:
            return diff_image

        data = (diff_image * overlap_mask).astype(np.float64)
        h, w = data.shape
        max_side = max(h, w)
        target_side = int(self.diff_params.get('rpca_target_side', 384))
        scale = target_side / float(max_side) if max_side > target_side else 1.0

        if scale < 1.0:
            small = zoom(data, zoom=scale, order=1)
        else:
            small = data.copy()

        lam = 1.0 / np.sqrt(max(small.shape))
        max_iter = int(self.diff_params.get('rpca_max_iter', 30))
        low_rank_small, _ = self._rpca_decompose(small, lam=lam, max_iter=max_iter, tol=1e-5)

        if scale < 1.0:
            zoom_y = h / float(low_rank_small.shape[0])
            zoom_x = w / float(low_rank_small.shape[1])
            bg_model = zoom(low_rank_small, zoom=(zoom_y, zoom_x), order=1)
            bg_model = bg_model[:h, :w]
        else:
            bg_model = low_rank_small

        bg_subtracted = (data - bg_model) * overlap_mask

        if diff_calc_mode == 'abs':
            # 绝对差分场景下，保留正残差，避免引入负值噪声。
            bg_subtracted = np.where(bg_subtracted < 0, 0, bg_subtracted)

        return bg_subtracted.astype(np.float32)
    
    def detect_differences(self, img1, img2, diff_calc_mode='abs', apply_diff_postprocess=False):
        """
        检测两个图像之间的差异

        Args:
            img1 (numpy.ndarray): 参考图像
            img2 (numpy.ndarray): 比较图像
            diff_calc_mode (str): 差异计算方式，'abs' 或 'signed'
            apply_diff_postprocess (bool): 是否对差异图执行后处理（负值置零+中值滤波）

        Returns:
            tuple: (差异图像, 二值化差异图像, 重叠区域掩码, 中间图像字典)
        """
        # 创建重叠区域掩码
        mask_start = time.time()
        overlap_mask = self.create_overlap_mask(img1, img2)
        overlap_mask = self.trim_overlap_mask_edge(
            overlap_mask, int(self.diff_params.get('overlap_edge_exclusion_px', 0))
        )
        self.logger.debug(f"  ⏱️  创建重叠掩码耗时: {time.time() - mask_start:.3f}秒")

        # 标准化图像
        norm_start = time.time()
        norm_img1 = self.normalize_image(img1)
        norm_img2 = self.normalize_image(img2)
        self.logger.debug(f"  ⏱️  标准化图像耗时: {time.time() - norm_start:.3f}秒")

        # 应用高斯模糊减少噪声
        blur_start = time.time()
        blurred_img1 = gaussian_filter(norm_img1, sigma=self.diff_params['gaussian_sigma'])
        blurred_img2 = gaussian_filter(norm_img2, sigma=self.diff_params['gaussian_sigma'])
        self.logger.debug(f"  ⏱️  高斯模糊耗时: {time.time() - blur_start:.3f}秒")

        # 计算差异（只在重叠区域）
        diff_start = time.time()
        diff_raw = blurred_img2 - blurred_img1
        if diff_calc_mode == 'signed':
            diff_image = diff_raw * overlap_mask
        else:
            diff_image = np.abs(diff_raw) * overlap_mask

        # 可选：对差异图执行后处理（仅影响 difference 产物与后续二值化）
        if apply_diff_postprocess:
            # 排除负值
            diff_image = np.where(diff_image < 0, 0, diff_image)
            # 3x3 中值滤波，抑制孤立噪声
            diff_image = median_filter(diff_image, size=3)

        # 对 difference 再做一次 RPCA 去背景，抑制低频背景/边缘残留亮结构。
        if self.diff_params.get('apply_rpca_bg_subtraction', True):
            rpca_start = time.time()
            diff_image = self.rpca_background_subtract_difference(
                diff_image, overlap_mask, diff_calc_mode=diff_calc_mode
            )
            self.logger.debug(f"  ⏱️  RPCA去背景耗时: {time.time() - rpca_start:.3f}秒")
        self.logger.debug(f"  ⏱️  计算差异耗时: {time.time() - diff_start:.3f}秒")

        # 二值化差异图像
        binary_start = time.time()
        binary_diff = (diff_image > self.diff_params['diff_threshold']).astype(np.uint8)
        self.logger.debug(f"  ⏱️  二值化耗时: {time.time() - binary_start:.3f}秒")

        intermediate_images = {
            'normalized_reference': norm_img1,
            'normalized_aligned': norm_img2,
            'blurred_reference': blurred_img1,
            'blurred_aligned': blurred_img2
        }

        return diff_image, binary_diff, overlap_mask, intermediate_images
    
    def save_fits_result(self, data, output_path, header=None):
        """
        保存数据为FITS文件
        
        Args:
            data (numpy.ndarray): 要保存的数据
            output_path (str): 输出路径
            header: FITS头信息（可选）
        """
        try:
            hdu = fits.PrimaryHDU(data=data.astype(np.float32), header=header)
            hdu.writeto(output_path, overwrite=True)
            self.logger.info(f"FITS文件已保存: {output_path}")
        except Exception as e:
            self.logger.error(f"保存FITS文件失败 {output_path}: {str(e)}")
    
    def get_overlap_bounding_box(self, overlap_mask):
        """
        获取重叠区域的边界框

        Args:
            overlap_mask (numpy.ndarray): 重叠区域掩码

        Returns:
            tuple: (x_min, y_min, x_max, y_max) 边界框坐标，如果没有重叠区域返回None
        """
        try:
            # 找到所有非零像素的坐标
            coords = np.argwhere(overlap_mask > 0)

            if len(coords) == 0:
                return None

            # 获取边界框 (注意：argwhere返回的是(row, col)即(y, x))
            y_min, x_min = coords.min(axis=0)
            y_max, x_max = coords.max(axis=0)

            return (x_min, y_min, x_max, y_max)

        except Exception as e:
            self.logger.error(f"计算重叠区域边界框失败: {str(e)}")
            return None

    def save_jpg_result(self, data, output_path, title="", colormap='viridis',
                       overlap_bbox=None, draw_alignment_box=False):
        """
        保存数据为JPG文件

        Args:
            data (numpy.ndarray): 要保存的数据
            output_path (str): 输出路径
            title (str): 图像标题
            colormap (str): 颜色映射
            overlap_bbox (tuple): 重叠区域边界框 (x_min, y_min, x_max, y_max)
            draw_alignment_box (bool): 是否绘制对齐区域方框
        """
        try:
            # 防御式切换为非交互后端，避免在子线程/Tk主循环之外触发错误
            try:
                import matplotlib.pyplot as _plt
                _plt.switch_backend('Agg')
            except Exception:
                pass
            fig, ax = plt.subplots(figsize=(10, 8))
            im = ax.imshow(data, cmap=colormap, origin='lower')
            plt.colorbar(im, ax=ax, label='强度')
            ax.set_title(title)

            # 如果需要绘制对齐区域方框
            if draw_alignment_box and overlap_bbox is not None:
                x_min, y_min, x_max, y_max = overlap_bbox
                width = x_max - x_min
                height = y_max - y_min

                # 绘制绿色方框表示对齐区域
                from matplotlib.patches import Rectangle
                rect = Rectangle((x_min, y_min), width, height,
                               linewidth=2, edgecolor='lime', facecolor='none',
                               linestyle='--', label='Alignment Region')
                ax.add_patch(rect)

                # 添加文本标注
                text_x = x_min + 10
                text_y = y_min + 20
                ax.text(text_x, text_y,
                       f'Align Box: ({x_min},{y_min})-({x_max},{y_max})\nSize: {width}x{height}',
                       color='lime', fontsize=9, weight='bold',
                       bbox=dict(boxstyle='round,pad=0.5', facecolor='black', alpha=0.7))

                # 添加图例
                ax.legend(loc='upper right', fontsize=9)

                self.logger.info(f"对齐区域边界框: ({x_min},{y_min})-({x_max},{y_max}), 大小: {width}x{height}")

            plt.tight_layout()
            plt.savefig(output_path, dpi=150, bbox_inches='tight')
            plt.close()
            self.logger.info(f"JPG文件已保存: {output_path}")
        except Exception as e:
            self.logger.error(f"保存JPG文件失败 {output_path}: {str(e)}")
    
    def find_aligned_fits_files(self, directory):
        """
        在指定目录中查找已对齐的FITS文件

        Args:
            directory (str): 目录路径

        Returns:
            tuple: (参考文件路径, 对齐文件路径)
        """
        # 只查找 noise_cleaned_aligned.fits 文件
        noise_cleaned_files = glob.glob(os.path.join(directory, "*noise_cleaned_aligned.fits"))

        if len(noise_cleaned_files) < 2:
            self.logger.error(f"目录中 noise_cleaned_aligned.fits 文件数量不足: {len(noise_cleaned_files)}")
            if noise_cleaned_files:
                for f in noise_cleaned_files:
                    self.logger.error(f"  - {os.path.basename(f)}")
            return None, None

        if len(noise_cleaned_files) > 2:
            self.logger.error(f"目录中 noise_cleaned_aligned.fits 文件数量过多: {len(noise_cleaned_files)}，应该只有2个")
            for f in noise_cleaned_files:
                self.logger.error(f"  - {os.path.basename(f)}")
            return None, None

        # 根据文件名模式识别模板文件和对齐文件
        # K***-*_noise_cleaned_aligned.fits 是模板文件
        # GY*_K***-*_noise_cleaned_aligned.fits 是对齐文件（下载文件）
        template_file = None
        aligned_file = None

        for file_path in noise_cleaned_files:
            filename = os.path.basename(file_path)

            # 判断是模板文件还是对齐文件
            # 模板文件格式: K***-*_noise_cleaned_aligned.fits (以K开头)
            # 对齐文件格式: GY*_K***-*_noise_cleaned_aligned.fits (以GY开头)
            if filename.startswith("K") and not filename.startswith("GY"):
                template_file = file_path
            elif filename.startswith("GY"):
                aligned_file = file_path
            else:
                self.logger.error(f"无法识别文件类型: {filename}")
                self.logger.error(f"模板文件应以 K 开头，对齐文件应以 GY 开头")
                return None, None

        # 检查是否找到了两个文件
        if not template_file or not aligned_file:
            self.logger.error("未找到完整的文件对")
            self.logger.error(f"模板文件: {os.path.basename(template_file) if template_file else '未找到'}")
            self.logger.error(f"对齐文件: {os.path.basename(aligned_file) if aligned_file else '未找到'}")
            return None, None

        self.logger.info("成功识别文件对")
        self.logger.info(f"模板文件: {os.path.basename(template_file)}")
        self.logger.info(f"对齐文件: {os.path.basename(aligned_file)}")

        return template_file, aligned_file

    def _get_astap_executable(self):
        """
        从项目配置中解析 ASTAP 可执行文件路径。

        Returns:
            str | None: ASTAP 可执行文件路径
        """
        try:
            script_dir = os.path.dirname(os.path.abspath(__file__))
            project_root = os.path.dirname(script_dir)
            config_path = os.path.join(project_root, 'config', 'url_config.json')
            if not os.path.exists(config_path):
                self.logger.error(f"未找到ASTAP配置文件: {config_path}")
                return None

            with open(config_path, 'r', encoding='utf-8') as f:
                cfg = json.load(f)

            template = cfg.get('astap_cmd_template', '')
            if not template:
                self.logger.error("配置中缺少 astap_cmd_template")
                return None

            parts = shlex.split(template, posix=False)
            if not parts:
                self.logger.error("astap_cmd_template 解析失败")
                return None

            exe_path = parts[0].strip('"')
            return exe_path
        except Exception as e:
            self.logger.error(f"解析ASTAP路径失败: {e}")
            return None

    def _normalize_cutout_to_u8(self, data):
        """把 cutout 归一化到 uint8 便于保存 PNG。"""
        arr = np.asarray(data, dtype=np.float32)
        finite = np.isfinite(arr)
        if not np.any(finite):
            return np.zeros(arr.shape, dtype=np.uint8)

        vals = arr[finite]
        p1, p99 = np.percentile(vals, [1, 99])
        if p99 <= p1:
            vmin, vmax = float(np.min(vals)), float(np.max(vals))
        else:
            vmin, vmax = float(p1), float(p99)
        if vmax <= vmin:
            return np.zeros(arr.shape, dtype=np.uint8)

        out = np.clip((arr - vmin) / (vmax - vmin), 0, 1)
        return (out * 255).astype(np.uint8)

    def _extract_cutout(self, image, cx, cy, size=100):
        """围绕中心点提取固定尺寸 cutout，不足部分补零。"""
        half = size // 2
        h, w = image.shape
        x0, x1 = int(cx - half), int(cx + half)
        y0, y1 = int(cy - half), int(cy + half)

        src_x0, src_x1 = max(0, x0), min(w, x1)
        src_y0, src_y1 = max(0, y0), min(h, y1)

        cut = np.zeros((size, size), dtype=np.float32)
        dst_x0, dst_y0 = src_x0 - x0, src_y0 - y0
        dst_x1, dst_y1 = dst_x0 + (src_x1 - src_x0), dst_y0 + (src_y1 - src_y0)
        if src_x1 > src_x0 and src_y1 > src_y0:
            cut[dst_y0:dst_y1, dst_x0:dst_x1] = image[src_y0:src_y1, src_x0:src_x1]
        return cut

    def _save_astap_marked_difference_png(self, diff_data, stars, output_png_path):
        """
        基于 ASTAP 坐标导出 difference.fits 标注图。
        """
        if diff_data is None or len(stars) == 0:
            return False

        try:
            gray_u8 = self._normalize_cutout_to_u8(diff_data)
            canvas = cv2.cvtColor(gray_u8, cv2.COLOR_GRAY2BGR)
            h, w = gray_u8.shape
            radius = max(4, int(min(h, w) * 0.004))
            thickness = max(1, radius // 2)

            for x, y in stars:
                ix, iy = int(round(x)), int(round(y))
                if ix < 0 or iy < 0 or ix >= w or iy >= h:
                    continue
                cv2.circle(canvas, (ix, iy), radius, (0, 255, 255), thickness, lineType=cv2.LINE_AA)
                cv2.drawMarker(canvas, (ix, iy), (0, 0, 255), markerType=cv2.MARKER_CROSS,
                               markerSize=max(8, radius * 2), thickness=1, line_type=cv2.LINE_AA)

            os.makedirs(os.path.dirname(output_png_path), exist_ok=True)
            return bool(cv2.imwrite(output_png_path, canvas))
        except Exception as e:
            self.logger.warning(f"生成ASTAP标注差分图失败: {e}")
            return False

    def _parse_astap_sources(self, output_directory, created_files, image_shape, diff_fits_path=None):
        """
        解析 ASTAP 产物中的星点坐标，返回 [(x, y), ...]。
        支持 csv/txt 的宽松解析。
        """
        h, w = image_shape
        coords = []

        def _append_xy(xv, yv):
            try:
                x, y = float(xv), float(yv)
            except Exception:
                return
            if 0 <= x < w and 0 <= y < h:
                coords.append((x, y))

        candidate_files = [
            os.path.join(output_directory, f)
            for f in created_files
            if f.lower().endswith(('.csv', '.txt'))
        ]
        # ASTAP 可能覆盖已存在文件，导致 created_files 中没有变化；补充同名 csv 探测。
        if diff_fits_path:
            base_csv = os.path.splitext(os.path.basename(diff_fits_path))[0] + ".csv"
            base_csv_path = os.path.join(output_directory, base_csv)
            if os.path.exists(base_csv_path) and base_csv_path not in candidate_files:
                candidate_files.append(base_csv_path)

        for path in candidate_files:
            try:
                if path.lower().endswith('.csv'):
                    with open(path, 'r', encoding='utf-8', errors='ignore') as f:
                        reader = csv.reader(f)
                        for row in reader:
                            if len(row) < 2:
                                continue
                            # 优先前两列；失败则扫描行中前两个数字
                            try:
                                _append_xy(row[0], row[1])
                                continue
                            except Exception:
                                pass

                            nums = []
                            for c in row:
                                try:
                                    nums.append(float(c))
                                except Exception:
                                    continue
                                if len(nums) >= 2:
                                    break
                            if len(nums) >= 2:
                                _append_xy(nums[0], nums[1])
                else:
                    with open(path, 'r', encoding='utf-8', errors='ignore') as f:
                        for line in f:
                            tokens = line.replace(',', ' ').replace(';', ' ').split()
                            nums = []
                            for t in tokens:
                                try:
                                    nums.append(float(t))
                                except Exception:
                                    continue
                                if len(nums) >= 2:
                                    break
                            if len(nums) >= 2:
                                _append_xy(nums[0], nums[1])
            except Exception:
                continue

        # 去重并限制数量，避免异常文件导致过多点
        unique = []
        seen = set()
        for x, y in coords:
            key = (int(round(x)), int(round(y)))
            if key in seen:
                continue
            seen.add(key)
            unique.append((x, y))
            if len(unique) >= 2000:
                break

        return unique

    def _write_astap_detection_outputs(
        self,
        diff_fits_path,
        output_directory,
        reference_file,
        aligned_file,
        created_files,
        detected_count=None,
        fast_mode=False
    ):
        """
        将 ASTAP 提取结果整理为 detection_* 目录，兼容后续 GUI/流程。
        """
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        detection_dir = os.path.join(output_directory, f"detection_{timestamp}")
        cutouts_dir = os.path.join(detection_dir, "cutouts")
        os.makedirs(cutouts_dir, exist_ok=True)

        # 读取图像用于 cutouts
        diff_data = self.load_fits_data(diff_fits_path)
        ref_data = self.load_fits_data(reference_file) if reference_file and os.path.exists(reference_file) else None
        aligned_data = self.load_fits_data(aligned_file) if aligned_file and os.path.exists(aligned_file) else None

        stars = []
        if diff_data is not None:
            stars = self._parse_astap_sources(
                output_directory, created_files, diff_data.shape, diff_fits_path=diff_fits_path
            )

        if detected_count is None or (int(detected_count) == 0 and len(stars) > 0):
            detected_count = len(stars)

        base_name = os.path.splitext(os.path.basename(diff_fits_path))[0]
        marked_png_path = os.path.join(output_directory, f"{base_name}_astap_marked.png")

        # 拷贝 ASTAP 原始输出，便于追溯
        raw_dir = os.path.join(detection_dir, "astap_raw")
        os.makedirs(raw_dir, exist_ok=True)
        for f in created_files:
            src = os.path.join(output_directory, f)
            if os.path.isfile(src):
                try:
                    shutil.copy2(src, os.path.join(raw_dir, os.path.basename(f)))
                except Exception:
                    pass

        # 生成 cutouts（如可解析到坐标）
        if diff_data is not None and stars:
            for idx, (x, y) in enumerate(stars, 1):
                ix, iy = int(round(x)), int(round(y))
                prefix = f"{idx:03d}_X{ix}_Y{iy}"

                det_cut = self._extract_cutout(diff_data, ix, iy, size=100)
                det_u8 = self._normalize_cutout_to_u8(det_cut)
                cv2.imwrite(os.path.join(cutouts_dir, f"{prefix}_3_detection.png"), det_u8)

                if ref_data is not None:
                    ref_cut = self._extract_cutout(ref_data, ix, iy, size=100)
                    ref_u8 = self._normalize_cutout_to_u8(ref_cut)
                    cv2.imwrite(os.path.join(cutouts_dir, f"{prefix}_1_reference.png"), ref_u8)

                if aligned_data is not None:
                    ali_cut = self._extract_cutout(aligned_data, ix, iy, size=100)
                    ali_u8 = self._normalize_cutout_to_u8(ali_cut)
                    cv2.imwrite(os.path.join(cutouts_dir, f"{prefix}_2_aligned.png"), ali_u8)

        # 额外输出 difference.fits 星点标注 PNG（坐标来源仅 ASTAP）
        marked_png_generated = False
        if diff_data is not None and stars:
            marked_png_generated = self._save_astap_marked_difference_png(
                diff_data=diff_data,
                stars=stars,
                output_png_path=marked_png_path
            )
            if marked_png_generated:
                self.logger.info(f"已生成ASTAP标注图: {os.path.basename(marked_png_path)}")

        # 生成分析文件（保持 GUI 依赖的关键文案）
        analysis_path = os.path.join(detection_dir, f"{base_name}_analysis_astap_extract2.txt")
        with open(analysis_path, 'w', encoding='utf-8') as f:
            f.write("ASTAP提取检测结果\n")
            f.write("=" * 80 + "\n")
            f.write(f"处理时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"输入差异图: {os.path.basename(diff_fits_path)}\n")
            f.write("检测方法: ASTAP -extract 2\n")
            f.write(f"检测到 {int(detected_count)} 个斑点\n\n")
            if stars:
                f.write(f"{'序号':<8}{'X坐标':<12}{'Y坐标':<12}\n")
                f.write("-" * 40 + "\n")
                for i, (x, y) in enumerate(stars, 1):
                    f.write(f"{i:<8}{x:<12.2f}{y:<12.2f}\n")

        return {
            'detection_dir': detection_dir,
            'analysis_file': analysis_path,
            'cutouts_dir': cutouts_dir,
            'parsed_stars': len(stars),
            'marked_png': marked_png_path if marked_png_generated else None
        }

    def run_astap_extractor(self, diff_fits_path, output_directory, reference_file=None, aligned_file=None, fast_mode=False, generate_gif=False):
        """
        对 difference.fits 执行 ASTAP 提取（-extract 2）。

        Args:
            diff_fits_path: difference.fits文件路径
            output_directory: 输出目录
            reference_file: 参考图像（模板）FITS文件路径
            aligned_file: 对齐图像（下载）FITS文件路径
            fast_mode: 快速模式，不生成hull和poly可视化图片，默认False
            generate_gif: 是否生成GIF动画，默认False

        Returns:
            dict: 检测结果信息
        """
        try:
            astap_exe = self._get_astap_executable()
            if not astap_exe:
                return {'success': False, 'error': 'ASTAP executable not found'}

            # 构建命令
            cmd = [
                astap_exe,
                '-f',
                diff_fits_path,
                '-extract',
                '2'
            ]
            self.logger.info(f"执行ASTAP提取命令: {' '.join(cmd)}")

            pre_files = set()
            if os.path.isdir(output_directory):
                pre_files = set(os.listdir(output_directory))

            # 执行检测
            result = subprocess.run(
                cmd,
                cwd=output_directory,
                capture_output=True,
                text=True,
                encoding='utf-8',
                errors='replace',  # 遇到无法解码的字符时替换为?
                timeout=300  # 5分钟超时
            )

            output_lines = (result.stdout + "\n" + result.stderr).split('\n')
            detected_count = None
            for line in output_lines:
                text = line.strip()
                if not text:
                    continue
                # 常见 ASTAP 输出里包含 "xxx stars"
                try:
                    import re
                    m = re.search(r'(\d+)\s+stars?', text, re.IGNORECASE)
                    if m:
                        detected_count = int(m.group(1))
                except Exception:
                    pass

            post_files = set(os.listdir(output_directory)) if os.path.isdir(output_directory) else set()
            created_files = sorted(list(post_files - pre_files))
            if created_files:
                self.logger.info(f"ASTAP新生成文件数: {len(created_files)}")

            detection_outputs = self._write_astap_detection_outputs(
                diff_fits_path=diff_fits_path,
                output_directory=output_directory,
                reference_file=reference_file,
                aligned_file=aligned_file,
                created_files=created_files,
                detected_count=detected_count,
                fast_mode=fast_mode
            )
            parsed_count = detection_outputs.get('parsed_stars', 0)

            # 仅使用 ASTAP 导出结果：要求至少能解析到一个坐标。
            if parsed_count <= 0:
                err_msg = f"ASTAP未导出可解析坐标(returncode={result.returncode})"
                self.logger.error(err_msg)
                return {
                    'success': False,
                    'error': err_msg + ("\n" + result.stderr if result.stderr else ""),
                    'detected_count': 0
                }

            if result.returncode == 0:
                self.logger.info("ASTAP提取完成")
            else:
                self.logger.warning(
                    f"ASTAP返回码非0({result.returncode})，但已解析到{parsed_count}个目标，按成功处理"
                )

            self.logger.info(
                f"检测目录已输出: {os.path.basename(detection_outputs.get('detection_dir', ''))} "
                f"(解析星点={parsed_count}, 统计数量={detected_count if detected_count is not None else parsed_count})"
            )

            return {
                'success': True,
                'output': result.stdout,
                'detected_count': detected_count if detected_count is not None else parsed_count,
                'created_files': created_files,
                'detection_dir': detection_outputs.get('detection_dir'),
                'analysis_file': detection_outputs.get('analysis_file')
            }

        except subprocess.TimeoutExpired:
            self.logger.error("ASTAP提取执行超时")
            return {'success': False, 'error': 'Timeout'}
        except Exception as e:
            self.logger.error(f"执行ASTAP提取时出错: {str(e)}")
            return {'success': False, 'error': str(e)}

    def process_aligned_fits_comparison(self, input_directory, output_directory=None, remove_bright_lines=True, fast_mode=False, max_jaggedness_ratio=2.0, detection_method='contour', overlap_edge_exclusion_px=40, generate_gif=False, diff_calc_mode='abs', apply_diff_postprocess=False):
        """
        处理已对齐FITS文件的差异比较

        Args:
            input_directory (str): 输入目录路径
            output_directory (str): 输出目录路径
            remove_bright_lines (bool): 是否去除亮线，默认True
            fast_mode (bool): 快速模式，减少中间文件输出，默认False
            max_jaggedness_ratio (float): 最大锯齿比率，默认2.0
            detection_method (str): 检测方法，'contour'=轮廓检测（默认）, 'simple_blob'=SimpleBlobDetector
            overlap_edge_exclusion_px (int): 重叠边界剔除宽度（像素），默认40
            generate_gif (bool): 是否生成GIF动画，默认False
            diff_calc_mode (str): 差异计算方式，'abs'（默认）或 'signed'
            apply_diff_postprocess (bool): 是否对差异图执行后处理（负值置零+中值滤波）

        Returns:
            dict: 处理结果信息
        """
        # 记录总体开始时间
        total_start_time = time.time()
        timing_stats = {}

        # 设置输出目录
        setup_start = time.time()
        if output_directory is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_directory = f"aligned_diff_results_{timestamp}"

        os.makedirs(output_directory, exist_ok=True)
        timing_stats['设置输出目录'] = time.time() - setup_start
        self.logger.info(f"⏱️  设置输出目录耗时: {timing_stats['设置输出目录']:.3f}秒")

        # 查找FITS文件
        find_start = time.time()
        reference_file, aligned_file = self.find_aligned_fits_files(input_directory)
        if not reference_file or not aligned_file:
            return None
        timing_stats['查找FITS文件'] = time.time() - find_start
        self.logger.info(f"⏱️  查找FITS文件耗时: {timing_stats['查找FITS文件']:.3f}秒")

        # 加载FITS数据
        load_start = time.time()
        self.logger.info("加载FITS文件...")
        ref_data = self.load_fits_data(reference_file)
        aligned_data = self.load_fits_data(aligned_file)

        if ref_data is None or aligned_data is None:
            self.logger.error("FITS文件加载失败")
            return None

        # 检查图像尺寸
        if ref_data.shape != aligned_data.shape:
            self.logger.error(f"图像尺寸不匹配: {ref_data.shape} vs {aligned_data.shape}")
            return None

        timing_stats['加载FITS数据'] = time.time() - load_start
        self.logger.info(f"⏱️  加载FITS数据耗时: {timing_stats['加载FITS数据']:.3f}秒")

        # 设置重叠边界剔除参数（差异检测时生效）
        try:
            edge_exclusion = max(0, int(overlap_edge_exclusion_px))
        except Exception:
            edge_exclusion = 40
        self.diff_params['overlap_edge_exclusion_px'] = edge_exclusion

        # 执行差异检测
        diff_start = time.time()
        self.logger.info("执行差异检测...")
        diff_image, binary_diff, overlap_mask, intermediate_images = self.detect_differences(
            ref_data, aligned_data,
            diff_calc_mode=diff_calc_mode,
            apply_diff_postprocess=apply_diff_postprocess
        )
        timing_stats['差异检测'] = time.time() - diff_start
        self.logger.info(f"⏱️  差异检测耗时: {timing_stats['差异检测']:.3f}秒")

        # 应用重叠掩码到所有输出图像（确保非重叠区域为黑色）
        mask_start = time.time()
        self.logger.info("应用重叠掩码，确保非重叠区域为黑色...")
        ref_data = ref_data * overlap_mask
        aligned_data = aligned_data * overlap_mask
        timing_stats['应用重叠掩码'] = time.time() - mask_start
        self.logger.info(f"⏱️  应用重叠掩码耗时: {timing_stats['应用重叠掩码']:.3f}秒")

        # 生成输出文件名
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        base_name = f"aligned_comparison_{timestamp}"

        # 保存FITS格式结果
        save_fits_start = time.time()
        self.logger.info("保存FITS格式结果...")

        # 保存差异图像（FITS）
        diff_fits_path = os.path.join(output_directory, f"{base_name}_difference.fits")
        self.save_fits_result(diff_image, diff_fits_path)

        # 初始化文件路径变量（快速模式下可能不会创建这些文件）
        binary_fits_path = None
        overlap_mask_fits_path = None
        normalized_ref_fits_path = None
        normalized_aligned_fits_path = None
        blurred_ref_fits_path = None
        blurred_aligned_fits_path = None
        ref_jpg_path = None
        aligned_jpg_path = None
        diff_jpg_path = None
        binary_jpg_path = None
        overlap_mask_jpg_path = None

        # 快速模式：跳过大部分中间文件保存
        if not fast_mode:
            # 保存二值化差异图像（FITS）
            binary_fits_path = os.path.join(output_directory, f"{base_name}_binary_diff.fits")
            self.save_fits_result(binary_diff.astype(np.float32), binary_fits_path)

            # 保存重叠掩码（FITS）
            overlap_mask_fits_path = os.path.join(output_directory, f"{base_name}_overlap_mask.fits")
            self.save_fits_result(overlap_mask.astype(np.float32), overlap_mask_fits_path)

            # 保存归一化后的图像（FITS）
            normalized_ref_fits_path = os.path.join(output_directory, f"{base_name}_normalized_reference.fits")
            self.save_fits_result(intermediate_images['normalized_reference'], normalized_ref_fits_path)

            normalized_aligned_fits_path = os.path.join(output_directory, f"{base_name}_normalized_aligned.fits")
            self.save_fits_result(intermediate_images['normalized_aligned'], normalized_aligned_fits_path)

            # 保存高斯平滑后的图像（FITS）
            blurred_ref_fits_path = os.path.join(output_directory, f"{base_name}_blurred_reference.fits")
            self.save_fits_result(intermediate_images['blurred_reference'], blurred_ref_fits_path)

            blurred_aligned_fits_path = os.path.join(output_directory, f"{base_name}_blurred_aligned.fits")
            self.save_fits_result(intermediate_images['blurred_aligned'], blurred_aligned_fits_path)

        timing_stats['保存FITS文件'] = time.time() - save_fits_start
        self.logger.info(f"⏱️  保存FITS文件耗时: {timing_stats['保存FITS文件']:.3f}秒")

        # 计算重叠区域边界框用于调试可视化
        bbox_start = time.time()
        self.logger.info("计算对齐区域边界框...")
        overlap_bbox = self.get_overlap_bounding_box(overlap_mask)
        timing_stats['计算边界框'] = time.time() - bbox_start
        self.logger.info(f"⏱️  计算边界框耗时: {timing_stats['计算边界框']:.3f}秒")

        if not fast_mode:
            # 保存JPG格式结果
            save_jpg_start = time.time()
            self.logger.info("保存JPG格式结果（包含对齐区域调试信息）...")

            # 保存参考图像（JPG）- 带对齐区域方框
            ref_jpg_path = os.path.join(output_directory, f"{base_name}_reference.jpg")
            self.save_jpg_result(self.normalize_image(ref_data), ref_jpg_path,
                               "参考图像（非重叠区域已设为黑色）", 'gray',
                               overlap_bbox=overlap_bbox, draw_alignment_box=True)

            # 保存对齐图像（JPG）- 带对齐区域方框
            aligned_jpg_path = os.path.join(output_directory, f"{base_name}_aligned.jpg")
            self.save_jpg_result(self.normalize_image(aligned_data), aligned_jpg_path,
                               "对齐图像（非重叠区域已设为黑色）", 'gray',
                               overlap_bbox=overlap_bbox, draw_alignment_box=True)

            # 保存差异图像（JPG）- 带对齐区域方框
            diff_jpg_path = os.path.join(output_directory, f"{base_name}_difference.jpg")
            self.save_jpg_result(diff_image, diff_jpg_path,
                               "差异图像（仅重叠区域）", 'hot',
                               overlap_bbox=overlap_bbox, draw_alignment_box=True)

            # 保存二值化差异图像（JPG）- 带对齐区域方框
            binary_jpg_path = os.path.join(output_directory, f"{base_name}_binary_diff.jpg")
            self.save_jpg_result(binary_diff, binary_jpg_path,
                               "二值化差异图像（仅重叠区域）", 'gray',
                               overlap_bbox=overlap_bbox, draw_alignment_box=True)

            # 保存重叠掩码（JPG）- 带对齐区域方框
            overlap_mask_jpg_path = os.path.join(output_directory, f"{base_name}_overlap_mask.jpg")
            self.save_jpg_result(overlap_mask, overlap_mask_jpg_path,
                               "重叠区域掩码（白色=重叠，黑色=非重叠）", 'gray',
                               overlap_bbox=overlap_bbox, draw_alignment_box=True)

            timing_stats['保存JPG文件'] = time.time() - save_jpg_start
            self.logger.info(f"⏱️  保存JPG文件耗时: {timing_stats['保存JPG文件']:.3f}秒")
        else:
            self.logger.info("快速模式：跳过中间文件保存")

        # 执行 ASTAP 提取检测
        blob_start = time.time()
        self.logger.info("执行ASTAP提取检测...")
        blob_detection_result = self.run_astap_extractor(
            diff_fits_path, output_directory,
            reference_file=reference_file,
            aligned_file=aligned_file,
            fast_mode=fast_mode,
            generate_gif=generate_gif
        )
        timing_stats['信号检测'] = time.time() - blob_start
        self.logger.info(f"⏱️  信号检测耗时: {timing_stats['信号检测']:.3f}秒")

        # 快速模式：检测完成后删除差异FITS文件
        cleanup_start = time.time()
        if fast_mode and os.path.exists(diff_fits_path):
            try:
                os.remove(diff_fits_path)
                self.logger.info(f"快速模式：已删除中间文件 {os.path.basename(diff_fits_path)}")
                diff_fits_path = None  # 标记为已删除
            except Exception as e:
                self.logger.warning(f"快速模式：删除中间文件失败: {e}")

        if fast_mode:
            timing_stats['清理中间文件'] = time.time() - cleanup_start
            self.logger.info(f"⏱️  清理中间文件耗时: {timing_stats['清理中间文件']:.3f}秒")

        # 返回处理结果
        output_files = {
            'fits': {},
            'jpg': {}
        }

        # 只包含实际存在的文件
        if diff_fits_path:  # 快速模式下可能已被删除
            output_files['fits']['difference'] = diff_fits_path
        if binary_fits_path:
            output_files['fits']['binary_diff'] = binary_fits_path
        if overlap_mask_fits_path:
            output_files['fits']['overlap_mask'] = overlap_mask_fits_path
        if normalized_ref_fits_path:
            output_files['fits']['normalized_reference'] = normalized_ref_fits_path
        if normalized_aligned_fits_path:
            output_files['fits']['normalized_aligned'] = normalized_aligned_fits_path
        if blurred_ref_fits_path:
            output_files['fits']['blurred_reference'] = blurred_ref_fits_path
        if blurred_aligned_fits_path:
            output_files['fits']['blurred_aligned'] = blurred_aligned_fits_path
        if ref_jpg_path:
            output_files['jpg']['reference'] = ref_jpg_path
        if aligned_jpg_path:
            output_files['jpg']['aligned'] = aligned_jpg_path
        if diff_jpg_path:
            output_files['jpg']['difference'] = diff_jpg_path
        if binary_jpg_path:
            output_files['jpg']['binary_diff'] = binary_jpg_path
        if overlap_mask_jpg_path:
            output_files['jpg']['overlap_mask'] = overlap_mask_jpg_path

        # 计算总耗时
        total_time = time.time() - total_start_time
        timing_stats['总耗时'] = total_time

        # 输出耗时统计摘要
        self.logger.info("=" * 60)
        self.logger.info("⏱️  差异比较耗时统计摘要:")
        self.logger.info(f"  设置输出目录: {timing_stats.get('设置输出目录', 0):.3f}秒")
        self.logger.info(f"  查找FITS文件: {timing_stats.get('查找FITS文件', 0):.3f}秒")
        self.logger.info(f"  加载FITS数据: {timing_stats.get('加载FITS数据', 0):.3f}秒")
        self.logger.info(f"  差异检测: {timing_stats.get('差异检测', 0):.3f}秒")
        self.logger.info(f"  应用重叠掩码: {timing_stats.get('应用重叠掩码', 0):.3f}秒")
        self.logger.info(f"  保存FITS文件: {timing_stats.get('保存FITS文件', 0):.3f}秒")
        self.logger.info(f"  计算边界框: {timing_stats.get('计算边界框', 0):.3f}秒")
        if not fast_mode:
            self.logger.info(f"  保存JPG文件: {timing_stats.get('保存JPG文件', 0):.3f}秒")
        self.logger.info(f"  信号检测: {timing_stats.get('信号检测', 0):.3f}秒")
        if fast_mode:
            self.logger.info(f"  清理中间文件: {timing_stats.get('清理中间文件', 0):.3f}秒")
        self.logger.info(f"  总耗时: {total_time:.3f}秒")
        self.logger.info("=" * 60)

        final_target_count = 0
        if blob_detection_result and blob_detection_result.get('success'):
            detected_count = blob_detection_result.get('detected_count')
            if isinstance(detected_count, int):
                final_target_count = detected_count

        result = {
            'success': True,
            'reference_file': reference_file,
            'aligned_file': aligned_file,
            'output_directory': output_directory,
            'new_bright_spots': final_target_count,
            'blob_detection': blob_detection_result,
            'output_files': output_files,
            'fast_mode': fast_mode,
            'timing_stats': timing_stats,  # 添加耗时统计信息
            'alignment_success': True  # 添加对齐成功标志
        }

        return result


def main():
    """主函数"""
    import argparse

    parser = argparse.ArgumentParser(
        description='已对齐FITS文件差异比较工具',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
使用示例:
  # 使用默认输入目录
  python compare_aligned_fits.py

  # 指定输入目录
  python compare_aligned_fits.py --input E:\\fix_data\\align-diff

  # 指定输入和输出目录
  python compare_aligned_fits.py --input E:\\fix_data\\align-diff --output results
        """
    )

    parser.add_argument('--input', '-i', default=r'E:\fix_data\align-diff',
                       help='包含已对齐FITS文件的输入目录')
    parser.add_argument('--output', '-o',
                       help='输出目录（默认自动生成时间戳目录）')
    parser.add_argument('--threshold', '-t', type=float, default=0.0,
                       help='差异检测阈值（默认0.0）')
    parser.add_argument('--gaussian-sigma', '-g', type=float, default=1.0,
                       help='高斯模糊参数（默认1.0）')

    args = parser.parse_args()

    # 检查输入目录
    if not os.path.exists(args.input):
        print(f"错误: 输入目录不存在 - {args.input}")
        sys.exit(1)

    # 创建比较器
    comparator = AlignedFITSComparator()

    # 更新参数
    comparator.diff_params['diff_threshold'] = args.threshold
    comparator.diff_params['gaussian_sigma'] = args.gaussian_sigma

    print("=" * 60)
    print("已对齐FITS文件差异比较工具")
    print("=" * 60)
    print(f"输入目录: {args.input}")
    print(f"输出目录: {args.output or '自动生成'}")
    print(f"差异阈值: {args.threshold}")
    print(f"高斯模糊: {args.gaussian_sigma}")
    print("=" * 60)

    # 执行比较
    try:
        result = comparator.process_aligned_fits_comparison(args.input, args.output)

        if result and result['success']:
            print("\n处理完成！")
            print("=" * 60)
            print(f"参考文件: {os.path.basename(result['reference_file'])}")
            print(f"对齐文件: {os.path.basename(result['aligned_file'])}")
            print(f"检测到新亮点: {result['new_bright_spots']} 个")

            print(f"\n输出文件已保存到: {result['output_directory']}")
            print("\nFITS格式文件:")
            for name, path in result['output_files']['fits'].items():
                print(f"  {name}: {os.path.basename(path)}")

            print("\nJPG格式文件:")
            for name, path in result['output_files']['jpg'].items():
                print(f"  {name}: {os.path.basename(path)}")

        else:
            print("处理失败！请检查日志文件了解详细错误信息。")
            sys.exit(1)

    except KeyboardInterrupt:
        print("\n用户中断处理")
        sys.exit(1)
    except Exception as e:
        print(f"处理过程中发生错误: {str(e)}")
        sys.exit(1)


if __name__ == "__main__":
    main()
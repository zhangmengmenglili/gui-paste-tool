#!/usr/bin/env python3
"""
交互式缺陷粘贴工具 - GUI 版
============================
带文件夹选择按钮、标签输入框的完整图形界面。
鼠标画出缺陷区域 → 拖到好图上手动放置 → 保存。

启动:
    python gui_paste.py

或指定初始路径:
    python gui_paste.py --good-dir ./data/good --defect-dir ./data/defect
"""

import os
import sys
import json
import random
import argparse
from pathlib import Path
from typing import List, Tuple, Optional

import cv2
import numpy as np

# tkinter
import tkinter as tk
from tkinter import ttk, filedialog, messagebox
from PIL import Image, ImageTk

# ============================================================================
# 图像处理函数（复用之前的逻辑）
# ============================================================================


def imread_unicode(path: str) -> Optional[np.ndarray]:
    """Unicode 安全地读取图片（解决 OpenCV 在 Windows 上中文路径无法读取的问题）。"""
    try:
        data = np.fromfile(path, dtype=np.uint8)
        img = cv2.imdecode(data, cv2.IMREAD_COLOR)
        return img
    except Exception:
        return None


def imwrite_unicode(path: str, img: np.ndarray, params=None) -> bool:
    """Unicode 安全地保存图片。"""
    try:
        ext = os.path.splitext(path)[1].lower()
        if ext in (".jpg", ".jpeg"):
            fmt = ".jpg"
        elif ext == ".png":
            fmt = ".png"
        else:
            fmt = ".png"
        _, encoded = cv2.imencode(fmt, img, params or [])
        if encoded is None:
            return False
        encoded.tofile(path)
        return True
    except Exception:
        return False


def create_soft_mask_from_polygon(shape, points, feather=15):
    h, w = shape[:2]
    mask = np.zeros((h, w), dtype=np.uint8)
    pts = points.reshape((-1, 1, 2))
    cv2.fillPoly(mask, [pts], 255)
    if feather > 0:
        ksize = min(feather * 2 + 1, min(h, w) // 4 * 2 + 1)
        if ksize >= 3:
            mask = cv2.GaussianBlur(mask, (ksize, ksize), feather / 3)
    return mask


def match_color(defect_patch, bg_region, mask_float, strength=1.0):
    """将缺陷颜色匹配到背景区域，解决亮缺陷贴暗背景的色差问题。
    strength: 0=不匹配(原色), 1=完全匹配背景色调
    在 LAB 空间做均值/标准差迁移，mask 不为0的像素参与统计。
    """
    if strength <= 0:
        return defect_patch

    # 转 LAB (颜色更均匀)
    d_lab = cv2.cvtColor(defect_patch.astype(np.uint8), cv2.COLOR_BGR2LAB).astype(np.float32)
    b_lab = cv2.cvtColor(bg_region.astype(np.uint8), cv2.COLOR_BGR2LAB).astype(np.float32)

    result = d_lab.copy()
    m = mask_float  # (H,W), 0~1

    for c in range(3):
        d_ch = d_lab[:, :, c]
        b_ch = b_lab[:, :, c]

        # 只在 mask 区域内计算缺陷的统计
        d_vals = d_ch[m > 0.1]
        if len(d_vals) < 10:
            continue
        d_mean, d_std = d_vals.mean(), max(d_vals.std(), 1.0)

        # 背景区域统计（取 mask 外扩一圈的区域）
        b_vals = b_ch[m < 0.3]
        if len(b_vals) < 10:
            b_vals = b_ch.flatten()
        b_mean, b_std = b_vals.mean(), max(b_vals.std(), 1.0)

        # 颜色迁移: (d - d_mean) / d_std * b_std + b_mean
        adjusted = (d_ch - d_mean) / d_std * b_std + b_mean
        # 混合原色和匹配色
        result[:, :, c] = d_ch * (1 - strength) + adjusted * strength

    result = np.clip(result, 0, 255).astype(np.uint8)
    return cv2.cvtColor(result, cv2.COLOR_LAB2BGR)


def blend_defect(bg, defect_img, mask, x, y, blend_mode="alpha",
                 alpha_strength=1.0, color_match=0.0):
    """将缺陷混合到背景上。
    alpha_strength: 1.0=边缘最锐利，缺陷颜色完全保留；越小边缘越软（中心始终不变浅）
    color_match:    0.0=保留缺陷原色；1.0=自动匹配背景色调（解决亮缺陷贴暗背景问题）
    """
    result = bg.copy()
    dh, dw = defect_img.shape[:2]
    bg_h, bg_w = bg.shape[:2]

    x1, y1 = max(0, x), max(0, y)
    x2, y2 = min(bg_w, x + dw), min(bg_h, y + dh)
    if x2 <= x1 or y2 <= y1:
        return result

    dx1, dy1 = x1 - x, y1 - y
    dx2, dy2 = dx1 + (x2 - x1), dy1 + (y2 - y1)

    d_patch = defect_img[dy1:dy2, dx1:dx2]
    m_patch = mask[dy1:dy2, dx1:dx2].astype(np.float32) / 255.0
    bg_region = result[y1:y2, x1:x2]

    # ---- 颜色匹配（在混合之前做）----
    if color_match > 0.01:
        d_patch = match_color(d_patch, bg_region, m_patch, strength=color_match)

    # ---- 计算混合权重 ----
    if blend_mode == "direct":
        m_float = (m_patch > 0.3).astype(np.float32)
    else:
        gamma = 1.0 / max(0.1, alpha_strength)
        m_float = np.power(m_patch, gamma)

    m_3ch = np.stack([m_float] * 3, axis=-1)

    # ---- Alpha 混合 ----
    bg_float = bg_region.astype(np.float32)
    blended = d_patch.astype(np.float32) * m_3ch + bg_float * (1.0 - m_3ch)
    result[y1:y2, x1:x2] = blended.astype(np.uint8)
    return result


def extract_defect_region(img, mask):
    coords = cv2.findNonZero(mask)
    if coords is None or len(coords) < 3:
        return None, None, 0, 0, 0, 0
    x, y, w, h = cv2.boundingRect(coords)
    pad = max(5, min(w, h) // 8)
    x1 = max(0, x - pad)
    y1 = max(0, y - pad)
    x2 = min(img.shape[1], x + w + pad)
    y2 = min(img.shape[0], y + h + pad)
    region = img[y1:y2, x1:x2].copy()
    region_mask = mask[y1:y2, x1:x2].copy()
    return region, region_mask, x1, y1, x2, y2


# ============================================================================
# 主 GUI 应用
# ============================================================================


MODE_SELECT = 0   # 抠缺陷模式
MODE_PLACE = 1    # 贴缺陷模式


class DefectPasteGUI:
    def __init__(self):
        self.root = tk.Tk()
        self.root.title("交互式缺陷粘贴工具")
        self.root.geometry("1400x950")

        # ---- 数据 ----
        self.good_dir = tk.StringVar(value="")
        self.defect_dir = tk.StringVar(value="")
        self.output_dir = tk.StringVar(value="./output/gui_output")
        self.class_label = tk.StringVar(value="0")

        self.good_images = []
        self.defect_images = []
        self.current_good_idx = -1
        self.current_defect_idx = -1

        # ---- 模式 ----
        self.mode = MODE_SELECT
        self.mode_text = tk.StringVar(value="🔍 抠缺陷模式")
        self.status_text = tk.StringVar(value="请先选择好图目录和缺陷图目录")

        # ---- 缺陷图相关 ----
        self.defect_img_original = None
        self.defect_img_display = None
        self.polygon_points = []
        self.defect_mask = None

        # ---- 抠出的缺陷 ----
        self.cropped_defect = None
        self.cropped_mask = None

        # ---- 背景图相关 ----
        self.bg_img_original = None
        self.bg_img_display = None
        self.bg_img_clean = None
        self.pasted_defects = []

        # ---- 变换参数 ----
        self.scale_val = 1.0
        self.rotation_val = 0.0
        self.flip_h = False
        self.flip_v = False
        # 弹性/拉伸/斜切形变
        self.stretch_x = 1.0       # 横向拉伸/压缩 (相对uniform scale叠加)
        self.stretch_y = 1.0       # 纵向拉伸/压缩
        self.shear_x = 0.0         # 横向斜切角度 (度)
        self.shear_y = 0.0         # 纵向斜切角度 (度)
        self.elastic_strength = 0.0  # 弹性形变强度 0~1
        self.elastic_seed = 0        # 弹性形变随机种子
        self._elastic_cache = None   # (seed, shape, dx, dy)
        self.transformed_defect = None
        self.transformed_mask = None
        self.defect_pos = (0, 0)

        # ---- 鼠标 ----
        self.is_dragging = False
        self.drag_start = (0, 0)
        self.drag_start_pos = (0, 0)
        self.is_panning = False       # 拖拽平移视图
        self.pan_start = (0, 0)

        # ---- 视图控制 (解决大图显示不全) ----
        self.view_scale = 0.3          # 视图缩放 (0.1~3.0), 默认30%让4K图能看全
        self.view_offset_x = 0         # 视图平移 X
        self.view_offset_y = 0         # 视图平移 Y
        self.zoom_step = 0.05          # 每次滚轮缩放步长

        # ---- 参数 ----
        self.blend_mode = "alpha"
        self.alpha_strength = 1.0    # 边缘柔和度：1.0=锐利，越小边缘越软
        self.feather_size = 15       # 羽化像素
        self.color_match = 0.0       # 颜色匹配：0=原色，1=完全匹配背景色调

        # ---- 构建界面 ----
        self._build_ui()

        # ---- 快捷键 ----
        self.root.bind('<Key>', self._on_key)
        self.root.bind('<Control-s>', lambda e: self._save_result())
        self.root.bind('<Control-z>', lambda e: self._undo_paste())

    # ================================================================
    # 界面构建
    # ================================================================

    def _build_ui(self):
        style = ttk.Style()
        style.configure("Load.TButton", font=("", 13, "bold"), padding=8)

        # ---- 第0行: 标题提示 ----
        header = ttk.Frame(self.root, padding=5)
        header.pack(side=tk.TOP, fill=tk.X)
        ttk.Label(header, text="第1步: 选好图和缺陷图目录 → 第2步: 输标签ID → 第3步: 点加载数据 → 第4步: 鼠标抠图贴图 → 第5步: 保存",
                  font=("", 10), foreground="#555").pack(side=tk.LEFT, padx=10)

        # ---- 第1行: 目录选择 ----
        row1 = ttk.Frame(self.root, padding=5)
        row1.pack(side=tk.TOP, fill=tk.X)

        ttk.Label(row1, text="好图目录:", font=("", 10)).grid(row=0, column=0, sticky=tk.W, padx=(5, 2), pady=2)
        ttk.Entry(row1, textvariable=self.good_dir, width=50).grid(row=0, column=1, padx=2, pady=2)
        ttk.Button(row1, text="📁 浏览...", command=self._browse_good_dir).grid(row=0, column=2, padx=2, pady=2)

        ttk.Label(row1, text="缺陷图目录:", font=("", 10)).grid(row=1, column=0, sticky=tk.W, padx=(5, 2), pady=2)
        ttk.Entry(row1, textvariable=self.defect_dir, width=50).grid(row=1, column=1, padx=2, pady=2)
        ttk.Button(row1, text="📁 浏览...", command=self._browse_defect_dir).grid(row=1, column=2, padx=2, pady=2)

        ttk.Label(row1, text="输出目录:", font=("", 10)).grid(row=2, column=0, sticky=tk.W, padx=(5, 2), pady=2)
        ttk.Entry(row1, textvariable=self.output_dir, width=50).grid(row=2, column=1, padx=2, pady=2)
        ttk.Button(row1, text="📁 浏览...", command=self._browse_output_dir).grid(row=2, column=2, padx=2, pady=2)

        # ---- 第2行: 标签ID + 大大的加载按钮 + 保存/撤销 ----
        row2 = ttk.Frame(self.root, padding=5)
        row2.pack(side=tk.TOP, fill=tk.X, pady=5)

        ttk.Label(row2, text="YOLO标签ID:", font=("", 11)).pack(side=tk.LEFT, padx=(10, 2))
        label_entry = ttk.Entry(row2, textvariable=self.class_label, width=5, font=("", 12))
        label_entry.pack(side=tk.LEFT, padx=5)

        ttk.Separator(row2, orient=tk.VERTICAL).pack(side=tk.LEFT, padx=15, fill=tk.Y)

        # ★ 核心按钮 - 大而醒目
        self.btn_load = ttk.Button(row2, text="★  加 载 数 据  ★",
                                   command=self._load_data, style="Load.TButton")
        self.btn_load.pack(side=tk.LEFT, padx=10)

        ttk.Separator(row2, orient=tk.VERTICAL).pack(side=tk.LEFT, padx=15, fill=tk.Y)

        ttk.Button(row2, text="💾 保存结果 (Ctrl+S)", command=self._save_result).pack(side=tk.LEFT, padx=3)
        ttk.Button(row2, text="↩ 撤销粘贴 (Ctrl+Z)", command=self._undo_paste).pack(side=tk.LEFT, padx=3)
        ttk.Button(row2, text="❓ 操作帮助", command=self._show_help).pack(side=tk.LEFT, padx=3)

        # 图片数量
        self.info_var = tk.StringVar(value="好图: 0/0 | 缺陷图: 0/0")
        ttk.Label(row2, textvariable=self.info_var, foreground="green", font=("", 10)).pack(side=tk.RIGHT, padx=10)

        # ---- 第3行: 模式 + 导航按钮 ----
        row3 = ttk.Frame(self.root, padding=3)
        row3.pack(side=tk.TOP, fill=tk.X)

        ttk.Label(row3, textvariable=self.mode_text, font=("", 12, "bold"),
                  foreground="blue", width=55).pack(side=tk.LEFT, padx=5)

        self.btn_switch_mode = ttk.Button(row3, text="切换贴图模式", command=self._switch_mode, width=14)
        self.btn_switch_mode.pack(side=tk.LEFT, padx=3)
        self.btn_switch_mode.config(state=tk.DISABLED)

        ttk.Button(row3, text="上一张缺陷图 (Shift+N)", command=self._prev_defect, width=18).pack(side=tk.LEFT, padx=2)
        ttk.Button(row3, text="下一张缺陷图 (N)", command=self._next_defect, width=16).pack(side=tk.LEFT, padx=2)
        ttk.Button(row3, text="上一张好图 (Shift+B)", command=self._prev_background, width=18).pack(side=tk.LEFT, padx=2)
        ttk.Button(row3, text="下一张好图 (B)", command=self._next_background, width=16).pack(side=tk.LEFT, padx=2)

        # ---- 第4行: 调节参数（单独一行，不会被挤走） ----
        row4 = ttk.Frame(self.root, padding=3)
        row4.pack(side=tk.TOP, fill=tk.X)

        ttk.Label(row4, text="━━ 粘贴参数 ━━", foreground="gray").pack(side=tk.LEFT, padx=(10, 5))

        ttk.Label(row4, text="羽化:").pack(side=tk.LEFT, padx=(5, 2))
        self.feather_var = tk.IntVar(value=self.feather_size)
        ttk.Scale(row4, from_=1, to=100, variable=self.feather_var,
                  command=self._on_feather_change, length=70).pack(side=tk.LEFT, padx=2)
        ttk.Label(row4, textvariable=tk.StringVar(value=str(self.feather_size)), width=3).pack(side=tk.LEFT)

        ttk.Label(row4, text="  边缘过渡:").pack(side=tk.LEFT, padx=(5, 2))
        self.alpha_var = tk.DoubleVar(value=self.alpha_strength)
        ttk.Scale(row4, from_=0.2, to=1.0, variable=self.alpha_var,
                  command=self._on_alpha_change, length=70).pack(side=tk.LEFT, padx=2)

        ttk.Label(row4, text="  颜色匹配:").pack(side=tk.LEFT, padx=(5, 2))
        self.cmatch_var = tk.DoubleVar(value=self.color_match)
        ttk.Scale(row4, from_=0.0, to=1.0, variable=self.cmatch_var,
                  command=self._on_cmatch_change, length=70).pack(side=tk.LEFT, padx=2)

        self.blend_var = tk.StringVar(value="alpha")
        ttk.Label(row4, text="  模式:").pack(side=tk.LEFT, padx=(10, 2))
        ttk.Combobox(row4, textvariable=self.blend_var, values=["alpha(自然)", "direct(硬边)"],
                     state="readonly", width=14).pack(side=tk.LEFT, padx=2)
        self.blend_var.trace_add("write", lambda *a: setattr(self, 'blend_mode',
            'alpha' if 'alpha' in self.blend_var.get() else 'direct'))

        # ---- 第5行: 形变参数（拉伸/压缩/斜切/弹性形变） ----
        row5 = ttk.Frame(self.root, padding=3)
        row5.pack(side=tk.TOP, fill=tk.X)

        ttk.Label(row5, text="━━ 形变参数 ━━", foreground="gray").pack(side=tk.LEFT, padx=(10, 5))

        ttk.Label(row5, text="横向拉伸:").pack(side=tk.LEFT, padx=(5, 2))
        self.stretchx_var = tk.DoubleVar(value=self.stretch_x)
        ttk.Scale(row5, from_=0.3, to=3.0, variable=self.stretchx_var,
                  command=self._on_stretchx_change, length=80).pack(side=tk.LEFT, padx=2)

        ttk.Label(row5, text="  纵向拉伸:").pack(side=tk.LEFT, padx=(5, 2))
        self.stretchy_var = tk.DoubleVar(value=self.stretch_y)
        ttk.Scale(row5, from_=0.3, to=3.0, variable=self.stretchy_var,
                  command=self._on_stretchy_change, length=80).pack(side=tk.LEFT, padx=2)

        ttk.Label(row5, text="  横向斜切:").pack(side=tk.LEFT, padx=(5, 2))
        self.shearx_var = tk.DoubleVar(value=self.shear_x)
        ttk.Scale(row5, from_=-60, to=60, variable=self.shearx_var,
                  command=self._on_shearx_change, length=80).pack(side=tk.LEFT, padx=2)

        ttk.Label(row5, text="  纵向斜切:").pack(side=tk.LEFT, padx=(5, 2))
        self.sheary_var = tk.DoubleVar(value=self.shear_y)
        ttk.Scale(row5, from_=-60, to=60, variable=self.sheary_var,
                  command=self._on_sheary_change, length=80).pack(side=tk.LEFT, padx=2)

        ttk.Label(row5, text="  弹性形变:").pack(side=tk.LEFT, padx=(5, 2))
        self.elastic_var = tk.DoubleVar(value=self.elastic_strength)
        ttk.Scale(row5, from_=0.0, to=1.0, variable=self.elastic_var,
                  command=self._on_elastic_change, length=80).pack(side=tk.LEFT, padx=2)

        ttk.Button(row5, text="🎲 换形态", command=self._reroll_elastic, width=9).pack(side=tk.LEFT, padx=4)
        ttk.Button(row5, text="↺ 重置形变", command=self._reset_shape, width=10).pack(side=tk.LEFT, padx=2)

        # -- 主画布 --
        canvas_frame = ttk.Frame(self.root)
        canvas_frame.pack(side=tk.TOP, fill=tk.BOTH, expand=True, padx=5, pady=5)

        self.canvas = tk.Canvas(canvas_frame, bg="#2b2b2b", cursor="crosshair")
        self.canvas.pack(fill=tk.BOTH, expand=True)

        # 画布事件绑定 — macOS tkinter 右键是 Button-2
        self.canvas.bind("<Button-1>", self._on_left_click)
        self.canvas.bind("<Button-2>", self._on_right_click)   # macOS 右键
        self.canvas.bind("<Button-3>", self._on_right_click)   # Windows/Linux 右键
        self.canvas.bind("<B1-Motion>", self._on_drag)
        self.canvas.bind("<B2-Motion>", self._on_right_drag)   # macOS 右键拖拽=平移
        self.canvas.bind("<B3-Motion>", self._on_right_drag)   # Windows 右键拖拽=平移
        self.canvas.bind("<ButtonRelease-1>", self._on_release)
        self.canvas.bind("<ButtonRelease-2>", self._on_release)
        self.canvas.bind("<ButtonRelease-3>", self._on_release)
        self.canvas.bind("<MouseWheel>", self._on_mouse_wheel)
        self.canvas.bind("<Control-MouseWheel>", self._on_ctrl_wheel)

        # -- 状态栏 --
        self.status_var = tk.StringVar(value="就绪")
        status_bar = ttk.Label(self.root, textvariable=self.status_var, relief=tk.SUNKEN, anchor=tk.W, padding=3)
        status_bar.pack(side=tk.BOTTOM, fill=tk.X)

    # ================================================================
    # 目录选择
    # ================================================================

    def _browse_good_dir(self):
        d = filedialog.askdirectory(title="选择好图（正常零件）目录")
        if d:
            self.good_dir.set(d)

    def _browse_defect_dir(self):
        d = filedialog.askdirectory(title="选择缺陷图目录")
        if d:
            self.defect_dir.set(d)

    def _browse_output_dir(self):
        d = filedialog.askdirectory(title="选择输出目录")
        if d:
            self.output_dir.set(d)

    # ================================================================
    # 数据加载
    # ================================================================

    def _load_data(self):
        """加载好图和缺陷图"""
        good_d = self.good_dir.get().strip()
        defect_d = self.defect_dir.get().strip()

        if not good_d or not defect_d:
            messagebox.showwarning("路径缺失", "请先选择好图目录和缺陷图目录")
            return

        good_path = Path(good_d)
        defect_path = Path(defect_d)

        if not good_path.is_dir():
            messagebox.showerror("错误", f"好图目录不存在:\n{good_d}")
            return
        if not defect_path.is_dir():
            messagebox.showerror("错误", f"缺陷图目录不存在:\n{defect_d}")
            return

        # 扫描图片
        self.good_images = self._scan_images(good_path)
        self.defect_images = self._scan_images(defect_path)

        random.shuffle(self.good_images)
        random.shuffle(self.defect_images)

        self.info_var.set(f"好图: {self.current_good_idx + 1}/{len(self.good_images)} | 缺陷图: {self.current_defect_idx + 1}/{len(self.defect_images)}")

        if not self.defect_images:
            messagebox.showwarning("警告", "缺陷图目录中没有找到图片")
            return
        if not self.good_images:
            messagebox.showwarning("警告", "好图目录中没有找到图片")
            return

        # 加载第一张
        self.current_defect_idx = 0
        self.current_good_idx = 0
        self.mode = MODE_SELECT
        self.polygon_points = []
        self.cropped_defect = None
        self.pasted_defects = []

        self._load_defect_image(self.defect_images[self.current_defect_idx])
        self._load_bg_image(self.good_images[self.current_good_idx])
        self._update_display()
        self._update_ui_state()

        self.status_var.set(f"加载完成! 好图 {len(self.good_images)} 张, 缺陷图 {len(self.defect_images)} 张")

    def _scan_images(self, directory: Path):
        result = []
        for ext in ("*.jpg", "*.jpeg", "*.png", "*.bmp", "*.JPG", "*.PNG"):
            result.extend(sorted(directory.glob(ext)))
        return [str(p) for p in result]

    def _load_defect_image(self, path: str):
        img = imread_unicode(path)
        if img is None:
            print(f"[WARN] 无法读取缺陷图: {path}")
            return
        self.defect_img_original = img
        self.polygon_points = []
        self.defect_mask = None
        self.cropped_defect = None
        self.cropped_mask = None
        self.transformed_defect = None
        self.transformed_mask = None
        self.status_var.set(f"缺陷图: {Path(path).name} | 左键画多边形 | 右键闭合")

    def _load_bg_image(self, path: str):
        img = imread_unicode(path)
        if img is None:
            print(f"[WARN] 无法读取背景图: {path}")
            return
        self.bg_img_original = img
        self.bg_img_clean = img.copy()
        self.pasted_defects = []
        self.status_var.set(f"背景图: {Path(path).name}")

    # ================================================================
    # 显示更新
    # ================================================================

    def _update_display(self):
        """更新画布显示"""
        if self.mode == MODE_SELECT:
            display = self._render_select_mode()
        else:
            display = self._render_place_mode()

        if display is not None:
            self._show_image(display)
        else:
            self.canvas.delete("all")
            self.canvas.create_text(400, 300, text="加载中...", fill="white", font=("", 20))

    def _show_image(self, img_bgr):
        """按当前视图缩放+平移显示图片"""
        h, w = img_bgr.shape[:2]

        # 计算画布显示尺寸
        disp_w = int(w * self.view_scale)
        disp_h = int(h * self.view_scale)

        if disp_w < 1 or disp_h < 1:
            return

        # 缩放图片
        if self.view_scale != 1.0:
            disp_img = cv2.resize(img_bgr, (disp_w, disp_h), interpolation=cv2.INTER_AREA if self.view_scale < 1 else cv2.INTER_LINEAR)
        else:
            disp_img = img_bgr

        # 转 tk PhotoImage
        img_rgb = cv2.cvtColor(disp_img, cv2.COLOR_BGR2RGB)
        img_pil = Image.fromarray(img_rgb)
        self._photo = ImageTk.PhotoImage(img_pil)

        # 计算画布上的位置 (居中偏移)
        cx = 10 + int(self.view_offset_x * self.view_scale)
        cy = 10 + int(self.view_offset_y * self.view_scale)

        self.canvas.delete("all")
        self.canvas.create_image(cx, cy, anchor=tk.NW, image=self._photo)

        # 记录图片在画布上的位置用于调试
        self._img_canvas_x = cx
        self._img_canvas_y = cy

    def _render_select_mode(self):
        if self.defect_img_original is None:
            return None
        display = self.defect_img_original.copy()

        # 画多边形
        if len(self.polygon_points) >= 2:
            pts = np.array(self.polygon_points, dtype=np.int32)
            cv2.polylines(display, [pts], False, (0, 255, 0), 3)

        for pt in self.polygon_points:
            cv2.circle(display, pt, 8, (0, 0, 255), -1)
            cv2.circle(display, pt, 10, (255, 255, 255), 2)

        if self.defect_mask is not None:
            mask_overlay = cv2.cvtColor(self.defect_mask, cv2.COLOR_GRAY2BGR)
            mask_overlay[:, :, 0] = 0
            mask_overlay[:, :, 2] = 0
            display = cv2.addWeighted(display, 0.7, mask_overlay, 0.3, 0)

        # 提示文字
        cv2.putText(display,
                    f"左键:添加顶点 | 右键:闭合多边形 | Z:撤销顶点 | N:下一张 | 滚轮:缩放(当前{self.view_scale:.0%})",
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

        return display

    def _render_place_mode(self):
        if self.bg_img_original is None:
            return None
        display = self.bg_img_original.copy()

        # 已经粘贴的背景（如果有）
        if self.pasted_defects:
            display = self.bg_img_clean.copy()
            for pasted in self.pasted_defects:
                display = blend_defect(
                    display,
                    pasted['defect_img'],
                    pasted['defect_mask'],
                    pasted['x'], pasted['y'],
                    blend_mode=self.blend_mode,
                    alpha_strength=self.alpha_strength,
                    color_match=self.color_match,
                )

        # 预览当前缺陷（半透明）
        if self.transformed_defect is not None and self.mode == MODE_PLACE:
            dh, dw = self.transformed_defect.shape[:2]
            px, py = self.defect_pos
            bg_h, bg_w = display.shape[:2]
            if px < bg_w and py < bg_h and px + dw > 0 and py + dh > 0:
                preview = blend_defect(
                    display,
                    self.transformed_defect,
                    self.transformed_mask,
                    px, py,
                    blend_mode=self.blend_mode,
                    alpha_strength=0.5,
                    color_match=self.color_match,
                )
                x1, y1 = max(0, px), max(0, py)
                x2, y2 = min(bg_w, px + dw), min(bg_h, py + dh)
                display[y1:y2, x1:x2] = preview[y1:y2, x1:x2]
            cv2.rectangle(display, (px, py), (px + dw, py + dh), (0, 255, 0), 3)

        # 已粘贴的标记
        for p in self.pasted_defects:
            cv2.drawMarker(display, (p['x'], p['y']), (0, 255, 0), cv2.MARKER_CROSS, 20, 2)

        # 信息
        cm_label = f"颜色匹配:{self.color_match:.0%}" if self.color_match > 0 else "颜色匹配:关"
        info = [
            f"视图: {self.view_scale:.0%} | 缺陷缩放: {self.scale_val:.2f}x | 旋转: {self.rotation_val:.0f}° | 翻转: H={self.flip_h} V={self.flip_v}",
            f"拉伸: X={self.stretch_x:.2f} Y={self.stretch_y:.2f} | 斜切: X={self.shear_x:.0f}° Y={self.shear_y:.0f}° | 弹性: {self.elastic_strength:.2f}",
            f"模式: {self.blend_mode} | 羽化: {self.feather_size} | 边缘过渡: {self.alpha_strength:.2f} | {cm_label}",
            f"已贴: {len(self.pasted_defects)} 个 | 标签ID: {self.class_label.get()}",
            "左键:移动 | Shift+左键:旋转 | 滚轮:缩放视图 | Ctrl+滚轮:缩放缺陷 | 右键拖拽:平移 | 右键单击:放置",
        ]
        y0 = 30
        for line in info:
            cv2.putText(display, line, (10, y0), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 0), 2)
            y0 += 26

        return display

    # ================================================================
    # 鼠标事件
    # ================================================================

    def _image_to_canvas(self, ix, iy):
        """图片坐标转画布坐标"""
        cx = 10 + (ix + self.view_offset_x) * self.view_scale
        cy = 10 + (iy + self.view_offset_y) * self.view_scale
        return cx, cy

    def _canvas_to_image(self, canvas_x, canvas_y):
        """画布坐标转图片坐标（考虑视图缩放和平移）"""
        ix = int((canvas_x - 10) / self.view_scale - self.view_offset_x)
        iy = int((canvas_y - 10) / self.view_scale - self.view_offset_y)
        return ix, iy

    # ---- 左键 ----

    def _on_left_click(self, event):
        ix, iy = self._canvas_to_image(event.x, event.y)
        if self.mode == MODE_SELECT:
            self.polygon_points.append((ix, iy))
            n = len(self.polygon_points)
            self.status_var.set(f"顶点数: {n} | 右键闭合多边形 | Z键撤销顶点 | 滚轮缩放视图")
            self._update_display()
        elif self.mode == MODE_PLACE:
            self.is_dragging = True
            self.drag_start = (event.x, event.y)
            self.drag_start_pos = self.defect_pos

    # ---- 右键 (macOS=Button-2, Windows=Button-3) ----

    def _on_right_click(self, event):
        if self.mode == MODE_SELECT and len(self.polygon_points) >= 3:
            self._extract_defect()
        elif self.mode == MODE_PLACE and self.transformed_defect is not None:
            # 右键直接贴在当前预览框位置，避免光标位置导致偏移
            self._place_defect_at()

    # ---- 右键拖拽 = 平移视图 ----

    def _on_right_drag(self, event):
        if not self.is_panning:
            self.is_panning = True
            self.pan_start = (event.x, event.y)
            self._pan_start_offset = (self.view_offset_x, self.view_offset_y)
        dx = (event.x - self.pan_start[0]) / self.view_scale
        dy = (event.y - self.pan_start[1]) / self.view_scale
        self.view_offset_x = self._pan_start_offset[0] + dx
        self.view_offset_y = self._pan_start_offset[1] + dy
        self._update_display()

    # ---- 左键拖拽 ----

    def _on_drag(self, event):
        if not self.is_dragging:
            return

        if self.mode == MODE_PLACE:
            if event.state & 0x0001:  # Shift = 旋转缺陷
                dx = event.x - self.drag_start[0]
                self.rotation_val = (self.rotation_val + dx * 0.5) % 360
                self.drag_start = (event.x, event.y)
            else:
                dx = (event.x - self.drag_start[0]) / self.view_scale
                dy = (event.y - self.drag_start[1]) / self.view_scale
                self.defect_pos = (
                    int(self.drag_start_pos[0] + dx),
                    int(self.drag_start_pos[1] + dy),
                )
            self._update_transformed()
            self._update_display()

    # ---- 释放 ----

    def _on_release(self, event):
        self.is_dragging = False
        self.is_panning = False

    # ---- 滚轮缩放视图 ----

    def _on_mouse_wheel(self, event):
        """普通滚轮 = 缩放视图（看全图/放大细节）"""
        # macOS 滚轮 delta 可能很大,统一处理
        delta = event.delta
        if delta > 0:
            self.view_scale = min(3.0, self.view_scale + self.zoom_step)
        else:
            self.view_scale = max(0.05, self.view_scale - self.zoom_step)
        self.status_var.set(f"视图缩放: {self.view_scale:.0%} | 右键拖拽平移")
        self._update_display()

    # ---- Ctrl+滚轮 = 缩放缺陷 (贴图模式) ----

    def _on_ctrl_wheel(self, event):
        """Ctrl+滚轮 = 缩放缺陷本身"""
        if self.mode != MODE_PLACE or self.transformed_defect is None:
            return
        if event.delta > 0:
            self.scale_val *= 1.1
        else:
            self.scale_val /= 1.1
        self.scale_val = max(0.1, min(5.0, self.scale_val))
        self._update_transformed()
        self._update_display()
        self.status_var.set(f"缺陷缩放: {self.scale_val:.1f}x | 视图: {self.view_scale:.0%}")

    # ================================================================
    # 缺陷提取与变换
    # ================================================================

    def _extract_defect(self):
        if len(self.polygon_points) < 3 or self.defect_img_original is None:
            return

        pts = np.array(self.polygon_points, dtype=np.int32)
        self.defect_mask = create_soft_mask_from_polygon(
            self.defect_img_original.shape, pts, self.feather_size
        )

        result = extract_defect_region(self.defect_img_original, self.defect_mask)
        if result[0] is None:
            self.status_var.set("未能提取有效区域，请重新选择")
            return

        self.cropped_defect, self.cropped_mask, x1, y1, x2, y2 = result

        # 重置变换
        self.scale_val = 1.0
        self.rotation_val = 0.0
        self.flip_h = False
        self.flip_v = False
        self._reset_shape()

        # 初始位置: 背景图中心
        if self.bg_img_original is not None:
            bh, bw = self.bg_img_original.shape[:2]
            dh, dw = self.cropped_defect.shape[:2]
            self.defect_pos = (bw // 2 - dw // 2, bh // 2 - dh // 2)

        self._update_transformed()

        # 切换到贴图模式
        self.mode = MODE_PLACE
        self.mode_text.set("📌 贴缺陷模式")
        self.btn_switch_mode.config(text="🔍 切换到抠图模式", state=tk.NORMAL)

        self.status_var.set(f"缺陷提取成功! 尺寸: {self.cropped_defect.shape[1]}x{self.cropped_defect.shape[0]} | 拖拽放置")
        self._update_display()
        self._update_ui_state()

    def _update_transformed(self):
        if self.cropped_defect is None:
            return

        d_img = self.cropped_defect.copy()
        d_mask = self.cropped_mask.copy()

        if self.flip_h:
            d_img = cv2.flip(d_img, 1)
            d_mask = cv2.flip(d_mask, 1)
        if self.flip_v:
            d_img = cv2.flip(d_img, 0)
            d_mask = cv2.flip(d_mask, 0)

        # ---- 弹性形变（在原始尺度上做，保持随机场稳定）----
        d_img, d_mask = self._apply_elastic(d_img, d_mask)

        # ---- 几何形变：uniform缩放 + 非等比拉伸 + 斜切 + 旋转（合成一次仿射，画质更好）----
        d_img, d_mask = self._apply_geometric(d_img, d_mask)

        self.transformed_defect = d_img
        self.transformed_mask = np.clip(d_mask, 0, 255).astype(np.uint8)

    def _apply_elastic(self, img, mask):
        """弹性形变：用高斯平滑的随机位移场对图像做 remap，得到自然的弹性扭曲。"""
        if self.elastic_strength <= 0.001:
            return img, mask

        h, w = img.shape[:2]
        # 缓存位移场，只有种子或尺寸变化时才重建（保证拖动时形态稳定）
        cache = self._elastic_cache
        if cache is None or cache[0] != self.elastic_seed or cache[1] != (h, w):
            rng = np.random.RandomState(self.elastic_seed & 0x7FFFFFFF)
            dx = (rng.rand(h, w).astype(np.float32) * 2 - 1)
            dy = (rng.rand(h, w).astype(np.float32) * 2 - 1)
            sigma = max(h, w) * 0.10
            dx = cv2.GaussianBlur(dx, (0, 0), sigma)
            dy = cv2.GaussianBlur(dy, (0, 0), sigma)
            # 归一化到 [-1, 1]
            dx /= (np.abs(dx).max() + 1e-6)
            dy /= (np.abs(dy).max() + 1e-6)
            self._elastic_cache = (self.elastic_seed, (h, w), dx, dy)
        else:
            dx, dy = cache[2], cache[3]

        amp = self.elastic_strength * max(h, w) * 0.18
        grid_x, grid_y = np.meshgrid(np.arange(w), np.arange(h))
        map_x = (grid_x + dx * amp).astype(np.float32)
        map_y = (grid_y + dy * amp).astype(np.float32)

        img2 = cv2.remap(img, map_x, map_y, cv2.INTER_LINEAR,
                         borderMode=cv2.BORDER_CONSTANT, borderValue=(0, 0, 0))
        mask2 = cv2.remap(mask, map_x, map_y, cv2.INTER_LINEAR,
                          borderMode=cv2.BORDER_CONSTANT, borderValue=0)
        return img2, mask2

    def _apply_geometric(self, img, mask):
        """合成 缩放(等比+非等比) + 斜切 + 旋转 为单次仿射变换，并自动扩展画布避免裁剪。"""
        h, w = img.shape[:2]
        sx = self.scale_val * self.stretch_x
        sy = self.scale_val * self.stretch_y
        shx = np.tan(np.deg2rad(self.shear_x))
        shy = np.tan(np.deg2rad(self.shear_y))
        theta = np.deg2rad(self.rotation_val)
        cos_t, sin_t = np.cos(theta), np.sin(theta)

        # 线性部分: R @ Shear @ Scale
        S = np.array([[sx, 0.0], [0.0, sy]], dtype=np.float64)
        Sh = np.array([[1.0, shx], [shy, 1.0]], dtype=np.float64)
        R = np.array([[cos_t, -sin_t], [sin_t, cos_t]], dtype=np.float64)
        M2 = R @ Sh @ S

        # 如果是恒等变换，直接返回
        if (abs(sx - 1) < 1e-3 and abs(sy - 1) < 1e-3 and
                abs(shx) < 1e-6 and abs(shy) < 1e-6 and abs(self.rotation_val) < 1e-3):
            return img, mask

        # 计算变换后包围盒
        corners = np.array([[-w / 2, -h / 2], [w / 2, -h / 2],
                            [w / 2, h / 2], [-w / 2, h / 2]], dtype=np.float64).T
        new_corners = M2 @ corners
        minx, miny = new_corners.min(axis=1)
        maxx, maxy = new_corners.max(axis=1)
        new_w = max(1, int(np.ceil(maxx - minx)))
        new_h = max(1, int(np.ceil(maxy - miny)))

        # 平移：使原中心映射到新画布中心
        center_old = np.array([w / 2, h / 2], dtype=np.float64)
        center_new = np.array([new_w / 2, new_h / 2], dtype=np.float64)
        t = center_new - M2 @ center_old
        M = np.hstack([M2, t.reshape(2, 1)])

        img2 = cv2.warpAffine(img, M, (new_w, new_h), flags=cv2.INTER_LINEAR,
                              borderMode=cv2.BORDER_CONSTANT, borderValue=(0, 0, 0))
        mask2 = cv2.warpAffine(mask, M, (new_w, new_h), flags=cv2.INTER_LINEAR,
                               borderMode=cv2.BORDER_CONSTANT, borderValue=0)
        return img2, mask2

    def _place_defect_at(self):
        """将当前预览位置的缺陷粘贴到背景上。"""
        if self.transformed_defect is None:
            return

        px, py = self.defect_pos
        dh, dw = self.transformed_defect.shape[:2]

        self.pasted_defects.append({
            'x': px, 'y': py,
            'w': dw, 'h': dh,
            'defect_img': self.transformed_defect.copy(),
            'defect_mask': self.transformed_mask.copy(),
        })

        # 更新背景
        self.bg_img_original = blend_defect(
            self.bg_img_original if not self.pasted_defects[:-1] else self._get_full_bg(),
            self.transformed_defect,
            self.transformed_mask,
            px, py,
            blend_mode=self.blend_mode,
            alpha_strength=self.alpha_strength,
            color_match=self.color_match,
        )

        self.status_var.set(f"缺陷已粘贴! 共 {len(self.pasted_defects)} 个 | 继续放置或 Ctrl+S 保存")
        self._update_display()

    def _get_full_bg(self):
        """重建完整背景（含所有已粘贴缺陷）"""
        if self.bg_img_clean is None:
            return None
        bg = self.bg_img_clean.copy()
        for p in self.pasted_defects:
            bg = blend_defect(bg, p['defect_img'], p['defect_mask'],
                              p['x'], p['y'],
                              blend_mode=self.blend_mode,
                              alpha_strength=self.alpha_strength,
                              color_match=self.color_match)
        return bg

    # ================================================================
    # 键盘事件
    # ================================================================

    def _on_key(self, event):
        key = event.keysym.lower()
        ctrl = event.state & 0x0004
        shift = event.state & 0x0001

        if key == 'n' and not shift:
            self._next_defect()
        elif key == 'n' and shift:
            self._prev_defect()
        elif key == 'b' and not shift:
            self._next_background()
        elif key == 'b' and shift:
            self._prev_background()
        elif key == 'z' and not ctrl and self.mode == MODE_SELECT:
            # Z = 撤销多边形顶点
            if self.polygon_points:
                self.polygon_points.pop()
                self.status_var.set(f"撤销顶点 | 当前: {len(self.polygon_points)}")
                self._update_display()
        elif key == 'r' and self.mode == MODE_SELECT:
            self.polygon_points = []
            self.defect_mask = None
            self._update_display()
            self.status_var.set("已重置所有顶点")
        elif key == '0' and ctrl:
            # Ctrl+0 重置视图
            self.view_scale = 0.3
            self.view_offset_x = 0
            self.view_offset_y = 0
            self._update_display()
            self.status_var.set(f"视图已重置 | 缩放: {self.view_scale:.0%}")
        elif key == 'f' and self.mode == MODE_PLACE:
            self.flip_h = not self.flip_h
            self._update_transformed()
            self._update_display()
        elif key == 'v' and self.mode == MODE_PLACE:
            self.flip_v = not self.flip_v
            self._update_transformed()
            self._update_display()
        elif key == 'r' and self.mode == MODE_PLACE:
            self.rotation_val = 0
            self.scale_val = 1.0
            self.flip_h = False
            self.flip_v = False
            self._reset_shape()
        elif key == 'c' and self.mode == MODE_PLACE:
            # C = 切换颜色匹配 (0→0.5→1.0→0)
            self.color_match = {0: 0.5, 0.5: 1.0, 1.0: 0}.get(self.color_match, 0)
            self.cmatch_var.set(self.color_match)
            self._update_display()
            labels = {0: "关(原色)", 0.5: "50%匹配", 1.0: "完全匹配背景"}
            self.status_var.set(f"颜色匹配: {labels[self.color_match]}")
        elif key == 's' and ctrl:
            self._save_result()
        elif key == 'z' and ctrl:
            self._undo_paste()
        elif key == 'h':
            self._show_help()
        elif key == 'escape':
            self.root.quit()

    # ================================================================
    # 按钮操作
    # ================================================================

    def _switch_mode(self):
        if self.mode == MODE_SELECT:
            # 切到贴图模式（需要已有缺陷）
            if self.cropped_defect is not None:
                self.mode = MODE_PLACE
        else:
            # 切回抠图模式
            self.mode = MODE_SELECT
            self.polygon_points = []
            self.defect_mask = None
        self._update_ui_state()
        self._update_display()

    def _update_ui_state(self):
        if self.mode == MODE_SELECT:
            self.mode_text.set("🔍 抠缺陷 [左键加点|右键闭合|Z撤销|滚轮缩放|右键平移]")
            self.btn_switch_mode.config(text="切换到贴图模式")
            self.btn_switch_mode.config(state=tk.NORMAL if self.cropped_defect is not None else tk.DISABLED)
            self.canvas.config(cursor="crosshair")
        else:
            self.mode_text.set("📌 贴缺陷 [左键移动|Shift旋转|右键平移|Ctrl+滚轮缩放缺陷|C颜色匹配]")
            self.btn_switch_mode.config(text="切回抠图模式 (选新缺陷)", state=tk.NORMAL)
            self.canvas.config(cursor="fleur")

    def _update_info_var(self):
        """更新顶部信息栏显示当前图片序号/总数。"""
        good_cur = self.current_good_idx + 1 if self.good_images else 0
        good_total = len(self.good_images)
        defect_cur = self.current_defect_idx + 1 if self.defect_images else 0
        defect_total = len(self.defect_images)
        self.info_var.set(f"好图: {good_cur}/{good_total} | 缺陷图: {defect_cur}/{defect_total}")

    def _next_defect(self):
        if not self.defect_images:
            return
        self.current_defect_idx = (self.current_defect_idx + 1) % len(self.defect_images)
        self.mode = MODE_SELECT
        self._load_defect_image(self.defect_images[self.current_defect_idx])
        self._update_ui_state()
        self._update_display()
        self._update_info_var()
        self.status_var.set(f"缺陷图 [{self.current_defect_idx + 1}/{len(self.defect_images)}]: {Path(self.defect_images[self.current_defect_idx]).name}")

    def _next_background(self):
        if not self.good_images:
            return
        self.current_good_idx = (self.current_good_idx + 1) % len(self.good_images)
        self._load_bg_image(self.good_images[self.current_good_idx])

        # 如果有缺陷，重新计算位置
        if self.cropped_defect is not None and self.bg_img_original is not None:
            bh, bw = self.bg_img_original.shape[:2]
            dh, dw = self.cropped_defect.shape[:2]
            self.defect_pos = (bw // 2 - dw // 2, bh // 2 - dh // 2)
            self._update_transformed()

        self._update_display()
        self._update_info_var()
        self.status_var.set(f"背景图 [{self.current_good_idx + 1}/{len(self.good_images)}]: {Path(self.good_images[self.current_good_idx]).name}")

    def _prev_defect(self):
        if not self.defect_images:
            return
        self.current_defect_idx = (self.current_defect_idx - 1) % len(self.defect_images)
        self.mode = MODE_SELECT
        self._load_defect_image(self.defect_images[self.current_defect_idx])
        self._update_ui_state()
        self._update_display()
        self._update_info_var()
        self.status_var.set(f"缺陷图 [{self.current_defect_idx + 1}/{len(self.defect_images)}]: {Path(self.defect_images[self.current_defect_idx]).name}")

    def _prev_background(self):
        if not self.good_images:
            return
        self.current_good_idx = (self.current_good_idx - 1) % len(self.good_images)
        self._load_bg_image(self.good_images[self.current_good_idx])

        # 如果有缺陷，重新计算位置
        if self.cropped_defect is not None and self.bg_img_original is not None:
            bh, bw = self.bg_img_original.shape[:2]
            dh, dw = self.cropped_defect.shape[:2]
            self.defect_pos = (bw // 2 - dw // 2, bh // 2 - dh // 2)
            self._update_transformed()

        self._update_display()
        self._update_info_var()
        self.status_var.set(f"背景图 [{self.current_good_idx + 1}/{len(self.good_images)}]: {Path(self.good_images[self.current_good_idx]).name}")

    def _undo_paste(self):
        if self.pasted_defects:
            removed = self.pasted_defects.pop()
            # 重建背景
            self.bg_img_original = self._get_full_bg()
            self.status_var.set(f"已撤销 1 个粘贴 | 剩余 {len(self.pasted_defects)} 个")
            self._update_display()

    def _on_feather_change(self, val):
        self.feather_size = int(float(val))

    def _on_alpha_change(self, val):
        self.alpha_strength = float(val)

    def _on_cmatch_change(self, val):
        self.color_match = float(val)
        self._update_display()

    # ---- 形变参数回调 ----

    def _on_stretchx_change(self, val):
        self.stretch_x = float(val)
        self._update_transformed()
        self._update_display()

    def _on_stretchy_change(self, val):
        self.stretch_y = float(val)
        self._update_transformed()
        self._update_display()

    def _on_shearx_change(self, val):
        self.shear_x = float(val)
        self._update_transformed()
        self._update_display()

    def _on_sheary_change(self, val):
        self.shear_y = float(val)
        self._update_transformed()
        self._update_display()

    def _on_elastic_change(self, val):
        self.elastic_strength = float(val)
        self._update_transformed()
        self._update_display()

    def _reroll_elastic(self):
        """随机换一个弹性形态。"""
        self.elastic_seed = random.randint(0, 1 << 30)
        self._elastic_cache = None
        if self.elastic_strength <= 0.001:
            # 自动给一点强度，让效果可见
            self.elastic_strength = 0.4
            self.elastic_var.set(self.elastic_strength)
        self._update_transformed()
        self._update_display()
        self.status_var.set(f"弹性形态已更换 (seed={self.elastic_seed}) | 强度 {self.elastic_strength:.2f}")

    def _reset_shape(self):
        """重置所有形变参数（拉伸/斜切/弹性），保留uniform缩放与旋转。"""
        self.stretch_x = 1.0
        self.stretch_y = 1.0
        self.shear_x = 0.0
        self.shear_y = 0.0
        self.elastic_strength = 0.0
        self.stretchx_var.set(1.0)
        self.stretchy_var.set(1.0)
        self.shearx_var.set(0.0)
        self.sheary_var.set(0.0)
        self.elastic_var.set(0.0)
        self._update_transformed()
        self._update_display()
        self.status_var.set("形变参数已重置")

    def _show_help(self):
        help_text = """
🖱 鼠标操作:
  抠缺陷模式:
    左键点击      → 添加多边形顶点
    右键单击      → 闭合多边形,提取缺陷
    Z键           → 撤销上一个顶点
    滚轮          → 缩放视图 (看全图/放大细节)
    右键拖拽      → 平移视图

  贴缺陷模式:
    左键拖拽       → 移动缺陷位置
    Shift+左键拖拽 → 旋转缺陷
    滚轮          → 缩放视图
    Ctrl+滚轮     → 缩放缺陷本身
    右键拖拽      → 平移视图
    右键单击      → 在光标位置放置缺陷

⌨ 键盘快捷键:
  N           → 下一张缺陷图
  Shift+N     → 上一张缺陷图
  B           → 下一张背景图
  Shift+B     → 上一张背景图
  F (贴图模式) → 水平翻转缺陷
  V (贴图模式) → 垂直翻转缺陷
  R (贴图模式) → 重置缺陷变换
  R (抠图模式) → 重置多边形顶点
  Z (抠图模式) → 撤销上一个顶点
  Ctrl+0      → 重置视图缩放
  Ctrl+S      → 保存结果
  Ctrl+Z      → 撤销上次粘贴
  H           → 显示此帮助
  ESC         → 退出

🌀 形变参数 (第5行滑块, 让缺陷形态更多样):
  横向/纵向拉伸 → 非等比拉伸或压缩 (X、Y独立)
  横向/纵向斜切 → 斜着拉伸 (shear/错切变形)
  弹性形变      → 自然的弹性扭曲 (波浪/不规则变形)
  🎲 换形态     → 随机生成一个新的弹性形态
  ↺ 重置形变    → 清空所有拉伸/斜切/弹性

💡 提示:
  1. 先选好图和缺陷图目录,点"加载数据"
  2. 滚轮缩小视图可看到图片全貌
  3. 在缺陷图上用鼠标画出缺陷轮廓
  4. 右键闭合后自动切换到贴图模式
  5. 在好图上调整位置/大小/旋转
  6. 右键放置缺陷,可贴多个
  7. Ctrl+S 保存
        """
        messagebox.showinfo("帮助", help_text)

    # ================================================================
    # 保存
    # ================================================================

    def _save_result(self):
        if not self.pasted_defects:
            messagebox.showinfo("提示", "还没有粘贴任何缺陷，请先放置缺陷再保存")
            return

        out_dir = Path(self.output_dir.get().strip())
        if not out_dir.exists():
            out_dir.mkdir(parents=True, exist_ok=True)

        # ---- 创建子目录 ----
        img_dir = out_dir / "images"
        lbl_dir = out_dir / "labels"
        vis_dir = out_dir / "vis"
        for d in (img_dir, lbl_dir, vis_dir):
            d.mkdir(parents=True, exist_ok=True)

        # ---- 找下一个可用序号（扫描三个子目录，取最大值+1，避免覆盖）----
        def _extract_index(path: Path) -> int:
            stem = path.stem  # e.g. "gui_synthetic_0003" or "gui_synthetic_0003_vis"
            suffix = stem.split("_")[-1]
            if suffix.isdigit():
                return int(suffix)
            # 处理 "..._vis" 的情况，往前再取一段
            parts = stem.split("_")
            if len(parts) >= 2 and parts[-2].isdigit():
                return int(parts[-2])
            return 0

        existing = []
        existing.extend(img_dir.glob("*_gui_synthetic_*.jpg"))
        existing.extend(lbl_dir.glob("*_gui_synthetic_*.txt"))
        existing.extend(vis_dir.glob("*_gui_synthetic_*.jpg"))
        indices = [_extract_index(p) for p in existing]
        idx = max(indices, default=0) + 1

        # 生成文件名（前缀加上类别标签号）
        class_id = self.class_label.get().strip() or "0"
        base_name = f"{class_id}_gui_synthetic_{idx:04d}"
        img_name = f"{base_name}.jpg"
        label_name = f"{base_name}.txt"
        vis_name = f"{base_name}_vis.jpg"

        img_path = img_dir / img_name
        label_path = lbl_dir / label_name
        vis_path = vis_dir / vis_name

        # 生成最终图
        final = self._get_full_bg()
        if final is None:
            return

        imwrite_unicode(str(img_path), final, [cv2.IMWRITE_JPEG_QUALITY, 95])

        # 生成 YOLO 标签
        bg_h, bg_w = final.shape[:2]

        labels = []
        for p in self.pasted_defects:
            px, py = p['x'], p['y']
            dw, dh = p['w'], p['h']
            cx = (px + dw / 2) / bg_w
            cy = (py + dh / 2) / bg_h
            w = dw / bg_w
            h = dh / bg_h
            cx = max(0.0, min(1.0, cx))
            cy = max(0.0, min(1.0, cy))
            w = min(1.0, w)
            h = min(1.0, h)
            if w > 0.001 and h > 0.001:
                labels.append(f"{class_id} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}")

        with open(label_path, 'w') as f:
            f.write("\n".join(labels))

        # 保存可视化
        vis = final.copy()
        for p in self.pasted_defects:
            px, py = p['x'], p['y']
            dw, dh = p['w'], p['h']
            cv2.rectangle(vis, (px, py), (px + dw, py + dh), (0, 255, 0), 3)
        imwrite_unicode(str(vis_path), vis, [cv2.IMWRITE_JPEG_QUALITY, 95])

        self.status_var.set(f"已保存: {img_name} | 标签: {label_name} | {len(labels)} 个缺陷")
        messagebox.showinfo("保存成功",
                            f"图片: {img_name}\n标签: {label_name}\n标注框: {len(labels)} 个\n\n"
                            f"保存到:\n  images: {img_dir}\n  labels: {lbl_dir}\n  vis: {vis_dir}")

    def run(self):
        self.root.mainloop()


# ============================================================================
# 入口
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description="交互式缺陷粘贴工具 - GUI 版")
    parser.add_argument("--good-dir", type=str, default="", help="好图目录")
    parser.add_argument("--defect-dir", type=str, default="", help="缺陷图目录")
    parser.add_argument("--output-dir", type=str, default="./output/gui_output", help="输出目录")
    args = parser.parse_args()

    app = DefectPasteGUI()
    if args.good_dir:
        app.good_dir.set(args.good_dir)
    if args.defect_dir:
        app.defect_dir.set(args.defect_dir)
    if args.output_dir:
        app.output_dir.set(args.output_dir)
    app.run()


if __name__ == "__main__":
    main()

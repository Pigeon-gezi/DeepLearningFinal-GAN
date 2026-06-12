# -*- coding: utf-8 -*-
"""工具函数与常量：字体、文件IO、图像处理、随机种子、设备管理。"""

import os
import csv
import json
import random
import warnings

import numpy as np
import torch
import torch.nn as nn
from torchvision.utils import make_grid

import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib import font_manager

warnings.filterwarnings("ignore")

IMAGE_EXTENSIONS = (".jpg", ".jpeg", ".png", ".bmp", ".webp")


# ---------------------------------------------------------------------------
# 中文字体 / matplotlib 全局设置
# ---------------------------------------------------------------------------


def get_chinese_font():
    font_paths = [
        "C:\\Windows\\Fonts\\msyh.ttc",
        "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
        "/usr/share/fonts/truetype/wqy/wqy-microhei.ttc",
        "/usr/share/fonts/truetype/arphic/uming.ttc",
    ]
    for font_path in font_paths:
        if os.path.exists(font_path):
            return font_manager.FontProperties(fname=font_path)
    return font_manager.FontProperties()


myfont = get_chinese_font()
sns.set_style("whitegrid")


# ---------------------------------------------------------------------------
# 文件/路径工具
# ---------------------------------------------------------------------------


def make_safe_filename(text):
    safe = str(text)
    for char in [" ", "-", "(", ")", "/", "\\", ":", "：", "|"]:
        safe = safe.replace(char, "_")
    while "__" in safe:
        safe = safe.replace("__", "_")
    return safe.strip("_") or "output"


def ensure_dir(path):
    os.makedirs(path, exist_ok=True)
    return path


def save_json(data, path):
    ensure_dir(os.path.dirname(path))
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def append_csv_row(path, fieldnames, row):
    ensure_dir(os.path.dirname(path))
    file_exists = os.path.exists(path)
    with open(path, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if not file_exists:
            writer.writeheader()
        writer.writerow(row)


# ---------------------------------------------------------------------------
# 随机种子 / 设备
# ---------------------------------------------------------------------------


def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    try:
        torch.backends.cudnn.benchmark = True
    except Exception:
        pass


def get_available_device():
    try:
        if torch.cuda.is_available():
            torch.empty(1, device="cuda")
            torch.cuda.empty_cache()
            return torch.device("cuda")
    except RuntimeError as exc:
        print(f"CUDA初始化失败，使用CPU: {exc}")
    return torch.device("cpu")


# ---------------------------------------------------------------------------
# 图像处理工具
# ---------------------------------------------------------------------------


def denormalize_images(images):
    """将[-1, 1]图像张量转换到[0, 1]。"""
    return torch.clamp((images.detach().cpu() + 1.0) / 2.0, 0.0, 1.0)


def make_image_grid(images, nrow=8, padding=2, pad_value=1.0):
    images = images.detach().cpu()
    if images.dim() != 4:
        raise ValueError("images需要是[N, C, H, W]张量")
    return make_grid(
        images,
        nrow=max(1, min(nrow, images.size(0))),
        padding=padding,
        normalize=True,
        value_range=(-1, 1),
        pad_value=pad_value,
    )


def save_image_grid(images, path, nrow=8, title=None):
    ensure_dir(os.path.dirname(path))
    grid = make_image_grid(images, nrow=nrow).permute(1, 2, 0).numpy()
    plt.figure(
        figsize=(max(6, nrow), max(4, grid.shape[0] / max(grid.shape[1], 1) * nrow))
    )
    plt.imshow(grid)
    plt.axis("off")
    if title:
        plt.title(title, fontproperties=myfont)
    plt.tight_layout(pad=0)
    plt.savefig(path, dpi=220, bbox_inches="tight", facecolor="white", pad_inches=0.05)
    plt.close()
    print(f"图像已保存到: {path}")


def count_trainable_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

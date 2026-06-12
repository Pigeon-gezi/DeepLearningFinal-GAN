# -*- coding: utf-8 -*-
"""数据增强工具：实例噪声注入、轻量可微增强。"""

import random
import torch


def add_instance_noise(x, std):
    if std <= 0:
        return x
    return torch.clamp(x + torch.randn_like(x) * std, -1.0, 1.0)


def diff_augment(x):
    """轻量可微增强；默认关闭，可通过--use_diff_augment启用。"""
    if random.random() < 0.5:
        x = torch.flip(x, dims=[3])

    if random.random() < 0.8:
        brightness = torch.empty(x.size(0), 1, 1, 1, device=x.device).uniform_(
            -0.1, 0.1
        )
        contrast = torch.empty(x.size(0), 1, 1, 1, device=x.device).uniform_(0.85, 1.15)
        x_mean = x.mean(dim=[2, 3], keepdim=True)
        x = (x - x_mean) * contrast + x_mean + brightness

    if random.random() < 0.5:
        shift_x = random.randint(-2, 2)
        shift_y = random.randint(-2, 2)
        x = torch.roll(x, shifts=(shift_y, shift_x), dims=(2, 3))
    return torch.clamp(x, -1.0, 1.0)

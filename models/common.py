# -*- coding: utf-8 -*-
"""模型公共组件：权重初始化、谱归一化包装、MinibatchStdDev。"""

import torch
import torch.nn as nn


def init_weights_dcgan(module):
    classname = module.__class__.__name__
    if "Conv" in classname or "Linear" in classname:
        if hasattr(module, "weight") and module.weight is not None:
            nn.init.normal_(module.weight.data, 0.0, 0.02)
        if hasattr(module, "bias") and module.bias is not None:
            nn.init.constant_(module.bias.data, 0.0)
    elif "BatchNorm" in classname:
        if hasattr(module, "weight") and module.weight is not None:
            nn.init.normal_(module.weight.data, 1.0, 0.02)
        if hasattr(module, "bias") and module.bias is not None:
            nn.init.constant_(module.bias.data, 0.0)


def spectral(module, use_spectral_norm=False):
    if use_spectral_norm:
        return nn.utils.spectral_norm(module)
    return module


class MinibatchStdDev(nn.Module):
    """给判别器加入批内标准差通道，用来缓解模式崩溃。"""

    def __init__(self, eps=1e-8):
        super(MinibatchStdDev, self).__init__()
        self.eps = eps

    def forward(self, x):
        if x.size(0) <= 1:
            std_channel = torch.zeros(
                x.size(0), 1, x.size(2), x.size(3), device=x.device, dtype=x.dtype
            )
            return torch.cat([x, std_channel], dim=1)
        std = torch.sqrt(x.var(dim=0, unbiased=False) + self.eps)
        mean_std = std.mean().view(1, 1, 1, 1)
        std_channel = mean_std.expand(x.size(0), 1, x.size(2), x.size(3))
        return torch.cat([x, std_channel], dim=1)

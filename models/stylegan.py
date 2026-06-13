# -*- coding: utf-8 -*-
"""轻量 StyleGAN 组件：PixelNorm, EqualLinear, EqualConv2d, MappingNetwork,
AdaIN, NoiseInjection, StyledConvBlock, StyleGANLiteGenerator。"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from models.common import init_weights_dcgan


class PixelNorm(nn.Module):
    def __init__(self, eps=1e-8):
        super(PixelNorm, self).__init__()
        self.eps = eps

    def forward(self, x):
        return x * torch.rsqrt(torch.mean(x * x, dim=1, keepdim=True) + self.eps)


class EqualLinear(nn.Module):
    def __init__(self, in_dim, out_dim, lr_mul=1.0):
        super(EqualLinear, self).__init__()
        self.weight = nn.Parameter(torch.randn(out_dim, in_dim).div_(lr_mul))
        self.bias = nn.Parameter(torch.zeros(out_dim))
        self.scale = (1.0 / math.sqrt(in_dim)) * lr_mul
        self.lr_mul = lr_mul

    def forward(self, x):
        return F.linear(x, self.weight * self.scale, self.bias * self.lr_mul)


class EqualConv2d(nn.Module):
    """等学习率卷积 — StyleGAN 核心组件，确保不同分辨率层学习率一致。"""

    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0):
        super(EqualConv2d, self).__init__()
        fan_in = kernel_size * kernel_size * in_channels
        self.scale = 1.0 / math.sqrt(fan_in)
        self.weight = nn.Parameter(
            torch.randn(out_channels, in_channels, kernel_size, kernel_size)
        )
        self.bias = nn.Parameter(torch.zeros(out_channels))
        self.stride = stride
        self.padding = padding

    def forward(self, x):
        return F.conv2d(
            x,
            self.weight * self.scale,
            self.bias,
            stride=self.stride,
            padding=self.padding,
        )


class MappingNetwork(nn.Module):
    """StyleGAN思想中的z->w映射网络。"""

    def __init__(self, latent_dim=128, w_dim=128, num_layers=4):
        super(MappingNetwork, self).__init__()
        layers: list[nn.Module] = [PixelNorm()]
        in_dim = latent_dim
        for _ in range(num_layers):
            layers.extend(
                [
                    EqualLinear(in_dim, w_dim, lr_mul=0.01),
                    nn.LeakyReLU(0.2, inplace=True),
                ]
            )
            in_dim = w_dim
        self.net = nn.Sequential(*layers)

    def forward(self, z):
        return self.net(z)


class AdaIN(nn.Module):
    def __init__(self, channels, w_dim):
        super(AdaIN, self).__init__()
        self.norm = nn.InstanceNorm2d(channels, affine=False)
        self.style = EqualLinear(w_dim, channels * 2)

    def forward(self, x, w):
        style = self.style(w).view(w.size(0), 2, x.size(1), 1, 1)
        gamma = style[:, 0] + 1.0
        beta = style[:, 1]
        return self.norm(x) * gamma + beta


class NoiseInjection(nn.Module):
    def __init__(self, channels):
        super(NoiseInjection, self).__init__()
        self.weight = nn.Parameter(torch.zeros(1, channels, 1, 1))

    def forward(self, x, noise=None):
        if noise is None:
            noise = torch.randn(
                x.size(0), 1, x.size(2), x.size(3), device=x.device, dtype=x.dtype
            )
        return x + self.weight * noise


class StyledConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, w_dim, upsample=False):
        super(StyledConvBlock, self).__init__()
        self.upsample = upsample
        self.conv1 = EqualConv2d(in_channels, out_channels, 3, 1, 1)
        self.noise1 = NoiseInjection(out_channels)
        self.adain1 = AdaIN(out_channels, w_dim)
        self.conv2 = EqualConv2d(out_channels, out_channels, 3, 1, 1)
        self.noise2 = NoiseInjection(out_channels)
        self.adain2 = AdaIN(out_channels, w_dim)
        self.activation = nn.LeakyReLU(0.2, inplace=True)

    def forward(self, x, w):
        if self.upsample:
            x = F.interpolate(x, scale_factor=2, mode="bilinear", align_corners=False)
        x = self.conv1(x)
        x = self.noise1(x)
        x = self.activation(self.adain1(x, w))
        x = self.conv2(x)
        x = self.noise2(x)
        x = self.activation(self.adain2(x, w))
        return x


class StyleGANLiteGenerator(nn.Module):
    """
    Bonus模型：轻量StyleGAN风格生成器。

    它保留映射网络、常量输入、噪声注入和AdaIN调制，训练上搭配WGAN-GP/EMA，
    比完整StyleGAN小很多，适合作业环境里的CelebA/LFW实验。
    """

    def __init__(
        self,
        latent_dim=128,
        w_dim=128,
        image_size=64,
        image_channels=3,
        base_channels=64,
        max_channels=512,
        mapping_layers=4,
    ):
        super(StyleGANLiteGenerator, self).__init__()
        if image_size < 16 or image_size & (image_size - 1) != 0:
            raise ValueError("image_size需要是>=16的2的幂，例如64或128")

        self.latent_dim = latent_dim
        self.w_dim = w_dim
        self.image_size = image_size
        self.mapping = MappingNetwork(latent_dim, w_dim, mapping_layers)

        resolutions = [2**i for i in range(2, int(math.log2(image_size)) + 1)]
        channels = {}
        for resolution in resolutions:
            multiplier = max(1, image_size // resolution)
            channels[resolution] = min(
                max_channels, max(base_channels, base_channels * multiplier)
            )

        self.constant = nn.Parameter(torch.randn(1, channels[4], 4, 4))
        blocks = []
        in_channels = channels[4]
        blocks.append(StyledConvBlock(in_channels, in_channels, w_dim, upsample=False))
        for resolution in resolutions[1:]:
            out_channels = channels[resolution]
            blocks.append(
                StyledConvBlock(in_channels, out_channels, w_dim, upsample=True)
            )
            in_channels = out_channels
        self.blocks = nn.ModuleList(blocks)
        self.to_rgb = nn.Sequential(
            EqualConv2d(in_channels, image_channels, 1, 1, 0),
            nn.Tanh(),
        )

    def forward(self, z, return_w=False):
        if z.dim() > 2:
            z = z.view(z.size(0), -1)
        w = self.mapping(z)
        x = self.constant.repeat(z.size(0), 1, 1, 1)
        for block in self.blocks:
            x = block(x, w)
        image = self.to_rgb(x)
        if return_w:
            return image, w
        return image


class WGANDiscriminator(nn.Module):
    """WGAN-GP 判別器 — IN + MinibatchStd + 线性输出。"""

    def __init__(self, image_channels=3, base_features=64, image_size=64):
        super(WGANDiscriminator, self).__init__()
        if image_size < 16 or image_size & (image_size - 1) != 0:
            raise ValueError("image_size需要是>=16的2的幂，例如64或128")

        num_downsamples = int(math.log2(image_size)) - 2
        blocks = []
        current_channels = base_features

        blocks.append(
            nn.Sequential(
                nn.Conv2d(image_channels, current_channels, 4, 2, 1),
                nn.LeakyReLU(0.2, inplace=True),
            )
        )

        for i in range(num_downsamples - 1):
            next_channels = min(base_features * (2 ** (i + 1)), base_features * 16)
            blocks.append(
                nn.Sequential(
                    nn.Conv2d(current_channels, next_channels, 4, 2, 1),
                    nn.LeakyReLU(0.2, inplace=True),
                )
            )
            current_channels = next_channels

        self.blocks = nn.ModuleList(blocks)
        self.final = nn.Conv2d(current_channels, 1, 4, 1, 0)
        self.apply(init_weights_dcgan)

    def forward(self, x, return_features=False):
        features = []
        for block in self.blocks:
            x = block(x)
            features.append(x)
        logits = self.final(x).view(x.size(0), -1).mean(dim=1)
        if return_features:
            return logits, features
        return logits

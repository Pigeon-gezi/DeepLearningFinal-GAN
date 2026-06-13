# -*- coding: utf-8 -*-
"""DCGAN 生成器与判別器（专用于 BCE loss）。"""

import math
import torch
import torch.nn as nn

from models.common import init_weights_dcgan


class DCGANGenerator(nn.Module):
    """基础模型：DCGAN生成器。"""

    def __init__(
        self, latent_dim=128, image_channels=3, base_features=64, image_size=64
    ):
        super(DCGANGenerator, self).__init__()
        if image_size < 16 or image_size & (image_size - 1) != 0:
            raise ValueError("image_size需要是>=16的2的幂，例如64或128")

        num_upsamples = int(math.log2(image_size)) - 2
        max_multiplier = min(2 ** (num_upsamples - 1), 16)
        current_channels = base_features * max_multiplier

        layers = [
            nn.ConvTranspose2d(latent_dim, current_channels, 4, 1, 0, bias=False),
            nn.BatchNorm2d(current_channels),
            nn.ReLU(True),
        ]

        for _ in range(num_upsamples - 1):
            next_channels = max(base_features, current_channels // 2)
            layers.extend(
                [
                    nn.ConvTranspose2d(
                        current_channels, next_channels, 4, 2, 1, bias=False
                    ),
                    nn.BatchNorm2d(next_channels),
                    nn.ReLU(True),
                ]
            )
            current_channels = next_channels

        layers.extend(
            [
                nn.ConvTranspose2d(
                    current_channels, image_channels, 4, 2, 1, bias=False
                ),
                nn.Tanh(),
            ]
        )

        self.net = nn.Sequential(*layers)
        self.apply(init_weights_dcgan)

    def forward(self, z):
        if z.dim() == 2:
            z = z[:, :, None, None]
        return self.net(z)


class DCGANDiscriminator(nn.Module):
    """DCGAN 判別器 — BN + LeakyReLU，输出 BCE logit。"""

    def __init__(self, image_channels=3, base_features=64, image_size=64):
        super(DCGANDiscriminator, self).__init__()
        if image_size < 16 or image_size & (image_size - 1) != 0:
            raise ValueError("image_size需要是>=16的2的幂，例如64或128")

        num_downsamples = int(math.log2(image_size)) - 2
        blocks = []
        current_channels = base_features

        blocks.append(
            nn.Sequential(
                nn.Conv2d(image_channels, current_channels, 4, 2, 1, bias=False),
                nn.BatchNorm2d(current_channels),
                nn.LeakyReLU(0.2, inplace=True),
            )
        )

        for i in range(num_downsamples - 1):
            next_channels = min(base_features * (2 ** (i + 1)), base_features * 16)
            blocks.append(
                nn.Sequential(
                    nn.Conv2d(current_channels, next_channels, 4, 2, 1, bias=False),
                    nn.BatchNorm2d(next_channels),
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

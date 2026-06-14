# -*- coding: utf-8 -*-
"""轻量 StyleGAN 组件：PixelNorm, EqualLinear, EqualConv2d, MappingNetwork,
AdaIN, NoiseInjection, StyledConvBlock, StyleGANLiteGenerator。"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from models.common import init_weights_dcgan, MinibatchStdDev


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


class ModulatedConv2d(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size,
        w_dim,
        padding=0,
        demodulate=True,
        eps=1e-8,
    ):
        super(ModulatedConv2d, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.padding = padding
        self.demodulate = demodulate
        self.eps = eps
        self.scale = 1.0 / math.sqrt(in_channels * kernel_size * kernel_size)
        self.weight = nn.Parameter(
            torch.randn(1, out_channels, in_channels, kernel_size, kernel_size)
        )
        self.style = EqualLinear(w_dim, in_channels)
        self.bias = nn.Parameter(torch.zeros(out_channels))

    def forward(self, x, w):
        batch, _, height, width = x.shape
        style = self.style(w).view(batch, 1, self.in_channels, 1, 1) + 1.0
        weight = self.weight * self.scale * style
        if self.demodulate:
            demod = torch.rsqrt(weight.pow(2).sum(dim=(2, 3, 4)) + self.eps)
            weight = weight * demod.view(batch, self.out_channels, 1, 1, 1)

        x = x.view(1, batch * self.in_channels, height, width)
        weight = weight.view(
            batch * self.out_channels,
            self.in_channels,
            self.kernel_size,
            self.kernel_size,
        )
        out = F.conv2d(x, weight, padding=self.padding, groups=batch)
        out = out.view(batch, self.out_channels, out.size(2), out.size(3))
        return out + self.bias.view(1, -1, 1, 1)


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
    def __init__(self, channels, w_dim, strength=1.0):
        super(AdaIN, self).__init__()
        self.norm = nn.InstanceNorm2d(channels, affine=False)
        self.style = EqualLinear(w_dim, channels * 2)
        self.strength = strength

    def forward(self, x, w):
        style = self.style(w).view(w.size(0), 2, x.size(1), 1, 1)
        gamma = style[:, 0] + 1.0
        beta = style[:, 1]
        styled = self.norm(x) * gamma + beta
        return x + self.strength * (styled - x)


class NoiseInjection(nn.Module):
    def __init__(self, channels, enabled=True):
        super(NoiseInjection, self).__init__()
        self.enabled = enabled
        self.weight = nn.Parameter(torch.zeros(1, channels, 1, 1))

    def forward(self, x, noise=None):
        if not self.enabled:
            return x
        if noise is None:
            noise = torch.randn(
                x.size(0), 1, x.size(2), x.size(3), device=x.device, dtype=x.dtype
            )
        return x + self.weight * noise


class StyledConvBlock(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        w_dim,
        upsample=False,
        use_noise=True,
        adain_strength=1.0,
        conv_mode="adain",
        extra_conv=False,
    ):
        super(StyledConvBlock, self).__init__()
        if conv_mode not in ["adain", "modulated"]:
            raise ValueError(f"unsupported style conv mode: {conv_mode}")
        self.upsample = upsample
        self.conv_mode = conv_mode
        self.extra_conv = extra_conv
        self.skip = None
        if upsample or in_channels != out_channels:
            self.skip = EqualConv2d(in_channels, out_channels, 1, 1, 0)
        self.res_scale = 1.0 / math.sqrt(2.0)
        self.conv1 = self._make_conv(in_channels, out_channels, w_dim)
        self.noise1 = NoiseInjection(out_channels, enabled=use_noise)
        self.adain1 = self._make_adain(out_channels, w_dim, adain_strength)
        self.conv2 = self._make_conv(out_channels, out_channels, w_dim)
        self.noise2 = NoiseInjection(out_channels, enabled=use_noise)
        self.adain2 = self._make_adain(out_channels, w_dim, adain_strength)
        if extra_conv:
            self.conv3 = self._make_conv(out_channels, out_channels, w_dim)
            self.noise3 = NoiseInjection(out_channels, enabled=use_noise)
            self.adain3 = self._make_adain(out_channels, w_dim, adain_strength)
        self.activation = nn.LeakyReLU(0.2, inplace=True)

    def _make_conv(self, in_channels, out_channels, w_dim):
        if self.conv_mode == "modulated":
            return ModulatedConv2d(
                in_channels, out_channels, 3, w_dim, padding=1, demodulate=True
            )
        return EqualConv2d(in_channels, out_channels, 3, 1, 1)

    def _make_adain(self, channels, w_dim, adain_strength):
        if self.conv_mode == "modulated":
            return None
        return AdaIN(channels, w_dim, strength=adain_strength)

    def _apply_styled_conv(self, x, w, conv, noise, adain):
        if self.conv_mode == "modulated":
            x = conv(x, w)
        else:
            x = conv(x)
        x = noise(x)
        if adain is not None:
            x = adain(x, w)
        return self.activation(x)

    def forward(self, x, w):
        residual = x
        if self.upsample:
            residual = F.interpolate(
                residual, scale_factor=2, mode="bilinear", align_corners=False
            )
            x = residual
        if self.skip is not None:
            residual = self.skip(residual)
        x = self._apply_styled_conv(x, w, self.conv1, self.noise1, self.adain1)
        x = self._apply_styled_conv(x, w, self.conv2, self.noise2, self.adain2)
        if self.extra_conv:
            x = self._apply_styled_conv(x, w, self.conv3, self.noise3, self.adain3)
        return (x + residual) * self.res_scale


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
        use_noise=True,
        adain_strength=1.0,
        conv_mode="adain",
        extra_highres_conv=False,
        extra_highres_min_resolution=32,
    ):
        super(StyleGANLiteGenerator, self).__init__()
        if conv_mode not in ["adain", "modulated"]:
            raise ValueError(f"unsupported style conv mode: {conv_mode}")
        if image_size < 16 or image_size & (image_size - 1) != 0:
            raise ValueError("image_size需要是>=16的2的幂，例如64或128")

        self.latent_dim = latent_dim
        self.w_dim = w_dim
        self.image_size = image_size
        self.use_noise = use_noise
        self.adain_strength = adain_strength
        self.conv_mode = conv_mode
        self.extra_highres_conv = extra_highres_conv
        self.extra_highres_min_resolution = extra_highres_min_resolution
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
        blocks.append(
            StyledConvBlock(
                in_channels,
                in_channels,
                w_dim,
                upsample=False,
                use_noise=use_noise,
                adain_strength=adain_strength,
                conv_mode=conv_mode,
                extra_conv=extra_highres_conv
                and 4 >= extra_highres_min_resolution,
            )
        )
        for resolution in resolutions[1:]:
            out_channels = channels[resolution]
            blocks.append(
                StyledConvBlock(
                    in_channels,
                    out_channels,
                    w_dim,
                    upsample=True,
                    use_noise=use_noise,
                    adain_strength=adain_strength,
                    conv_mode=conv_mode,
                    extra_conv=extra_highres_conv
                    and resolution >= extra_highres_min_resolution,
                )
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
    """WGAN-GP 判別器 — IN + 线性输出。"""

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
                    nn.Conv2d(current_channels, next_channels, 4, 2, 1, bias=False),
                    nn.InstanceNorm2d(next_channels),
                    nn.LeakyReLU(0.2, inplace=True),
                )
            )
            current_channels = next_channels

        self.blocks = nn.ModuleList(blocks)
        self.minibatch_std = MinibatchStdDev()
        self.final = nn.Conv2d(current_channels + 1, 1, 4, 1, 0)
        self.apply(init_weights_dcgan)

    def forward(self, x, return_features=False):
        features = []
        for block in self.blocks:
            x = block(x)
            features.append(x)
        x = self.minibatch_std(x)
        logits = self.final(x).view(x.size(0), -1).mean(dim=1)
        if return_features:
            return logits, features
        return logits

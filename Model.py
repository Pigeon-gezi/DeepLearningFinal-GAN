# -*- coding: utf-8 -*-
import os

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")
if os.path.exists("/etc/fonts/fonts.conf"):
    os.environ.setdefault("FONTCONFIG_FILE", "/etc/fonts/fonts.conf")
os.makedirs(os.environ["MPLCONFIGDIR"], exist_ok=True)

import argparse
import copy
import csv
import json
import math
import random
import time
import warnings

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from PIL import Image
from torch.autograd import grad
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from torchvision.models import Inception_V3_Weights, inception_v3
from torchvision.transforms import InterpolationMode
from torchvision.utils import make_grid

import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib import font_manager

warnings.filterwarnings("ignore")


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


IMAGE_EXTENSIONS = (".jpg", ".jpeg", ".png", ".bmp", ".webp")


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


class Config:
    """集中存储GAN实验设置。"""

    def __init__(self):
        script_dir = os.path.dirname(os.path.abspath(__file__))

        # 路径设置
        self.output_dir = ensure_dir(script_dir)
        self.data_path = os.path.join(script_dir, "faces")
        self.checkpoint_dir = ensure_dir(os.path.join(script_dir, "checkpoints"))
        self.sample_dir = ensure_dir(os.path.join(script_dir, "samples"))
        self.metric_dir = ensure_dir(os.path.join(script_dir, "metrics"))

        # 数据设置
        self.image_size = 64
        self.image_channels = 3
        self.batch_size = 64
        self.num_workers = 0
        self.max_images = None
        self.center_crop = True
        self.random_flip = True

        # 模型设置
        self.latent_dim = 128
        self.dcgan_g_features = 64
        self.dcgan_d_features = 64
        self.style_w_dim = 128
        self.style_base_channels = 64
        self.style_max_channels = 512
        self.style_mapping_layers = 4

        # 基础DCGAN训练设置
        self.epochs = 20
        self.learning_rate_g = 2e-4
        self.learning_rate_d = 2e-4
        self.beta1 = 0.5
        self.beta2 = 0.999
        self.weight_decay = 0.0
        self.grad_clip = 10.0

        # Bonus/模式崩溃改进设置
        self.improved_learning_rate_g = 1e-4
        self.improved_learning_rate_d = 1e-4
        self.improved_beta1 = 0.0
        self.improved_beta2 = 0.99
        self.n_critic = 5
        self.lambda_gp = 10.0
        self.drift_weight = 1e-3
        self.feature_matching_weight = 2.0
        self.ema_decay = 0.999
        self.use_label_smoothing = True
        self.instance_noise_std = 0.03
        self.instance_noise_decay = 0.95
        self.use_diff_augment = False

        # 记录、采样、评估设置
        self.sample_interval = 1
        self.checkpoint_interval = 5
        self.metric_interval = 5
        self.fixed_noise_count = 64
        self.eval_num_images = 1024
        self.eval_batch_size = 32
        self.fid_feature_batch_size = 32
        self.metric_backend = (
            "inception"  # 新版torchvision环境下默认使用标准Inception FID/IS
        )
        self.interpolation_steps = 12

        # 复现实验
        self.seed = 42
        self.device = get_available_device()


class FaceTransform:
    """基于新版torchvision.transforms的图像预处理。"""

    def __init__(self, image_size=64, center_crop=True, random_flip=True):
        self.image_size = image_size
        self.center_crop = center_crop
        self.random_flip = random_flip
        transform_steps = []

        if center_crop:
            transform_steps.extend(
                [
                    transforms.Resize(
                        image_size, interpolation=InterpolationMode.BICUBIC
                    ),
                    transforms.CenterCrop(image_size),
                ]
            )
        else:
            transform_steps.append(
                transforms.Resize(
                    (image_size, image_size), interpolation=InterpolationMode.BICUBIC
                )
            )

        if random_flip:
            transform_steps.append(transforms.RandomHorizontalFlip(p=0.5))

        transform_steps.extend(
            [
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
            ]
        )
        self.transform = transforms.Compose(transform_steps)

    def __call__(self, image):
        return self.transform(image.convert("RGB"))


class FaceImageDataset(Dataset):
    """兼容CelebA、LFW或普通递归图片文件夹的数据集。"""

    def __init__(self, root_dir, transform=None, max_images=None, seed=42):
        self.root_dir = root_dir
        self.transform = transform
        self.image_paths = self._scan_images(root_dir)

        if max_images is not None and len(self.image_paths) > max_images:
            rng = random.Random(seed)
            self.image_paths = list(self.image_paths)
            rng.shuffle(self.image_paths)
            self.image_paths = sorted(self.image_paths[:max_images])

        if len(self.image_paths) == 0:
            raise FileNotFoundError(
                f"未在 {root_dir} 下找到图片。请将CelebA的img_align_celeba或LFW图片目录放到这里，"
                "或通过 --data_path 指定数据集路径。"
            )

        print(f"发现图片数量: {len(self.image_paths)}")
        print(f"示例图片: {self.image_paths[0]}")

    def _scan_images(self, root_dir):
        if not os.path.exists(root_dir):
            raise FileNotFoundError(
                f"数据路径不存在: {root_dir}\n"
                "当前代码已经完整，但需要数据集后才能训练。可以先运行 --mode smoke_test 检查模型。"
            )

        if os.path.isfile(root_dir):
            if root_dir.lower().endswith(IMAGE_EXTENSIONS):
                return [root_dir]
            raise ValueError(f"给定路径是文件但不是支持的图片格式: {root_dir}")

        paths = []
        for current_dir, _, filenames in os.walk(root_dir):
            for filename in filenames:
                if filename.lower().endswith(IMAGE_EXTENSIONS):
                    paths.append(os.path.join(current_dir, filename))
        return sorted(paths)

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        for offset in range(8):
            image_path = self.image_paths[(idx + offset) % len(self.image_paths)]
            try:
                with Image.open(image_path) as image:
                    if self.transform is not None:
                        return self.transform(image)
                    return image.convert("RGB")
            except Exception as exc:
                print(f"读取图片失败，已跳过: {image_path}, 原因: {exc}")
        raise RuntimeError(f"连续多张图片读取失败，请检查数据集: {self.root_dir}")


class FaceDataProcessor:
    def __init__(self, config):
        self.config = config

    def create_dataloader(self, shuffle=True, drop_last=True):
        transform = FaceTransform(
            image_size=self.config.image_size,
            center_crop=self.config.center_crop,
            random_flip=self.config.random_flip and shuffle,
        )
        dataset = FaceImageDataset(
            self.config.data_path,
            transform=transform,
            max_images=self.config.max_images,
            seed=self.config.seed,
        )
        loader = DataLoader(
            dataset,
            batch_size=self.config.batch_size,
            shuffle=shuffle,
            num_workers=self.config.num_workers,
            drop_last=drop_last,
        )
        print(
            f"DataLoader创建完成: batch_size={self.config.batch_size}, 批次数={len(loader)}"
        )
        return loader


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


class DCGANDiscriminator(nn.Module):
    """DCGAN判别器；可选谱归一化和minibatch stddev用于Bonus实验。"""

    def __init__(
        self,
        image_channels=3,
        base_features=64,
        image_size=64,
        use_spectral_norm=False,
        use_minibatch_std=False,
    ):
        super(DCGANDiscriminator, self).__init__()
        if image_size < 16 or image_size & (image_size - 1) != 0:
            raise ValueError("image_size需要是>=16的2的幂，例如64或128")

        num_downsamples = int(math.log2(image_size)) - 2
        blocks = []
        current_channels = base_features

        blocks.append(
            nn.Sequential(
                spectral(
                    nn.Conv2d(image_channels, current_channels, 4, 2, 1),
                    use_spectral_norm,
                ),
                nn.LeakyReLU(0.2, inplace=True),
            )
        )

        for i in range(num_downsamples - 1):
            next_channels = min(base_features * (2 ** (i + 1)), base_features * 16)
            layers: list[nn.Module] = [
                spectral(
                    nn.Conv2d(current_channels, next_channels, 4, 2, 1, bias=False),
                    use_spectral_norm,
                ),
            ]
            if not use_spectral_norm:
                layers.append(nn.BatchNorm2d(next_channels))
            layers.append(nn.LeakyReLU(0.2, inplace=True))
            blocks.append(nn.Sequential(*layers))
            current_channels = next_channels

        self.blocks = nn.ModuleList(blocks)
        self.use_minibatch_std = use_minibatch_std
        self.minibatch_std = MinibatchStdDev()
        final_in_channels = current_channels + (1 if use_minibatch_std else 0)
        self.final = spectral(
            nn.Conv2d(final_in_channels, 1, 4, 1, 0), use_spectral_norm
        )
        self.apply(init_weights_dcgan)

    def forward(self, x, return_features=False):
        features = []
        for block in self.blocks:
            x = block(x)
            features.append(x)
        if self.use_minibatch_std:
            x = self.minibatch_std(x)
        logits = self.final(x).view(x.size(0), -1).mean(dim=1)
        if return_features:
            return logits, features
        return logits


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
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, 1, 1)
        self.noise1 = NoiseInjection(out_channels)
        self.adain1 = AdaIN(out_channels, w_dim)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, 1, 1)
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
            nn.Conv2d(in_channels, image_channels, 1, 1, 0),
            nn.Tanh(),
        )

        self.apply(init_weights_dcgan)

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


class GANTrainer:
    def __init__(
        self, generator, discriminator, config, model_name="DCGAN", loss_type="bce"
    ):
        self.generator = generator.to(config.device)
        self.discriminator = discriminator.to(config.device)
        self.config = config
        self.device = config.device
        self.model_name = model_name
        self.loss_type = loss_type

        if loss_type == "wgan-gp":
            lr_g = config.improved_learning_rate_g
            lr_d = config.improved_learning_rate_d
            beta1 = config.improved_beta1
            beta2 = config.improved_beta2
        else:
            lr_g = config.learning_rate_g
            lr_d = config.learning_rate_d
            beta1 = config.beta1
            beta2 = config.beta2

        self.optimizer_g = optim.Adam(
            self.generator.parameters(),
            lr=lr_g,
            betas=(beta1, beta2),
            weight_decay=config.weight_decay,
        )
        self.optimizer_d = optim.Adam(
            self.discriminator.parameters(),
            lr=lr_d,
            betas=(beta1, beta2),
            weight_decay=config.weight_decay,
        )
        self.criterion = nn.BCEWithLogitsLoss()

        self.fixed_noise = torch.randn(
            config.fixed_noise_count, config.latent_dim, device=self.device
        )
        self.global_step = 0
        self.current_instance_noise = config.instance_noise_std
        self.history = {
            "epoch": [],
            "g_loss": [],
            "d_loss": [],
            "d_real": [],
            "d_fake": [],
            "gp": [],
            "feature_matching": [],
            "pixel_diversity": [],
            "epoch_time": [],
        }

        self.generator_ema = None
        if loss_type == "wgan-gp" and config.ema_decay > 0:
            self.generator_ema = copy.deepcopy(self.generator).to(self.device).eval()
            for param in self.generator_ema.parameters():
                param.requires_grad_(False)

        print(f"\n模型: {model_name}")
        print(f"生成器参数量: {count_trainable_parameters(self.generator):,}")
        print(f"判别器参数量: {count_trainable_parameters(self.discriminator):,}")

    def sample_noise(self, batch_size):
        return torch.randn(batch_size, self.config.latent_dim, device=self.device)

    def update_ema(self):
        if self.generator_ema is None:
            return
        decay = self.config.ema_decay
        with torch.no_grad():
            for ema_param, param in zip(
                self.generator_ema.parameters(), self.generator.parameters()
            ):
                ema_param.data.mul_(decay).add_(param.data, alpha=1.0 - decay)
            for ema_buffer, buffer in zip(
                self.generator_ema.buffers(), self.generator.buffers()
            ):
                ema_buffer.copy_(buffer)

    def _prepare_for_discriminator(self, images):
        if self.current_instance_noise > 0:
            images = add_instance_noise(images, self.current_instance_noise)
        if self.config.use_diff_augment:
            images = diff_augment(images)
        return images

    def train_discriminator_bce(self, real_images):
        batch_size = real_images.size(0)
        self.optimizer_d.zero_grad()

        real_images_for_d = self._prepare_for_discriminator(real_images)
        real_logits = self.discriminator(real_images_for_d)
        if self.config.use_label_smoothing:
            real_labels = torch.empty(batch_size, device=self.device).uniform_(
                0.85, 1.0
            )
            fake_labels = torch.empty(batch_size, device=self.device).uniform_(
                0.0, 0.15
            )
        else:
            real_labels = torch.ones(batch_size, device=self.device)
            fake_labels = torch.zeros(batch_size, device=self.device)
        real_loss = self.criterion(real_logits, real_labels)

        with torch.no_grad():
            fake_images = self.generator(self.sample_noise(batch_size))
        fake_logits = self.discriminator(self._prepare_for_discriminator(fake_images))
        fake_loss = self.criterion(fake_logits, fake_labels)

        d_loss = real_loss + fake_loss
        d_loss.backward()
        if self.config.grad_clip is not None:
            torch.nn.utils.clip_grad_norm_(
                self.discriminator.parameters(), self.config.grad_clip
            )
        self.optimizer_d.step()

        return (
            d_loss.item(),
            real_logits.detach().mean().item(),
            fake_logits.detach().mean().item(),
            0.0,
        )

    def train_generator_bce(self, batch_size):
        self.optimizer_g.zero_grad()
        fake_images = self.generator(self.sample_noise(batch_size))
        fake_logits = self.discriminator(self._prepare_for_discriminator(fake_images))
        target_labels = torch.ones(batch_size, device=self.device)
        g_loss = self.criterion(fake_logits, target_labels)
        g_loss.backward()
        if self.config.grad_clip is not None:
            torch.nn.utils.clip_grad_norm_(
                self.generator.parameters(), self.config.grad_clip
            )
        self.optimizer_g.step()
        self.update_ema()
        diversity = fake_images.detach().std(dim=0).mean().item()
        return g_loss.item(), 0.0, diversity

    def gradient_penalty(self, real_images, fake_images):
        batch_size = real_images.size(0)
        alpha = torch.rand(batch_size, 1, 1, 1, device=self.device)
        interpolates = (alpha * real_images + (1 - alpha) * fake_images).requires_grad_(
            True
        )
        scores = self.discriminator(interpolates)
        gradients = grad(
            outputs=scores,
            inputs=interpolates,
            grad_outputs=torch.ones_like(scores),
            create_graph=True,
            retain_graph=True,
            only_inputs=True,
        )[0]
        gradients = gradients.view(batch_size, -1)
        return ((gradients.norm(2, dim=1) - 1.0) ** 2).mean()

    def train_discriminator_wgan(self, real_images):
        batch_size = real_images.size(0)
        self.optimizer_d.zero_grad()
        fake_images = self.generator(self.sample_noise(batch_size)).detach()

        real_input = self._prepare_for_discriminator(real_images)
        fake_input = self._prepare_for_discriminator(fake_images)
        real_scores = self.discriminator(real_input)
        fake_scores = self.discriminator(fake_input)
        gp = self.gradient_penalty(real_images, fake_images)
        drift = self.config.drift_weight * torch.mean(real_scores**2)
        d_loss = (
            fake_scores.mean() - real_scores.mean() + self.config.lambda_gp * gp + drift
        )

        d_loss.backward()
        if self.config.grad_clip is not None:
            torch.nn.utils.clip_grad_norm_(
                self.discriminator.parameters(), self.config.grad_clip
            )
        self.optimizer_d.step()

        return (
            d_loss.item(),
            real_scores.detach().mean().item(),
            fake_scores.detach().mean().item(),
            gp.item(),
        )

    def train_generator_wgan(self, real_images):
        batch_size = real_images.size(0)
        self.optimizer_g.zero_grad()
        fake_images = self.generator(self.sample_noise(batch_size))
        fake_scores, fake_features = self.discriminator(
            self._prepare_for_discriminator(fake_images),
            return_features=True,
        )
        g_loss = -fake_scores.mean()
        feature_matching = torch.tensor(0.0, device=self.device)

        if self.config.feature_matching_weight > 0:
            with torch.no_grad():
                _, real_features = self.discriminator(
                    self._prepare_for_discriminator(real_images),
                    return_features=True,
                )
            for fake_feat, real_feat in zip(fake_features, real_features):
                feature_matching = feature_matching + F.l1_loss(
                    fake_feat.mean(dim=0),
                    real_feat.detach().mean(dim=0),
                )
            g_loss = g_loss + self.config.feature_matching_weight * feature_matching

        g_loss.backward()
        if self.config.grad_clip is not None:
            torch.nn.utils.clip_grad_norm_(
                self.generator.parameters(), self.config.grad_clip
            )
        self.optimizer_g.step()
        self.update_ema()
        diversity = fake_images.detach().std(dim=0).mean().item()
        return g_loss.item(), feature_matching.item(), diversity

    def train_epoch(self, train_loader, epoch):
        start_time = time.time()
        self.generator.train()
        self.discriminator.train()

        epoch_g_losses = []
        epoch_d_losses = []
        epoch_d_real = []
        epoch_d_fake = []
        epoch_gp = []
        epoch_feature_matching = []
        epoch_diversity = []

        for batch_idx, real_images in enumerate(train_loader):
            real_images = real_images.to(self.device)
            batch_size = real_images.size(0)
            self.global_step += 1

            if self.loss_type == "wgan-gp":
                d_loss, d_real, d_fake, gp_value = self.train_discriminator_wgan(
                    real_images
                )
                if self.global_step % self.config.n_critic == 0:
                    g_loss, feature_matching, diversity = self.train_generator_wgan(
                        real_images
                    )
                    epoch_g_losses.append(g_loss)
                    epoch_feature_matching.append(feature_matching)
                    epoch_diversity.append(diversity)
            else:
                d_loss, d_real, d_fake, gp_value = self.train_discriminator_bce(
                    real_images
                )
                g_loss, feature_matching, diversity = self.train_generator_bce(
                    batch_size
                )
                epoch_g_losses.append(g_loss)
                epoch_feature_matching.append(feature_matching)
                epoch_diversity.append(diversity)

            epoch_d_losses.append(d_loss)
            epoch_d_real.append(d_real)
            epoch_d_fake.append(d_fake)
            epoch_gp.append(gp_value)

            if (batch_idx + 1) % 100 == 0:
                latest_g = epoch_g_losses[-1] if epoch_g_losses else float("nan")
                print(
                    f"Epoch {epoch}/{self.config.epochs} | Batch {batch_idx + 1}/{len(train_loader)} | "
                    f"D: {d_loss:.4f} | G: {latest_g:.4f} | D(real): {d_real:.3f} | D(fake): {d_fake:.3f}"
                )

        self.current_instance_noise *= self.config.instance_noise_decay

        epoch_time = time.time() - start_time
        summary = {
            "epoch": epoch,
            "g_loss": (
                float(np.mean(epoch_g_losses)) if epoch_g_losses else float("nan")
            ),
            "d_loss": (
                float(np.mean(epoch_d_losses)) if epoch_d_losses else float("nan")
            ),
            "d_real": float(np.mean(epoch_d_real)) if epoch_d_real else float("nan"),
            "d_fake": float(np.mean(epoch_d_fake)) if epoch_d_fake else float("nan"),
            "gp": float(np.mean(epoch_gp)) if epoch_gp else 0.0,
            "feature_matching": (
                float(np.mean(epoch_feature_matching))
                if epoch_feature_matching
                else 0.0
            ),
            "pixel_diversity": (
                float(np.mean(epoch_diversity)) if epoch_diversity else 0.0
            ),
            "epoch_time": epoch_time,
        }

        for key, value in summary.items():
            self.history[key].append(value)
        return summary

    def train(self, train_loader):
        print(
            f"开始训练 {self.model_name}，epochs={self.config.epochs}, device={self.device}"
        )
        fieldnames = [
            "epoch",
            "g_loss",
            "d_loss",
            "d_real",
            "d_fake",
            "gp",
            "feature_matching",
            "pixel_diversity",
            "epoch_time",
        ]
        csv_path = os.path.join(
            self.config.metric_dir,
            f"Train_Log_{make_safe_filename(self.model_name)}.csv",
        )

        for epoch in range(1, self.config.epochs + 1):
            summary = self.train_epoch(train_loader, epoch)
            append_csv_row(csv_path, fieldnames, summary)
            print(
                f"Epoch {epoch}/{self.config.epochs}: "
                f"G Loss={summary['g_loss']:.4f}, D Loss={summary['d_loss']:.4f}, "
                f"D(real)={summary['d_real']:.3f}, D(fake)={summary['d_fake']:.3f}, "
                f"GP={summary['gp']:.4f}, Diversity={summary['pixel_diversity']:.4f}, "
                f"Time={summary['epoch_time']:.1f}s"
            )

            if epoch % self.config.sample_interval == 0:
                self.save_samples(epoch)
            if (
                epoch % self.config.checkpoint_interval == 0
                or epoch == self.config.epochs
            ):
                self.save_checkpoint(epoch)

        self.plot_training_curves()
        self.save_samples("final")
        self.save_checkpoint("final")
        return self.generator_ema if self.generator_ema is not None else self.generator

    def get_sampling_generator(self):
        if self.generator_ema is not None:
            return self.generator_ema
        return self.generator

    def save_samples(self, epoch):
        generator = self.get_sampling_generator()
        generator.eval()
        with torch.no_grad():
            fake_images = generator(self.fixed_noise).detach()
        filename = f"Samples_{make_safe_filename(self.model_name)}_epoch_{epoch}.png"
        path = os.path.join(self.config.sample_dir, filename)
        save_image_grid(
            fake_images, path, nrow=8, title=f"{self.model_name} Epoch {epoch}"
        )

    def save_checkpoint(self, epoch):
        checkpoint = {
            "epoch": epoch,
            "model_name": self.model_name,
            "loss_type": self.loss_type,
            "generator_state_dict": self.generator.state_dict(),
            "discriminator_state_dict": self.discriminator.state_dict(),
            "optimizer_g_state_dict": self.optimizer_g.state_dict(),
            "optimizer_d_state_dict": self.optimizer_d.state_dict(),
            "history": self.history,
            "config": {
                "image_size": self.config.image_size,
                "latent_dim": self.config.latent_dim,
                "image_channels": self.config.image_channels,
            },
        }
        if self.generator_ema is not None:
            checkpoint["generator_ema_state_dict"] = self.generator_ema.state_dict()
        filename = f"Checkpoint_{make_safe_filename(self.model_name)}_epoch_{epoch}.pth"
        path = os.path.join(self.config.checkpoint_dir, filename)
        torch.save(checkpoint, path)
        print(f"检查点已保存到: {path}")

    def plot_training_curves(self):
        epochs = self.history["epoch"]
        if not epochs:
            return
        plt.figure(figsize=(16, 10))

        plt.subplot(2, 2, 1)
        plt.plot(epochs, self.history["g_loss"], label="生成器损失", color="#4C78A8")
        plt.plot(epochs, self.history["d_loss"], label="判别器损失", color="#F58518")
        plt.xlabel("Epoch", fontproperties=myfont)
        plt.ylabel("Loss", fontproperties=myfont)
        plt.title("GAN训练损失", fontproperties=myfont)
        plt.legend(prop=myfont)
        plt.grid(True, alpha=0.3)

        plt.subplot(2, 2, 2)
        plt.plot(epochs, self.history["d_real"], label="D(real)", color="#54A24B")
        plt.plot(epochs, self.history["d_fake"], label="D(fake)", color="#E45756")
        plt.xlabel("Epoch", fontproperties=myfont)
        plt.ylabel("Score/Logit", fontproperties=myfont)
        plt.title("判别器输出", fontproperties=myfont)
        plt.legend(prop=myfont)
        plt.grid(True, alpha=0.3)

        plt.subplot(2, 2, 3)
        plt.plot(epochs, self.history["gp"], label="梯度惩罚", color="#72B7B2")
        plt.plot(
            epochs, self.history["feature_matching"], label="特征匹配", color="#B279A2"
        )
        plt.xlabel("Epoch", fontproperties=myfont)
        plt.ylabel("Penalty", fontproperties=myfont)
        plt.title("模式崩溃改进项", fontproperties=myfont)
        plt.legend(prop=myfont)
        plt.grid(True, alpha=0.3)

        plt.subplot(2, 2, 4)
        plt.plot(epochs, self.history["pixel_diversity"], color="#FF9DA6")
        plt.xlabel("Epoch", fontproperties=myfont)
        plt.ylabel("Mean pixel std", fontproperties=myfont)
        plt.title("生成样本多样性", fontproperties=myfont)
        plt.grid(True, alpha=0.3)

        plt.suptitle(f"{self.model_name} 训练过程", fontsize=16, fontproperties=myfont)
        plt.tight_layout()
        path = os.path.join(
            self.config.metric_dir,
            f"Training_{make_safe_filename(self.model_name)}.png",
        )
        plt.savefig(path, dpi=300, bbox_inches="tight", facecolor="white")
        plt.close()
        print(f"训练曲线已保存到: {path}")


class InceptionMetricBackend:
    """基于新版torchvision预训练Inception-v3计算FID/IS。"""

    def __init__(self, device):
        self.device = device
        self._load_models()

    def _load_models(self):
        weights = Inception_V3_Weights.DEFAULT
        self.feature_model = inception_v3(weights=weights, transform_input=False)
        self.logit_model = inception_v3(weights=weights, transform_input=False)

        self.feature_model.fc = nn.Identity()
        self.feature_model.eval().to(self.device)
        self.logit_model.eval().to(self.device)
        for model in [self.feature_model, self.logit_model]:
            for param in model.parameters():
                param.requires_grad_(False)

    def preprocess(self, images):
        images = torch.clamp((images + 1.0) / 2.0, 0.0, 1.0)
        images = F.interpolate(
            images, size=(299, 299), mode="bilinear", align_corners=False
        )
        mean = torch.tensor([0.485, 0.456, 0.406], device=images.device).view(
            1, 3, 1, 1
        )
        std = torch.tensor([0.229, 0.224, 0.225], device=images.device).view(1, 3, 1, 1)
        return (images - mean) / std

    def features(self, images):
        with torch.no_grad():
            images = self.preprocess(images)
            feats = self.feature_model(images)
            if feats.dim() > 2:
                feats = feats.view(feats.size(0), -1)
            return feats

    def probabilities(self, images):
        with torch.no_grad():
            images = self.preprocess(images)
            logits = self.logit_model(images)
            if isinstance(logits, tuple):
                logits = logits[0]
            return F.softmax(logits, dim=1)


def calculate_fid_from_features(real_features, fake_features):
    real_features = np.asarray(real_features, dtype=np.float64)
    fake_features = np.asarray(fake_features, dtype=np.float64)

    if real_features.ndim != 2 or fake_features.ndim != 2:
        raise ValueError("FID特征必须是二维数组")
    if real_features.shape[0] < 2 or fake_features.shape[0] < 2:
        return float("nan")

    mu_real = np.mean(real_features, axis=0)
    mu_fake = np.mean(fake_features, axis=0)
    sigma_real = np.cov(real_features, rowvar=False)
    sigma_fake = np.cov(fake_features, rowvar=False)

    diff = mu_real - mu_fake
    try:
        from scipy import linalg

        covmean = linalg.sqrtm(sigma_real.dot(sigma_fake))
        if np.iscomplexobj(covmean):
            covmean = covmean.real
    except Exception:
        product = sigma_real.dot(sigma_fake)
        eigenvalues, eigenvectors = np.linalg.eig(product)
        eigenvalues = np.maximum(eigenvalues.real, 0.0)
        covmean = eigenvectors.real.dot(np.diag(np.sqrt(eigenvalues))).dot(
            np.linalg.inv(eigenvectors.real)
        )

    fid = diff.dot(diff) + np.trace(sigma_real + sigma_fake - 2.0 * covmean)
    return float(np.real(fid))


def calculate_inception_score(probabilities, splits=10, eps=1e-16):
    probabilities = np.asarray(probabilities, dtype=np.float64)
    if probabilities.ndim != 2 or probabilities.shape[0] < splits:
        return float("nan"), float("nan")
    split_scores = []
    for part in np.array_split(probabilities, splits):
        py = np.mean(part, axis=0, keepdims=True)
        kl = part * (np.log(part + eps) - np.log(py + eps))
        split_scores.append(np.exp(np.mean(np.sum(kl, axis=1))))
    return float(np.mean(split_scores)), float(np.std(split_scores))


class Evaluator:
    def __init__(self, config):
        self.config = config
        self.device = config.device

    def _build_metric_backend(self):
        if self.config.metric_backend != "inception":
            raise ValueError(
                "新版实现默认使用标准Inception评估，请将metric_backend设为'inception'"
            )
        backend = InceptionMetricBackend(self.device)
        print("使用torchvision Inception-v3计算FID/IS")
        return "inception", backend

    def collect_real_features(self, data_loader, feature_extractor, max_images):
        features = []
        collected = 0
        for real_images in data_loader:
            real_images = real_images.to(self.device)
            with torch.no_grad():
                feats = feature_extractor.features(real_images)
            features.append(feats.detach().cpu().numpy())
            collected += real_images.size(0)
            if collected >= max_images:
                break
        return np.concatenate(features, axis=0)[:max_images]

    def collect_fake_outputs(self, generator, feature_extractor, num_images):
        generator.eval()
        features = []
        probabilities = []
        pixel_diversities = []
        generated = 0

        while generated < num_images:
            batch_size = min(self.config.eval_batch_size, num_images - generated)
            noise = torch.randn(batch_size, self.config.latent_dim, device=self.device)
            with torch.no_grad():
                fake_images = generator(noise)
                feats = feature_extractor.features(fake_images)
                probs = feature_extractor.probabilities(fake_images)
                probabilities.append(probs.detach().cpu().numpy())
                features.append(feats.detach().cpu().numpy())
                pixel_diversities.append(fake_images.detach().std(dim=0).mean().item())
            generated += batch_size

        outputs = {
            "features": np.concatenate(features, axis=0)[:num_images],
            "pixel_diversity": float(np.mean(pixel_diversities)),
        }
        if probabilities:
            outputs["probabilities"] = np.concatenate(probabilities, axis=0)[
                :num_images
            ]
        return outputs

    def evaluate(self, generator, data_loader, model_name="Model"):
        backend_name, feature_extractor = self._build_metric_backend()
        max_images = min(self.config.eval_num_images, len(data_loader.dataset))
        print(
            f"开始评估 {model_name}: 使用 {max_images} 张真实图像和 {max_images} 张生成图像"
        )

        real_features = self.collect_real_features(
            data_loader, feature_extractor, max_images
        )
        fake_outputs = self.collect_fake_outputs(
            generator, feature_extractor, max_images
        )
        fake_features = fake_outputs["features"]

        fid_value = calculate_fid_from_features(real_features, fake_features)
        metrics = {
            "model": model_name,
            "metric_backend": backend_name,
            "num_eval_images": int(max_images),
            "FID": fid_value,
            "Pixel_Diversity": fake_outputs["pixel_diversity"],
        }

        if "probabilities" in fake_outputs:
            is_mean, is_std = calculate_inception_score(fake_outputs["probabilities"])
            metrics["Inception_Score_Mean"] = is_mean
            metrics["Inception_Score_Std"] = is_std

        metric_path = os.path.join(
            self.config.metric_dir, f"Metrics_{make_safe_filename(model_name)}.json"
        )
        save_json(metrics, metric_path)
        print(f"{model_name} 评估指标:")
        for key, value in metrics.items():
            if isinstance(value, float):
                print(f"  {key}: {value:.4f}")
            else:
                print(f"  {key}: {value}")
        return metrics

    def save_interpolation(self, generator, model_name="Model", steps=None):
        steps = steps or self.config.interpolation_steps
        generator.eval()
        z1 = torch.randn(1, self.config.latent_dim, device=self.device)
        z2 = torch.randn(1, self.config.latent_dim, device=self.device)
        alphas = torch.linspace(0, 1, steps, device=self.device).view(steps, 1)
        latents = (1 - alphas) * z1 + alphas * z2
        with torch.no_grad():
            images = generator(latents)
        path = os.path.join(
            self.config.sample_dir,
            f"Interpolation_{make_safe_filename(model_name)}.png",
        )
        save_image_grid(images, path, nrow=steps, title=f"{model_name} 线性插值")
        return path


class FaceGANExperiment:
    def __init__(self, config):
        self.config = config
        self.data_processor = FaceDataProcessor(config)
        self.results = {}

    def build_models(self, model_type):
        model_type = model_type.lower()
        if model_type in ["dcgan", "baseline"]:
            generator = DCGANGenerator(
                latent_dim=self.config.latent_dim,
                image_channels=self.config.image_channels,
                base_features=self.config.dcgan_g_features,
                image_size=self.config.image_size,
            )
            discriminator = DCGANDiscriminator(
                image_channels=self.config.image_channels,
                base_features=self.config.dcgan_d_features,
                image_size=self.config.image_size,
                use_spectral_norm=False,
                use_minibatch_std=False,
            )
            return generator, discriminator, "DCGAN", "bce"

        if model_type in ["stylegan_lite", "stylegan-lite", "improved", "bonus"]:
            generator = StyleGANLiteGenerator(
                latent_dim=self.config.latent_dim,
                w_dim=self.config.style_w_dim,
                image_size=self.config.image_size,
                image_channels=self.config.image_channels,
                base_channels=self.config.style_base_channels,
                max_channels=self.config.style_max_channels,
                mapping_layers=self.config.style_mapping_layers,
            )
            discriminator = DCGANDiscriminator(
                image_channels=self.config.image_channels,
                base_features=self.config.dcgan_d_features,
                image_size=self.config.image_size,
                use_spectral_norm=True,
                use_minibatch_std=True,
            )
            return generator, discriminator, "StyleGAN-lite + WGAN-GP", "wgan-gp"

        raise ValueError(f"未知模型类型: {model_type}")

    def run_single_model(self, model_type, evaluate=True):
        train_loader = self.data_processor.create_dataloader(
            shuffle=True, drop_last=True
        )
        generator, discriminator, model_name, loss_type = self.build_models(model_type)
        trainer = GANTrainer(
            generator,
            discriminator,
            self.config,
            model_name=model_name,
            loss_type=loss_type,
        )
        trained_generator = trainer.train(train_loader)

        evaluator = Evaluator(self.config)
        interpolation_path = evaluator.save_interpolation(
            trained_generator, model_name=model_name
        )
        metrics = {}
        if evaluate:
            eval_loader = self.data_processor.create_dataloader(
                shuffle=False, drop_last=False
            )
            metrics = evaluator.evaluate(
                trained_generator, eval_loader, model_name=model_name
            )
        metrics["interpolation_path"] = interpolation_path

        self.results[model_name] = {
            "metrics": metrics,
            "history": trainer.history,
        }
        return trained_generator, metrics

    def compare_models(self):
        print("\n开始对比基础DCGAN与改进GAN")
        _, dcgan_metrics = self.run_single_model("dcgan", evaluate=True)
        _, improved_metrics = self.run_single_model("improved", evaluate=True)

        self.results["DCGAN"]["metrics"] = dcgan_metrics
        self.results["StyleGAN-lite + WGAN-GP"]["metrics"] = improved_metrics

        comparison = {
            model_name: result["metrics"] for model_name, result in self.results.items()
        }
        comparison_path = os.path.join(
            self.config.metric_dir, "Model_Comparison_GAN.json"
        )
        save_json(comparison, comparison_path)
        self.plot_model_comparison(comparison)
        return comparison

    def plot_model_comparison(self, comparison):
        metric_candidates = ["FID", "Inception_Score_Mean", "Pixel_Diversity"]
        available_metrics = []
        for metric in metric_candidates:
            if any(metric in metrics for metrics in comparison.values()):
                available_metrics.append(metric)
        if not available_metrics:
            return

        plt.figure(figsize=(5 * len(available_metrics), 4))
        model_names = list(comparison.keys())
        for idx, metric in enumerate(available_metrics):
            plt.subplot(1, len(available_metrics), idx + 1)
            values = [comparison[name].get(metric, np.nan) for name in model_names]
            if metric == "FID":
                best_idx = int(np.nanargmin(values))
            else:
                best_idx = int(np.nanargmax(values))
            colors = ["#4C78A8"] * len(model_names)
            colors[best_idx] = "#F58518"
            bars = plt.bar(model_names, values, color=colors)
            for bar in bars:
                height = bar.get_height()
                plt.text(
                    bar.get_x() + bar.get_width() / 2.0,
                    height,
                    f"{height:.3f}",
                    ha="center",
                    va="bottom",
                    fontproperties=myfont,
                )
            plt.title(f"{metric} 比较", fontproperties=myfont)
            plt.xticks(rotation=20)
            plt.grid(True, alpha=0.3)
        plt.tight_layout()
        path = os.path.join(self.config.metric_dir, "Model_Comparison_GAN.png")
        plt.savefig(path, dpi=300, bbox_inches="tight", facecolor="white")
        plt.close()
        print(f"模型比较图已保存到: {path}")

    def smoke_test(self):
        """无数据集时也能运行的网络连通性检查。"""
        print("运行smoke_test：构建两个模型并保存随机采样图。")
        for model_type in ["dcgan", "improved"]:
            generator, discriminator, model_name, _ = self.build_models(model_type)
            generator = generator.to(self.config.device).eval()
            discriminator = discriminator.to(self.config.device).eval()
            noise = torch.randn(8, self.config.latent_dim, device=self.config.device)
            with torch.no_grad():
                fake_images = generator(noise)
                scores = discriminator(fake_images)
            print(
                f"{model_name}: fake_images={tuple(fake_images.shape)}, "
                f"D_scores={tuple(scores.shape)}, G参数={count_trainable_parameters(generator):,}, "
                f"D参数={count_trainable_parameters(discriminator):,}"
            )
            path = os.path.join(
                self.config.sample_dir,
                f"Smoke_Test_{make_safe_filename(model_name)}.png",
            )
            save_image_grid(fake_images, path, nrow=4, title=f"{model_name} Smoke Test")
            Evaluator(self.config).save_interpolation(
                generator, model_name=f"{model_name}_smoke", steps=8
            )
        print("smoke_test完成。")

    def load_generator_from_checkpoint(self, checkpoint_path, model_type):
        generator, _, model_name, _ = self.build_models(model_type)
        checkpoint = torch.load(checkpoint_path, map_location=self.config.device)
        state_key = (
            "generator_ema_state_dict"
            if "generator_ema_state_dict" in checkpoint
            else "generator_state_dict"
        )
        generator.load_state_dict(checkpoint[state_key])
        generator = generator.to(self.config.device).eval()
        print(f"已加载生成器: {checkpoint_path} ({model_name}, {state_key})")
        return generator, model_name


def parse_args():
    parser = argparse.ArgumentParser(description="AI2602任务D：基于GAN的人头图像生成")
    parser.add_argument(
        "--mode",
        type=str,
        default="smoke_test",
        choices=[
            "smoke_test",
            "train_dcgan",
            "train_improved",
            "compare",
            "evaluate",
            "interpolate",
        ],
        help="运行模式。无数据时推荐先用smoke_test。",
    )
    parser.add_argument(
        "--data_path", type=str, default=None, help="CelebA/LFW/普通图片文件夹路径"
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default=None,
        help="evaluate/interpolate模式下加载的检查点",
    )
    parser.add_argument(
        "--model_type",
        type=str,
        default="dcgan",
        choices=["dcgan", "improved"],
        help="检查点对应模型类型",
    )
    parser.add_argument("--epochs", type=int, default=None, help="训练轮数")
    parser.add_argument("--batch_size", type=int, default=None, help="批大小")
    parser.add_argument(
        "--image_size", type=int, default=None, help="图像尺寸，建议64或128"
    )
    parser.add_argument("--latent_dim", type=int, default=None, help="潜变量维度")
    parser.add_argument(
        "--max_images", type=int, default=None, help="只使用部分图片，便于快速实验"
    )
    parser.add_argument(
        "--num_workers", type=int, default=None, help="DataLoader workers"
    )
    parser.add_argument(
        "--eval_num_images", type=int, default=None, help="FID/IS评估图片数量"
    )
    parser.add_argument(
        "--metric_backend",
        type=str,
        default=None,
        choices=["inception"],
        help="评估后端，默认使用torchvision Inception-v3",
    )
    parser.add_argument(
        "--use_diff_augment", action="store_true", help="启用轻量DiffAugment"
    )
    parser.add_argument("--seed", type=int, default=None, help="随机种子")
    return parser.parse_args()


def apply_args_to_config(config, args):
    if args.data_path is not None:
        config.data_path = args.data_path
    if args.epochs is not None:
        config.epochs = args.epochs
    if args.batch_size is not None:
        config.batch_size = args.batch_size
    if args.image_size is not None:
        config.image_size = args.image_size
    if args.latent_dim is not None:
        config.latent_dim = args.latent_dim
    if args.max_images is not None:
        config.max_images = args.max_images
    if args.num_workers is not None:
        config.num_workers = args.num_workers
    if args.eval_num_images is not None:
        config.eval_num_images = args.eval_num_images
    if args.metric_backend is not None:
        config.metric_backend = args.metric_backend
    if args.use_diff_augment:
        config.use_diff_augment = True
    if args.seed is not None:
        config.seed = args.seed
    return config


def main():
    args = parse_args()
    config = apply_args_to_config(Config(), args)
    set_seed(config.seed)

    print("AI2602任务D：基于GAN的人头图像生成")
    print(f"使用设备: {config.device}")
    print(f"数据路径: {config.data_path}")
    print(f"输出目录: {config.output_dir}")

    experiment = FaceGANExperiment(config)

    if args.mode == "smoke_test":
        experiment.smoke_test()
        return

    if args.mode == "train_dcgan":
        experiment.run_single_model("dcgan", evaluate=True)
        return

    if args.mode == "train_improved":
        experiment.run_single_model("improved", evaluate=True)
        return

    if args.mode == "compare":
        experiment.compare_models()
        return

    if args.mode in ["evaluate", "interpolate"]:
        if args.checkpoint is None:
            raise ValueError(f"{args.mode}模式需要提供 --checkpoint")
        generator, model_name = experiment.load_generator_from_checkpoint(
            args.checkpoint, args.model_type
        )
        evaluator = Evaluator(config)
        if args.mode == "interpolate":
            evaluator.save_interpolation(generator, model_name=model_name)
        else:
            eval_loader = experiment.data_processor.create_dataloader(
                shuffle=False, drop_last=False
            )
            evaluator.evaluate(generator, eval_loader, model_name=model_name)
        return


if __name__ == "__main__":
    main()

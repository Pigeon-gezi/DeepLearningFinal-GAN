# -*- coding: utf-8 -*-
"""数据管线：图像预处理、数据集加载、DataLoader 工厂。"""

from __future__ import annotations

import os
import random

from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from torchvision.transforms import InterpolationMode

from utils import IMAGE_EXTENSIONS


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

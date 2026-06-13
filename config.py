# -*- coding: utf-8 -*-
"""集中存储GAN实验设置的配置类。"""

import os
import torch

from utils import ensure_dir, get_available_device


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
        self.improved_beta1 = 0.0  # WGAN-GP 原论文: β₁=0 for both G & D
        self.improved_beta2 = 0.99
        self.n_critic = 5
        self.lambda_gp = 5.0
        self.drift_weight = 1e-1  # 无 SN 时需较强 drift 约束 D(real)
        self.feature_matching_weight = 0.5
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

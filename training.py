# -*- coding: utf-8 -*-
"""GAN 训练器：支持 BCE 和 WGAN-GP 两种 loss 类型。"""

from __future__ import annotations

import copy
import os
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import grad

from augmentation import add_instance_noise, diff_augment
from utils import (
    append_csv_row,
    count_trainable_parameters,
    make_safe_filename,
    make_image_grid,
    save_image_grid,
    myfont,
)
import matplotlib.pyplot as plt


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
        if self.loss_type != "wgan-gp" and self.current_instance_noise > 0:
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
        # WGAN-GP 的 gradient penalty 已约束 Lipschitz，无需 clip
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
        # WGAN-GP 模式下跳过 grad_clip，让 GP 独立约束梯度
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
        if os.path.exists(csv_path):
            os.remove(csv_path)

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

# -*- coding: utf-8 -*-
"""顶层实验编排：模型构建、训练、对比、评估。"""

from __future__ import annotations

import os

import numpy as np
import torch

from config import Config
from data import FaceDataProcessor
from models import DCGANGenerator, DCGANDiscriminator, StyleGANLiteGenerator
from training import GANTrainer
from metrics import Evaluator
from utils import (
    ensure_dir,
    make_safe_filename,
    save_image_grid,
    save_json,
    count_trainable_parameters,
    myfont,
)
import matplotlib.pyplot as plt


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

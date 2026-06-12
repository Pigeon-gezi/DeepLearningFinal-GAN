# -*- coding: utf-8 -*-
"""评估模块：InceptionMetricBackend, FID/IS 计算, Evaluator。"""

from __future__ import annotations

import os

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import Inception_V3_Weights, inception_v3

from utils import ensure_dir, make_safe_filename, save_image_grid, save_json, myfont
import matplotlib.pyplot as plt


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

# -*- coding: utf-8 -*-
"""CLI 参数解析、配置更新、主函数。"""

import argparse
import os

from config import Config
from experiment import FaceGANExperiment
from metrics import Evaluator
from utils import set_seed


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
    parser.add_argument(
        "--n_critic",
        type=int,
        default=None,
        help="WGAN-GP判别器每轮更新次数（默认5，降为1-2可缓解欠训练）",
    )
    parser.add_argument(
        "--lr_g",
        type=float,
        default=None,
        help="WGAN-GP生成器学习率（默认1e-4，太小会导致欠训练，建议2e-4）",
    )
    parser.add_argument(
        "--drift_weight",
        type=float,
        default=None,
        help="WGAN-GP drift惩罚权重（默认1e-1，小数据集判別器过强时可加大）",
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
    if args.n_critic is not None:
        config.n_critic = args.n_critic
    if args.lr_g is not None:
        config.improved_learning_rate_g = args.lr_g
    if args.drift_weight is not None:
        config.drift_weight = args.drift_weight
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

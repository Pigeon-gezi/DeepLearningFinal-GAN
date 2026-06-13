# -*- coding: utf-8 -*-
"""
拆分验证测试: 验证每个新模块可独立导入并正常工作。
RED 阶段: 模块尚未创建，这些测试必定失败。
GREEN 阶段: 模块创建后全部通过。
"""

from __future__ import annotations

import os
import sys
import unittest
import tempfile

import numpy as np
import torch


class TestConfig(unittest.TestCase):
    """测试 config.py 模块"""

    def test_import_config(self):
        from config import Config

        cfg = Config()
        self.assertIsNotNone(cfg)

    def test_config_defaults(self):
        from config import Config

        cfg = Config()
        self.assertEqual(cfg.image_size, 64)
        self.assertEqual(cfg.latent_dim, 128)
        self.assertEqual(cfg.batch_size, 64)
        self.assertEqual(cfg.epochs, 20)
        self.assertIsInstance(cfg.device, torch.device)

    def test_config_dirs_exist(self):
        from config import Config

        cfg = Config()
        self.assertTrue(os.path.isdir(cfg.output_dir))
        self.assertTrue(os.path.isdir(cfg.checkpoint_dir))
        self.assertTrue(os.path.isdir(cfg.sample_dir))
        self.assertTrue(os.path.isdir(cfg.metric_dir))


class TestUtils(unittest.TestCase):
    """测试 utils.py 模块"""

    def test_import_utils(self):
        import utils

        self.assertTrue(hasattr(utils, "set_seed"))
        self.assertTrue(hasattr(utils, "get_available_device"))
        self.assertTrue(hasattr(utils, "make_safe_filename"))
        self.assertTrue(hasattr(utils, "ensure_dir"))
        self.assertTrue(hasattr(utils, "count_trainable_parameters"))

    def test_set_seed(self):
        from utils import set_seed

        set_seed(42)
        a = torch.randn(10)
        set_seed(42)
        b = torch.randn(10)
        self.assertTrue(torch.equal(a, b))

    def test_get_available_device(self):
        from utils import get_available_device

        device = get_available_device()
        self.assertIsInstance(device, torch.device)

    def test_make_safe_filename(self):
        from utils import make_safe_filename

        self.assertEqual(make_safe_filename("hello world"), "hello_world")
        self.assertEqual(make_safe_filename("a/b:c"), "a_b_c")
        self.assertNotEqual(make_safe_filename("test"), "")

    def test_ensure_dir(self):
        from utils import ensure_dir

        with tempfile.TemporaryDirectory() as tmpdir:
            p = os.path.join(tmpdir, "a", "b", "c")
            result = ensure_dir(p)
            self.assertTrue(os.path.isdir(result))

    def test_count_trainable_parameters(self):
        import torch.nn as nn
        from utils import count_trainable_parameters

        m = nn.Linear(10, 5)
        n = count_trainable_parameters(m)
        self.assertEqual(n, 55)  # 10*5 weights + 5 biases


class TestModelsDCGAN(unittest.TestCase):
    """测试 models/dcgan.py 模块"""

    def test_import_dcgan_generator(self):
        from models import DCGANGenerator

        g = DCGANGenerator(latent_dim=128, image_size=64)
        self.assertIsNotNone(g)

    def test_dcgan_generator_forward(self):
        from models import DCGANGenerator

        g = DCGANGenerator(latent_dim=128, image_size=64)
        z = torch.randn(4, 128)
        with torch.no_grad():
            out = g(z)
        self.assertEqual(out.shape, (4, 3, 64, 64))
        self.assertTrue((-1.0 <= out).all() and (out <= 1.0).all())

    def test_import_dcgan_discriminator(self):
        from models import DCGANDiscriminator

        d = DCGANDiscriminator(image_size=64)
        self.assertIsNotNone(d)

    def test_dcgan_discriminator_forward(self):
        from models import DCGANDiscriminator

        d = DCGANDiscriminator(image_size=64)
        x = torch.randn(4, 3, 64, 64)
        with torch.no_grad():
            out = d(x)
        self.assertEqual(out.shape, (4,))


class TestModelsStyleGAN(unittest.TestCase):
    """测试 models/stylegan.py 模块"""

    def test_import_stylegan_generator(self):
        from models import StyleGANLiteGenerator

        g = StyleGANLiteGenerator(latent_dim=128, w_dim=128, image_size=64)
        self.assertIsNotNone(g)

    def test_stylegan_generator_forward(self):
        from models import StyleGANLiteGenerator

        g = StyleGANLiteGenerator(latent_dim=128, w_dim=128, image_size=64)
        z = torch.randn(2, 128)
        with torch.no_grad():
            out = g(z)
        self.assertEqual(out.shape, (2, 3, 64, 64))

    def test_stylegan_return_w(self):
        from models import StyleGANLiteGenerator

        g = StyleGANLiteGenerator(latent_dim=128, w_dim=128, image_size=64)
        z = torch.randn(2, 128)
        with torch.no_grad():
            img, w = g(z, return_w=True)
        self.assertEqual(img.shape, (2, 3, 64, 64))
        self.assertEqual(w.shape, (2, 128))


class TestAugmentation(unittest.TestCase):
    """测试 augmentation.py 模块"""

    def test_import_augmentation(self):
        from augmentation import add_instance_noise, diff_augment

        self.assertTrue(callable(add_instance_noise))
        self.assertTrue(callable(diff_augment))

    def test_add_instance_noise(self):
        from augmentation import add_instance_noise

        x = torch.zeros(4, 3, 64, 64)
        noisy = add_instance_noise(x, std=0.1)
        self.assertEqual(noisy.shape, x.shape)
        self.assertFalse(torch.equal(noisy, x))

    def test_add_instance_noise_zero_std(self):
        from augmentation import add_instance_noise

        x = torch.randn(4, 3, 64, 64)
        same = add_instance_noise(x, std=0.0)
        self.assertTrue(torch.equal(same, x))

    def test_diff_augment_shape(self):
        from augmentation import diff_augment

        x = torch.randn(4, 3, 64, 64)
        out = diff_augment(x)
        self.assertEqual(out.shape, x.shape)


class TestData(unittest.TestCase):
    """测试 data.py 模块"""

    def test_import_data(self):
        from data import FaceTransform, FaceImageDataset, FaceDataProcessor

        self.assertTrue(callable(FaceTransform))
        self.assertIsNotNone(FaceImageDataset)
        self.assertIsNotNone(FaceDataProcessor)

    def test_face_transform(self):
        from data import FaceTransform
        from PIL import Image

        transform = FaceTransform(image_size=64)
        img = Image.new("RGB", (128, 128), color=(100, 150, 200))
        tensor = transform(img)
        self.assertEqual(tensor.shape, (3, 64, 64))
        self.assertTrue((-1.0 <= tensor).all() and (tensor <= 1.0).all())

    def test_face_data_processor(self):
        from config import Config
        from data import FaceDataProcessor

        cfg = Config()
        processor = FaceDataProcessor(cfg)
        self.assertIsNotNone(processor)


class TestTraining(unittest.TestCase):
    """测试 training.py 模块"""

    def test_import_trainer(self):
        from training import GANTrainer

        self.assertIsNotNone(GANTrainer)

    def test_trainer_dcgan_bce(self):
        from config import Config
        from models import DCGANGenerator, DCGANDiscriminator
        from training import GANTrainer

        cfg = Config()
        cfg.epochs = 1
        cfg.fixed_noise_count = 4
        g = DCGANGenerator(latent_dim=cfg.latent_dim, image_size=cfg.image_size)
        d = DCGANDiscriminator(image_size=cfg.image_size)
        trainer = GANTrainer(g, d, cfg, model_name="Test_DCGAN", loss_type="bce")
        self.assertEqual(trainer.model_name, "Test_DCGAN")
        self.assertEqual(trainer.loss_type, "bce")

    def test_trainer_wgan_gp(self):
        from config import Config
        from models import StyleGANLiteGenerator, WGANDiscriminator
        from training import GANTrainer

        cfg = Config()
        cfg.epochs = 1
        g = StyleGANLiteGenerator(
            latent_dim=cfg.latent_dim, w_dim=128, image_size=cfg.image_size
        )
        d = WGANDiscriminator(image_size=cfg.image_size)
        trainer = GANTrainer(g, d, cfg, model_name="Test_WGAN", loss_type="wgan-gp")
        self.assertEqual(trainer.loss_type, "wgan-gp")
        self.assertIsNotNone(trainer.generator_ema)

    def test_trainer_sample_noise(self):
        from config import Config
        from models import DCGANGenerator, DCGANDiscriminator
        from training import GANTrainer

        cfg = Config()
        g = DCGANGenerator(latent_dim=cfg.latent_dim, image_size=cfg.image_size)
        d = DCGANDiscriminator(image_size=cfg.image_size)
        trainer = GANTrainer(g, d, cfg, model_name="Test", loss_type="bce")
        z = trainer.sample_noise(8)
        self.assertEqual(z.shape, (8, cfg.latent_dim))


class TestMetrics(unittest.TestCase):
    """测试 metrics.py 模块"""

    def test_import_evaluator(self):
        from metrics import Evaluator

        self.assertIsNotNone(Evaluator)

    def test_calculate_fid(self):
        from metrics import calculate_fid_from_features

        rng = np.random.RandomState(42)
        real = rng.randn(100, 2048).astype(np.float64)
        fake = rng.randn(100, 2048).astype(np.float64)
        fid = calculate_fid_from_features(real, fake)
        self.assertIsInstance(fid, float)
        self.assertFalse(np.isnan(fid))

    def test_calculate_is(self):
        from metrics import calculate_inception_score

        rng = np.random.RandomState(42)
        probs = rng.rand(100, 1000).astype(np.float64)
        probs = probs / probs.sum(axis=1, keepdims=True)
        mean, std = calculate_inception_score(probs, splits=5)
        self.assertIsInstance(mean, float)
        self.assertFalse(np.isnan(mean))


class TestExperiment(unittest.TestCase):
    """测试 experiment.py 模块"""

    def test_import_experiment(self):
        from experiment import FaceGANExperiment

        self.assertIsNotNone(FaceGANExperiment)

    def test_build_models_dcgan(self):
        from config import Config
        from experiment import FaceGANExperiment

        cfg = Config()
        exp = FaceGANExperiment(cfg)
        g, d, name, loss = exp.build_models("dcgan")
        self.assertIsNotNone(g)
        self.assertIsNotNone(d)
        self.assertEqual(loss, "bce")

    def test_build_models_improved(self):
        from config import Config
        from experiment import FaceGANExperiment

        cfg = Config()
        exp = FaceGANExperiment(cfg)
        g, d, name, loss = exp.build_models("improved")
        self.assertIsNotNone(g)
        self.assertIsNotNone(d)
        self.assertEqual(loss, "wgan-gp")


class TestCLI(unittest.TestCase):
    """测试 cli.py 模块"""

    def test_import_cli(self):
        import cli

        self.assertTrue(hasattr(cli, "parse_args"))
        self.assertTrue(hasattr(cli, "apply_args_to_config"))
        self.assertTrue(hasattr(cli, "main"))

    def test_apply_args(self):
        from config import Config
        import cli

        cfg = Config()
        # 避免 unittest 命令行参数干扰 argparse
        args = (
            cli.parse_args.__wrapped__
            if hasattr(cli.parse_args, "__wrapped__")
            else cli.parse_args
        )
        import sys

        orig_argv = sys.argv
        try:
            sys.argv = ["main.py", "--epochs", "5"]
            parsed = cli.parse_args()
            self.assertEqual(parsed.epochs, 5)
            cfg = cli.apply_args_to_config(cfg, parsed)
            self.assertEqual(cfg.epochs, 5)
        finally:
            sys.argv = orig_argv


class TestMainEntry(unittest.TestCase):
    """测试 main.py 入口"""

    def test_import_main(self):
        """确保 main.py 可被解释器无语法错误地导入"""
        import importlib

        with open("main.py", "r", encoding="utf-8") as f:
            compile(f.read(), "main.py", "exec")
        self.assertTrue(True)


if __name__ == "__main__":
    unittest.main(verbosity=2)

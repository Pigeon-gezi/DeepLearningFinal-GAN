# GAN 人脸图像生成 — 完整训练流程

> **当前环境**: Windows 11 | Python 3.13.5 | PyTorch 2.11.0+cu130 | RTX 4060 Laptop GPU (8GB)

---

## 目录

1. [GAN 人脸图像生成 — 完整训练流程](#gan-人脸图像生成--完整训练流程)
   1. [目录](#目录)
   2. [1. 环境准备](#1-环境准备)
      1. [当前环境（无需额外操作）](#当前环境无需额外操作)
      2. [安装依赖（仅当缺少包时）](#安装依赖仅当缺少包时)
      3. [依赖版本清单](#依赖版本清单)
   3. [2. 数据集准备](#2-数据集准备)
      1. [支持的数据集](#支持的数据集)
      2. [目录结构示例](#目录结构示例)
      3. [如果数据集在其他位置](#如果数据集在其他位置)
   4. [3. Smoke Test（无数据验证）](#3-smoke-test无数据验证)
   5. [4. 快速实验](#4-快速实验)
   6. [5. 完整训练 DCGAN](#5-完整训练-dcgan)
      1. [自定义训练参数](#自定义训练参数)
      2. [训练过程输出](#训练过程输出)
      3. [输出位置](#输出位置)
   7. [6. 完整训练 StyleGAN-lite + WGAN-GP](#6-完整训练-stylegan-lite--wgan-gp)
      1. [训练特点](#训练特点)
      2. [与 DCGAN 对比](#与-dcgan-对比)
   8. [7. 模型对比实验](#7-模型对比实验)
   9. [8. 评估已有检查点](#8-评估已有检查点)
      1. [指标说明](#指标说明)
   10. [9. 潜变量插值可视化](#9-潜变量插值可视化)
   11. [10. 输出文件说明](#10-输出文件说明)
   12. [常见问题](#常见问题)
       1. [Q: 显存不足 (OOM)](#q-显存不足-oom)
       2. [Q: 训练太慢](#q-训练太慢)
       3. [Q: 数据集路径错误](#q-数据集路径错误)
       4. [Q: 先验证代码正确性](#q-先验证代码正确性)
   13. [推荐训练命令一览](#推荐训练命令一览)

---

## 1. 环境准备

### 当前环境（无需额外操作）

```powershell
# 确认 Python 版本
python --version
# Python 3.13.5

# 确认 PyTorch 和 CUDA
python -c "import torch; print(f'PyTorch {torch.__version__}, CUDA {torch.cuda.is_available()}')"
# PyTorch 2.11.0+cu130, CUDA True

# 确认 GPU
python -c "import torch; print(torch.cuda.get_device_name(0))"
# NVIDIA GeForce RTX 4060 Laptop GPU
```

### 安装依赖（仅当缺少包时）

```powershell
# 切换到项目目录
cd E:\SJTU\DeepLearning\DL_final_god

# 检查当前虚拟环境
python -c "import sys; print(sys.executable)"
# E:\Python_env\AI\Scripts\python.exe  (已激活)

# 安装依赖（已有则跳过）
pip install -r requirements.txt
```

### 依赖版本清单

| 包          | 版本         | 用途                      |
| ----------- | ------------ | ------------------------- |
| PyTorch     | 2.11.0+cu130 | 深度学习框架              |
| torchvision | 0.26.0+cu130 | 图像预处理 + Inception-v3 |
| numpy       | 2.2.6        | 数值计算                  |
| scipy       | 1.16.3       | FID 矩阵平方根            |
| Pillow      | 12.0.0       | 图像读写                  |
| matplotlib  | 3.10.7       | 绘图                      |
| seaborn     | 0.13.2       | 训练曲线                  |

---

## 2. 数据集准备

### 支持的数据集

- **CelebA**: 将 `img_align_celeba` 目录放到 `faces/` 下
- **LFW**: 将 LFW 图片目录放到 `faces/` 下
- **任意图片文件夹**: 递归扫描所有 `.jpg/.jpeg/.png/.bmp/.webp` 文件

### 目录结构示例

```txt
DL_final_god/
├── faces/                    ← 把图片放在这里
│   ├── 000001.jpg
│   ├── 000002.jpg
│   └── ...
├── main.py
├── config.py
└── ...
```

### 如果数据集在其他位置

```powershell
# 通过命令行参数指定
python main.py --mode train_dcgan --data_path "D:\datasets\celeba\img_align_celeba"
```

---

## 3. Smoke Test（无数据验证）

**无需数据集**，验证所有模型可以正常构建和前向传播。

```powershell
cd E:\SJTU\DeepLearning\DL_final_god
python main.py --mode smoke_test
```

**预期输出**:

- DCGAN 生成器和判别器参数量信息
- StyleGAN-lite 生成器和判别器参数量信息
- `samples/Smoke_Test_DCGAN.png`
- `samples/Smoke_Test_StyleGAN_lite_+_WGAN_GP.png`
- `samples/Interpolation_DCGAN_smoke.png`
- `samples/Interpolation_StyleGAN_lite_+_WGAN_GP_smoke.png`

---

## 4. 快速实验

使用少量图片快速验证训练流程是否正常（~2-5 分钟）。

```powershell
# DCGAN 快速实验：仅用 512 张图片，训练 5 个 epoch
python main.py --mode train_dcgan --max_images 512 --epochs 5 --batch_size 32
```

```powershell
# StyleGAN-lite 快速实验：仅用 512 张图片，训练 5 个 epoch
python main.py --mode train_improved --max_images 512 --epochs 5 --batch_size 32
```

---

## 5. 完整训练 DCGAN

**基础模型**，使用 BCE loss，训练快，适合作为 baseline。

```powershell
cd E:\SJTU\DeepLearning\DL_final_god

# 默认配置训练 (64x64 图片, 20 epochs, batch_size=64)
python main.py --mode train_dcgan
```

### 自定义训练参数

```powershell
# 更大分辨率 + 更多 epoch
python main.py --mode train_dcgan --image_size 128 --epochs 50 --batch_size 32

# 限制数据集数量（加速）
python main.py --mode train_dcgan --max_images 10000 --epochs 30

# 使用指定数据集路径
python main.py --mode train_dcgan --data_path "D:\datasets\celeba\img_align_celeba"
```

### 训练过程输出

- **每 100 batch**: 打印当前 D loss / G loss / D(real) / D(fake)
- **每个 epoch 结束时**: 打印 epoch 总结
- **每 5 个 epoch**: 保存检查点 + 训练曲线
- **每个 epoch**: 保存生成样本图

### 输出位置

| 内容     | 路径                                       |
| -------- | ------------------------------------------ |
| 生成样本 | `samples/Samples_DCGAN_epoch_N.png`        |
| 检查点   | `checkpoints/Checkpoint_DCGAN_epoch_N.pth` |
| 训练日志 | `metrics/Train_Log_DCGAN.csv`              |
| 训练曲线 | `metrics/Training_DCGAN.png`               |
| 评估指标 | `metrics/Metrics_DCGAN.json`               |

---

## 6. 完整训练 StyleGAN-lite + WGAN-GP

**改进模型**，使用 WGAN-GP + 谱归一化 + EMA + 特征匹配，生成质量更高但训练更慢。

```powershell
cd E:\SJTU\DeepLearning\DL_final_god

# 默认配置训练
python main.py --mode train_improved
```

### 训练特点

- **n_critic=5**: 判别器每更新 5 次，生成器更新 1 次
- **EMA**: 滑动平均生成器，用于最终采样（更稳定）
- **梯度惩罚 + 特征匹配**: 防止模式崩溃

### 与 DCGAN 对比

| 特性             | DCGAN | StyleGAN-lite + WGAN-GP |
| ---------------- | ----- | ----------------------- |
| Loss             | BCE   | WGAN-GP                 |
| 谱归一化         | ❌    | ✅                      |
| Minibatch StdDev | ❌    | ✅                      |
| EMA              | ❌    | ✅                      |
| 特征匹配         | ❌    | ✅                      |
| 生成器参数量     | ~3.8M | ~12.6M                  |
| 训练速度         | 快    | 慢 (~2x)                |
| 生成质量         | 基准  | 更优                    |

---

## 7. 模型对比实验

依次训练 DCGAN 和 StyleGAN-lite，自动计算 FID/IS 并生成对比图。

```powershell
cd E:\SJTU\DeepLearning\DL_final_god

# 完整对比（需要数据集，耗时较长）
python main.py --mode compare --epochs 20 --max_images 5000
```

**输出**:

- `metrics/Model_Comparison_GAN.json` — FID/IS 对比数值
- `metrics/Model_Comparison_GAN.png` — 对比柱状图

---

## 8. 评估已有检查点

对训练好的模型计算 FID 和 Inception Score。

```powershell
# 评估 DCGAN 检查点
python main.py --mode evaluate --model_type dcgan --checkpoint "checkpoints/Checkpoint_DCGAN_epoch_final.pth"
```

```powershell
# 评估 StyleGAN-lite 检查点
python main.py --mode evaluate --model_type improved --checkpoint "checkpoints/Checkpoint_StyleGAN-lite_+_WGAN-GP_epoch_final.pth"
```

```powershell
# 限制评估图片数量（加速 FID 计算）
python main.py --mode evaluate --model_type dcgan --checkpoint "checkpoints/Checkpoint_DCGAN_epoch_final.pth" --eval_num_images 512
```

### 指标说明

| 指标          | 含义                                 | 优秀  | 一般    | 差    |
| ------------- | ------------------------------------ | ----- | ------- | ----- |
| **FID** ↓     | Fréchet Inception Distance，越低越好 | < 50  | 50-100  | > 100 |
| **IS Mean** ↑ | Inception Score 均值，越高越好       | > 3.0 | 2.0-3.0 | < 2.0 |

> ⚠️ FID 对小样本量敏感，建议至少 1024 张图片以获得稳定结果。

---

## 9. 潜变量插值可视化

在潜空间中线性插值，观察生成人脸的平滑过渡。

```powershell
# 从 DCGAN 检查点生成插值
python main.py --mode interpolate --model_type dcgan --checkpoint "checkpoints/Checkpoint_DCGAN_epoch_final.pth"
```

```powershell
# 从 StyleGAN-lite 检查点生成插值
python main.py --mode interpolate --model_type improved --checkpoint "checkpoints/Checkpoint_StyleGAN-lite_+_WGAN-GP_epoch_final.pth"
```

**输出**: `samples/Interpolation_<模型名>.png`（12 帧插值序列）

---

## 10. 输出文件说明

```txt
DL_final_god/
├── checkpoints/
│   ├── Checkpoint_DCGAN_epoch_5.pth          # 每 5 epoch 的模型权重
│   ├── Checkpoint_DCGAN_epoch_final.pth      # 最终模型权重
│   └── Checkpoint_StyleGAN-lite_+_WGAN-GP_epoch_final.pth
├── samples/
│   ├── Samples_DCGAN_epoch_1.png             # 固定噪声的生成样本
│   ├── Samples_DCGAN_epoch_final.png
│   ├── Interpolation_DCGAN.png               # 潜空间插值
│   └── Smoke_Test_*.png                      # smoke test 输出
├── metrics/
│   ├── Train_Log_DCGAN.csv                   # 逐 epoch 训练指标
│   ├── Training_DCGAN.png                    # 训练曲线图
│   ├── Metrics_DCGAN.json                    # FID/IS 评估结果
│   ├── Metrics_StyleGAN-lite_+_WGAN-GP.json
│   ├── Model_Comparison_GAN.json             # 模型对比
│   └── Model_Comparison_GAN.png              # 模型对比图
└── test_split.py                             # 单元测试（36 个）
```

---

## 常见问题

### Q: 显存不足 (OOM)

```powershell
# 减小 batch_size
python main.py --mode train_dcgan --batch_size 16 --image_size 64
```

### Q: 训练太慢

```powershell
# 限制 epoch 和数据集大小
python main.py --mode train_dcgan --epochs 10 --max_images 2000 --batch_size 32
```

### Q: 数据集路径错误

```powershell
FileNotFoundError: 数据路径不存在: E:\SJTU\DeepLearning\DL_final_god\faces
```

**解决**: 在项目根目录创建 `faces` 文件夹，放入图片；或使用 `--data_path` 指定路径。

### Q: 先验证代码正确性

```powershell
# 运行单元测试
cd E:\SJTU\DeepLearning\DL_final_god
python -m unittest test_split -v
# 或
pytest test_split.py -v
```

---

## 推荐训练命令一览

```powershell
# ===== 第一步：smoke test（30 秒） =====
cd E:\SJTU\DeepLearning\DL_final_god
python main.py --mode smoke_test

# ===== 第二步：快速实验（2-5 分钟） =====
python main.py --mode train_dcgan --max_images 512 --epochs 5 --batch_size 32

# ===== 第三步：完整 DCGAN 训练（~20 分钟 for 20 epochs） =====
python main.py --mode train_dcgan

# ===== 第四步：完整 StyleGAN-lite 训练（~40 分钟 for 20 epochs） =====
python main.py --mode train_improved

# ===== 第五步：对比评估 =====
python main.py --mode evaluate --model_type dcgan --checkpoint "checkpoints/Checkpoint_DCGAN_epoch_final.pth"
python main.py --mode evaluate --model_type improved --checkpoint "checkpoints/Checkpoint_StyleGAN-lite_+_WGAN-GP_epoch_final.pth"

# ===== 第六步：插值可视化 =====
python main.py --mode interpolate --model_type dcgan --checkpoint "checkpoints/Checkpoint_DCGAN_epoch_final.pth"
python main.py --mode interpolate --model_type improved --checkpoint "checkpoints/Checkpoint_StyleGAN-lite_+_WGAN-GP_epoch_final.pth"
```

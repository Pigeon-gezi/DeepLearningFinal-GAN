# StyleGAN-lite + WGAN-GP 训练问题排查手册

> 记录从最初到收敛过程中遇到的所有训练问题、根因分析及最终解决方案。

---

## 目录

1. [StyleGAN-lite + WGAN-GP 训练问题排查手册](#stylegan-lite--wgan-gp-训练问题排查手册)
   1. [目录](#目录)
   2. [问题一：极端模式崩溃（Diversity=0.017）](#问题一极端模式崩溃diversity0017)
   3. [问题二：D(real) 持续飙升（8→12）](#问题二dreal-持续飙升812)
   4. [问题三：D Loss 震荡/不收敛](#问题三d-loss-震荡不收敛)
   5. [问题四：G Loss 不降反增/震荡](#问题四g-loss-不降反增震荡)
   6. [问题五：Batch 级别指标剧烈波动](#问题五batch-级别指标剧烈波动)
   7. [问题六：EqualConv2d 缺失导致 AdaIN 学不动](#问题六equalconv2d-缺失导致-adain-学不动)
   8. [问题七：init\_weights\_dcgan 破坏等学习率初始化](#问题七init_weights_dcgan-破坏等学习率初始化)
   9. [问题八：Instance Noise + WGAN-GP 冲突](#问题八instance-noise--wgan-gp-冲突)
   10. [问题九：Grad Clip + Gradient Penalty 冲突](#问题九grad-clip--gradient-penalty-冲突)
   11. [问题十：Feature Matching 权重过大](#问题十feature-matching-权重过大)
   12. [问题十一：Batch Size 过大导致训练变慢](#问题十一batch-size-过大导致训练变慢)
   13. [问题十二：LFW 数据集规模不足](#问题十二lfw-数据集规模不足)
   14. [最终配置与参数速查](#最终配置与参数速查)
       1. [config.py 关键默认值](#configpy-关键默认值)
       2. [CLI 可调参数](#cli-可调参数)
       3. [推荐训练命令](#推荐训练命令)
       4. [健康训练指标参考](#健康训练指标参考)
       5. [代码改动清单](#代码改动清单)

---

## 问题一：极端模式崩溃（Diversity=0.017）

**现象**：StyleGAN-lite 训练 5 epoch 后生成图几乎完全相同，Pixel_Diversity=0.017，IS=1.10。

**根因**：`n_critic=5` 导致生成器更新严重不足。

| 参数 | 值 |
|------|:---:|
| 数据量 | 2000 |
| batch_size | 32 |
| 每 epoch batch 数 | 62 |
| n_critic=5 后 G 更新/epoch | 62÷5 ≈ 12 |
| 5 epoch 总 G 更新 | ≈ 60 |

生成器总共只被训练了 60 次，远不足以学到多样性。

**解决**：

```python
# config.py
self.n_critic = 2  # 从 5 降到 2
```

```bash
# 或 CLI 参数
--n_critic 2
```

---

## 问题二：D(real) 持续飙升（8→12）

**现象**：WGAN-GP 训练中 D(real) 从 3 一路涨到 8.4，甚至 12。D(fake) ≈ 0（判別器不在乎假图）。

**根因**：`drift_weight` 太小（1e-3），判別器走捷径——抬高 D(real) 带来的 loss 收益远超 drift 惩罚。

公式：`drift = drift_weight × E[D(real)²]`

| drift_weight | D(real) 临界点 | 效果 |
|:---:|:---:|------|
| 1e-3 | ≈ 500 | 几乎无约束 |
| 1e-2 | ≈ 50 | 约束不足 |
| **1e-1** | **≈ 5** | ✅ 目标区间 |

**解决**：

```python
# config.py
self.drift_weight = 1e-1  # 从 1e-3 提高
```

---

## 问题三：D Loss 震荡/不收敛

**现象**：D Loss 在 −20 ~ −9 间震荡，用户误以为训练失败。

**根因**：WGAN-GP 的 D Loss **本就不该收敛到 0**。它由三项组成：

```python
D_loss = E[D(fake)] - E[D(real)] + λ_GP × GP + ε_drift × E[D(real)²]
```

健康训练中：

- `E[D(fake)] - E[D(real)]` 越来越负（判別器区分力增强）
- GP 在 0.05-0.3 间波动
- D Loss 震荡是 G/D 博弈的正常现象

**正确的健康指标**：

| 指标 | 健康范围 |
|------|:---:|
| D(real) − D(fake) | 5-15 |
| D(real) | 2-5 |
| GP | 0.05-0.3 |
| Pixel_Diversity | > 0.3 |
| 生成样本图 | 肉眼可见改善 |

---

## 问题四：G Loss 不降反增/震荡

**现象一**：n_critic=5 时 G Loss 不降反增。

**根因**：判別器更新 5 次 G 才更新 1 次，判別器碾压生成器。G Loss = `-E[D(G(z))]`，判別器越强，`D(G(z))` 越负，G Loss 自然上升。

**解决**：`n_critic` 降至 2。

**现象二**：调整后 G Loss 仍震荡在 0-10 之间。

**根因**：Feature Matching 权重过大（2.0），导致生成器优化方向偏离。

```python
G_loss = -E[D(G(z))] + feature_matching_weight × L1(feature_mean)
```

`feature_matching_weight=2.0` 时，特征匹配项主导梯度，生成器努力模仿中间层特征而非骗过判別器。

**结果**：Diversity 不错但图像模糊，FID 下不去。

**解决**：

```python
# config.py
self.feature_matching_weight = 0.5  # 从 2.0 降低
```

---

## 问题五：Batch 级别指标剧烈波动

**现象**：Epoch 前 100 batch 的 D(fake) = −10.8，epoch 平均却变成 +2.5。

**根因**：`n_critic=2` 时，最后几个 batch 恰逢 G 更新。G 权重突变后，随后 batch 的判別器评估的是全新假图，D(fake) 瞬间跳变。

```
batch 100: 旧 G 的假图 → D(fake) = -10.8
batch 101: G 更新 → 新假图
batch 102: 新假图 → D(fake) 可能跳变到正数
```

epoch 平均是简单算术平均，最后几个异常值拉偏整体。

**影响**：**不影响训练质量。** 看 batch 100 的快照即可判断训练状态。最终 FID/Diversity/IS 才是真实指标。

---

## 问题六：EqualConv2d 缺失导致 AdaIN 学不动

**现象**：修复超参后 Pixel_Diversity 0.5 但 FID 仍 280，IS=1.68。生成图视觉效果差。

**根因**：StyleGAN-lite 使用普通 `nn.Conv2d`，不同分辨率层的有效学习率不统一。4×4 层的权重更新幅度和 64×64 层天差地别，导致 AdaIN 调制几乎不生效——所有 z 映射到相似的 w。

**解决**：新增 `EqualConv2d` 替代所有 `nn.Conv2d`。

```python
# models/stylegan.py
class EqualConv2d(nn.Module):
    """等学习率卷积"""
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0):
        fan_in = kernel_size * kernel_size * in_channels
        self.scale = 1.0 / math.sqrt(fan_in)
        self.weight = nn.Parameter(torch.randn(out_channels, in_channels, kernel_size, kernel_size))

    def forward(self, x):
        return F.conv2d(x, self.weight * self.scale, self.bias, ...)
```

同时在 `StyledConvBlock` 和 `to_rgb` 中将 `nn.Conv2d` 替换为 `EqualConv2d`。

---

## 问题七：init_weights_dcgan 破坏等学习率初始化

**现象**：加了 `EqualConv2d` 后训练仍然有问题。

**根因**：`StyleGANLiteGenerator.__init__` 调用了 `self.apply(init_weights_dcgan)`。`init_weights_dcgan` 检测到 `classname` 包含 "Conv" 就执行 `N(0, 0.02)` 覆盖权重，恰好抹掉了 `EqualConv2d` 精心设计的 scale 初始化。

**解决**：从 `models/stylegan.py` 中移除对 `init_weights_dcgan` 的导入和调用。`EqualConv2d` 和 `EqualLinear` 构造函数中已有正确的初始化逻辑。

```python
# 删除这两行
# from models.common import init_weights_dcgan
# self.apply(init_weights_dcgan)
```

---

## 问题八：Instance Noise + WGAN-GP 冲突

**现象**：WGAN-GP 训练不稳定，D(fake) 方差极大。

**根因**：`instance_noise_std=0.03` 对 BCELoss 有益（防止判別器过强），但对 WGAN-GP 有害——GP 计算的是插值点梯度，输入加了噪声后 GP 不知道噪声的存在，惩罚目标歪了。

**解决**：WGAN-GP 训练时跳过 instance noise。

```python
# training.py _prepare_for_discriminator
def _prepare_for_discriminator(self, images):
    if self.loss_type != "wgan-gp" and self.current_instance_noise > 0:
        images = add_instance_noise(images, self.current_instance_noise)
    ...
```

---

## 问题九：Grad Clip + Gradient Penalty 冲突

**现象**：D(real) 波动大，G/D 博弈不稳定。

**根因**：WGAN-GP 的 gradient penalty 已通过惩罚梯度 norm 偏离 1 来约束 Lipschitz。再额外做 `clip_grad_norm_(10.0)` 是双重要求——GP 要 1.0，clip 限制到 10.0，两者互相干扰。

**解决**：WGAN-GP 的两个训练方法中移除 `clip_grad_norm_`。

```python
# training.py train_discriminator_wgan / train_generator_wgan
# 删除以下代码块：
# if self.config.grad_clip is not None:
#     torch.nn.utils.clip_grad_norm_(parameters, config.grad_clip)
```

同时将 `lambda_gp` 从 10.0 降到 5.0——EqualConv2d 的梯度天然更均匀，不需要那么强的 GP。

---

## 问题十：Feature Matching 权重过大

**现象**：Diversity 正常但 FID 降不下来，IS 偏低。

**根因**：`feature_matching_weight=2.0` 时，生成器 loss 中 ~80% 来自特征匹配项。生成器被优化去模仿判別器中间层特征，而非生成高质量图像。

**解决**：

```python
# config.py
self.feature_matching_weight = 0.5  # 从 2.0 降低
```

---

## 问题十一：Batch Size 过大导致训练变慢

**现象**：batch_size=256 时 FID 反而变差。

**根因**：batch_size 越大，每 epoch batch 数越少。

| batch_size | batch/epoch | G 更新/epoch (n_critic=2) |
|:---:|:---:|:---:|
| 64 | 31 | 15 |
| 128 | 15 | 7 |
| 256 | 7 | **3** ← 太少 |

WGAN-GP + Small Dataset 中大 batch 的梯度估计反而比中等 batch 更"平均化"，且 G 更新次数急剧减少。

**解决**：batch_size 保持在 64-128，优先保证足够的 G 更新频率。

---

## 问题十二：LFW 数据集规模不足

**现象**：DCGAN 在 LFW 上 FID≈100、IS≈2.86，全面优于 StyleGAN-lite 的 FID≈182、IS≈2.52。

**根因**：StyleGAN-lite 的 mapping network 需要学习从高斯噪声到"人脸流形"的映射。LFW 只 13K 张图、每人平均 ~2 张，不存在流形。

| 数据集 | 图片数 | 每人张数 | DCGAN 适用 | StyleGAN 适用 |
|------|:---:|:---:|:---:|:---:|
| LFW | 13K | ~2 | ✅ | ❌ 流形不存在 |
| **CelebA** | **202K** | ~20 | ✅ | ✅ |

**解决**：StyleGAN-lite 必须用 CelebA 级别的数据集（≥50K 张、每身份≥10 张）。

---

## 最终配置与参数速查

### config.py 关键默认值

```python
# WGAN-GP 训练
self.n_critic = 2              # G 每次 epoch 约 15× 更新
self.lambda_gp = 5.0           # EqualConv 梯度均匀，不需要 10
self.drift_weight = 1e-1       # D(real) 控制在 2-5
self.feature_matching_weight = 0.5  # 辅助而非主导
self.improved_learning_rate_g = 2e-4
self.improved_learning_rate_d = 1e-4
self.ema_decay = 0.999

# DCGAN
self.learning_rate_g = 2e-4
self.learning_rate_d = 2e-4
self.beta1 = 0.5
```

### CLI 可调参数

```bash
--n_critic        # WGAN-GP 判別器更新频率
--lr_g            # WGAN-GP 生成器学习率
--drift_weight    # D(real) 约束强度
--batch_size      # 批大小（建议 64-128）
--epochs          # 训练轮次
--max_images      # 限制数据集规模（调试用）
```

### 推荐训练命令

```bash
# 服务器 A6000
python main.py --mode train_improved \
  --n_critic 2 --lr_g 2e-4 \
  --epochs 50 --batch_size 128 \
  --data_path ~/DL_final_god/faces

# 笔记本 4060（轻量调试）
python main.py --mode train_improved \
  --n_critic 2 --lr_g 2e-4 \
  --epochs 20 --batch_size 32 --max_images 5000 \
  --data_path <your_data_path>
```

### 健康训练指标参考

| 指标 | WGAN-GP 健康范围 | 异常信号 |
|------|:---:|------|
| D(real) | 2-5 | >8: drift 太小 |
| D(fake) | -5 ~ -1 | >0: 判別器太弱或 G 过强 |
| GP | 0.05-0.3 | >1: λ_GP 太大或模型不稳定 |
| Pixel_Diversity | >0.35 | <0.1: 模式崩溃 |
| IS (CelebA 50ep) | >2.5 | <1.5: 训练失败 |
| FID (CelebA 50ep) | <100 | >200: 训练不足或架构问题 |

### 代码改动清单

| 文件 | 改动 | 行数 |
|------|------|:---:|
| `models/stylegan.py` | 新增 `EqualConv2d`，替换 `nn.Conv2d` | +20 |
| `models/stylegan.py` | 移除 `init_weights_dcgan` 导入和调用 | −2 |
| `models/common.py` | 无需修改 | 0 |
| `training.py` | WGAN 跳过 instance noise | 1 |
| `training.py` | WGAN D/G 跳过 `clip_grad_norm_` | −8 |
| `config.py` | 调整 λ_GP、drift、feature_matching 默认值 | 3 |
| `cli.py` | 新增 `--n_critic`、`--lr_g`、`--drift_weight` | +12 |

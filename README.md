# Project 1 — 用 NumPy 实现 ViT-B/32 图像编码器

只用 **NumPy + Pillow** 复现 OpenCLIP **ViT-B/32** 的图像编码器前向传播：输入一批 224×224 的 RGB 图片，输出每张图 **512 维的单位向量**（CLIP 图像嵌入）。

本仓库包含完整实现、逐阶段自检脚本与实验报告，8 个数值检查阶段全部通过。

---

## 1. 目录结构

```
提交包/
├── image_encoder.py        # ★ 核心实现（本次作业补全的部分）
├── preprocess_image.py     # 图片预处理：缩放 / 中心裁剪 / 归一化
├── runtime.py              # 读取 config.json 与权重，组装模型
├── check.py                # 逐阶段数值自检（8 个 stage）
├── run.py                  # 批量推理入口，输出 output_tokens.npz
├── helpers.py              # LayerNorm / GELU / softmax 等辅助函数（课程提供）
├── config.json             # 模型超参数（层数、头数、维度）
├── requirements.txt
├── README.md               # 本文件
├── numpy_homework.py       # 附加：12 道 NumPy 练习（含运行结果）
├── fig_pipeline.png        # 报告用流程图
├── 实验报告_Project1.docx   # 报告源文件
└── 实验报告_Project1.pdf    # 报告 PDF（可直接打印/提交）
```

> 课程提供的 `weights/image_encoder.npz`（326 MB）与 `assets/`（10 张示例图 + 参考输出）
> **不随本包提交**。复现时把本包解压到课程附件解压目录中（与 `weights/`、`assets/` 同级），
> 再执行下面的命令即可。

## 2. 环境要求

| 依赖 | 版本 |
| --- | --- |
| Python | 3.10 及以上（本机验证 3.14.10） |
| NumPy | `>=1.24,<3`（本机验证 2.4.2） |
| Pillow | `>=9.2,<13`（本机验证 12.3.0） |
| imageio | `>=2.31,<3`（仅 run.py 写图时用到） |

安装：

```bash
pip install -r requirements.txt
# 国内网络可加清华源：
pip install -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple
```

## 3. 复现步骤

```bash
# ① 把本包解压到课程附件目录（该目录应含 weights/ 与 assets/）
cd project1-numpy-student

# ② 逐阶段自检：会依次比对 预处理 → 切块 → tokens → block0 → 最后一层 → 输出 → batch
python check.py --stage all

# ③ 批量推理 10 张示例图，生成 output_tokens.npz
python run.py --output output_tokens.npz
```

期望结果：

```
assets       PASS: 152 matching visual tensors and ten image/reference pairs
preprocess   PASS shape=(10, 3, 224, 224) max_abs=0.000e+00 mean_abs=0.000e+00 (atol=2e-06, rtol=0.0002)
patches      PASS shape=(2, 49, 3072) max_abs=0.000e+00 mean_abs=0.000e+00 (atol=0.0002, rtol=0.0002)
tokens       PASS shape=(2, 50, 768) max_abs=4.005e-05 mean_abs=1.289e-06 (atol=0.0002, rtol=0.0002)
block0       PASS shape=(2, 50, 768) max_abs=7.063e-06 mean_abs=3.250e-07 (atol=0.0002, rtol=0.0002)
last_tokens  PASS shape=(2, 50, 768) max_abs=4.768e-05 mean_abs=2.696e-06 (atol=0.0002, rtol=0.0002)
output       PASS shape=(10, 512) max_abs=5.439e-07 mean_abs=8.311e-08 (atol=2e-05, rtol=0.0002)
batch        PASS shape=(3, 512) max_abs=4.619e-07 mean_abs=7.632e-08 (atol=2e-05, rtol=0.0002)
Requested numerical checks passed. The reference file is never an encoder input.
```

`run.py` 保存的 `output_tokens.npz` 里键名是 `output_tokens` 和 `ids`，形状 `(10, 512)`，`dtype=float32`，每一行的 L2 范数都等于 1：

```python
import numpy as np
z = np.load("output_tokens.npz")["output_tokens"]
print(z.shape, z.dtype, np.linalg.norm(z, axis=1).round(6))
# (10, 512) float32 [1. 1. 1. 1. 1. 1. 1. 1. 1. 1.]
```

## 4. 实现要点（对应 `image_encoder.py`）

整条链路只有两处需要“改形状”，其余都是 `(B, 50, 768)` 上的矩阵乘法。

| 步骤 | 关键代码 | 形状变化 |
| --- | --- | --- |
| 切块 | `grid = x.reshape(B,3,7,32,7,32)` → `transpose(0,2,4,1,3,5)` → `reshape(B,49,3072)` | (B,3,224,224) → (B,49,3072) |
| 线性投影 | `patches @ self.patch_weight.T` | (B,49,3072) → (B,49,768) |
| 拼 CLS | `np.concatenate([cls, patch_tokens], axis=1) + position` | (B,49,768) → (B,50,768) |
| 多头拆分 | `reshape(B,N,12,64).transpose(0,2,1,3)` | (B,50,768) → (B,12,50,64) |
| 注意力 | `(q @ k.transpose(0,1,3,2)) / sqrt(64)` → softmax → `@ v` | (B,12,50,50) |
| 合并多头 | `transpose(0,2,1,3).reshape(B,N,768)` | (B,12,50,64) → (B,50,768) |
| 输出投影 | `tokens[:,0,:] @ self.projection` | (B,768) → (B,512) |
| 单位化 | `v / np.linalg.norm(v, axis=-1, keepdims=True)` | (B,512) → (B,512) |

三个容易踩的坑：

1. **切块必须先 transpose**：直接 `reshape(B,49,3072)` 得到的是“按行横切”，形状对但语义错。
2. **权重矩阵是 `[out, in]`**：`x @ W.T + b`，漏掉 `.T` 有时形状仍兼容（如 768×768），会静默出错。
3. **`visual.proj` 已是 `(768,512)`**：不要再转置。

## 5. 验证结果

| 阶段 | 最大绝对误差 | 平均绝对误差 | 容差 |
| --- | --- | --- | --- |
| 预处理 / 切块 | 0 | 0 | 2e-6 / 2e-4 |
| tokens | 4.005e-05 | 1.289e-06 | 2e-4 |
| block0 | 7.063e-06 | 3.250e-07 | 2e-4 |
| last_tokens | 4.768e-05 | 2.696e-06 | 2e-4 |
| output | 5.439e-07 | 8.311e-08 | 2e-5 |
| batch (B=10) | 4.619e-07 | 7.632e-08 | 2e-5 |

三组“只改一行”的对照实验（详见报告）：把多头 reshape 写反、切块漏 transpose、输出投影漏 `.T`，误差分别从 1e-07 量级涨到 1e-01 量级，可以用来理解每个阶段在防什么错。

## 6. 常见问题

**Q: `ModuleNotFoundError: No module named 'runtime'`**
说明 Python 处于 isolated 模式（目录里有 `python3xx._pth`）。删除或改名该文件，或直接用 `python -m` 方式运行。

**Q: `DLL load failed while importing _multiarray_umath`**
NumPy 与当前 Python 小版本不匹配，先升级 pip 再装官方 wheel；仍失败则退一个 NumPy 小版本（本机 2.5.x 失败、2.4.2 正常）。

**Q: `ImportError: cannot import name 'Image' from 'PIL'`**
Pillow 装成了 `pillow-simd` 或版本过低，执行 `pip install --force-reinstall "pillow>=9.2,<13"`。

## 7. 报告与练习

- `实验报告_Project1.pdf`：5 页（含封面），含流程图、逐阶段误差表与三组调试实验。
- `numpy_homework.py`：配套 NumPy 数值练习（12 题），覆盖 shape、广播、批量夹角、top-k 检索、距离矩阵与两层前向。

## 8. 说明

模型权重 `weights/image_encoder.npz` 体积 326 MB，来自课程提供，**未包含在本包中**；
`assets/` 里的参考数组同样不重复提交。把本包解压到课程附件目录、保持 `weights/` 与
`assets/` 原样，即可复现第 3 节的完整输出。


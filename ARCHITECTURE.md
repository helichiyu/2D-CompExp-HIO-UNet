# 架构说明

> 本文档描述 `迭代模型/对比` 项目的模块划分、依赖方向、数据流与关键设计决策。项目背景见 [README.md](README.md)，理论见 [调研报告.md](调研报告.md)，初测发现见 [初测汇报.md](初测汇报.md)。

---

## 1. 概述

本项目用**控制变量法**对比两种二维相位恢复方法：甲方（传统严格 HIO）与乙方（未训练 UNet）。两者共享同一套实空间约束，唯一区别在像空间处理——甲方硬替换振幅，乙方用 UNet 学习振幅。系统职责分三层：工具层（`utils`）、方法层（`hio` / `unet_pr`）、编排层（`main`）。

---

## 2. 模块划分

| 模块 | 职责 | 主要导出 |
|---|---|---|
| `utils.py`（工具层） | 甲乙共用基础设施：图像预处理、可微 FFT/IFFT、随机相位、shrinkwrap、直方图匹配、全部评估指标 | `load_and_preprocess`、`fft_amp_phase`、`ifft_real`、`make_random_phase`、`init_density`、`shrinkwrap_support`、`estimate_reference_histogram`、`histogram_match`、`evaluate_all`、`device` |
| `hio.py`（甲方方法层） | 严格 HIO 迭代：像空间硬替换振幅 + 实空间约束 | `run_hio` |
| `unet_pr.py`（乙方方法层） | 未训练 UNet（Deep Image Prior）：UNet 结构 + 像空间振幅 MSE 反传迭代 | `UNet`、`run_unet` |
| `main.py`（编排层） | 串起整个实验：读图 → 共享初始化 → 跑甲 → 跑乙 → 对比出图 + 存结果 | `main` |

`smoke_test.py` 是验证脚本，不属于核心架构，用于复现初测结果。

---

## 3. 依赖方向

```
main.py
  ├── hio.py ──┐
  └── unet_pr.py ──┤
                   └── utils.py
```

- **单向依赖、无环**：`main` 调 `hio` / `unet_pr`，两者都只调 `utils`；`utils` 不依赖任何上层模块。
- 方法层（`hio` / `unet_pr`）之间**完全独立**，互不引用——这是控制变量的体现：甲乙是两条平行路径，只在 `main` 汇合对比。

---

## 4. 数据流

```
567.png
  │  load_and_preprocess（灰度→扩2倍画布→极性翻转→pad32）
  ▼
rho_work（暗背景工作表示，[1,1,H,W]，物体亮背景≈0）
  │
  ├──► fft_amp_phase ──► amp_orig（唯一实验数据：去相位后的振幅）
  │
  ├──► make_random_phase(seed=42) ──► phase0
  │           │  init_density(amp_orig, phase0)
  │           ▼
  │      rho_init（甲乙共享同一初始密度 → 控制变量）
  │
  ├──► shrinkwrap_support(rho_work) ──► support_gt（评估用真值支撑域）
  │
  └──► estimate_reference_histogram ──► ref_edges（HM 参考直方图，从原图统计）

       ┌──────────── amp_orig, rho_init, ref_edges, rho_work, support_gt ────────────┐
       │                                                                              │
       ▼                                                                              ▼
   run_hio（甲方）                                                              run_unet（乙方）
   每轮：FFT→振幅替换→IFFT→HIO公式→HM                                        每轮：UNet→正值+support→FFT→振幅MSE→反传→HM喂回
       │                                                                              │
       ▼                                                                              ▼
   best_hio + hist_hio                                                        best_unet + hist_unet
       │                                                                              │
       └────────────────── to_visual（极性翻回 + unpad）+ evaluate_all ───────────────┘
                                            ▼
                       result_compare.png / result_convergence.png / result_metrics.csv
```

---

## 5. 关键设计决策

### 5.1 控制变量法（核心原则）
甲乙双方**实空间约束集合完全一致**（正值 + 实数 + 直方图匹配 + shrinkwrap 支撑域），且**共享初始条件**（同一随机相位种子 → 同一 ρ_init、同一参考直方图）。唯一变量是像空间处理：
- 甲方：FFT 后用 |A_orig| 硬替换新振幅，保留新相位；
- 乙方：UNet 输出做 FFT 取振幅，与 |A_orig| 算 MSE 反传更新网络。

### 5.2 极性翻转（暗背景表示）
567.png 是"白背景 + 黑物体"。直接归一化会让所有实空间约束方向反向。内部统一用 `ρ_work = bg_val − img`（物体亮、背景≈0）。数学依据：`FFT(c−f)` 与 `FFT(f)` 的非直流振幅完全相同，且项目保留直流——故**振幅约束一字不变，控制变量不破**。显示时再 `bg_val − ρ` 翻回视觉原貌。

### 5.3 保留直流
初测发现去直流（`fft[0,0]=0`）会导致 HIO 不收敛——相位恢复需要平均密度信息。`fft_amp_phase` 默认 `keep_dc=True`。

### 5.4 支撑域策略（应对"鸡生蛋"）
`run_hio` 的 support 策略可配：
- `fixed_support`：全程固定（调试用，如真值 support）；
- `init_support` + `warmup`：前 warmup 轮用初始 support（如全图 positivity）让 ρ_k 粗成形，之后动态 shrinkwrap 收紧；
- 默认（都为 None）：动态 shrinkwrap（初测发现不收敛，见已知问题）。

shrinkwrap 用高斯模糊 + 阈值二值化，σ 线性衰减（3.0→1.0），每 20 轮更新。

### 5.5 直方图匹配（HM）时机
HM 在宽松 support 下会扰动过大破坏收敛，故 `run_hio` 提供 `hm_start` 参数延后开启（紧 support 形成后再做 HM）。HM 用 Zhang-Main 300 等数量 bin + a·ρ+b 分段线性映射，只在 support 内做，非可微（`@torch.no_grad()`）。

### 5.6 评估前投影
HIO 输出的 support 外是反馈值（可能为负），直接评估会失真。`evaluate_all` 先把 ρ 投影到可行域 `clamp_min(0) × support_pred` 再算指标，甲乙双方一致。

### 5.7 数值约定
- 图像全程 [0,1]（暗背景下背景≈0、物体峰≈0.9）；
- 振幅用 `amp/amp.max()` 归一化算 loss（让 UNet loss 尺度合理，初测修复：原 `amp/(H·W)` 归一化让 loss ≈1e-8 太小）；
- 张量形状统一 `[1,1,H,W]`，float32；
- 图像 pad 到 32 的倍数（UNet 5 级下采样需要）。

### 5.8 UNet 细节
- 5 级下采样，bilinear 上采样 + 跳连，瓶颈通道减半；
- 用 `InstanceNorm2d` 而非 `BatchNorm2d`（单图训练 batch=1，BN 吃方差）；
- `OutConv` 用 Sigmoid，输出约束到 [0,1]（天然非负，实现正值约束）。

---

## 6. 技术栈

| 用途 | 选型 |
|---|---|
| 深度学习 / 可微 FFT | PyTorch（`torch.fft.fft2/ifft2`） |
| 数值计算 | numpy |
| 图像 IO | Pillow（PIL） |
| 可视化 | matplotlib（配置 SimHei 中文） |
| 评估 | 全部自实现（PSNR/SSIM/Pearson CC/amp_cc/Δφ/IoU），**无 skimage / cv2 依赖** |

运行环境：`D:\anaconda3\envs\use\python.exe`。

---

## 7. 目录结构

```
对比/
├── 567.png              输入图
├── utils.py             工具层（甲乙共用）
├── hio.py               甲方：严格 HIO
├── unet_pr.py           乙方：未训练 UNet
├── main.py              编排层：对比入口
├── smoke_test.py        初测验证脚本
├── README.md            项目说明
├── ARCHITECTURE.md      本文档
├── 调研报告.md          文献调研 + 参考文献
├── 实现方案.md          函数级实现方案
├── 初测汇报.md          初测发现与实验价值
├── CLAUDE.md            项目行为准则
└── .gitignore
```

运行后还会生成：`result_compare.png`、`result_convergence.png`、`result_metrics.csv`（结果文件，应加入 `.gitignore`）。

---

## 8. 已知问题与待办

1. **HIO 不偷看 support 时不收敛**：support 估计的"鸡生蛋"循环依赖（详见 [初测汇报.md](初测汇报.md)）。这是实验价值所在——HIO 强依赖准确 support，凸显 UNet 的潜力。
2. **UNet 完整收敛性待验证**：loss 归一化修复后方向正确，但需跑完整 5000 轮判断它能否脱 support 收敛。
3. **结果文件未加入 `.gitignore`**：`result_*.png/csv` 生成后应忽略上传。

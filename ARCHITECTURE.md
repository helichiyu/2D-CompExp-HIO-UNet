# 架构说明

> 四测（2026-07）后更新：三实验对比 + 配准评估 + 取末轮。项目背景见 [README.md](README.md)，理论见 [调研报告.md](调研报告.md)，四测发现见 [四测汇报.md](四测汇报.md)。

---

## 1. 概述

控制变量法对比**三种**二维相位恢复方法（HIO / UNet+sigmoid+置0 / UNet+tanh+HIO）。三者共享实空间约束，差异在像空间处理、support 外策略、输出层。**评估前配准**（消除平凡歧义）。三层职责：工具层（`utils`）、方法层（`hio` / `unet_pr`）、编排层（`main`）。

---

## 2. 模块划分

| 模块 | 职责 | 主要导出 |
|---|---|---|
| `utils.py`（工具层） | 共用基础设施：预处理、可微 FFT/IFFT、随机相位、shrinkwrap、直方图匹配、**配准**、评估 | `load_and_preprocess`、`fft_amp_phase`、`ifft_real`、`make_random_phase`、`init_density`、`shrinkwrap_support`、`estimate_reference_histogram`、`histogram_match`、**`register_to_gt`**、`evaluate_all`、`device` |
| `hio.py`（方法层·实验1） | 严格 HIO：像空间硬替换振幅 + 实空间 HIO 反馈 + HM | `run_hio` |
| `unet_pr.py`（方法层·实验2/3） | 未训练 UNet：UNet（输出层 sigmoid/tanh 可选）+ 像空间振幅 MSE 反传（support 外置0/HIO反馈可选） | `UNet`、`run_unet` |
| `main.py`（编排层） | 读图 → 三实验 → 各出图 + 横向对比 | `main` |

`smoke_test.py` / `_probe_unet.py` 是验证/探测脚本，非核心架构。

---

## 3. 依赖方向

```
main.py
  ├── hio.py ──┐
  └── unet_pr.py ──┤
                   └── utils.py
```

单向依赖、无环。`hio` / `unet_pr` 互不引用（控制变量的体现：三条平行路径，只在 `main` 汇合对比）。

---

## 4. 数据流

```
567.png
  │ load_and_preprocess（灰度→扩2倍画布→极性翻转→pad32）
  ▼
rho_work（暗背景，[1,1,H,W]）
  │
  ├── fft_amp_phase ──► amp_orig（唯一实验数据：去相位的振幅）
  ├── make_random_phase(seed=42) + init_density ──► rho_init（三实验共享）
  ├── shrinkwrap_support ──► support_gt（评估真值）
  └── estimate_reference_histogram ──► ref_edges（HM 参考）

       amp_orig, rho_init, ref_edges, rho_work, support_gt
       ┌────────────────────┬─────────────────────┐
       ▼                    ▼                     ▼
   run_hio（实验1）      run_unet(实验2)       run_unet(实验3)
   硬替换+HIO反馈        sigmoid+置0           tanh+HIO反馈
       │                    │                     │
       ▼                    ▼                     ▼
   末轮 rho + history   末轮 rho + history    末轮 rho + history
       └────────────────────┴─────────────────────┘
                            ▼
        dump_experiment：register_to_gt（配准出图）+ evaluate_all（配准评估）
        + 存 history.csv（全程）/ metrics.csv（末轮）/ state.pt（张量）
                            ▼
                  comparison.png / comparison.csv（三实验末轮对比）
```

---

## 5. 关键设计决策

### 5.1 控制变量（三实验）
共享实空间约束 + 初始条件（同 ρ_init、同 ref_edges、同 support_gt、UNet 同权重 seed）。唯一变量：像空间（硬替换 vs 振幅 MSE）+ support 外（置0 vs HIO 反馈）+ 输出层（sigmoid vs tanh）。

### 5.2 极性翻转
`ρ_work = bg_val − img`。`FFT(c−f)` 与 `FFT(f)` 非直流振幅相同，振幅约束不变，控制变量不破。

### 5.3 保留直流
去直流导致 HIO 不收敛。`fft_amp_phase` 默认 `keep_dc=True`。

### 5.4 输出层参数化（新）
`OutConv` 激活可选：`sigmoid`（[0,1] 恒正，配 support 外置0）/ `tanh`（[−1,1] 可负，配 HIO 反馈）。

### 5.5 support 外策略（新）
`run_unet(use_hio_feedback)`：`False` → `rho_next = raw×support`（置0，天然有界）；`True` → `rho_next = γ·current_input − β·raw`（**relaxed HIO**，γ<1 防累加发散）。实验1 HIO 用同公式（γ·ρ_k − β·ρ′）。γ=`GAMMA`（main，默认 0.9）。详见 [调研报告.md](调研报告.md) §2.3。

### 5.6 评估前配准（★ 四测新增）
相位恢复的平凡歧义（平移 / 共轭反转 / 全局符号）使未配准的 Δφ/amp_cc 被平移严重污染。`evaluate_all(align_to_gt=True)` 先 `register_to_gt`（歧义群枚举 + FFT 互相关求平移）再算指标。**配准只作用于评估副本，不进迭代**（迭代主线 `raw→rho_c→loss→rho_next→HM→support` 不变）。来源见 [调研报告.md](调研报告.md) §5.1/§5.2 + [28-30]。

### 5.7 评估前投影
`rho_eval = clamp_min(0, rho_aligned) × support_pred`，消除 support 外反馈值干扰，甲乙一致。

### 5.8 取末轮（新）
`best_rho` / `best_point` 取末轮（收敛态），不取最优瞬间（cherry-pick，可能选不稳定峰值）。

### 5.9 轮次公平（新）
同 wall-clock（HIO 10000×0.101 s ≈ UNet 3000×0.34 s ≈ 17 min），非同轮次（单轮计算量不可比）。

### 5.10 数值约定
图像全程 [0,1]；振幅 `amp/amp.max()` 归一化算 loss；张量 `[1,1,H,W]` float32；pad 到 32 的倍数（UNet 5 级下采样）。

---

## 6. 技术栈

| 用途 | 选型 |
|---|---|
| 深度学习 / 可微 FFT | PyTorch（`torch.fft.fft2/ifft2`） |
| 数值计算 | numpy |
| 图像 IO | Pillow |
| 可视化 | matplotlib（SimHei 中文） |
| 评估 | 全部自实现（PSNR/SSIM/Pearson CC/amp_cc/Δφ/IoU + 配准），无 skimage/cv2 |

运行环境：`D:\anaconda3\envs\use\python.exe`。

---

## 7. 目录结构

```
对比/
├── 567.png              输入图
├── utils.py             工具层（含配准 register_to_gt）
├── hio.py               实验1：严格 HIO
├── unet_pr.py           实验2/3：UNet（输出层/support外可选）
├── main.py              编排层：三实验对比入口
├── smoke_test.py        初测验证脚本
├── _probe_unet.py       二测 UNet 探测脚本
├── README.md / ARCHITECTURE.md / CLAUDE.md
├── 调研报告.md          文献调研 + 30 篇参考文献
├── 实现方案.md          函数级实现方案
├── 初测/二测/四测汇报.md
├── 实验猜想与结论.md    猜想与结论（四测后修订）
├── .gitignore
└── results/run_<时间戳>/  产物（gitignore）
```

---

## 8. 已知问题（四测发现，详见 [实验猜想与结论.md](实验猜想与结论.md)）

1. **HIO 不收敛**（C2）：support 鸡生蛋，10000 轮仍不收敛。
2. **support 自锁**（C7）：实验2 置0 → support 焊死，形态固化。
3. **尺度发散**（C8）：实验3 tanh+HIO → rho 爆炸（psnr −38）。
4. **CUDA 非确定性**（C6）：UNet 不可复现，最后阶段需多次跑取统计。
5. **amp_cc 不可靠**（C9）：和形态脱钩，横向比较无区分力。

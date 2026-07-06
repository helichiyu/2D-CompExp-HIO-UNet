# 架构说明

> 七测（2026-07）后更新：大规模重复取统计（10 组 × 5 实验 = 50 个，HIO 5000 轮 ×3 + UNet 1500 轮 ×2）。核心：三方法命中率统计（C12）、sigmoid 相变质量最好（C13）、HIO 能恢复但长迭代必出周期斜线（C2/C14，FFT 卷绕）。项目背景见 [README.md](README.md)，理论见 [调研报告.md](调研报告.md)，七测发现见 [七测汇报.md](七测汇报.md)，九测发现见 [九测汇报.md](九测汇报.md)。

> **九测（2026-07）后更新**：矩阵扩 5 法（HIO/RAAR/DM + tanh/sigmoid）。**RAAR 反射框架破 C14（C15）**——5/5 稳定 ssim 0.95、背景干净；DM β=1.1 排除；γ=0.7 抑制 HIO 斜线但治不了震荡；tanh/sigmoid 中心好但背景斑杂（C16）。

> **十测（2026-07）后更新**：矩阵收 4 法（HIO / tanh_full / RAAR / unet_raar），核心对照 P_M 实现（raar 硬替换 vs unet_raar UNet 软约束），轮次对齐 2000。**unet_raar 命题失败（C18）**——UNet 软约束替代 P_M 引入不稳定性（5 组 ssim 0.619–0.938，组4 1700 轮相位发散突崩）、中心未超 RAAR；RAAR 对 P_M 精确性敏感，近似投影不能替代硬替换。**C16 归因到 loss 作用域（C17）**——全图 loss 背景干净但分摊中心梯度。γ=0.8 折中失败。详见 [十测汇报.md](十测汇报.md)。

> **十一测（2026-07）后更新**：4 法矩阵 + **Yoshida 2024 自适应 σ**（support 剧变大 σ、稳定小 σ 锁定，新增 `AdaptiveSigma` 类 + 4 个 `run_*` 加 `use_adaptive_sigma`）。**自适应 σ 消除 unet_raar 突崩（C19）**——4/4 无中途暴跌（vs 十测 1/5 突崩）；但"防崩"≠"提质"，unet_raar 仍不如 RAAR（ssim 0.77–0.93 vs 0.94–0.96）。RAAR 自适应 σ 下仍 4/4 稳（C15）；HIO 10000 轮+自适应 σ 仍 4/4 崩（C14 再证）。详见 [十一测汇报.md](十一测汇报.md)。

---

## 1. 概述

控制变量法对比**三种**二维相位恢复方法（实验1 HIO / 实验2 UNet+tanh+HIO / 实验3 UNet+sigmoid+置0，均开 HM），并**多次独立重复取统计**（七测：10 组 × 5 实验 = 50 个；HIO 5000 轮 ×3 + UNet 1500 轮 ×2）。三者共享实空间约束，差异在像空间处理、support 外策略、输出层。**评估前配准**（消除平凡歧义）。三层职责：工具层（`utils`）、方法层（`hio` / `unet_pr`）、编排层（`main`）。

---

## 2. 模块划分

| 模块 | 职责 | 主要导出 |
|---|---|---|
| `utils.py`（工具层） | 共用基础设施：预处理、可微 FFT/IFFT、随机相位、shrinkwrap、直方图匹配、**实空间投影 `proj_S`**、**配准**、评估 | `load_and_preprocess`、`fft_amp_phase`、`ifft_real`、`make_random_phase`、`init_density`、`shrinkwrap_support`、`estimate_reference_histogram`、`histogram_match`、**`proj_S`**、**`register_to_gt`**、`evaluate_all`、`device` |
| `hio.py`（方法层·HIO） | 严格 HIO：像空间硬替换振幅 + 实空间 relaxed HIO 反馈 + HM | `run_hio` |
| `raar.py`（方法层·RAAR，九测） | RAAR 反射迭代：两空间反射 R=2P−I + β 松弛（破 C14，C15） | `run_raar` |
| `dm.py`（方法层·DM，九测） | Difference Map 投影差迭代（β=1.1，已排除） | `run_dm` |
| `unet_pr.py`（方法层·UNet） | 未训练 UNet：UNet（输出层 sigmoid/tanh 可选）+ 像空间振幅 MSE 反传（`loss_scope` support/full 验 C16）+ `run_unet_raar`（RAAR 骨架 + P_M 换 UNet 软约束，十测） | `UNet`、`run_unet`、`run_unet_raar` |
| `main.py`（编排层） | 读图 → 10 组 × 5 实验（HIO×3 + UNet×2）→ 各出图 + 每组并排/对比 + summary + 断点续跑 | `main` |

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
  ├── make_random_phase() + init_density ──► rho_init（三实验共享；种子不固定，C6）
  ├── shrinkwrap_support ──► support_gt（评估真值）
  └── estimate_reference_histogram ──► ref_edges（HM 参考）

       amp_orig, rho_init, ref_edges, rho_work, support_gt
       ┌────────────────────┬─────────────────────┐
       ▼                    ▼                     ▼
   run_hio（实验1×3）    run_unet(实验2)       run_unet(实验3)
   硬替换+relaxed HIO    tanh+HIO反馈          sigmoid+置0
   3 个不同 phase seed
       │                    │                     │
       ▼                    ▼                     ▼
   3 个 rho+history     末轮 rho + history    末轮 rho + history
   （取末轮ssim最高代表）
       └────────────────────┴─────────────────────┘
                            ▼
        每个实验 dump_experiment：register_to_gt（配准出图）+ evaluate_all（配准评估）
        + hio_3runs.png（HIO 3 次并排挑收敛）+ history.csv（全程）/ metrics.csv（末轮）/ state.pt
        + 每组 comparison_gXx.png/csv（组内三方法末轮对比）
                            ▼
        外层 10 组循环（progress.json 记每组完成，支持断点续跑）
                            ▼
                  summary.csv（50 实验末轮指标汇总）
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

### 5.9 轮次、重复与断点续跑（七测改）
一组 = HIO 5000 轮 ×3 + UNet 1500 轮 ×2；外层 10 组独立重复取统计（C12）。组间差异来自随机性（phase seed 不固定 + CUDA 非确定性 C6），即 10 次独立样本。每组 HIO 取末轮 ssim 最高者作该组 comparison 代表，另出 `hio_3runs.png` 供看图挑收敛。**断点续跑**：`progress.json` 记 `done_groups`，续跑时跳过已完成组、校验 iter 配置一致。

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
├── main.py              编排层：10 组 × 5 实验（HIO×3+UNet×2）对比入口 + 断点续跑
├── smoke_test.py        初测验证脚本
├── _probe_unet.py       二测 UNet 探测脚本
├── README.md / ARCHITECTURE.md / CLAUDE.md
├── 调研报告.md          文献调研 + 30 篇参考文献
├── 实现方案.md          函数级实现方案
├── 各测汇报.md（初测/二测/四测/五测/六测/七测）
├── 各测计划.md（三测/六测/七测）
├── 实验猜想与结论.md    猜想与结论（七测后修订）
├── .gitignore
└── results/run_<时间戳>/  产物（gitignore，含 group01..10/ + summary.csv）
```

---

## 8. 已知问题（九测状态，详见 [实验猜想与结论.md](实验猜想与结论.md)）

1. **三方法概率恢复，已有命中率**（C12 统计性）：HIO 0/30、tanh+HIO 7/10（指标虚高）、sigmoid 2/10（质量最好）。
2. **HIO 能恢复但长迭代必出周期斜线**（C2/C14 大推翻）：5000 轮 × 30 次每张中间有物体但被斜线淹没；主因 = HIO 负反馈震荡/stagnation（sigmoid 无反馈丝滑为证），FFT 卷绕次要（padding 充足）。
3. **sigmoid 相变质量最佳**（C13）：曲线丝滑无震荡，UNet 先验抑制卷绕条纹；但相变仅 20% 概率。
4. **~~support 自锁~~（C7 已推翻）**：sigmoid 慢相变收敛，非死锁。
5. **尺度发散已修复**（C8）：relaxed HIO 防 −38 爆炸；末轮尺度漂移方差大（tanh+HIO psnr 七测 −11.8 ~ +14.3）。
6. **twin image / 十字星**（C11，待查）：七测 tanh+HIO 70% 有轮廓但未完全恢复，典型 twin image 未再见。
7. **CUDA 非确定性**（C6）：UNet 不可复现，多次跑取统计。
8. **指标虚高**（R5 强化）：iou/amp_cc 高的 tanh+HIO 也只“一点轮廓”；曲线丝滑才是真收敛信号。
9. **RAAR 破 C14（C15，九测）**：反射框架 5/5 稳定 ssim 0.95 + 背景干净，项目首次拿到不靠概率的稳定高质量恢复；~~但中心物体质量不如 UNet~~（误读误差图，R13，实际所有方法中心都只有轮廓；RAAR 真正优势是背景≈0 + 稳定）。
10. **DM β=1.1 排除（九测）**：5/5 ssim 0.10，support+positivity+HM setup 下不工作。
11. **UNet 路线背景斑杂（C16，九测）**：~~tanh/sigmoid 中心物体好~~（误读误差图，R13，实际无效），但背景杂项多（方向对）；RAAR 干净背景反衬。**十测归因到 loss 作用域（C17）：全图 loss 背景干净但分摊中心梯度。**
12. **unet_raar 命题失败（C18，十测）**：UNet 软约束替代 P_M 引入不稳定性（5 组 ssim 0.619–0.938，组4 1700 轮相位发散突崩）、中心未超 RAAR；RAAR 对 P_M 精确性敏感，近似投影不能替代硬替换。
13. **γ=0.8 折中失败（十测）**：HIO 5/5 ssim 0.215–0.292，与 γ=0.7 无改善，HIO 调参无救，已被 RAAR 全面碾压。
14. **自适应 σ 消除 unet_raar 突崩（C19，十一测）**：Yoshida 自适应 σ 4/4 无中途突崩（vs 十测固定 σ 1/5）；但 unet_raar 仍不如 RAAR（防崩≠提质，ssim 0.77–0.93 vs 0.94–0.96）。RAAR 自适应 σ 下仍 4/4 稳（C15）、HIO 10000 轮+自适应 σ 仍 4/4 崩（C14 三证）。

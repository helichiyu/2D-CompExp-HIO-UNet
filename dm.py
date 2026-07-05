"""
dm.py —— Difference Map 相位恢复迭代

Elser 2003 投影差框架。与 HIO 同层（纯迭代），复用 utils 原子操作。
公式（调研报告 §2.5）：
  D(x) = x + β·[π1(x+γ2·(π2(x)−x)) − π2(x+γ1·(π1(x)−x))]
  π1 = P_S（实空间投影），π2 = P_M（像空间投影）

参数：β=1.1（Elser 推荐 β≈1.1–1.2 创造"非退化 wandering 不动点"以逃 stagnation；
  标准 β=1、γ₁=−1、γ₂=±1/β 时退化为 Douglas–Rachford ≡ HIO，Bauschke 2002 [31]
  凸优化证明、Fienup 2013 与 Wikipedia 确认——故取 β=1.1 脱离退化点）。
  γ₁=−1/β、γ₂=1/β（Wikipedia 标准 convention；γ 符号文献有 ±1/β 两种写法，
  实现时按"β=1.1 下 DM 结果应明显不同于 HIO"反验，见 九测计划.md §7.1）。
计划见 九测计划.md §3.4。
"""

import time
import torch

from utils import (fft_amp_phase, ifft_real, shrinkwrap_support, sigma_schedule,
                   histogram_match, evaluate_all)


def _proj_S(x, support):
    """实空间投影 P_S：support 外 0、内 clamp_min(0)。幂等。"""
    return torch.where(support > 0.5, torch.clamp_min(x, 0), torch.zeros_like(x))


def _proj_M(x, amp_orig):
    """像空间投影 P_M：用 amp_orig 替换 FFT(x) 振幅、保留相位，IFFT 实部。幂等。"""
    _, phase = fft_amp_phase(x)
    return ifft_real(amp_orig, phase)


def run_dm(amp_orig, rho_init, ref_edges, gt, support_gt,
           max_iter=3000, beta=1.1, gamma1=None, gamma2=None,
           sigma0=3.0, sigma_end=1.0, sigma_interval=20, sigma_total=200,
           eval_every=20, do_hm=True, hm_start=0,
           fixed_support=None, init_support=None, warmup=0, align_eval=True):
    """
    Difference Map 迭代。签名对齐 run_hio，额外 gamma1/gamma2（默认按 β 推标准关系）。
    support/HM/评估流水线沿用 HIO，只换迭代核。返回 (rho_end, history)，取末轮。
    """
    # γ 默认按 Wikipedia 标准 convention：γ1=−1/β、γ2=1/β（β=1 时退化为 HIO）
    if gamma1 is None:
        gamma1 = -1.0 / beta
    if gamma2 is None:
        gamma2 = 1.0 / beta

    _, phase_orig = fft_amp_phase(gt)

    x = rho_init.detach().clone()
    if fixed_support is not None:
        support = fixed_support
    elif init_support is not None:
        support = init_support
    else:
        support = shrinkwrap_support(x, sigma0)

    metric_keys = ["psnr", "ssim", "pearson_cc", "amp_cc", "phase_err", "support_iou"]
    history = {"iter": [], "amp_residual": []}
    for k in metric_keys:
        history[k] = []
    start = time.time()
    for it in range(max_iter):
        # === 动态 support（warmup 后周期更新，σ 线性衰减）===
        if fixed_support is None and it >= warmup and it % sigma_interval == 0:
            sigma = sigma_schedule(it - warmup, sigma0, sigma_end, sigma_total)
            support = shrinkwrap_support(x, sigma)

        # === Difference Map 迭代核 ===
        p1 = _proj_S(x, support)                       # π1(x)
        p2 = _proj_M(x, amp_orig)                      # π2(x)
        psi1 = x + gamma1 * (p1 - x)                   # x + γ1·(π1(x) − x)
        p2_psi1 = _proj_M(psi1, amp_orig)              # π2(x + γ1·(π1(x) − x))
        psi2 = x + gamma2 * (p2 - x)                   # x + γ2·(π2(x) − x)
        p1_psi2 = _proj_S(psi2, support)               # π1(x + γ2·(π2(x) − x))
        x_next = x + beta * (p1_psi2 - p2_psi1)        # x + β·[π1(ψ2) − π2(ψ1)]

        # === 直方图匹配（迭代核后，support 内，hm_start 后开启）===
        if do_hm and it >= hm_start:
            x_next = histogram_match(x_next, support, ref_edges)

        # === 评估 ===
        if it % eval_every == 0:
            m = evaluate_all(x_next, gt, amp_orig, phase_orig, support_gt, align_to_gt=align_eval)
            amp_res = ((fft_amp_phase(x_next)[0] - amp_orig).norm() / (amp_orig.norm() + 1e-12)).item()
            history["iter"].append(it)
            history["amp_residual"].append(amp_res)
            for k in metric_keys:
                history[k].append(m[k])
            elapsed = time.time() - start
            print(f"[DM] {it}/{max_iter} | psnr {m['psnr']:.2f} | ssim {m['ssim']:.3f} | "
                  f"amp_cc {m['amp_cc']:.3f} | Δφ {m['phase_err']:.3f} | iou {m['support_iou']:.3f} | "
                  f"{elapsed:.1f}s")

        x = x_next

    return x, history

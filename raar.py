"""
raar.py —— RAAR（Relaxed Averaged Alternating Reflections）相位恢复迭代

Luke 2005 反射算子框架。与 HIO 同层（纯迭代），复用 utils 原子操作，
迭代核替换为反射：两空间（实 S + 像 M）都用反射 R=2P−I + β 松弛平均，
替换 HIO 的"投影+反馈"。反射无累加项 → 理论上避 HIO 负反馈震荡（C14）。

公式（调研报告 §2.5）：
  x_{k+1} = (β/2)·(R_M(R_S(x_k)) + x_k) + (1−β)·R_S(x_k)，R = 2P − I
  P_S（实空间投影）：support 外 = 0、support 内 = clamp_min(0)
  P_M（像空间投影）：FFT(ρ) → 用 amp_orig 替换振幅、保留相位 → IFFT 实部

参数 β=0.7（Luke 2005 给 β∈(0,1)，CDI 实践常用 0.7–0.9；取 0.7 与项目 HIO 的 β=0.7
一致好对照）。relaxed HIO（γ·ρ−β·ρ′）是其工程简化，本测上严格 RAAR。
计划见 九测计划.md §3.3。
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


def run_raar(amp_orig, rho_init, ref_edges, gt, support_gt,
             max_iter=3000, beta=0.7,
             sigma0=3.0, sigma_end=1.0, sigma_interval=20, sigma_total=200,
             eval_every=20, do_hm=True, hm_start=0,
             fixed_support=None, init_support=None, warmup=0, align_eval=True):
    """
    RAAR 迭代。签名对齐 run_hio（去掉 gamma/mode/er_* —— RAAR 反射无累加项、
    无需松弛、不做 ER 交替）。support/HM/评估流水线沿用 HIO，只换迭代核。
    返回 (rho_end, history)，取末轮（收敛态）。
    """
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

        # === RAAR 迭代核 ===
        r_s = 2 * _proj_S(x, support) - x                       # R_S(x) = 2·P_S(x) − x
        r_m_rs = 2 * _proj_M(r_s, amp_orig) - r_s               # R_M(R_S(x)) = 2·P_M(R_S(x)) − R_S(x)
        x_next = (beta / 2) * (r_m_rs + x) + (1 - beta) * r_s   # (β/2)(R_M(R_S)+x) + (1−β)·R_S

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
            print(f"[RAAR] {it}/{max_iter} | psnr {m['psnr']:.2f} | ssim {m['ssim']:.3f} | "
                  f"amp_cc {m['amp_cc']:.3f} | Δφ {m['phase_err']:.3f} | iou {m['support_iou']:.3f} | "
                  f"{elapsed:.1f}s")

        x = x_next

    return x, history

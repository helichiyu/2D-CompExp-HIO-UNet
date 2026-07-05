"""
hio.py —— 甲方：传统严格 HIO 相位恢复迭代

像空间：硬替换振幅（用 |A_orig| 换掉新振幅，保留新相位）。
实空间：正值 + 实数约束（HIO 公式内置）+ 直方图匹配 + shrinkwrap 支撑域。

support 策略（解决"鸡生蛋"：support 要 ρ_k 成形才准，ρ_k 要 support 对才成形）：
  - warmup 阶段用 init_support（通常全图=positivity）让 ρ_k 粗成形；
  - warmup 后周期性 shrinkwrap 动态收紧 support（不偷看原图）。
HM 在 support 内做；早期 support 宽松时若开 HM 会扰动过大，可由 hm_start 控制开启时机。
"""

import time
import torch

from utils import (fft_amp_phase, ifft_real, shrinkwrap_support, sigma_schedule,
                   histogram_match, evaluate_all)


def run_hio(amp_orig, rho_init, ref_edges, gt, support_gt,
            max_iter=3000, beta=0.7,
            sigma0=3.0, sigma_end=1.0, sigma_interval=20, sigma_total=200,
            eval_every=20, do_hm=True, hm_start=0,
            fixed_support=None, init_support=None, warmup=0, align_eval=True, gamma=0.9,
            mode='hio', er_every=20, er_len=5):
    """
    严格 HIO 迭代。
      amp_orig:       原始振幅 [1,1,H,W]
      rho_init:       初始实空间密度 ρ_0
      ref_edges:      HM 参考直方图分箱边界
      gt:             真值密度（评估用）
      support_gt:     真值支撑域（评估用）
      do_hm:          是否做直方图匹配
      hm_start:       HM 从第几轮开始（早期 support 宽松时 HM 扰动大，可延后）
      fixed_support:  非 None 时全程用此支撑域（调试用）
      init_support:   warmup 阶段的初始支撑域（None 则用 shrinkwrap(rho_init)）
      warmup:         前 warmup 轮固定 init_support，之后动态 shrinkwrap
      mode:           'hio'（默认，纯 relaxed HIO）/ 'er_hio'（ER+HIO 交替，⚠️ 八测⑤废弃——周期穿插
                      ER 侵蚀物体，见 八测阶段汇报.md §5；保留参数供参考，不再使用）
      er_every:       mode='er_hio' 时的 HIO 段长度（轮）
      er_len:         mode='er_hio' 时的 ER 段长度（轮）
    返回 (best_rho, history)。
    """
    _, phase_orig = fft_amp_phase(gt)

    rho_k = rho_init.detach().clone()
    if fixed_support is not None:
        support = fixed_support
    elif init_support is not None:
        support = init_support
    else:
        support = shrinkwrap_support(rho_k, sigma0)

    metric_keys = ["psnr", "ssim", "pearson_cc", "amp_cc", "phase_err", "support_iou"]
    history = {"iter": [], "amp_residual": []}
    for k in metric_keys:
        history[k] = []
    start = time.time()
    for it in range(max_iter):
        # === 像空间：FFT → 振幅硬替换 → IFFT ===
        _, phase_k = fft_amp_phase(rho_k)
        rho_prime = ifft_real(amp_orig, phase_k)  # ρ'_k

        # === 动态 support 更新（warmup 后周期更新，σ 线性衰减）===
        if fixed_support is None and it >= warmup and it % sigma_interval == 0:
            sigma = sigma_schedule(it - warmup, sigma0, sigma_end, sigma_total)
            support = shrinkwrap_support(rho_k, sigma)

        # === 实空间：按 mode 选 ER / relaxed HIO ===
        keep = (support > 0.5) & (rho_prime >= 0)
        in_er = (mode == 'er_hio' and er_len > 0 and er_every > 0
                 and (it % (er_every + er_len)) >= er_every)
        if in_er:
            # ER 段：违反约束的像素置 0（硬投影，稳定；穿插在 HIO 间收敛局部，Latychevskaia §3.4）
            rho_next = torch.where(keep, rho_prime, torch.zeros_like(rho_prime))
        else:
            # relaxed HIO：support 外 γ·ρ_k − β·ρ′（γ<1 防累加发散）
            rho_next = torch.where(keep, rho_prime, gamma * rho_k - beta * rho_prime)

        # === 实空间：直方图匹配（只在 support 内，hm_start 后开启）===
        if do_hm and it >= hm_start:
            rho_next = histogram_match(rho_next, support, ref_edges)

        # === 评估 ===
        if it % eval_every == 0:
            m = evaluate_all(rho_next, gt, amp_orig, phase_orig, support_gt, align_to_gt=align_eval)
            amp_res = ((fft_amp_phase(rho_next)[0] - amp_orig).norm() / (amp_orig.norm() + 1e-12)).item()
            history["iter"].append(it)
            history["amp_residual"].append(amp_res)
            for k in metric_keys:
                history[k].append(m[k])
            elapsed = time.time() - start
            print(f"[HIO] {it}/{max_iter} | psnr {m['psnr']:.2f} | ssim {m['ssim']:.3f} | "
                  f"amp_cc {m['amp_cc']:.3f} | Δφ {m['phase_err']:.3f} | iou {m['support_iou']:.3f} | "
                  f"{elapsed:.1f}s")

        rho_k = rho_next

    return rho_k, history               # 取末轮（收敛态），不取最优瞬间

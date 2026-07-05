"""
unet_raar_probe.py —— 十测单跑验证：纯 RAAR 对照 + unet_raar + tanh_full（全图 loss）

同 rho_init、同轮次跑三法，各自 dump 全套 + 末轮并排 compare.png + 末轮指标表。验证：
  ① unet_raar 能跑通、数值不爆炸（头号风险）；
  ② unet_raar 中心物体是否比纯 RAAR 清晰 + 背景是否干净（C15+C16 互补命题）；
  ③ tanh_full（全图 loss）背景是否比九测 tanh+HIO（support 内 loss）干净（C16 根源猜想）。

用法：
  python unet_raar_probe.py [MAX_ITER]
  python unet_raar_probe.py 200        # 短测验证逻辑 + 数值不爆炸
  python unet_raar_probe.py 1500       # 默认 1500 轮，看趋势/看图

输出 results/unet_raar_<时间戳>/：raar/ + unet_raar/ + tanh_full/ 各全套 + compare.png
"""

import os
import sys
from datetime import datetime

import matplotlib.pyplot as plt

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from utils import (load_and_preprocess, fft_amp_phase, make_random_phase, init_density,
                   shrinkwrap_support, estimate_reference_histogram, register_to_gt, unpad)
from raar import run_raar
from unet_pr import run_unet, run_unet_raar
from main import dump_experiment, to_visual

plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

SIGMA0 = 3.0


def make_run_dir():
    run_dir = os.path.join('results', f"unet_raar_{datetime.now().strftime('%Y%m%d_%H%M%S')}")
    os.makedirs(run_dir, exist_ok=True)
    return run_dir


def plot_compare(gt_vis, runs, rho_work, bg_val, pad_info, save_path):
    """原图 + 各方法末轮并排。runs: [(label, best, hist), ...]"""
    n = len(runs)
    fig, axes = plt.subplots(1, n + 1, figsize=(5 * (n + 1), 5))
    axes[0].imshow(gt_vis, cmap='gray', vmin=0, vmax=1)
    axes[0].set_title('原图（真值）', fontsize=14); axes[0].axis('off')
    for i, (label, best, hist) in enumerate(runs):
        best_a = register_to_gt(best, rho_work)
        vis = unpad(bg_val - best_a, pad_info).squeeze().detach().cpu().numpy()
        axes[i + 1].imshow(vis, cmap='gray', vmin=0, vmax=1)
        axes[i + 1].set_title(f'{label}\nssim={hist["ssim"][-1]:.3f} psnr={hist["psnr"][-1]:.1f}',
                              fontsize=12)
        axes[i + 1].axis('off')
    plt.tight_layout(); plt.savefig(save_path, dpi=150, bbox_inches='tight'); plt.close()
    print(f"  已保存 compare.png")


def main(max_iter=1500, run_dir=None, phase_seed=None):
    run_dir = run_dir or make_run_dir()
    print(f"UNet+RAAR 单跑验证：{run_dir}/  轮次={max_iter}\n")
    with open(os.path.join(run_dir, 'config.txt'), 'w', encoding='utf-8') as f:
        f.write(f"MAX_ITER={max_iter}\nPHASE_SEED={phase_seed}\n"
                f"RAAR: beta=0.7（对照）\nunet_raar: beta=0.7, tanh\n"
                f"tanh_full: gamma=0.8, beta=0.7, tanh, loss_scope=full\n")

    rho_work, bg_val, pad_info, _, _ = load_and_preprocess('567.png')
    amp_orig, _ = fft_amp_phase(rho_work)
    support_gt = shrinkwrap_support(rho_work, SIGMA0)
    ref_edges = estimate_reference_histogram(rho_work, support_gt, n_bins=300)
    gt_vis = to_visual(rho_work, bg_val, pad_info)

    # 同一个 rho_init（控制变量：三法仅算法核/loss 不同）
    phase = make_random_phase(rho_work.shape, seed=phase_seed)
    rho_init = init_density(amp_orig, phase)

    runs = []
    print(f"\n--- RAAR 对照（β=0.7，{max_iter} 轮）---")
    best_r, hist_r = run_raar(amp_orig, rho_init, ref_edges, rho_work, support_gt,
                              max_iter=max_iter, beta=0.7, sigma0=SIGMA0, eval_every=100)
    dump_experiment('raar', 'RAAR 对照（β=0.7）', best_r, hist_r,
                    gt_vis, rho_work, bg_val, pad_info, support_gt, run_dir,
                    monitor_key='amp_residual', monitor_label='振幅残差')
    runs.append(('RAAR β=0.7', best_r, hist_r))

    print(f"\n--- unet_raar（β=0.7，{max_iter} 轮）---")
    best_u, hist_u = run_unet_raar(amp_orig, rho_init, ref_edges, rho_work, support_gt,
                                   max_iter=max_iter, beta=0.7, sigma0=SIGMA0, eval_every=100,
                                   out_act='tanh')
    dump_experiment('unet_raar', 'UNet+RAAR（β=0.7）', best_u, hist_u,
                    gt_vis, rho_work, bg_val, pad_info, support_gt, run_dir,
                    monitor_key='loss', monitor_label='UNet loss')
    runs.append(('unet_raar', best_u, hist_u))

    print(f"\n--- tanh_full（γ=0.8 全图 loss，{max_iter} 轮）---")
    best_t, hist_t = run_unet(amp_orig, rho_init, ref_edges, rho_work, support_gt,
                              max_iter=max_iter, lr=1e-4, sigma0=SIGMA0, eval_every=100,
                              out_act='tanh', use_hio_feedback=True, beta=0.7, gamma=0.8,
                              loss_scope='full')
    dump_experiment('tanh_full', 'tanh+全图loss（γ=0.8）', best_t, hist_t,
                    gt_vis, rho_work, bg_val, pad_info, support_gt, run_dir,
                    monitor_key='loss', monitor_label='UNet loss')
    runs.append(('tanh_full γ=0.8', best_t, hist_t))

    plot_compare(gt_vis, runs, rho_work, bg_val, pad_info, os.path.join(run_dir, 'compare.png'))

    print(f"\n===== 末轮指标对比 =====")
    print(f"{'方法':>16} | {'psnr':>7} | {'ssim':>6} | {'amp_cc':>7} | {'iou':>5}")
    print("-" * 56)
    for label, _, hist in runs:
        print(f"{label:>16} | {hist['psnr'][-1]:>7.2f} | {hist['ssim'][-1]:>6.3f} | "
              f"{hist['amp_cc'][-1]:>7.3f} | {hist['support_iou'][-1]:>5.3f}")

    print(f"\n完成。结果在 {run_dir}/  看 compare.png + 各 real_space.png")
    print(f"判断：unet_raar 中心是否比 RAAR 清晰 + 背景干净；tanh_full 背景是否比九测 tanh+HIO 干净")


if __name__ == '__main__':
    _max_iter = int(sys.argv[1]) if len(sys.argv) > 1 else 1500
    _run_dir = sys.argv[2] if len(sys.argv) > 2 else None
    main(_max_iter, _run_dir)

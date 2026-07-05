"""
raar_dm_probe.py —— 九测步骤①②：RAAR / DM 单跑验证（同起点 vs HIO 对照）

同 rho_init、同轮次跑【HIO（γ=0.7 对照）+ RAAR（β=0.7）+ DM（β=1.1）】三条，
各自 dump 全套 + 末轮并排 compare.png + 末轮指标表。验证：
  ① RAAR/DM 能跑通、数值不爆炸（步骤1）；
  ② RAAR/DM 的 real_space 周期斜线是否弱于 HIO（C14，步骤2）；
  ③ DM 末轮应明显不同于 HIO（β=1.1 脱离退化点 + γ 符号正确性反验，§7.1）。

用法：
  python raar_dm_probe.py [MAX_ITER]
  python raar_dm_probe.py 200        # 短测验证逻辑 + 数值不爆炸
  python raar_dm_probe.py 1500       # 默认 1500 轮，看趋势

输出 results/raar_dm_<时间戳>/：hio_ctrl/ + raar/ + dm/ 各全套 + compare.png
"""

import os
import sys
from datetime import datetime

import matplotlib.pyplot as plt

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from utils import (load_and_preprocess, fft_amp_phase, make_random_phase, init_density,
                   shrinkwrap_support, estimate_reference_histogram, register_to_gt, unpad)
from hio import run_hio
from raar import run_raar
from dm import run_dm
from main import dump_experiment, to_visual

plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

SIGMA0 = 3.0


def make_run_dir():
    run_dir = os.path.join('results', f"raar_dm_{datetime.now().strftime('%Y%m%d_%H%M%S')}")
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
    print(f"RAAR/DM 单跑验证：{run_dir}/  轮次={max_iter}\n")
    with open(os.path.join(run_dir, 'config.txt'), 'w', encoding='utf-8') as f:
        f.write(f"MAX_ITER={max_iter}\nPHASE_SEED={phase_seed}\n"
                f"HIO: beta=0.7 gamma=0.7（九测 γ=0.7）\nRAAR: beta=0.7\nDM: beta=1.1\n")

    rho_work, bg_val, pad_info, _, _ = load_and_preprocess('567.png')
    amp_orig, _ = fft_amp_phase(rho_work)
    support_gt = shrinkwrap_support(rho_work, SIGMA0)
    ref_edges = estimate_reference_histogram(rho_work, support_gt, n_bins=300)
    gt_vis = to_visual(rho_work, bg_val, pad_info)

    # 同一个 rho_init（控制变量：三法仅迭代核不同）
    phase = make_random_phase(rho_work.shape, seed=phase_seed)
    rho_init = init_density(amp_orig, phase)

    runs = []
    print(f"\n--- HIO 对照（γ=0.7，{max_iter} 轮）---")
    best_h, hist_h = run_hio(amp_orig, rho_init, ref_edges, rho_work, support_gt,
                             max_iter=max_iter, beta=0.7, gamma=0.7, sigma0=SIGMA0,
                             eval_every=100)
    dump_experiment('hio_ctrl', 'HIO 对照（γ=0.7）', best_h, hist_h,
                    gt_vis, rho_work, bg_val, pad_info, support_gt, run_dir,
                    monitor_key='amp_residual', monitor_label='振幅残差')
    runs.append(('HIO γ=0.7', best_h, hist_h))

    print(f"\n--- RAAR（β=0.7，{max_iter} 轮）---")
    best_r, hist_r = run_raar(amp_orig, rho_init, ref_edges, rho_work, support_gt,
                              max_iter=max_iter, beta=0.7, sigma0=SIGMA0, eval_every=100)
    dump_experiment('raar', 'RAAR（β=0.7）', best_r, hist_r,
                    gt_vis, rho_work, bg_val, pad_info, support_gt, run_dir,
                    monitor_key='amp_residual', monitor_label='振幅残差')
    runs.append(('RAAR β=0.7', best_r, hist_r))

    print(f"\n--- DM（β=1.1，{max_iter} 轮）---")
    best_d, hist_d = run_dm(amp_orig, rho_init, ref_edges, rho_work, support_gt,
                            max_iter=max_iter, beta=1.1, sigma0=SIGMA0, eval_every=100)
    dump_experiment('dm', 'DM（β=1.1）', best_d, hist_d,
                    gt_vis, rho_work, bg_val, pad_info, support_gt, run_dir,
                    monitor_key='amp_residual', monitor_label='振幅残差')
    runs.append(('DM β=1.1', best_d, hist_d))

    plot_compare(gt_vis, runs, rho_work, bg_val, pad_info, os.path.join(run_dir, 'compare.png'))

    # 末轮指标对比（DM 反验：应明显不同于 HIO）
    print(f"\n===== 末轮指标对比 =====")
    print(f"{'方法':>14} | {'psnr':>7} | {'ssim':>6} | {'amp_cc':>7} | {'iou':>5}")
    print("-" * 52)
    for label, _, hist in runs:
        print(f"{label:>14} | {hist['psnr'][-1]:>7.2f} | {hist['ssim'][-1]:>6.3f} | "
              f"{hist['amp_cc'][-1]:>7.3f} | {hist['support_iou'][-1]:>5.3f}")

    print(f"\n完成。结果在 {run_dir}/  看 compare.png + 各 real_space.png")
    print(f"DM 反验：DM 末轮应明显不同于 HIO（ssim/psnr 差异大）；若几乎一致→γ 符号或退化问题（§7.1）")


if __name__ == '__main__':
    _max_iter = int(sys.argv[1]) if len(sys.argv) > 1 else 1500
    _run_dir = sys.argv[2] if len(sys.argv) > 2 else None
    main(_max_iter, _run_dir)

"""
er_hio_probe.py —— 八测阶段⑤：ER+HIO 交替单跑（⚠️ 已废弃）

⚠️ 八测阶段⑤测得 20H+5E 周期穿插失败（ER 在 HIO 未建立物体时穿插、硬清零侵蚀物体、
中心图像丢失），方向废弃。保留脚本作参考，不再使用。详见 八测阶段汇报.md §5。

原计划：文献（Latychevskaia [33] §3.4）ER 与 HIO 交替可避免 stagnation / 周期斜线；
本脚本同起点同轮次跑【纯 HIO + ER+HIO】并排出图，看是否改善周期斜线。

用法：
  python er_hio_probe.py [MAX_ITER] [ER_EVERY] [ER_LEN] [RUN_DIR]
  python er_hio_probe.py 3000 20 5      # 默认：3000 轮，每 20 轮 HIO 插 5 轮 ER
  python er_hio_probe.py 200            # 短测验证逻辑

输出 results/er_hio_<时间戳>/：
  hio_ctrl/  纯 HIO 全套（对照）   er_hio/  ER+HIO 全套   compare.png  并排对比
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
from main import dump_experiment, to_visual

plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

SIGMA0 = 3.0


def make_run_dir():
    run_dir = os.path.join('results', f"er_hio_{datetime.now().strftime('%Y%m%d_%H%M%S')}")
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


def main(max_iter=3000, er_every=20, er_len=5, run_dir=None, phase_seed=None):
    run_dir = run_dir or make_run_dir()
    print(f"ER+HIO 交替单跑：{run_dir}/  轮次={max_iter}  节奏={er_every}H+{er_len}E\n")
    with open(os.path.join(run_dir, 'config.txt'), 'w', encoding='utf-8') as f:
        f.write(f"MAX_ITER={max_iter}\nER_EVERY={er_every} ER_LEN={er_len}\n"
                f"PHASE_SEED={phase_seed}\nbeta=0.7 gamma=0.9\n")

    rho_work, bg_val, pad_info, _, _ = load_and_preprocess('567.png')
    amp_orig, _ = fft_amp_phase(rho_work)
    support_gt = shrinkwrap_support(rho_work, SIGMA0)
    ref_edges = estimate_reference_histogram(rho_work, support_gt, n_bins=300)
    gt_vis = to_visual(rho_work, bg_val, pad_info)

    # 同一个 rho_init（控制变量：两条仅 mode 不同）
    phase = make_random_phase(rho_work.shape, seed=phase_seed)
    rho_init = init_density(amp_orig, phase)

    runs = []
    print(f"\n--- 纯 HIO（对照，{max_iter} 轮）---")
    best_h, hist_h = run_hio(amp_orig, rho_init, ref_edges, rho_work, support_gt,
                             max_iter=max_iter, beta=0.7, gamma=0.9, sigma0=SIGMA0,
                             eval_every=100, mode='hio')
    dump_experiment('hio_ctrl', '纯 HIO（对照）', best_h, hist_h,
                    gt_vis, rho_work, bg_val, pad_info, support_gt, run_dir,
                    monitor_key='amp_residual', monitor_label='振幅残差')
    runs.append(('纯 HIO', best_h, hist_h))

    print(f"\n--- ER+HIO 交替（{max_iter} 轮，每 {er_every} HIO 插 {er_len} ER）---")
    best_e, hist_e = run_hio(amp_orig, rho_init, ref_edges, rho_work, support_gt,
                             max_iter=max_iter, beta=0.7, gamma=0.9, sigma0=SIGMA0,
                             eval_every=100, mode='er_hio', er_every=er_every, er_len=er_len)
    dump_experiment('er_hio', f'ER+HIO（{er_every}H+{er_len}E）', best_e, hist_e,
                    gt_vis, rho_work, bg_val, pad_info, support_gt, run_dir,
                    monitor_key='amp_residual', monitor_label='振幅残差')
    runs.append((f'ER+HIO（{er_every}H+{er_len}E）', best_e, hist_e))

    plot_compare(gt_vis, runs, rho_work, bg_val, pad_info, os.path.join(run_dir, 'compare.png'))
    print(f"\n完成。结果在 {run_dir}/  看 hio_ctrl/real_space.png vs er_hio/real_space.png + compare.png")


if __name__ == '__main__':
    _max_iter = int(sys.argv[1]) if len(sys.argv) > 1 else 3000
    _er_every = int(sys.argv[2]) if len(sys.argv) > 2 else 20
    _er_len = int(sys.argv[3]) if len(sys.argv) > 3 else 5
    _run_dir = sys.argv[4] if len(sys.argv) > 4 else None
    main(_max_iter, _er_every, _er_len, _run_dir)

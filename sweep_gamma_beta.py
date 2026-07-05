"""
sweep_gamma_beta.py —— 八测阶段③：tanh+HIO 的 γ/β 扫描

七测 tanh+HIO（γ=0.9, β=0.7）末轮 PSNR 漂移 −11.8~+14.3、曲线震荡、看图只"一点轮廓"（C8 延续）。
扫 γ×β 网格，量化每个组合的【末轮 PSNR】+【稳态震荡幅度】，找震荡成因 + 是否有更稳组合。

控制变量：所有组合共用同一个 ρ_init（phase seed 固定）+ UNet seed 固定 → 差异只来自 γ/β
（CUDA 非确定性 C6 仍在，但起点一致）。

用法：
  python sweep_gamma_beta.py [UNET_ITER] [GAMMAS] [BETAS] [RUN_DIR]
  python sweep_gamma_beta.py                          # 全量 1500 轮，4γ×3β=12
  python sweep_gamma_beta.py 500 0.7,0.9,0.95 0.7     # 短测：3 组合 500 轮

输出 results/sweep_<时间戳>/：
  config.txt / sweep_summary.csv
  g<γ>_b<β>/  history.csv + state.pt（末轮 best_rho 已配准）
  heatmap_psnr.png（末轮 PSNR，越高越好）/ heatmap_std.png（稳态震荡 std，越低越稳）
  curves_psnr.png（各组合 PSNR 收敛曲线叠加）
"""

import csv
import os
import sys
from datetime import datetime

import numpy as np
import torch
import matplotlib.pyplot as plt

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from utils import (load_and_preprocess, fft_amp_phase, make_random_phase, init_density,
                   shrinkwrap_support, estimate_reference_histogram, register_to_gt, device)
from unet_pr import run_unet

plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

SIGMA0 = 3.0
UNET_LR = 1e-4
PHASE_SEED = 42          # 固定相位起点（γ/β 间控制变量；C6 下 UNet 仍不可复现，但相位起点一致）
UNET_SEED = 0            # 固定 UNet 权重起点
DEFAULT_GAMMAS = [0.7, 0.8, 0.9, 0.95]
DEFAULT_BETAS = [0.5, 0.7, 0.9]


def parse_list(s):
    return [float(x) for x in s.split(',') if x.strip()]


def make_run_dir():
    run_dir = os.path.join('results', f"sweep_{datetime.now().strftime('%Y%m%d_%H%M%S')}")
    os.makedirs(run_dir, exist_ok=True)
    return run_dir


def late_metrics(hist):
    """末轮指标 + 稳态震荡幅度（后 2/3 段 PSNR 的 std）+ 全程极差。"""
    psnr = np.array(hist['psnr'], dtype=float)
    late = psnr[max(1, len(psnr) // 3):]
    return {
        'psnr_final': float(psnr[-1]),
        'ssim_final': float(hist['ssim'][-1]),
        'psnr_std_late': float(np.std(late)) if len(late) > 1 else 0.0,
        'psnr_range': float(psnr.max() - psnr.min()),
    }


def save_history(hist, save_path):
    metric_keys = ['psnr', 'ssim', 'pearson_cc', 'amp_cc', 'phase_err', 'support_iou']
    rows = [['iter', 'loss'] + metric_keys]
    for i in range(len(hist['iter'])):
        rows.append([hist['iter'][i], hist['loss'][i]] + [hist[k][i] for k in metric_keys])
    with open(save_path, 'w', newline='', encoding='utf-8-sig') as f:
        csv.writer(f).writerows(rows)


def plot_heatmap(values, gammas, betas, title, save_path, cmap, fmt='{:.2f}'):
    """values[g][b]：行=γ，列=β。缺失格画 '—'。"""
    fig, ax = plt.subplots(figsize=(2.2 * len(betas) + 1.5, 2.2 * len(gammas) + 1))
    data = np.array([[v if v is not None else np.nan for v in row] for row in values], dtype=float)
    im = ax.imshow(data, cmap=cmap, aspect='auto')
    ax.set_xticks(range(len(betas))); ax.set_xticklabels([f'β={b}' for b in betas], fontsize=11)
    ax.set_yticks(range(len(gammas))); ax.set_yticklabels([f'γ={g}' for g in gammas], fontsize=11)
    for i in range(len(gammas)):
        for j in range(len(betas)):
            v = values[i][j]
            ax.text(j, i, fmt.format(v) if v is not None else '—',
                    ha='center', va='center', fontsize=11, color='black')
    ax.set_title(title, fontsize=13)
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    plt.tight_layout(); plt.savefig(save_path, dpi=150, bbox_inches='tight'); plt.close()
    print(f"  已保存 {os.path.basename(save_path)}")


def plot_curves(curves_data, save_path):
    """curves_data: [(label, iters, psnr), ...]"""
    fig, ax = plt.subplots(figsize=(10, 6))
    for label, iters, psnr in curves_data:
        ax.plot(iters, psnr, lw=1.5, alpha=0.8, label=label)
    ax.set_xlabel('迭代轮数'); ax.set_ylabel('PSNR (dB)')
    ax.set_title('各 γ/β 组合 PSNR 收敛曲线（tanh+HIO）', fontsize=13)
    ax.grid(True, alpha=0.3); ax.legend(loc='best', fontsize=8, ncol=2)
    plt.tight_layout(); plt.savefig(save_path, dpi=150, bbox_inches='tight'); plt.close()
    print(f"  已保存 curves_psnr.png")


def main(run_dir=None, unet_iter=1500, gammas=None, betas=None, n_repeats=1):
    gammas = gammas or DEFAULT_GAMMAS
    betas = betas or DEFAULT_BETAS
    run_dir = run_dir or make_run_dir()
    n_runs = len(gammas) * len(betas) * n_repeats
    print(f"γ/β 扫描：{run_dir}/")
    print(f"  γ={gammas}\n  β={betas}\n  轮次={unet_iter} ×{n_repeats} = {n_runs} 个 run\n")

    with open(os.path.join(run_dir, 'config.txt'), 'w', encoding='utf-8') as f:
        f.write(f"UNET_ITER={unet_iter}\nGAMMAS={gammas}\nBETAS={betas}\nN_REPEATS={n_repeats}\n"
                f"UNET_LR={UNET_LR}\nPHASE_SEED={PHASE_SEED}\nUNET_SEED={UNET_SEED}\n"
                f"out_act=tanh, use_hio_feedback=True（tanh+HIO）\n")

    # 共享预处理（控制变量：同图、同 amp_orig、同 ref_edges、同 support_gt）
    rho_work, bg_val, pad_info, _, _ = load_and_preprocess('567.png')
    amp_orig, _ = fft_amp_phase(rho_work)
    support_gt = shrinkwrap_support(rho_work, SIGMA0)
    ref_edges = estimate_reference_histogram(rho_work, support_gt, n_bins=300)
    phase = make_random_phase(rho_work.shape, seed=PHASE_SEED)   # 固定相位起点
    rho_init = init_density(amp_orig, phase)

    rows = [['gamma', 'beta', 'rep', 'psnr_final', 'ssim_final', 'psnr_std_late', 'psnr_range']]
    curves = []
    heat_psnr = [[None] * len(betas) for _ in gammas]
    heat_std = [[None] * len(betas) for _ in gammas]

    for gi, gamma in enumerate(gammas):
        for bi, beta in enumerate(betas):
            for rep in range(n_repeats):
                tag = f"g{gamma}_b{beta}" + (f"_r{rep}" if n_repeats > 1 else "")
                print(f"\n--- {tag}（tanh+HIO，{unet_iter} 轮）---")
                best, hist = run_unet(amp_orig, rho_init, ref_edges, rho_work, support_gt,
                                      max_iter=unet_iter, lr=UNET_LR, sigma0=SIGMA0,
                                      out_act='tanh', use_hio_feedback=True, beta=beta,
                                      gamma=gamma, unet_seed=UNET_SEED, eval_every=20)
                exp_dir = os.path.join(run_dir, tag); os.makedirs(exp_dir, exist_ok=True)
                save_history(hist, os.path.join(exp_dir, 'history.csv'))
                best_a = register_to_gt(best, rho_work)
                torch.save({'best_rho': best_a, 'best_rho_raw': best, 'hist': hist,
                            'gamma': gamma, 'beta': beta}, os.path.join(exp_dir, 'state.pt'))
                m = late_metrics(hist)
                rows.append([gamma, beta, rep, m['psnr_final'], m['ssim_final'],
                             m['psnr_std_late'], m['psnr_range']])
                curves.append((tag, hist['iter'], hist['psnr']))
                if n_repeats == 1:                       # 单次才有单格热图意义
                    heat_psnr[gi][bi] = m['psnr_final']
                    heat_std[gi][bi] = m['psnr_std_late']
                print(f"  [{tag}] psnr_final={m['psnr_final']:.2f} ssim={m['ssim_final']:.3f} "
                      f"std_late={m['psnr_std_late']:.2f} range={m['psnr_range']:.2f}")

    with open(os.path.join(run_dir, 'sweep_summary.csv'), 'w', newline='', encoding='utf-8-sig') as f:
        csv.writer(f).writerows(rows)
    print("\n已保存 sweep_summary.csv")
    if n_repeats == 1:
        plot_heatmap(heat_psnr, gammas, betas, '末轮 PSNR (dB，越高越好)',
                     os.path.join(run_dir, 'heatmap_psnr.png'), cmap='RdYlGn')
        plot_heatmap(heat_std, gammas, betas, '稳态 PSNR 震荡 std (越低越稳)',
                     os.path.join(run_dir, 'heatmap_std.png'), cmap='RdYlGn_r')
    plot_curves(curves, os.path.join(run_dir, 'curves_psnr.png'))
    print(f"\n完成。结果在 {run_dir}/")


if __name__ == '__main__':
    _unet_iter = int(sys.argv[1]) if len(sys.argv) > 1 else 1500
    _gammas = parse_list(sys.argv[2]) if len(sys.argv) > 2 else None
    _betas = parse_list(sys.argv[3]) if len(sys.argv) > 3 else None
    _run_dir = sys.argv[4] if len(sys.argv) > 4 else None
    main(_run_dir, _unet_iter, _gammas, _betas)

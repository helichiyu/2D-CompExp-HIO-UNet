"""
main.py —— 跑甲方 HIO + 乙方 UNet，对比出图 + 存结果

运行：D:\\anaconda3\\envs\\use\\python.exe main.py
结果输出到 results/run_<时间戳>/ 文件夹，含：
  real_space.png  实空间对比（原图/HIO/UNet + 各自误差）
  spectra.png     频域振幅谱对比（log，低频居中）
  support.png     支撑域对比（真值/HIO估/UNet估）
  convergence.png 各指标收敛曲线
  metrics.csv     综合分数最优点的指标对比表
"""

import csv
import os
import sys
from datetime import datetime

import numpy as np
import torch
import matplotlib.pyplot as plt

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")  # 避免 Windows GBK 编码报错

from utils import (load_and_preprocess, fft_amp_phase, make_random_phase, init_density,
                   shrinkwrap_support, estimate_reference_histogram, unpad)
from hio import run_hio
from unet_pr import run_unet

# ===================== 中文显示 =====================
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

# ===================== 超参 =====================
SIGMA0 = 3.0          # shrinkwrap 初始 σ
HIO_ITER = 30000      # 甲方迭代轮数（HIO 每轮快，给足轮次才与 UNet 算力相当）
UNET_ITER = 5000      # 乙方迭代轮数
UNET_LR = 1e-4        # 乙方学习率
PHASE_SEED = 42       # 随机相位种子（甲乙共享，控制变量）
UNET_SEED = 0         # UNet 权重种子（方法内生差异）


def make_run_dir():
    """创建本次运行的结果文件夹 results/run_<时间戳>/，每次运行独立保留。"""
    run_dir = os.path.join('results', f"run_{datetime.now().strftime('%Y%m%d_%H%M%S')}")
    os.makedirs(run_dir, exist_ok=True)
    return run_dir


def to_visual(rho, bg_val, pad_info):
    """暗背景 rho → 视觉白背景（bg_val − rho）→ unpad → numpy [H,W]。"""
    vis = unpad(bg_val - rho, pad_info)
    return vis.squeeze().detach().cpu().numpy()


def plot_real_space(gt_vis, hio_vis, unet_vis, save_path):
    """实空间对比 2×3：第 1 行 原图/HIO/UNet，第 2 行 各自与原图的绝对误差。"""
    imgs = [gt_vis, hio_vis, unet_vis]
    titles = ['原图（真值）', '甲方 HIO', '乙方 UNet']
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    for j, (img, t) in enumerate(zip(imgs, titles)):
        axes[0, j].imshow(img, cmap='gray', vmin=0, vmax=1)
        axes[0, j].set_title(t, fontsize=14)
        axes[0, j].axis('off')
        err = np.abs(img - gt_vis)
        im = axes[1, j].imshow(err, cmap='hot', vmin=0, vmax=max(err.max(), 1e-6))
        axes[1, j].set_title(f'|{t} - 原图|', fontsize=12)
        axes[1, j].axis('off')
        plt.colorbar(im, ax=axes[1, j], fraction=0.046, pad=0.04)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"已保存实空间对比: {save_path}")


def plot_spectra(gt, hio, unet, save_path):
    """频域振幅谱对比（log，fftshift 居中，低频在中心）。gt/hio/unet 为暗背景密度张量 [1,1,H,W]。"""
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    for ax, rho, t in zip(axes, [gt, hio, unet], ['原图振幅谱', 'HIO 振幅谱', 'UNet 振幅谱']):
        amp = torch.fft.fftshift(torch.abs(torch.fft.fft2(rho)))  # 居中
        ax.imshow(np.log1p(amp.squeeze().detach().cpu().numpy()), cmap='magma')
        ax.set_title(t, fontsize=14)
        ax.axis('off')
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"已保存振幅谱对比: {save_path}")


def plot_support(support_gt, support_hio, support_unet, pad_info, save_path):
    """支撑域对比：真值 / HIO 估 / UNet 估（unpad 显示，与原图视觉对齐）。"""
    masks = [support_gt, support_hio, support_unet]
    titles = ['真值 support', 'HIO 估 support', 'UNet 估 support']
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    for ax, m, t in zip(axes, masks, titles):
        vis = unpad(m, pad_info).squeeze().detach().cpu().numpy()
        ax.imshow(vis, cmap='gray')
        ax.set_title(t, fontsize=14)
        ax.axis('off')
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"已保存支撑域对比: {save_path}")


def plot_convergence(hist_hio, hist_unet, save_path):
    """各指标随迭代收敛曲线对比。"""
    metrics = ['psnr', 'ssim', 'amp_cc', 'phase_err', 'support_iou']
    titles = ['PSNR (dB)', 'SSIM', '振幅域 CC', '平均相位误差 (rad)', '支撑域 IoU']
    fig, axes = plt.subplots(1, len(metrics), figsize=(5 * len(metrics), 4))
    for ax, k, t in zip(axes, metrics, titles):
        ax.plot(hist_hio['iter'], hist_hio[k], label='甲方 HIO', color='#2E86AB', lw=2)
        ax.plot(hist_unet['iter'], hist_unet[k], label='乙方 UNet', color='#A23B72', lw=2)
        ax.set_xlabel('迭代轮数')
        ax.set_ylabel(t)
        ax.set_title(t)
        ax.grid(True, alpha=0.3)
        ax.legend()
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"已保存收敛曲线: {save_path}")


def save_metrics_table(hist_hio, hist_unet, save_path):
    """指标对比表：取综合分数（SSIM+振幅CC）最大点的各项指标。"""
    keys = ['psnr', 'ssim', 'pearson_cc', 'amp_cc', 'phase_err', 'support_iou']

    def best_point(hist):
        scores = [s + a for s, a in zip(hist['ssim'], hist['amp_cc'])]
        idx = int(np.argmax(scores))
        return {k: hist[k][idx] for k in keys}

    bh, bu = best_point(hist_hio), best_point(hist_unet)
    rows = [['指标', '甲方 HIO', '乙方 UNet']]
    for k in keys:
        rows.append([k, f"{bh[k]:.4f}", f"{bu[k]:.4f}"])

    with open(save_path, 'w', newline='', encoding='utf-8-sig') as f:
        csv.writer(f).writerows(rows)

    print("\n===== 指标对比（综合分数最优点的各项指标）=====")
    print(f"{'指标':>14} | {'甲方 HIO':>10} | {'乙方 UNet':>10}")
    print("-" * 42)
    for k, vh, vu in rows[1:]:
        print(f"{k:>14} | {vh:>10} | {vu:>10}")
    print(f"\n已保存指标表: {save_path}")


def main():
    run_dir = make_run_dir()
    print(f"本次结果将保存到: {run_dir}/\n")

    # 1. 读图预处理
    print("=" * 60)
    print("加载 567.png ...")
    rho_work, bg_val, pad_info, H0, W0 = load_and_preprocess('567.png')
    Hp, Wp = rho_work.shape[-2], rho_work.shape[-1]
    print(f"原始 {W0}×{H0} → 扩边+pad 后 {Wp}×{Hp}，bg_val={bg_val:.3f}")

    # 2. 振幅（唯一实验数据）+ 初始相位 + 初始密度 + support + 参考直方图
    amp_orig, _ = fft_amp_phase(rho_work)
    phase0 = make_random_phase(rho_work.shape, seed=PHASE_SEED)
    rho_init = init_density(amp_orig, phase0)
    support_gt = shrinkwrap_support(rho_work, SIGMA0)
    ref_edges = estimate_reference_histogram(rho_work, support_gt, n_bins=300)

    # 3. 甲方 HIO（eval_every=100：轮次多，评估间隔放大，收敛曲线点数与 UNet 相当）
    print("\n" + "=" * 60)
    print("甲方：严格 HIO")
    print("=" * 60)
    best_hio, hist_hio = run_hio(amp_orig, rho_init, ref_edges, rho_work, support_gt,
                                 max_iter=HIO_ITER, beta=0.7, sigma0=SIGMA0, eval_every=100)

    # 4. 乙方 UNet
    print("\n" + "=" * 60)
    print("乙方：未训练 UNet")
    print("=" * 60)
    best_unet, hist_unet = run_unet(amp_orig, rho_init, ref_edges, rho_work, support_gt,
                                    max_iter=UNET_ITER, lr=UNET_LR, sigma0=SIGMA0,
                                    unet_seed=UNET_SEED)

    # 5. 可视化数据：翻回视觉极性 + unpad + 估 support（与 evaluate_all 一致用 σ=1.0）
    gt_vis = to_visual(rho_work, bg_val, pad_info)
    hio_vis = to_visual(best_hio, bg_val, pad_info)
    unet_vis = to_visual(best_unet, bg_val, pad_info)
    support_hio = shrinkwrap_support(best_hio, 1.0)
    support_unet = shrinkwrap_support(best_unet, 1.0)

    # 6. 出图 + 存表（全部到 run_dir）
    plot_real_space(gt_vis, hio_vis, unet_vis, os.path.join(run_dir, 'real_space.png'))
    plot_spectra(rho_work, best_hio, best_unet, os.path.join(run_dir, 'spectra.png'))
    plot_support(support_gt, support_hio, support_unet, pad_info, os.path.join(run_dir, 'support.png'))
    plot_convergence(hist_hio, hist_unet, os.path.join(run_dir, 'convergence.png'))
    save_metrics_table(hist_hio, hist_unet, os.path.join(run_dir, 'metrics.csv'))

    print(f"\n完成！所有结果在: {run_dir}/")


if __name__ == '__main__':
    main()

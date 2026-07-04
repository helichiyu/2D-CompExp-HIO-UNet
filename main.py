"""
main.py —— 跑甲方 HIO + 乙方 UNet，对比出图 + 存结果

运行：D:\\anaconda3\\envs\\use\\python.exe main.py
"""

import csv
import sys

import numpy as np
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
HIO_ITER = 3000       # 甲方迭代轮数
UNET_ITER = 5000      # 乙方迭代轮数
UNET_LR = 1e-4        # 乙方学习率
PHASE_SEED = 42       # 随机相位种子（甲乙共享，控制变量）
UNET_SEED = 0         # UNet 权重种子（方法内生差异）


def to_visual(rho, bg_val, pad_info):
    """暗背景 rho → 视觉白背景（bg_val − rho）→ unpad → numpy [H,W]。"""
    vis = unpad(bg_val - rho, pad_info)
    return vis.squeeze().detach().cpu().numpy()


def plot_triple(gt_vis, hio_vis, unet_vis, save_path):
    """原图 / HIO / UNet 三连图。"""
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    for ax, img, title in zip(axes, [gt_vis, hio_vis, unet_vis],
                              ['原图（真值）', '甲方 HIO', '乙方 UNet']):
        ax.imshow(img, cmap='gray', vmin=0, vmax=1)
        ax.set_title(title, fontsize=14)
        ax.axis('off')
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"已保存三连图: {save_path}")


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
    # 1. 读图预处理
    print("=" * 60)
    print("加载 567.png ...")
    rho_work, bg_val, pad_info, H0, W0 = load_and_preprocess('567.png')
    _, _, Hp, Wp = rho_work.shape
    print(f"原始 {W0}×{H0} → 扩边+pad 后 {Wp}×{Hp}，bg_val={bg_val:.3f}")

    # 2. 振幅（唯一实验数据）+ 初始相位 + 初始密度 + support + 参考直方图
    amp_orig, _ = fft_amp_phase(rho_work)
    phase0 = make_random_phase(rho_work.shape, seed=PHASE_SEED)
    rho_init = init_density(amp_orig, phase0)
    support_gt = shrinkwrap_support(rho_work, SIGMA0)
    ref_edges = estimate_reference_histogram(rho_work, support_gt, n_bins=300)

    # 3. 甲方 HIO
    print("\n" + "=" * 60)
    print("甲方：严格 HIO")
    print("=" * 60)
    best_hio, hist_hio = run_hio(amp_orig, rho_init, ref_edges, rho_work, support_gt,
                                 max_iter=HIO_ITER, beta=0.7, sigma0=SIGMA0)

    # 4. 乙方 UNet
    print("\n" + "=" * 60)
    print("乙方：未训练 UNet")
    print("=" * 60)
    best_unet, hist_unet = run_unet(amp_orig, rho_init, ref_edges, rho_work, support_gt,
                                    max_iter=UNET_ITER, lr=UNET_LR, sigma0=SIGMA0,
                                    unet_seed=UNET_SEED)

    # 5. 翻回视觉极性 + unpad
    gt_vis = to_visual(rho_work, bg_val, pad_info)
    hio_vis = to_visual(best_hio, bg_val, pad_info)
    unet_vis = to_visual(best_unet, bg_val, pad_info)

    # 6. 出图 + 存表
    plot_triple(gt_vis, hio_vis, unet_vis, 'result_compare.png')
    plot_convergence(hist_hio, hist_unet, 'result_convergence.png')
    save_metrics_table(hist_hio, hist_unet, 'result_metrics.csv')

    print("\n完成！")


if __name__ == '__main__':
    main()

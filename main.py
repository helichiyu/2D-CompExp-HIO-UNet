"""
main.py —— 三测：传统 HIO + UNet(sigmoid+置0) + UNet(tanh+HIO反馈) 对比

运行：
  D:\\anaconda3\\envs\\use\\python.exe main.py [HIO_ITER] [UNET_ITER]
  不传参数则用默认 HIO_ITER=30000 / UNET_ITER=5000（长测试）；
  短测试验证可传小值，如 main.py 100 50。

结果输出到 results/run_<时间戳>/，含：
  exp1_hio/、exp2_unet_sigmoid/、exp3_unet_tanh/  各实验的实空间/振幅谱/support/收敛/指标
  comparison.png  三实验主指标柱状对比
  comparison.csv  三实验指标汇总
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

# ===================== 超参（默认长测试；可被命令行覆盖）=====================
SIGMA0 = 3.0
HIO_ITER = 30000      # 甲方轮次（HIO 每轮快，给足轮次才与 UNet 算力相当）
UNET_ITER = 5000      # 乙方轮次（实验2/3 同）
UNET_LR = 1e-4
PHASE_SEED = 42
UNET_SEED = 0


def make_run_dir():
    """创建本次运行的结果文件夹 results/run_<时间戳>/，每次运行独立保留。"""
    run_dir = os.path.join('results', f"run_{datetime.now().strftime('%Y%m%d_%H%M%S')}")
    os.makedirs(run_dir, exist_ok=True)
    return run_dir


def to_visual(rho, bg_val, pad_info):
    """暗背景 rho → 视觉白背景（bg_val − rho）→ unpad → numpy [H,W]。"""
    vis = unpad(bg_val - rho, pad_info)
    return vis.squeeze().detach().cpu().numpy()


def best_point(hist):
    """取综合分数（SSIM+振幅CC）最大点的各项指标。"""
    keys = ['psnr', 'ssim', 'pearson_cc', 'amp_cc', 'phase_err', 'support_iou']
    scores = [s + a for s, a in zip(hist['ssim'], hist['amp_cc'])]
    idx = int(np.argmax(scores))
    return {k: hist[k][idx] for k in keys}


# ===================== 单实验可视化 =====================
def plot_real_space(gt_vis, res_vis, title, save_path):
    """1×3：原图 / 该实验结果 / 绝对误差。"""
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    axes[0].imshow(gt_vis, cmap='gray', vmin=0, vmax=1); axes[0].set_title('原图（真值）', fontsize=14)
    axes[1].imshow(res_vis, cmap='gray', vmin=0, vmax=1); axes[1].set_title(title, fontsize=14)
    err = np.abs(res_vis - gt_vis)
    im = axes[2].imshow(err, cmap='hot', vmin=0, vmax=max(err.max(), 1e-6))
    axes[2].set_title(f'|{title} - 原图|', fontsize=14)
    plt.colorbar(im, ax=axes[2], fraction=0.046, pad=0.04)
    for ax in axes:
        ax.axis('off')
    plt.tight_layout(); plt.savefig(save_path, dpi=150, bbox_inches='tight'); plt.close()
    print(f"  已保存 real_space.png")


def plot_spectra(gt, res, title, save_path):
    """1×2：原图振幅谱 / 该实验振幅谱（log，fftshift 居中）。"""
    fig, axes = plt.subplots(1, 2, figsize=(10, 5))
    for ax, rho, t in zip(axes, [gt, res], ['原图振幅谱', f'{title} 振幅谱']):
        amp = torch.fft.fftshift(torch.abs(torch.fft.fft2(rho)))
        ax.imshow(np.log1p(amp.squeeze().detach().cpu().numpy()), cmap='magma')
        ax.set_title(t, fontsize=14); ax.axis('off')
    plt.tight_layout(); plt.savefig(save_path, dpi=150, bbox_inches='tight'); plt.close()
    print(f"  已保存 spectra.png")


def plot_support(support_gt, support_res, title, pad_info, save_path):
    """1×2：真值 support / 该实验估 support（unpad 显示）。"""
    fig, axes = plt.subplots(1, 2, figsize=(10, 5))
    for ax, m, t in zip(axes, [support_gt, support_res], ['真值 support', f'{title} 估']):
        vis = unpad(m, pad_info).squeeze().detach().cpu().numpy()
        ax.imshow(vis, cmap='gray'); ax.set_title(t, fontsize=14); ax.axis('off')
    plt.tight_layout(); plt.savefig(save_path, dpi=150, bbox_inches='tight'); plt.close()
    print(f"  已保存 support.png")


def plot_convergence_single(hist, title, save_path):
    """单实验 5 指标收敛曲线（2×3，末格空）。"""
    metrics = ['psnr', 'ssim', 'amp_cc', 'phase_err', 'support_iou']
    titles = ['PSNR (dB)', 'SSIM', '振幅域 CC', '平均相位误差 (rad)', '支撑域 IoU']
    fig, axes = plt.subplots(2, 3, figsize=(15, 8))
    axes = axes.flatten()
    for ax, k, t in zip(axes, metrics, titles):
        ax.plot(hist['iter'], hist[k], color='#2E86AB', lw=2)
        ax.set_xlabel('迭代轮数'); ax.set_ylabel(t); ax.set_title(t); ax.grid(True, alpha=0.3)
    axes[5].axis('off')
    fig.suptitle(f'{title} 收敛曲线', fontsize=16)
    plt.tight_layout(); plt.savefig(save_path, dpi=150, bbox_inches='tight'); plt.close()
    print(f"  已保存 convergence.png")


def save_metrics_single(hist, save_path):
    bp = best_point(hist)
    keys = ['psnr', 'ssim', 'pearson_cc', 'amp_cc', 'phase_err', 'support_iou']
    rows = [['指标', '值']]
    for k in keys:
        rows.append([k, f"{bp[k]:.4f}"])
    with open(save_path, 'w', newline='', encoding='utf-8-sig') as f:
        csv.writer(f).writerows(rows)
    print(f"  已保存 metrics.csv")


def dump_experiment(folder, label, best, hist, gt_vis, rho_work, bg_val, pad_info, support_gt, run_dir):
    """单个实验出全套图 + 指标表。"""
    exp_dir = os.path.join(run_dir, folder)
    os.makedirs(exp_dir, exist_ok=True)
    print(f"\n--- {label} 出图 ---")
    res_vis = to_visual(best, bg_val, pad_info)
    support_res = shrinkwrap_support(best, 1.0)
    plot_real_space(gt_vis, res_vis, label, os.path.join(exp_dir, 'real_space.png'))
    plot_spectra(rho_work, best, label, os.path.join(exp_dir, 'spectra.png'))
    plot_support(support_gt, support_res, label, pad_info, os.path.join(exp_dir, 'support.png'))
    plot_convergence_single(hist, label, os.path.join(exp_dir, 'convergence.png'))
    save_metrics_single(hist, os.path.join(exp_dir, 'metrics.csv'))


# ===================== 三实验横向对比 =====================
def plot_comparison(metrics_list, save_path):
    """三实验主指标柱状对比。metrics_list: [(label, {metric:val}), ...]"""
    labels = [m[0] for m in metrics_list]
    keys = ['amp_cc', 'phase_err', 'support_iou', 'ssim']
    titles = ['振幅域 CC (amp_cc)', '平均相位误差 Δφ (rad)', '支撑域 IoU', 'SSIM']
    colors = ['#2E86AB', '#F18F01', '#A23B72']  # 三实验色
    fig, axes = plt.subplots(1, 4, figsize=(20, 5))
    for ax, k, t in zip(axes, keys, titles):
        vals = [m[1][k] for m in metrics_list]
        ax.bar(range(len(labels)), vals, color=colors)
        ax.set_xticks(range(len(labels)))
        ax.set_xticklabels(labels, rotation=15, fontsize=10)
        ax.set_title(t, fontsize=13); ax.grid(True, alpha=0.3, axis='y')
        for i, v in enumerate(vals):
            ax.text(i, v, f'{v:.3f}', ha='center', va='bottom', fontsize=10)
    plt.tight_layout(); plt.savefig(save_path, dpi=150, bbox_inches='tight'); plt.close()
    print(f"已保存三实验对比图: {os.path.basename(save_path)}")


def save_comparison_table(metrics_list, save_path):
    keys = ['psnr', 'ssim', 'pearson_cc', 'amp_cc', 'phase_err', 'support_iou']
    labels = [m[0] for m in metrics_list]
    rows = [['指标'] + labels]
    for k in keys:
        rows.append([k] + [f"{m[1][k]:.4f}" for m in metrics_list])
    with open(save_path, 'w', newline='', encoding='utf-8-sig') as f:
        csv.writer(f).writerows(rows)
    print("\n===== 三实验对比（综合分数最优点的各项指标）=====")
    print(f"{'指标':>12} | " + " | ".join(f"{l:>18}" for l in labels))
    print("-" * (16 + 21 * len(labels)))
    for row in rows[1:]:
        print(f"{row[0]:>12} | " + " | ".join(f"{v:>18}" for v in row[1:]))
    print(f"\n已保存对比表: {os.path.basename(save_path)}")


def main():
    run_dir = make_run_dir()
    print(f"本次结果将保存到: {run_dir}/")
    print(f"轮次：HIO={HIO_ITER}，UNet={UNET_ITER}（实验2/3 同）\n")

    # 共享预处理与初始化（控制变量：三实验同图、同 ρ_init、同 ref_edges、同 support_gt）
    print("=" * 60); print("加载 567.png ...")
    rho_work, bg_val, pad_info, H0, W0 = load_and_preprocess('567.png')
    Hp, Wp = rho_work.shape[-2], rho_work.shape[-1]
    print(f"原始 {W0}×{H0} → 扩边+pad 后 {Wp}×{Hp}，bg_val={bg_val:.3f}")

    amp_orig, _ = fft_amp_phase(rho_work)
    phase0 = make_random_phase(rho_work.shape, seed=PHASE_SEED)
    rho_init = init_density(amp_orig, phase0)
    support_gt = shrinkwrap_support(rho_work, SIGMA0)
    ref_edges = estimate_reference_histogram(rho_work, support_gt, n_bins=300)
    gt_vis = to_visual(rho_work, bg_val, pad_info)

    # 实验1：传统 HIO（eval_every=100：轮次多，评估间隔放大）
    print("\n" + "=" * 60); print("实验1：传统 HIO"); print("=" * 60)
    best1, hist1 = run_hio(amp_orig, rho_init, ref_edges, rho_work, support_gt,
                           max_iter=HIO_ITER, beta=0.7, sigma0=SIGMA0, eval_every=100)

    # 实验2：UNet + sigmoid + 置0（无 HIO 反馈）
    print("\n" + "=" * 60); print("实验2：UNet + sigmoid + 置0（无 HIO 反馈）"); print("=" * 60)
    best2, hist2 = run_unet(amp_orig, rho_init, ref_edges, rho_work, support_gt,
                            max_iter=UNET_ITER, lr=UNET_LR, sigma0=SIGMA0, unet_seed=UNET_SEED,
                            out_act='sigmoid', use_hio_feedback=False)

    # 实验3：UNet + tanh + HIO 反馈
    print("\n" + "=" * 60); print("实验3：UNet + tanh + HIO 反馈"); print("=" * 60)
    best3, hist3 = run_unet(amp_orig, rho_init, ref_edges, rho_work, support_gt,
                            max_iter=UNET_ITER, lr=UNET_LR, sigma0=SIGMA0, unet_seed=UNET_SEED,
                            out_act='tanh', use_hio_feedback=True, beta=0.7)

    # 各实验出图
    experiments = [
        ('exp1_hio', '传统 HIO', best1, hist1),
        ('exp2_unet_sigmoid', 'UNet+sigmoid+置0', best2, hist2),
        ('exp3_unet_tanh', 'UNet+tanh+HIO', best3, hist3),
    ]
    for folder, label, best, hist in experiments:
        dump_experiment(folder, label, best, hist, gt_vis, rho_work, bg_val, pad_info, support_gt, run_dir)

    # 三实验横向对比
    metrics_list = [(label, best_point(hist)) for _, label, _, hist in experiments]
    plot_comparison(metrics_list, os.path.join(run_dir, 'comparison.png'))
    save_comparison_table(metrics_list, os.path.join(run_dir, 'comparison.csv'))

    print(f"\n完成！所有结果在: {run_dir}/")


if __name__ == '__main__':
    # 支持命令行传轮次：python main.py [HIO_ITER] [UNET_ITER]，便于短测试验证
    if len(sys.argv) > 1:
        HIO_ITER = int(sys.argv[1])
    if len(sys.argv) > 2:
        UNET_ITER = int(sys.argv[2])
    main()

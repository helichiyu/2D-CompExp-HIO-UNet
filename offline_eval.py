"""
offline_eval.py —— 八测阶段①②：多次重建对齐平均 + PRTF 客观评估

离线脚本，不重跑迭代。读 run_dir 下 50 个 state.pt（best_rho 已 register_to_gt 配准，
平移/共轭反转/符号三歧义已消除），按方法分类做：
  阶段① 实空间平均（clamp_min(0) 后栈平均）—— 抑制随机条纹，验 C14（HIO 斜线倾角随机→平均抵消）。
  阶段② PRTF（|<F>|/<|F|> 径向平均）—— 频域相位一致性，替代人眼看收敛（Latychevskaia §3.5）。

avg 用 clamp（评估态平均，HIO 负反馈值不抵消物体）；PRTF 不 clamp（保傅里叶线性）。

用法：
  python offline_eval.py <run_dir> [n_groups]
  python offline_eval.py results/run_20260705_000538

输出到 <run_dir>/offline/：
  avg_<type>.png / avg_metrics.csv                    阶段①
  prtf_<type>.png / prtf_curves.csv / prtf_resolution.csv   阶段②
  overview_avg.png                                    三类平均并排对比
"""

import csv
import os
import sys

import numpy as np
import torch
import matplotlib.pyplot as plt

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from utils import (load_and_preprocess, fft_amp_phase, evaluate_all,
                   shrinkwrap_support, unpad, device)

plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

# 每组的实验位布局（folder 名 → 类型）；HIO 占 3 位
EXP_LAYOUT = [
    ('HIO', ['hio_1', 'hio_2', 'hio_3']),
    ('tanh+HIO', ['tanh_hio']),
    ('sigmoid', ['sigmoid']),
]
SIGMOID_PHASE_THRESH = 0.8   # sigmoid 末轮 ssim 阈值，划"相变子组"（七测 g6/g8 预期入）


# ===================== 加载 =====================
def load_all_recons(run_dir, n_groups):
    """扫 run_dir 下所有 state.pt，返回 {type: [(group, folder, best_rho, hist), ...]}。"""
    recons = {t: [] for t, _ in EXP_LAYOUT}
    folder2type = {f: t for t, fs in EXP_LAYOUT for f in fs}
    for g in range(1, n_groups + 1):
        gdir = os.path.join(run_dir, f'group{g:02d}')
        if not os.path.isdir(gdir):
            print(f"  警告：缺 {gdir}，跳过该组")
            continue
        for folder in sorted(os.listdir(gdir)):
            state_path = os.path.join(gdir, folder, 'state.pt')
            if not (os.path.isfile(state_path) and folder in folder2type):
                continue
            d = torch.load(state_path, map_location=device, weights_only=False)
            t = folder2type[folder]
            recons[t].append((g, folder, d['best_rho'], d['hist']))
    for t, _ in EXP_LAYOUT:
        print(f"  {t}: {len(recons[t])} 个")
    return recons


def split_sigmoid_phase(sigmoid_list, thresh=SIGMOID_PHASE_THRESH):
    """sigmoid 按末轮 ssim 分相变子组 / 未相变子组。"""
    phase = [(g, f, r, h) for (g, f, r, h) in sigmoid_list if h['ssim'][-1] > thresh]
    nophase = [(g, f, r, h) for (g, f, r, h) in sigmoid_list if h['ssim'][-1] <= thresh]
    return phase, nophase


# ===================== 阶段①：实空间平均 =====================
def average_recons(rho_list):
    """clamp_min(0) 后栈平均（HIO/tanh 末轮有负反馈值；sigmoid 已 [0,1] 无影响）。"""
    stacked = torch.stack([r.clamp_min(0) for r in rho_list])
    return stacked.mean(dim=0)


def to_visual(rho, bg_val, pad_info):
    """暗背景 rho → 视觉白背景 → unpad → numpy [H,W]。"""
    return unpad(bg_val - rho, pad_info).squeeze().detach().cpu().numpy()


def plot_avg(gt_vis, res_vis, label, n, save_path):
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    axes[0].imshow(gt_vis, cmap='gray', vmin=0, vmax=1); axes[0].set_title('原图（真值）', fontsize=14)
    axes[1].imshow(res_vis, cmap='gray', vmin=0, vmax=1); axes[1].set_title(f'{label} 平均（n={n}）', fontsize=14)
    err = np.abs(res_vis - gt_vis)
    im = axes[2].imshow(err, cmap='hot', vmin=0, vmax=max(err.max(), 1e-6))
    axes[2].set_title(f'|{label} 平均 − 原图|', fontsize=14)
    plt.colorbar(im, ax=axes[2], fraction=0.046, pad=0.04)
    for ax in axes:
        ax.axis('off')
    plt.tight_layout(); plt.savefig(save_path, dpi=150, bbox_inches='tight'); plt.close()
    print(f"  已保存 {os.path.basename(save_path)}")


def run_avg(subsets, gt_data, out_dir):
    """对每个子集做实空间平均 + 出图 + 指标。subsets: [(key, label, [rho...]), ...]"""
    rho_work, amp_orig, phase_orig, support_gt, bg_val, pad_info, gt_vis = gt_data
    rows = [['subset', 'n', 'psnr', 'ssim', 'pearson_cc', 'amp_cc', 'phase_err', 'support_iou']]
    overview = []  # (label, n, vis) 用于并排总览
    for key, label, rhos in subsets:
        if not rhos:
            print(f"  [{label}] 无样本，跳过"); continue
        avg_rho = average_recons(rhos)
        m = evaluate_all(avg_rho, rho_work, amp_orig, phase_orig, support_gt, align_to_gt=False)
        vis = to_visual(avg_rho, bg_val, pad_info)
        plot_avg(gt_vis, vis, label, len(rhos), os.path.join(out_dir, f'avg_{key}.png'))
        rows.append([key, len(rhos)] + [f"{m[k]:.4f}" for k in
                     ['psnr', 'ssim', 'pearson_cc', 'amp_cc', 'phase_err', 'support_iou']])
        overview.append((label, len(rhos), vis))
        print(f"  [{label} n={len(rhos)}] ssim={m['ssim']:.3f} psnr={m['psnr']:.2f} "
              f"iou={m['support_iou']:.3f} amp_cc={m['amp_cc']:.3f}")
    with open(os.path.join(out_dir, 'avg_metrics.csv'), 'w', newline='', encoding='utf-8-sig') as f:
        csv.writer(f).writerows(rows)
    return overview


def plot_overview_avg(gt_vis, overview, save_path):
    """原图 + 各子集平均并排。"""
    n = len(overview)
    fig, axes = plt.subplots(1, n + 1, figsize=(5 * (n + 1), 5))
    axes[0].imshow(gt_vis, cmap='gray', vmin=0, vmax=1); axes[0].set_title('原图', fontsize=14); axes[0].axis('off')
    for i, (label, k, vis) in enumerate(overview):
        axes[i + 1].imshow(vis, cmap='gray', vmin=0, vmax=1)
        axes[i + 1].set_title(f'{label}\n(n={k})', fontsize=13); axes[i + 1].axis('off')
    plt.tight_layout(); plt.savefig(save_path, dpi=150, bbox_inches='tight'); plt.close()
    print(f"  已保存 overview_avg.png")


# ===================== 阶段②：PRTF =====================
def compute_prtf_radial(rho_list):
    """PRTF = |<F>| / <|F|>（N 次已配准重建的复傅里叶一致性），径向平均得 1D 曲线。
    用 best_rho 原值（不 clamp，保傅里叶线性）。返回 (prtf_radial[Nyquist+1], r_max)。"""
    Fs = torch.stack([torch.fft.fft2(r) for r in rho_list])        # [N,1,1,H,W] complex
    mean_F = Fs.mean(dim=0)
    mean_abs = Fs.abs().mean(dim=0)
    prtf = (mean_F.abs() / (mean_abs + 1e-12))[0, 0]               # [H,W]
    H, W = prtf.shape
    prtf = torch.fft.fftshift(prtf)                                # DC 居中
    cy, cx = H // 2, W // 2
    yy = torch.arange(H, device=prtf.device).view(H, 1) - cy
    xx = torch.arange(W, device=prtf.device).view(1, W) - cx
    r = torch.sqrt(yy ** 2 + xx ** 2)
    r_max = min(cy, cx)
    r_int = r.round().long().clamp(0, r_max).flatten()
    flat = prtf.flatten()
    numer = torch.bincount(r_int, weights=flat, minlength=r_max + 1)
    denom = torch.bincount(r_int, minlength=r_max + 1).clamp(min=1)
    return (numer / denom).cpu().numpy(), r_max


def resolution_at(prtf_radial, thresh):
    """PRTF 首次降到 ≤ thresh 的频率像素（找不到返回 r_max，表示全程高于阈值）。"""
    idx = np.where(prtf_radial <= thresh)[0]
    return int(idx[0]) if len(idx) else len(prtf_radial) - 1


def plot_prtf_curves(curves, save_path):
    """curves: [(key, label, prtf_radial, r_max), ...]。横轴频率像素，标 0.5 / 1/e 阈值线。"""
    fig, ax = plt.subplots(figsize=(9, 6))
    colors = ['#2E86AB', '#F18F01', '#A23B72', '#5CB85C', '#777777']
    for i, (key, label, rad, r_max) in enumerate(curves):
        ax.plot(np.arange(len(rad)), rad, color=colors[i % len(colors)], lw=2, label=label)
    ax.axhline(0.5, color='gray', ls='--', lw=1, alpha=0.7)
    ax.axhline(1 / np.e, color='gray', ls=':', lw=1, alpha=0.7)
    ax.text(5, 0.52, '0.5', fontsize=10, color='gray')
    ax.text(5, 1 / np.e + 0.02, '1/e≈0.368', fontsize=10, color='gray')
    ax.set_xlabel('频率（到 DC 的像素距离）'); ax.set_ylabel('PRTF')
    ax.set_title('PRTF 径向平均（越接近 1 = 多次重建相位越一致 = 越收敛）', fontsize=13)
    ax.set_ylim(0, 1.05); ax.grid(True, alpha=0.3); ax.legend(loc='best')
    plt.tight_layout(); plt.savefig(save_path, dpi=150, bbox_inches='tight'); plt.close()
    print(f"  已保存 {os.path.basename(save_path)}")


def run_prtf(subsets, out_dir):
    """对每个样本≥2 的子集算 PRTF，出曲线图 + 曲线 csv + 分辨率 csv。"""
    curves = []
    rows = [['subset', 'n', 'f_at_1/e', 'f_at_0.5', 'r_max(Nyquist)']]
    for key, label, rhos in subsets:
        if len(rhos) < 2:
            print(f"  [{label} PRTF] 样本 <2，无法算一致性，跳过"); continue
        rad, r_max = compute_prtf_radial(rhos)
        f_e = resolution_at(rad, 1 / np.e)
        f_5 = resolution_at(rad, 0.5)
        curves.append((key, label, rad, r_max))
        rows.append([key, len(rhos), f_e, f_5, r_max])
        print(f"  [{label} n={len(rhos)}] PRTF 跌到 1/e @ f={f_e} / 0.5 @ f={f_5} (Nyquist={r_max})")
    if not curves:
        return
    plot_prtf_curves(curves, os.path.join(out_dir, 'prtf_curves.png'))
    maxlen = max(len(c[2]) for c in curves)
    with open(os.path.join(out_dir, 'prtf_curves.csv'), 'w', newline='', encoding='utf-8-sig') as f:
        w = csv.writer(f); w.writerow(['freq'] + [c[0] for c in curves])
        for i in range(maxlen):
            w.writerow([i] + [f"{c[2][i]:.6f}" if i < len(c[2]) else '' for c in curves])
    with open(os.path.join(out_dir, 'prtf_resolution.csv'), 'w', newline='', encoding='utf-8-sig') as f:
        csv.writer(f).writerows(rows)
    print(f"  已保存 prtf_curves.csv / prtf_resolution.csv")


# ===================== 主流程 =====================
def main(run_dir, n_groups=10):
    out_dir = os.path.join(run_dir, 'offline')
    os.makedirs(out_dir, exist_ok=True)
    print(f"离线评估：{run_dir}/ → 输出 {out_dir}/\n")

    # 共享真值（控制变量：同图、同 amp_orig、同 support_gt）
    rho_work, bg_val, pad_info, _, _ = load_and_preprocess('567.png')
    amp_orig, phase_orig = fft_amp_phase(rho_work)
    support_gt = shrinkwrap_support(rho_work, 3.0)
    gt_vis = to_visual(rho_work, bg_val, pad_info)
    gt_data = (rho_work, amp_orig, phase_orig, support_gt, bg_val, pad_info, gt_vis)

    print("加载 50 个 state.pt ...")
    recons = load_all_recons(run_dir, n_groups)

    # sigmoid 拆相变子组
    sig_phase, sig_nophase = split_sigmoid_phase(recons['sigmoid'])
    print(f"  sigmoid 相变子组（ssim>{SIGMOID_PHASE_THRESH}）: {len(sig_phase)} 个 "
          f"（组号 {[r[0] for r in sig_phase]}）")

    def rhos_of(items):
        return [r for (_, _, r, _) in items]

    # 子集列表（avg 与 PRTF 共用）
    subsets = [
        ('HIO_all', 'HIO 全部', rhos_of(recons['HIO'])),
        ('tanh_all', 'tanh+HIO 全部', rhos_of(recons['tanh+HIO'])),
        ('sigmoid_all', 'sigmoid 全部', rhos_of(recons['sigmoid'])),
        ('sigmoid_phase', 'sigmoid 相变子组', rhos_of(sig_phase)),
        ('sigmoid_nophase', 'sigmoid 未相变', rhos_of(sig_nophase)),
    ]

    print("\n=== 阶段① 实空间平均 ===")
    overview = run_avg(subsets, gt_data, out_dir)
    if overview:
        plot_overview_avg(gt_vis, overview, os.path.join(out_dir, 'overview_avg.png'))

    print("\n=== 阶段② PRTF ===")
    run_prtf(subsets, out_dir)

    print(f"\n完成。结果在 {out_dir}/")


if __name__ == '__main__':
    if len(sys.argv) < 2:
        print("用法：python offline_eval.py <run_dir> [n_groups]")
        sys.exit(1)
    _run_dir = sys.argv[1]
    _n_groups = int(sys.argv[2]) if len(sys.argv) > 2 else 10
    main(_run_dir, _n_groups)

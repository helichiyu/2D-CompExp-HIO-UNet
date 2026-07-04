"""
main.py —— 七测：大规模重复取统计（10 组 × 5 实验 = 50 个）

六测核心认知 C12：三种方法（HIO / sigmoid / tanh+HIO）的恢复都是概率性事件，单次结果随机。
七测做多次独立重复取统计：10 组 ×（HIO 5000 轮 ×3 + UNet 1500 轮 ×2）= 50 个实验。
组间差异完全来自随机性（phase seed 不固定 + CUDA 非确定性），正是统计样本。

代码改造三处（相对六测）：
  ① 完整保存：HIO 3 次都各自 dump 全套（不只代表）；50 个实验每个独立全套。
  ② 多组循环 + 断点续跑：progress.json 记进度，中断后 main.py ... results/run_xxx 可续跑。
  ③ summary.csv：跑完扫 50 实验 metrics 汇总，供按类型排序推荐看图。

每组 5 个实验位（沿用六测矩阵）：
  hio_1/2/3    传统 HIO（relaxed γ=0.9），各 5000 轮，独立 phase seed（撞收敛）
  tanh_hio     UNet tanh + HIO 反馈，1500 轮
  sigmoid      UNet sigmoid + 置 0，1500 轮

运行：
  D:\\anaconda3\\envs\\use\\python.exe main.py [HIO_ITER] [UNET_ITER] [RUN_DIR]
  不传 RUN_DIR → 新建 results/run_<时间戳>/；传 RUN_DIR → 续跑该目录（跳过已完成组）。
  默认 HIO_ITER=5000、UNET_ITER=1500。

结果输出到 results/run_<时间戳>/，含：
  config.txt / progress.json
  group01..group10/  各组 hio_1/2/3 + tanh_hio + sigmoid（各全套）+ hio_3runs.png + comparison_gXx.png/csv
  summary.csv        50 实验末轮指标汇总（组号/类型/实验/6 指标）

注：跑完按 summary.csv 各类型排序推荐 top，但 HIO 场景 ssim 会失真（R5/R8），
    最终以 real_space.png 看图为准（见 六测汇报.md / 七测计划.md）。
"""

import csv
import json
import os
import sys
from datetime import datetime

import numpy as np
import torch
import matplotlib.pyplot as plt

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")  # 避免 Windows GBK 编码报错

from utils import (load_and_preprocess, fft_amp_phase, make_random_phase, init_density,
                   shrinkwrap_support, estimate_reference_histogram, unpad, register_to_gt)
from hio import run_hio
from unet_pr import run_unet

# ===================== 中文显示 =====================
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

# ===================== 超参（默认全量；可被命令行覆盖）=====================
SIGMA0 = 3.0
HIO_ITER = 5000     # 甲方轮次（每组重复 N_HIO_RUNS 次）
UNET_ITER = 1500    # 乙方轮次（tanh+HIO / sigmoid 同）
N_GROUPS = 10       # 独立重复组数（10 组 × 5 实验 = 50 个，取统计）
N_HIO_RUNS = 3      # 每组 HIO 重复次数（收敛是概率性事件，多次撞收敛）
UNET_LR = 1e-4
GAMMA = 0.9        # relaxed HIO 松弛系数（support 外 γ·ρ − β·ρ′，γ<1 防累加发散；HIO/tanh+HIO 一致）


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
    """取末轮（收敛态）的各项指标——不取最优瞬间，最优瞬间未必稳定真实。"""
    keys = ['psnr', 'ssim', 'pearson_cc', 'amp_cc', 'phase_err', 'support_iou']
    return {k: hist[k][-1] for k in keys}


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


def plot_convergence_single(hist, title, save_path, monitor_key=None, monitor_label=None):
    """单实验收敛曲线（2×3）：首格为 monitor(loss/amp_residual)，余 5 格为评估指标。"""
    metrics = ['psnr', 'ssim', 'amp_cc', 'phase_err', 'support_iou']
    titles = ['PSNR (dB)', 'SSIM', '振幅域 CC', '平均相位误差 (rad)', '支撑域 IoU']
    if monitor_key:
        metrics = [monitor_key] + metrics
        titles = [monitor_label or monitor_key] + titles
    fig, axes = plt.subplots(2, 3, figsize=(15, 8))
    axes = axes.flatten()
    for ax, k, t in zip(axes, metrics, titles):
        ax.plot(hist['iter'], hist[k], color='#2E86AB', lw=2)
        ax.set_xlabel('迭代轮数'); ax.set_ylabel(t); ax.set_title(t); ax.grid(True, alpha=0.3)
    if len(metrics) < 6:
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


def save_history_csv(hist, monitor_key, save_path):
    """全程 history 存 csv（每次 eval 一行）：iter, monitor(loss/amp_residual), 6 指标。"""
    metric_keys = ['psnr', 'ssim', 'pearson_cc', 'amp_cc', 'phase_err', 'support_iou']
    cols = ['iter', monitor_key] + metric_keys
    rows = [cols]
    for i in range(len(hist['iter'])):
        rows.append([hist['iter'][i], hist[monitor_key][i]] + [hist[k][i] for k in metric_keys])
    with open(save_path, 'w', newline='', encoding='utf-8-sig') as f:
        csv.writer(f).writerows(rows)
    print(f"  已保存 history.csv")


def dump_experiment(folder, label, best, hist, gt_vis, rho_work, bg_val, pad_info, support_gt, run_dir,
                    monitor_key=None, monitor_label=None):
    """单个实验出全套图 + 指标表（best 先配准再出图，消除平移歧义）。folder 相对 run_dir。"""
    exp_dir = os.path.join(run_dir, folder)
    os.makedirs(exp_dir, exist_ok=True)
    print(f"\n--- {label} 出图 ---")
    best_aligned = register_to_gt(best, rho_work)            # 出图前配准，消除平移歧义
    res_vis = to_visual(best_aligned, bg_val, pad_info)
    support_res = shrinkwrap_support(best_aligned, 1.0)
    plot_real_space(gt_vis, res_vis, label, os.path.join(exp_dir, 'real_space.png'))
    plot_spectra(rho_work, best_aligned, label, os.path.join(exp_dir, 'spectra.png'))
    plot_support(support_gt, support_res, label, pad_info, os.path.join(exp_dir, 'support.png'))
    plot_convergence_single(hist, label, os.path.join(exp_dir, 'convergence.png'),
                            monitor_key, monitor_label)
    save_metrics_single(hist, os.path.join(exp_dir, 'metrics.csv'))
    if monitor_key:
        save_history_csv(hist, monitor_key, os.path.join(exp_dir, 'history.csv'))
    torch.save({'best_rho': best_aligned, 'best_rho_raw': best, 'hist': hist},
               os.path.join(exp_dir, 'state.pt'))
    print(f"  已保存 state.pt（best_rho + history）")


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
    print(f"  已保存对比图: {os.path.basename(save_path)}")


def save_comparison_table(metrics_list, save_path):
    keys = ['psnr', 'ssim', 'pearson_cc', 'amp_cc', 'phase_err', 'support_iou']
    labels = [m[0] for m in metrics_list]
    rows = [['指标'] + labels]
    for k in keys:
        rows.append([k] + [f"{m[1][k]:.4f}" for m in metrics_list])
    with open(save_path, 'w', newline='', encoding='utf-8-sig') as f:
        csv.writer(f).writerows(rows)
    print(f"===== 组内对比（末轮各项指标）=====")
    print(f"{'指标':>12} | " + " | ".join(f"{l:>18}" for l in labels))
    print("-" * (16 + 21 * len(labels)))
    for row in rows[1:]:
        print(f"{row[0]:>12} | " + " | ".join(f"{v:>18}" for v in row[1:]))
    print(f"  已保存对比表: {os.path.basename(save_path)}")


def plot_hio_3runs(hio_runs, g, gt_vis, bg_val, pad_info, rho_work, save_path):
    """HIO 3 次实空间并排（原图 + 3 次），供看图挑收敛：收敛=轮廓，不收敛=斜线/中间态。"""
    n = len(hio_runs)
    fig, axes = plt.subplots(1, n + 1, figsize=(5 * (n + 1), 5))
    axes[0].imshow(gt_vis, cmap='gray', vmin=0, vmax=1); axes[0].set_title('原图', fontsize=14); axes[0].axis('off')
    for i, (best, hist) in enumerate(hio_runs):
        best_a = register_to_gt(best, rho_work)
        vis = unpad(bg_val - best_a, pad_info).squeeze().detach().cpu().numpy()
        axes[i + 1].imshow(vis, cmap='gray', vmin=0, vmax=1)
        axes[i + 1].set_title(f'组{g} 第{i + 1}次\nssim={hist["ssim"][-1]:.3f}', fontsize=13)
        axes[i + 1].axis('off')
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight'); plt.close()
    print(f"  已保存 hio_3runs.png（3 次并排，供挑收敛）")


# ===================== 断点续跑 / 配置 / 汇总 =====================
def load_progress(run_dir, hio_iter, unet_iter, n_groups):
    """读 progress.json；不存在则新建。续跑时校验配置一致。返回进度 dict。"""
    path = os.path.join(run_dir, 'progress.json')
    if os.path.exists(path):
        with open(path, encoding='utf-8') as f:
            prog = json.load(f)
        if (prog['hio_iter'] != hio_iter or prog['unet_iter'] != unet_iter
                or prog['n_groups'] != n_groups):
            raise RuntimeError(
                f"续跑配置与 progress.json 不一致：记录 hio={prog['hio_iter']} unet={prog['unet_iter']} "
                f"groups={prog['n_groups']}，当前 hio={hio_iter} unet={unet_iter} groups={n_groups}。"
                f"请用一致参数续跑，或换新目录。")
        return prog
    prog = {'done_groups': [], 'n_groups': n_groups, 'hio_iter': hio_iter, 'unet_iter': unet_iter}
    save_progress(run_dir, prog)
    return prog


def save_progress(run_dir, prog):
    with open(os.path.join(run_dir, 'progress.json'), 'w', encoding='utf-8') as f:
        json.dump(prog, f, ensure_ascii=False, indent=2)


def write_config(run_dir, hio_iter, unet_iter, n_groups):
    lines = [
        f"HIO_ITER = {hio_iter}（每组 ×{N_HIO_RUNS} 次）",
        f"UNET_ITER = {unet_iter}（tanh+HIO / sigmoid 同）",
        f"N_GROUPS = {n_groups}",
        f"UNET_LR = {UNET_LR}",
        f"GAMMA = {GAMMA}（relaxed HIO）",
        f"实验总数 = {n_groups * (N_HIO_RUNS + 2)}（{n_groups} 组 × {N_HIO_RUNS + 2}）",
    ]
    with open(os.path.join(run_dir, 'config.txt'), 'w', encoding='utf-8') as f:
        f.write('\n'.join(lines))


def read_metrics_csv(path):
    """读 save_metrics_single 写的两列 csv（指标/值），返回 {指标: 值}。"""
    d = {}
    with open(path, encoding='utf-8-sig') as f:
        for row in csv.reader(f):
            if len(row) < 2 or row[0] == '指标':
                continue
            try:
                d[row[0]] = float(row[1])
            except ValueError:
                pass
    return d


def write_summary(run_dir, n_groups):
    """扫所有组的 5 实验 metrics.csv，汇总 summary.csv（每个实验一行）。"""
    types = [(f'hio_{j}', 'HIO') for j in range(1, N_HIO_RUNS + 1)] + [('tanh_hio', 'tanh+HIO'), ('sigmoid', 'sigmoid')]
    keys = ['psnr', 'ssim', 'pearson_cc', 'amp_cc', 'phase_err', 'support_iou']
    rows = [['group', 'type', 'exp'] + keys]
    for g in range(1, n_groups + 1):
        for folder, typ in types:
            path = os.path.join(run_dir, f'group{g:02d}', folder, 'metrics.csv')
            if not os.path.exists(path):
                print(f"  警告：缺 {path}，跳过")
                continue
            m = read_metrics_csv(path)
            rows.append([g, typ, folder] + [f"{m.get(k, float('nan')):.4f}" for k in keys])
    out = os.path.join(run_dir, 'summary.csv')
    with open(out, 'w', newline='', encoding='utf-8-sig') as f:
        csv.writer(f).writerows(rows)
    print(f"\n已保存 summary.csv（{len(rows) - 1} 个实验末轮指标汇总）")


# ===================== 单组实验 =====================
def run_one_group(g, run_dir, group_dir, shared, hio_iter, unet_iter):
    """跑一组 5 个实验：HIO×3（各独立 phase）+ tanh+HIO + sigmoid（共享该组 rho_init）。
    各自 dump 全套，出该组 hio_3runs.png + comparison_gXx.png/csv。"""
    amp_orig = shared['amp_orig']; rho_work = shared['rho_work']
    ref_edges = shared['ref_edges']; support_gt = shared['support_gt']
    gt_vis = shared['gt_vis']; bg_val = shared['bg_val']; pad_info = shared['pad_info']

    # 该组 UNet 起点 rho_init（组间独立 → 组间样本独立）
    phase_g = make_random_phase(rho_work.shape)
    rho_init_g = init_density(amp_orig, phase_g)

    # HIO ×3（各独立 phase seed，撞收敛）
    hio_runs = []
    for j in range(N_HIO_RUNS):
        print(f"\n--- 组{g} HIO 第 {j + 1}/{N_HIO_RUNS} 次（{hio_iter} 轮）---")
        phase_j = make_random_phase(rho_work.shape)
        rho_init_j = init_density(amp_orig, phase_j)
        best_j, hist_j = run_hio(amp_orig, rho_init_j, ref_edges, rho_work, support_gt,
                                 max_iter=hio_iter, beta=0.7, sigma0=SIGMA0, eval_every=100, gamma=GAMMA)
        dump_experiment(os.path.join(f'group{g:02d}', f'hio_{j + 1}'), f'组{g} HIO 第{j + 1}次',
                        best_j, hist_j, gt_vis, rho_work, bg_val, pad_info, support_gt, run_dir,
                        monitor_key='amp_residual', monitor_label='振幅残差')
        hio_runs.append((best_j, hist_j))

    # tanh+HIO（UNet tanh + HIO 反馈）
    print(f"\n--- 组{g} tanh+HIO（{unet_iter} 轮）---")
    best_t, hist_t = run_unet(amp_orig, rho_init_g, ref_edges, rho_work, support_gt,
                              max_iter=unet_iter, lr=UNET_LR, sigma0=SIGMA0,
                              out_act='tanh', use_hio_feedback=True, beta=0.7, gamma=GAMMA)
    dump_experiment(os.path.join(f'group{g:02d}', 'tanh_hio'), f'组{g} tanh+HIO',
                    best_t, hist_t, gt_vis, rho_work, bg_val, pad_info, support_gt, run_dir,
                    monitor_key='loss', monitor_label='UNet loss')

    # sigmoid（UNet sigmoid + support 外置 0）
    print(f"\n--- 组{g} sigmoid（{unet_iter} 轮）---")
    best_s, hist_s = run_unet(amp_orig, rho_init_g, ref_edges, rho_work, support_gt,
                              max_iter=unet_iter, lr=UNET_LR, sigma0=SIGMA0,
                              out_act='sigmoid', use_hio_feedback=False, gamma=GAMMA)
    dump_experiment(os.path.join(f'group{g:02d}', 'sigmoid'), f'组{g} sigmoid',
                    best_s, hist_s, gt_vis, rho_work, bg_val, pad_info, support_gt, run_dir,
                    monitor_key='loss', monitor_label='UNet loss')

    # HIO 3 次实空间并排（供看图挑收敛）
    plot_hio_3runs(hio_runs, g, gt_vis, bg_val, pad_info, rho_work, os.path.join(group_dir, 'hio_3runs.png'))

    # 该组 comparison（HIO 取末轮 ssim 最高者作代表，仅初筛；是否真收敛以 hio_3runs.png 看图为准）
    _, hist_hio_rep = max(hio_runs, key=lambda r: r[1]['ssim'][-1])
    metrics_list = [(f'HIO g{g}', best_point(hist_hio_rep)),
                    (f'tanh+HIO g{g}', best_point(hist_t)),
                    (f'sigmoid g{g}', best_point(hist_s))]
    plot_comparison(metrics_list, os.path.join(group_dir, f'comparison_g{g:02d}.png'))
    save_comparison_table(metrics_list, os.path.join(group_dir, f'comparison_g{g:02d}.csv'))


def main(run_dir=None, hio_iter=HIO_ITER, unet_iter=UNET_ITER, n_groups=N_GROUPS):
    run_dir = run_dir or make_run_dir()
    print(f"结果目录：{run_dir}/")
    print(f"轮次：HIO={hio_iter}（每组 ×{N_HIO_RUNS}），UNet={unet_iter}，组数={n_groups}，"
          f"共 {n_groups * (N_HIO_RUNS + 2)} 个实验\n")

    prog = load_progress(run_dir, hio_iter, unet_iter, n_groups)
    if prog['done_groups']:
        print(f"续跑：已完成组 {sorted(prog['done_groups'])}，从第 {max(prog['done_groups']) + 1} 组继续\n")
    write_config(run_dir, hio_iter, unet_iter, n_groups)

    # 共享预处理（控制变量：各组同图、同 amp_orig、同 ref_edges、同 support_gt）
    print("=" * 60); print("加载 567.png ...")
    rho_work, bg_val, pad_info, H0, W0 = load_and_preprocess('567.png')
    Hp, Wp = rho_work.shape[-2], rho_work.shape[-1]
    print(f"原始 {W0}×{H0} → 扩边+pad 后 {Wp}×{Hp}，bg_val={bg_val:.3f}")
    amp_orig, _ = fft_amp_phase(rho_work)
    support_gt = shrinkwrap_support(rho_work, SIGMA0)
    ref_edges = estimate_reference_histogram(rho_work, support_gt, n_bins=300)
    gt_vis = to_visual(rho_work, bg_val, pad_info)
    shared = {'amp_orig': amp_orig, 'rho_work': rho_work, 'ref_edges': ref_edges,
              'support_gt': support_gt, 'gt_vis': gt_vis, 'bg_val': bg_val, 'pad_info': pad_info}

    # 组循环（断点续跑：跳过已完成组）
    for g in range(1, n_groups + 1):
        if g in prog['done_groups']:
            print(f"\n=== 组 {g}/{n_groups} 已完成，跳过 ===")
            continue
        print("\n" + "=" * 60); print(f"=== 组 {g}/{n_groups} ==="); print("=" * 60)
        group_dir = os.path.join(run_dir, f'group{g:02d}')
        os.makedirs(group_dir, exist_ok=True)
        run_one_group(g, run_dir, group_dir, shared, hio_iter, unet_iter)
        prog['done_groups'].append(g)
        save_progress(run_dir, prog)
        print(f"\n=== 组 {g} 完成，进度 {len(prog['done_groups'])}/{n_groups} ===")

    # 全部完成：汇总
    write_summary(run_dir, n_groups)
    print(f"\n全部完成！结果在: {run_dir}/")


if __name__ == '__main__':
    # 支持命令行：python main.py [HIO_ITER] [UNET_ITER] [RUN_DIR]
    # 续跑：python main.py 5000 1500 results/run_<时间戳>
    _hio_iter, _unet_iter, _run_dir = HIO_ITER, UNET_ITER, None
    if len(sys.argv) > 1:
        _hio_iter = int(sys.argv[1])
    if len(sys.argv) > 2:
        _unet_iter = int(sys.argv[2])
    if len(sys.argv) > 3:
        _run_dir = sys.argv[3]
    main(_run_dir, _hio_iter, _unet_iter)

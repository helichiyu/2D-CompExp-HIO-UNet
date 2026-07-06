"""
utils.py —— 甲乙双方共用的工具函数

包含：图像预处理（扩画布 + 极性翻转 + pad）、可微 FFT/IFFT、随机相位初始化、
shrinkwrap 动态支撑域估计、直方图匹配（Zhang-Main）、评估指标。

约定：所有图像张量形状 [1,1,H,W]，float32；ρ 始终指"暗背景工作表示"（物体亮、背景≈0）。
"""

import math
from collections import Counter

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

# ===================== 设备 =====================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ===================== 尺寸对齐 =====================
def pad_to_multiple(x, m=32):
    """
    将 [1,1,H,W] 张量 pad 到 H,W 均为 m 的倍数（UNet 5 级下采样需要）。
    暗背景下背景值=0，故用常数 0 填充。
    返回 (padded, pad_info)，pad_info=(left,right,top,bottom) 供 unpad 使用。
    """
    _, _, H, W = x.shape
    new_H = math.ceil(H / m) * m
    new_W = math.ceil(W / m) * m
    top = (new_H - H) // 2
    bottom = new_H - H - top
    left = (new_W - W) // 2
    right = new_W - W - left
    padded = F.pad(x, (left, right, top, bottom), mode="constant", value=0.0)
    return padded, (left, right, top, bottom)


def unpad(x, pad_info):
    """pad_to_multiple 的逆操作。"""
    left, right, top, bottom = pad_info
    _, _, H, W = x.shape
    return x[..., top:H - bottom, left:W - right]


# ===================== 图像预处理 =====================
def load_and_preprocess(path, expand=2, m=32):
    """
    读图 → 灰度 → 估背景 → 扩 expand 倍画布 → 归一化 [0,1] → 极性翻转 → pad。
    返回 (rho_work, bg_val, pad_info, H0, W0)。
      rho_work: [1,1,H,W] 暗背景工作表示（物体亮、背景≈0），范围约 [0,1]
      bg_val:   背景灰度（归一化后），显示时用 bg_val - rho_work 还原视觉原貌
    """
    gray = np.asarray(Image.open(path).convert("L"), dtype=np.float32)
    H0, W0 = gray.shape

    # 众数估背景（参考 轮廓/main.py）
    bg_val_255 = float(Counter(gray.flatten().astype(np.uint8)).most_common(1)[0][0])

    # 扩 expand 倍画布，居中填背景（缓解 FFT 卷绕、保证过采样）
    new_H, new_W = H0 * expand, W0 * expand
    canvas = np.full((new_H, new_W), bg_val_255, dtype=np.float32)
    sh, sw = (new_H - H0) // 2, (new_W - W0) // 2
    canvas[sh:sh + H0, sw:sw + W0] = gray

    # 归一化 + 极性翻转（白背景黑物体 → 暗背景亮物体）
    img_01 = canvas / 255.0
    bg_val = bg_val_255 / 255.0
    rho_work = bg_val - img_01  # 物体亮、背景≈0

    rho_work = torch.from_numpy(rho_work).float().unsqueeze(0).unsqueeze(0).to(device)
    rho_work, pad_info = pad_to_multiple(rho_work, m)
    return rho_work, bg_val, pad_info, H0, W0


# ===================== 可微 FFT / IFFT =====================
def fft_amp_phase(x, keep_dc=True):
    """
    可微 FFT。x:[1,1,H,W] (real) -> (amp_raw, phase)，均 [1,1,H,W]。
    keep_dc=True 保留直流（相位恢复需要平均密度信息，去直流会导致迭代不收敛）。
    """
    F = torch.fft.fft2(x)
    if not keep_dc:
        F[..., 0, 0] = 0
    return torch.abs(F), torch.angle(F)


def ifft_real(amp, phase):
    """
    可微 IFFT。由 (amp, phase) 重建实空间密度，[1,1,H,W]。
    torch.ifft2 默认 norm='backward'（已除以 N），故与 fft_amp_phase 往返一致。
    """
    F = amp * torch.exp(1j * phase)
    return torch.real(torch.fft.ifft2(F))


# ===================== 投影算子（RAAR 反射用）=====================
def proj_S(x, support):
    """实空间投影 P_S：support 外 = 0、support 内 clamp_min(0)。幂等。
    RAAR 反射 R_S = 2·P_S − I 用（raar.py / unet_pr.run_unet_raar 共享）。"""
    return torch.where(support > 0.5, torch.clamp_min(x, 0), torch.zeros_like(x))


# ===================== 随机相位初始化 =====================
def make_random_phase(shape, seed=None):
    """
    生成随机相位谱，保证共轭对称（使 IFFT 结果为实数）。
    做法：实噪声 → FFT → 取相位。比手工拼接共轭更稳。
    seed=None（默认）不固定——GPU 非确定性（C6）下固定种子已无复现意义，每次自然随机；
    传 seed 则固定（调试用）。实验2/3 共享 ρ_init 是因 main 只算一次，非因种子。
    """
    if seed is not None:
        torch.manual_seed(seed)
    noise = torch.randn(shape, device=device)
    F = torch.fft.fft2(noise)
    F[..., 0, 0] = 0
    return torch.angle(F)


def init_density(amp_orig, phase_init):
    """由 (原始振幅, 初始相位) → IFFT → 初始实空间密度 ρ_0。"""
    return ifft_real(amp_orig, phase_init)


# ===================== shrinkwrap 动态支撑域 =====================
def gaussian_kernel_2d(sigma, dev):
    """生成 2D 高斯卷积核 [1,1,ks,ks]（参考 验证/custom.py）。"""
    ks = int(2 * math.ceil(3 * sigma) + 1)
    ax = torch.arange(ks, dtype=torch.float32, device=dev) - (ks - 1) / 2
    xx, yy = torch.meshgrid(ax, ax, indexing="ij")
    kernel = torch.exp(-(xx ** 2 + yy ** 2) / (2 * sigma ** 2))
    return (kernel / kernel.sum()).unsqueeze(0).unsqueeze(0)


def shrinkwrap_support(rho, sigma, thresh_frac=0.15):
    """
    shrinkwrap 动态估计支撑域。暗背景下物体=亮区，阈值方向"大于"。
    返回 mask:[1,1,H,W] {0,1}，物体区=1、背景区=0。
    自带 degenerate 防护（mask 占比过高/过低自动调阈值）。
    """
    kernel = gaussian_kernel_2d(sigma, rho.device)
    blurred = F.conv2d(rho, kernel, padding="same")
    thr = thresh_frac * blurred.max()
    mask = (blurred > thr).float()

    # 防 mask 过大（把背景也圈进来）
    if mask.mean() > 0.5:
        thr = thresh_frac * 2 * blurred.max()
        mask = (blurred > thr).float()
    # 防 mask 过小（物体没识别出来）
    if mask.mean() < 0.02:
        thr = thresh_frac * 0.5 * blurred.max()
        mask = (blurred > thr).float()
    return mask


def sigma_schedule(it, sigma0, sigma_end, total):
    """线性衰减 σ：sigma0 → sigma_end，在 total 步内。"""
    return sigma0 + (sigma_end - sigma0) * min(it / max(total, 1), 1.0)


class AdaptiveSigma:
    """Yoshida 2024 自适应 σ（论文 §3.6）：监控过采样比 OS ratio（total/support）变化幅度，
    剧变（|Δ|>delta_thresh）用大 σ 平滑、稳定切小 σ 锁定。替代固定 sigma_schedule，
    防 support 剧变→发散（十测组4 unet_raar 突崩即此类）。详见 调研报告.md §3.9、十一测计划.md §2.2。

    sigma_large/small 用项目 sigma0/sigma_end（保与固定 schedule 同值域，唯一变量是 σ 决定逻辑）；
    delta_thresh=2.0 是 Yoshida 论文原值。
    """

    def __init__(self, sigma_large=3.0, sigma_small=1.0, delta_thresh=2.0):
        self.sigma_large = sigma_large
        self.sigma_small = sigma_small
        self.delta_thresh = delta_thresh
        self.prev_os = None           # 上次 SW 的 OS ratio
        self.locked = False           # 一旦切小 σ 就锁死
        self.sigma = sigma_large      # 首次用大 σ

    def next(self, support):
        """根据当前 support mask [1,1,H,W] 返回本次 shrinkwrap 用的 σ。"""
        os_ratio = support.numel() / max(support.sum().item(), 1.0)  # OS ratio = total/support，>2
        if self.locked:
            return self.sigma_small
        if self.prev_os is not None:
            if abs(os_ratio - self.prev_os) > self.delta_thresh:   # support 剧变→大 σ 平滑
                self.sigma = self.sigma_large
            else:                                                    # support 稳定→小 σ 紧贴，锁定
                self.sigma = self.sigma_small
                self.locked = True
        self.prev_os = os_ratio
        return self.sigma


# ===================== 直方图匹配（Zhang-Main 分箱线性映射）=====================
def _quantile_edges(vals, n_bins):
    """对 vals 计算等数量分箱的 n_bins+1 个边界（每个 bin 含相同数量像素）。"""
    quantiles = torch.linspace(0, 1, n_bins + 1, device=vals.device)
    # torch.quantile 需要 float，且对大张量在 GPU 上可能慢；转 CPU 算再转回以保证稳定
    edges = torch.quantile(vals.detach().float().cpu(), quantiles.cpu()).to(vals.device)
    return edges


def estimate_reference_histogram(img_ref, support, n_bins=300):
    """
    从暗背景原图在 support 内统计参考直方图（用户决策：参考来源 = 原图统计）。
    返回 ref_edges：n_bins+1 个等数量分箱边界。
    """
    vals = img_ref[support > 0.5]
    return _quantile_edges(vals, n_bins)


@torch.no_grad()
def histogram_match(rho, support, ref_edges, n_bins=300):
    """
    Zhang-Main 直方图匹配：把 rho 在 support 内的密度直方图拉平到 ref_edges 定义的分布。
    每个像素按其所属 bin 做分段线性映射 cur_edges → ref_edges。support 外保持不变。
    非可微，对 detach 的迭代点施加。
    """
    mask = support > 0.5
    result = rho.clone()

    cur_vals = rho[mask]
    if cur_vals.numel() == 0:
        return rho

    cur_edges = _quantile_edges(cur_vals, n_bins)  # support 内的当前等数量边界
    vals = cur_vals.float()

    # 找每个值落在 cur 的哪个 bin（searchsorted 要求有序序列）
    idx = torch.searchsorted(cur_edges, vals, right=True) - 1
    idx = idx.clamp(0, n_bins - 1)

    c_lo = cur_edges[idx]
    c_hi = cur_edges[idx + 1]
    r_lo = ref_edges[idx]
    r_hi = ref_edges[idx + 1]

    # 分段线性映射，防除零
    denom = (c_hi - c_lo).clamp(min=1e-8)
    mapped = r_lo + (vals - c_lo) * (r_hi - r_lo) / denom

    result[mask] = mapped.to(result.dtype)
    return result


# ===================== 评估指标 =====================
def psnr(a, b):
    """峰值信噪比，[0,1] 范围下 MAX=1。a,b: [1,1,H,W] 张量。"""
    mse = torch.mean((a - b) ** 2)
    return (-10.0 * torch.log10(mse + 1e-12)).item()


def ssim(a, b, data_range=1.0):
    """
    结构相似性（Wang 2004），11x11 高斯窗口，torch 实现（不依赖 skimage）。
    复用 gaussian_kernel_2d(1.5) 恰好生成 11x11 核。a,b: [1,1,H,W] 张量。
    """
    C1 = (0.01 * data_range) ** 2
    C2 = (0.03 * data_range) ** 2
    window = gaussian_kernel_2d(1.5, a.device)  # sigma=1.5 → ks=11
    pad = window.shape[-1] // 2
    mu_a = F.conv2d(a, window, padding=pad)
    mu_b = F.conv2d(b, window, padding=pad)
    mu_a_sq = mu_a * mu_a
    mu_b_sq = mu_b * mu_b
    mu_ab = mu_a * mu_b
    sigma_a = F.conv2d(a * a, window, padding=pad) - mu_a_sq
    sigma_b = F.conv2d(b * b, window, padding=pad) - mu_b_sq
    sigma_ab = F.conv2d(a * b, window, padding=pad) - mu_ab
    num = (2 * mu_ab + C1) * (2 * sigma_ab + C2)
    den = (mu_a_sq + mu_b_sq + C1) * (sigma_a + sigma_b + C2)
    return (num / (den + 1e-12)).mean().item()


def pearson_cc(a, b):
    """Pearson 相关系数。a,b: [1,1,H,W] 张量。"""
    a = a.flatten().float()
    b = b.flatten().float()
    a_c = a - a.mean()
    b_c = b - b.mean()
    denom = (torch.sqrt((a_c ** 2).sum()) * torch.sqrt((b_c ** 2).sum())) + 1e-12
    return (a_c * b_c).sum().item() / denom.item()


def amplitude_cc(amp_exp, phase_exp, amp_calc, phase_calc):
    """
    振幅域相关系数（参考 验证/custom.py calculate_CC_torch）：
      CC = Σ(Ae·Ac·cosΔφ) / √(ΣAe²·ΣAc²)
    """
    delta = phase_exp - phase_calc
    numerator = (amp_exp * amp_calc * torch.cos(delta)).sum()
    denom = torch.sqrt((amp_exp ** 2).sum() * (amp_calc ** 2).sum()) + 1e-12
    return (numerator / denom).item()


def mean_phase_error(phase_exp, phase_calc):
    """
    平均相位误差（弧度），用 arccos(cos(·)) 防 2π 卷绕（参考 验证/custom.py）。
    注：受原点平移歧义影响，作为辅助指标。
    """
    delta = phase_exp - phase_calc
    return torch.mean(torch.arccos(torch.cos(delta))).item()


def support_iou(mask_a, mask_b):
    """支撑域交并比。mask_a, mask_b: [1,1,H,W] {0,1}。"""
    a = (mask_a > 0.5)
    b = (mask_b > 0.5)
    inter = (a & b).float().sum()
    union = (a | b).float().sum() + 1e-12
    return (inter / union).item()


def register_to_gt(rho, gt):
    """在傅里叶振幅等价的平凡歧义群上把 rho 配准到 gt。

    相位恢复的平凡歧义——整体平移、共轭反转(twin image)、全局相位因子(实信号即 ±符号)，
    三者傅里叶振幅谱完全相同（Fannjiang 2020；Guizar-Sicairos & Fienup 2012；见调研报告 §5.1）。
    评估前必须在此歧义群上对齐，否则 Δφ/amp_cc 等被平移污染（Δφ→π/2、amp_cc→0）。

    做法：对 4 个变体（±符号 × 正/共轭反转）各做一次 FFT 互相关（Kuglin & Hines 1975）
    定位最佳整数平移，再就平移正负两种 roll 取与 gt 归一化内积最大者，规避 FFT/roll
    符号约定歧义。返回对齐后的 rho（不修改 gt）。
    """
    G = torch.fft.fft2(gt)
    _, _, H, W = rho.shape
    g_norm = gt.norm() + 1e-12
    best_score = -2.0
    best_aligned = rho
    for sign in (1.0, -1.0):
        for flip in (False, True):
            var = sign * rho
            if flip:
                var = torch.flip(var, dims=(-2, -1))
            V = torch.fft.fft2(var)
            cc = torch.fft.ifft2(G.conj() * V).real            # 互相关，峰值位置=位移
            dy, dx = divmod(cc.argmax().item(), W)
            if dy > H // 2:
                dy -= H                                        # 折到 [-H/2, H/2)
            if dx > W // 2:
                dx -= W
            var_norm = var.norm() + 1e-12
            for sy, sx in ((dy, dx), (-dy, -dx)):              # 两种 roll 方向取优
                aligned = torch.roll(var, shifts=(sy, sx), dims=(-2, -1))
                score = ((aligned * gt).sum() / (g_norm * var_norm)).item()
                if score > best_score:
                    best_score = score
                    best_aligned = aligned
    return best_aligned


def evaluate_all(rho, gt, amp_orig, phase_orig, support_gt, hm_sigma=1.0, align_to_gt=False):
    """
    一次性计算全部指标。返回 dict。
    评估前先把 rho 投影到可行域（正值 + support 内），消除 HIO support 外反馈值的影响；
    甲乙双方评估方式一致。
      rho:        当前恢复的实空间密度 [1,1,H,W]（暗背景）
      gt:         真值密度（暗背景原图）
      amp_orig:   原始振幅（去直流）
      phase_orig: 原始相位
      support_gt: 真值支撑域
    """
    if align_to_gt:
        rho = register_to_gt(rho, gt)                  # 平凡歧义群配准（调研报告 §5.2）
    support_pred = shrinkwrap_support(rho, hm_sigma)
    rho_eval = rho.clamp_min(0) * support_pred  # 投影到可行域
    amp_calc, phase_calc = fft_amp_phase(rho_eval)
    return {
        "psnr": psnr(rho_eval, gt),
        "ssim": ssim(rho_eval, gt),
        "pearson_cc": pearson_cc(rho_eval, gt),
        "amp_cc": amplitude_cc(amp_orig, phase_orig, amp_calc, phase_calc),
        "phase_err": mean_phase_error(phase_orig, phase_calc),
        "support_iou": support_iou(support_pred, support_gt),
    }

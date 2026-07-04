"""
unet_pr.py —— 乙方：未训练 UNet 相位恢复（Deep Image Prior 路线）

像空间：UNet 输出做 FFT 取振幅，与 |A_orig| 算 MSE，反传更新网络（软学习，对照甲方硬替换）。
实空间：HIO 反馈公式（support 内保留 ρ̃ / support 外 ρ_k−β·ρ̃）+ 直方图匹配 + shrinkwrap，
        与甲方逐字对齐（控制变量；二测修复：补齐此前缺失的 HIO 反馈项）。
"""

import time
import torch
import torch.nn as nn
import torch.nn.functional as F

from utils import (fft_amp_phase, shrinkwrap_support, sigma_schedule,
                   histogram_match, evaluate_all, device)


# ===================== UNet 结构（自己重写，InstanceNorm 适配单图训练）=====================
class DoubleConv(nn.Module):
    """Conv3x3 → InstanceNorm → ReLU × 2。"""

    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1),
            nn.InstanceNorm2d(out_ch, affine=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1),
            nn.InstanceNorm2d(out_ch, affine=True),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.double_conv(x)


class Down(nn.Module):
    """MaxPool → DoubleConv。"""

    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.maxpool_conv = nn.Sequential(nn.MaxPool2d(2), DoubleConv(in_ch, out_ch))

    def forward(self, x):
        return self.maxpool_conv(x)


class Up(nn.Module):
    """bilinear 上采样 + 跳连拼接 + DoubleConv。"""

    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.up = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True)
        self.conv = DoubleConv(in_ch, out_ch)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        diffY = x2.size(2) - x1.size(2)
        diffX = x2.size(3) - x1.size(3)
        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2, diffY // 2, diffY - diffY // 2])
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)


class OutConv(nn.Module):
    """1x1 卷积 + Sigmoid，输出约束到 [0,1]。"""

    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.conv = nn.Conv2d(in_ch, out_ch, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        return self.sigmoid(self.conv(x))


class UNet(nn.Module):
    """5 级下采样 UNet，bilinear 上采样，瓶颈通道减半。"""

    def __init__(self, n_channels=1, n_classes=1):
        super().__init__()
        factor = 2  # bilinear 时瓶颈通道减半
        self.inc = DoubleConv(n_channels, 64)
        self.down1 = Down(64, 128)
        self.down2 = Down(128, 256)
        self.down3 = Down(256, 512)
        self.down4 = Down(512, 1024 // factor)  # 瓶颈 512

        self.up1 = Up(1024, 512 // factor)
        self.up2 = Up(512, 256 // factor)
        self.up3 = Up(256, 128 // factor)
        self.up4 = Up(128, 64)
        self.outc = OutConv(64, n_classes)

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        return self.outc(x)


# ===================== 未训练 UNet 迭代 =====================
def run_unet(amp_orig, rho_init, ref_edges, gt, support_gt,
             max_iter=5000, lr=1e-4, beta=0.7,
             sigma0=3.0, sigma_end=1.0, sigma_interval=20, sigma_total=500,
             eval_every=20, unet_seed=0):
    """
    未训练 UNet（DIP）迭代。
      amp_orig:   原始振幅（去直流）[1,1,H,W]
      rho_init:   初始实空间密度 ρ_0（第一轮"先实空间"起手）
      ref_edges:  HM 参考直方图分箱边界
      gt:         真值密度（评估用）
      support_gt: 真值支撑域（评估用）
      beta:       HIO 反馈系数（与甲方一致，默认 0.7）
    返回 (best_rho, history)。
    """
    torch.manual_seed(unet_seed)
    model = UNet().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    mse = nn.MSELoss()

    amp_max = amp_orig.max().detach() + 1e-12  # 振幅归一化基准（让 loss 尺度合理）
    amp_orig_norm = amp_orig / amp_max
    _, phase_orig = fft_amp_phase(gt)

    current_input = rho_init.detach().clone()  # 第一轮"先实空间"：从 ρ_0 起手
    support = shrinkwrap_support(rho_init, sigma0)

    metric_keys = ["psnr", "ssim", "pearson_cc", "amp_cc", "phase_err", "support_iou"]
    history = {"iter": []}
    for k in metric_keys:
        history[k] = []
    best_score = -1.0
    best_rho = rho_init.clone()

    start = time.time()
    for it in range(max_iter):
        optimizer.zero_grad()

        # === 像空间（乙方唯一变量）：UNet 前向 → 振幅 MSE 反传（对照甲方硬替换）===
        raw = model(current_input)                              # ρ̃，sigmoid [0,1]，天然非负
        amp_pred = fft_amp_phase(raw)[0] / amp_max              # 全局振幅（不预先 ×support，对齐甲方全局替换）
        loss = mse(amp_pred, amp_orig_norm)
        loss.backward()
        optimizer.step()

        # === 实空间（与甲方对齐）：HIO 反馈公式 + 动态 support + HM ===
        with torch.no_grad():
            raw_d = raw.detach()
            keep = (support > 0.5) & (raw_d >= 0)
            rho_next = torch.where(keep, raw_d, current_input - beta * raw_d)  # 补上 HIO 反馈项
            rho_next = histogram_match(rho_next, support, ref_edges)

            if it > 0 and it % sigma_interval == 0:
                sigma = sigma_schedule(it, sigma0, sigma_end, sigma_total)
                support = shrinkwrap_support(rho_next, sigma)

            if it % eval_every == 0:
                m = evaluate_all(rho_next, gt, amp_orig, phase_orig, support_gt)
                history["iter"].append(it)
                for k in metric_keys:
                    history[k].append(m[k])
                score = m["ssim"] + m["amp_cc"]
                if score > best_score:
                    best_score = score
                    best_rho = rho_next.clone()
                elapsed = time.time() - start
                print(f"[UNet] {it}/{max_iter} | loss {loss.item():.3e} | "
                      f"psnr {m['psnr']:.2f} | ssim {m['ssim']:.3f} | "
                      f"amp_cc {m['amp_cc']:.3f} | Δφ {m['phase_err']:.3f} | "
                      f"iou {m['support_iou']:.3f} | {elapsed:.1f}s")

            current_input = rho_next.detach()

    return best_rho, history

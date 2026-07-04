"""smoke test：少轮次验证代码能跑通（验证环境/数据流，避免一上来跑完整轮次太久）。

用法：D:\\anaconda3\\envs\\use\\python.exe smoke_test.py
用途：跑通基础函数 + HIO/UNet 少轮次迭代，确认环境无误后再用 main.py 跑完整实验。
      （HIO 100 轮 + UNet 50 轮只需几分钟；不要求收敛，只验证数据流正常）
"""
import os
import sys

import torch

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
os.chdir(os.path.dirname(os.path.abspath(__file__)))

from utils import (load_and_preprocess, fft_amp_phase, ifft_real, make_random_phase,
                   init_density, shrinkwrap_support, estimate_reference_histogram, device)
from hio import run_hio
from unet_pr import run_unet

print(f"设备: {device}\n")

# 1. 读图 + 基础函数验证
rho_work, bg_val, pad_info, H0, W0 = load_and_preprocess('567.png')
print(f"rho_work {tuple(rho_work.shape)}, bg_val={bg_val:.3f}")

# FFT 往返（保留直流，应精确复原 rho_work）
amp, phase = fft_amp_phase(rho_work)
rho_back = ifft_real(amp, phase)
err = (rho_back - rho_work).abs().max().item()
print(f"FFT 往返误差（应≈0，数值精度）: {err:.2e}")

# 极性翻转：去直流后，非直流振幅应完全相同（控制变量的数学依据）
amp_work, _ = fft_amp_phase(rho_work, keep_dc=False)
amp_img, _ = fft_amp_phase(bg_val - rho_work, keep_dc=False)
polar_rel = (amp_work - amp_img).abs().max().item() / (amp_work.max().item() + 1e-12)
print(f"极性翻转非直流振幅相对误差（应≈0）: {polar_rel:.2e}")

mask = shrinkwrap_support(rho_work, 3.0)
print(f"support 占比: {mask.mean():.3f}")

# 2. 共享初始化
phase0 = make_random_phase(rho_work.shape, seed=42)
rho_init = init_density(amp, phase0)
ref_edges = estimate_reference_histogram(rho_work, mask)
print(f"rho_init range: [{rho_init.min():.3f}, {rho_init.max():.3f}]\n")

# 3. HIO 少轮次跑通验证（100 轮，几分钟内）
print("--- HIO 100 轮（跑通验证）---")
run_hio(amp, rho_init, ref_edges, rho_work, mask, max_iter=100, eval_every=50)

# 4. UNet 少轮次跑通验证（50 轮）
print("\n--- UNet 50 轮（跑通验证）---")
run_unet(amp, rho_init, ref_edges, rho_work, mask, max_iter=50, eval_every=25)

print("\n[通过] smoke test 结束，环境与数据流正常，可用 main.py 跑完整实验")

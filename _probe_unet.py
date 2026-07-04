"""临时探测脚本：验证 unet_pr.py 二测修复（补 HIO 反馈）是否生效。
看 Δφ 是否离开 π/2、amp_cc 是否启动。用完即删，不入 Git。
"""
import os
import sys

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
os.chdir(os.path.dirname(os.path.abspath(__file__)))

from utils import (load_and_preprocess, fft_amp_phase, make_random_phase,
                   init_density, shrinkwrap_support, estimate_reference_histogram)
from unet_pr import run_unet

SIGMA0 = 3.0
N_ITER = int(sys.argv[1]) if len(sys.argv) > 1 else 2000

rho_work, bg_val, pad_info, H0, W0 = load_and_preprocess('567.png')
amp_orig, _ = fft_amp_phase(rho_work)
phase0 = make_random_phase(rho_work.shape, seed=42)
rho_init = init_density(amp_orig, phase0)
support_gt = shrinkwrap_support(rho_work, SIGMA0)
ref_edges = estimate_reference_histogram(rho_work, support_gt, n_bins=300)

print(f"\n===== UNet（二测修复后）探测 {N_ITER} 轮 =====")
run_unet(amp_orig, rho_init, ref_edges, rho_work, support_gt,
         max_iter=N_ITER, lr=1e-4, beta=0.7, sigma0=SIGMA0, eval_every=50)

"""Correctness + speed of the Triton NF4 kernel vs bitsandbytes, on layer 0's real weights.

Timing uses triton.testing.do_bench, which flushes L2 between runs: the RTX 4050's 24 MB L2 would otherwise hold
the 8 MB q/o weight matrices and report impossible bandwidth.

Run:  uv run python profiling/kernels/bench_nf4.py
"""
import sys
import warnings
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
warnings.filterwarnings("ignore")

import bitsandbytes as bnb
import bitsandbytes.functional as bnbF
import torch
from triton.testing import do_bench

from common import load_model, measure_hardware, save
from kernels.nf4_gemm import NF4Weight

BATCHES = [1, 2, 3, 4, 6, 8]


def main():
    peak_bw = measure_hardware()["bandwidth_GBps"]
    _, model = load_model()
    layer = model.model.layers[0]
    rows = []
    for name in ["self_attn.q_proj", "self_attn.k_proj", "mlp.gate_proj", "mlp.down_proj"]:
        mod = layer.get_submodule(name)
        w = NF4Weight(mod.weight)
        ref_w = bnbF.dequantize_4bit(mod.weight.data, mod.weight.quant_state).float()
        bytes_read = w.packed.numel() + mod.weight.quant_state.absmax.numel()  # what bnb reads per call
        for M in BATCHES:
            x = torch.randn(M, 1, w.K, device="cuda", dtype=torch.bfloat16)
            ref = (x.float() @ ref_w.T)
            mine, theirs = w.matmul(x).float(), mod(x).float()
            err_mine = ((mine - ref).norm() / ref.norm()).item()
            err_bnb = ((theirs - ref).norm() / ref.norm()).item()
            t_bnb = do_bench(lambda: mod(x), warmup=25, rep=200)
            t_mine = do_bench(lambda: w.matmul(x), warmup=25, rep=200)
            row = {"layer": name, "N": w.N, "K": w.K, "M": M,
                   "bnb_us": round(t_bnb * 1e3, 1), "triton_us": round(t_mine * 1e3, 1),
                   "speedup": round(t_bnb / t_mine, 2),
                   "bnb_pct_bw": round(100 * bytes_read / (t_bnb * 1e-3) / 1e9 / peak_bw, 1),
                   "triton_pct_bw": round(100 * bytes_read / (t_mine * 1e-3) / 1e9 / peak_bw, 1),
                   "rel_err_triton": f"{err_mine:.1e}", "rel_err_bnb": f"{err_bnb:.1e}"}
            rows.append(row)
            print(f"{name:17s} {w.N:5d}x{w.K:<5d} M={M}: bnb {row['bnb_us']:7.1f} us ({row['bnb_pct_bw']:5.1f}% BW)"
                  f" | triton {row['triton_us']:7.1f} us ({row['triton_pct_bw']:5.1f}% BW)  x{row['speedup']}"
                  f"  err {row['rel_err_triton']} (bnb {row['rel_err_bnb']})")
    save("kernel_nf4_gemm", {"peak_bw_GBps": peak_bw, "rows": rows})


if __name__ == "__main__":
    main()

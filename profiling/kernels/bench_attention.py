"""Correctness + speed of the Triton decode-attention kernel vs the stock path, on Mistral's shapes.

Stock path = what transformers does on this setup at decode: repeat_kv (8 -> 32 heads) + torch SDPA, which on the
Windows wheel lands on the math backend. Bandwidth is measured against the KV bytes that must be read once.

Run:  uv run python profiling/kernels/bench_attention.py
"""
import sys
import warnings
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
warnings.filterwarnings("ignore")

import torch
import torch.nn.functional as F
from triton.testing import do_bench
from transformers.integrations.sdpa_attention import repeat_kv

from common import measure_hardware, save
from kernels.decode_attention import decode_attention

H, KVH, D = 32, 8, 128
SCALE = D ** -0.5


def stock(q, k, v):
    out = F.scaled_dot_product_attention(q, repeat_kv(k, H // KVH), repeat_kv(v, H // KVH), scale=SCALE)
    return out.transpose(1, 2).contiguous()


def main():
    peak_bw = measure_hardware()["bandwidth_GBps"]
    rows = []
    for B in (1, 4, 8):
        for S in (128, 512, 2048, 4096):
            q = torch.randn(B, H, 1, D, device="cuda", dtype=torch.bfloat16)
            k = torch.randn(B, KVH, S, D, device="cuda", dtype=torch.bfloat16)
            v = torch.randn(B, KVH, S, D, device="cuda", dtype=torch.bfloat16)
            ref = F.scaled_dot_product_attention(q.float(), repeat_kv(k, 4).float(), repeat_kv(v, 4).float(),
                                                 scale=SCALE).transpose(1, 2)
            err = ((decode_attention(q, k, v, SCALE).float() - ref).norm() / ref.norm()).item()
            err_stock = ((stock(q, k, v).float() - ref).norm() / ref.norm()).item()
            t_stock = do_bench(lambda: stock(q, k, v), warmup=25, rep=200)
            t_mine = do_bench(lambda: decode_attention(q, k, v, SCALE), warmup=25, rep=200)
            kv_bytes = 2 * k.numel() * k.element_size()
            row = {"B": B, "S": S, "stock_us": round(t_stock * 1e3, 1), "triton_us": round(t_mine * 1e3, 1),
                   "speedup": round(t_stock / t_mine, 1),
                   "triton_pct_bw": round(100 * kv_bytes / (t_mine * 1e-3) / 1e9 / peak_bw, 1),
                   "rel_err_triton": f"{err:.1e}", "rel_err_stock": f"{err_stock:.1e}"}
            rows.append(row)
            print(f"B={B} S={S:5d}: stock {row['stock_us']:7.1f} us | triton {row['triton_us']:6.1f} us "
                  f"({row['triton_pct_bw']:5.1f}% BW)  x{row['speedup']}  err {row['rel_err_triton']} (stock {row['rel_err_stock']})")
    save("kernel_decode_attention", {"peak_bw_GBps": peak_bw, "rows": rows})


if __name__ == "__main__":
    main()

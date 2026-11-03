"""Fused NF4 dequant + skinny matmul in Triton, for decode (a few tokens per forward).

Computes  Y[M, N] = X[M, K] @ W[N, K]^T  directly from bitsandbytes' NF4 storage:
  * W flattened row-major, two 4-bit codes per byte (high nibble = even element, low = odd)
  * code -> value via the 16-entry NF4 table
  * one scale ("absmax") per 64 consecutive weights

Why it exists: for M > 1 bitsandbytes switches from its fused GEMV to `gemm_4bit_sm80`, which streams the weights
at only ~30% of DRAM bandwidth on this GPU. Decode is memory-bound, so the goal is to read each packed byte once.

Design, arrived at by ablation (REPORT.md section 9.1):
  * 2D tile [BLOCK_N rows, BLOCK_K/2 bytes]; scales loaded once per 64-weight block and broadcast
  * X's even / odd columns are loaded separately, so the nibbles never need interleaving:
        X @ W^T = X_even @ W_hi^T + X_odd @ W_lo^T
  * CUDA-core FMAs with one reduction per token per tile. tl.dot was ~3x slower here: converting freshly
    dequantized registers into tensor-core operand layout every tile costs more than it saves at M <= 16.
The cost grows with M, so it beats bitsandbytes only in a band of batch sizes; `NF4Weight` measures both
at load time and dispatches per (shape, M).
"""
import torch
import triton
import triton.language as tl
from triton.testing import do_bench


@triton.autotune(
    configs=[
        triton.Config({"BLOCK_N": bn, "BLOCK_K": bk}, num_warps=w, num_stages=s)
        for bn in (8, 16, 32, 64)
        for bk in (128, 256)
        for w in (2, 4)
        for s in (2, 3)
    ],
    key=["M", "N", "K"],
)
@triton.jit
def _nf4_gemm_kernel(x_ptr, w_ptr, absmax_ptr, code2_ptr, absmax2_ptr, offset, code_ptr, y_ptr,
                     M, N, K, stride_xm, stride_ym,
                     NESTED: tl.constexpr, M_PAD: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr):
    pid = tl.program_id(0)
    rows = pid * BLOCK_N + tl.arange(0, BLOCK_N)              # output features handled by this program
    j = tl.arange(0, BLOCK_K // 2)                             # packed-byte column within the K tile
    m = tl.arange(0, M_PAD)
    NB: tl.constexpr = BLOCK_K // 64                           # quant blocks per tile
    acc = tl.zeros((M_PAD, BLOCK_N), dtype=tl.float32)

    for k0 in range(0, K, BLOCK_K):
        p = tl.load(w_ptr + rows[:, None] * (K // 2) + (k0 // 2 + j)[None, :])         # [BN, BK/2] uint8
        blk = rows[:, None] * (K // 64) + (k0 // 64 + tl.arange(0, NB))[None, :]           # [BN, NB] quant-block ids
        if NESTED:  # bitsandbytes double quant: scale = code2[u8] * absmax2[blk // 256] + offset
            sc = tl.load(code2_ptr + tl.load(absmax_ptr + blk).to(tl.int32)) * tl.load(absmax2_ptr + blk // 256) + offset
        else:
            sc = tl.load(absmax_ptr + blk).to(tl.float32)
        sc = tl.reshape(tl.broadcast_to(sc[:, :, None], (BLOCK_N, NB, 32)), (BLOCK_N, BLOCK_K // 2))
        w_hi = tl.load(code_ptr + (p >> 4).to(tl.int32)) * sc
        w_lo = tl.load(code_ptr + (p & 0xF).to(tl.int32)) * sc
        for mi in tl.static_range(M_PAD):
            if mi < M:
                xe = tl.load(x_ptr + mi * stride_xm + k0 + 2 * j).to(tl.float32)
                xo = tl.load(x_ptr + mi * stride_xm + k0 + 2 * j + 1).to(tl.float32)
                part = tl.sum(w_hi * xe[None, :] + w_lo * xo[None, :], axis=1)          # [BN]
                acc += tl.where(m[:, None] == mi, part[None, :], 0.0)

    tl.store(y_ptr + m[:, None] * stride_ym + rows[None, :], acc.to(tl.bfloat16), mask=(m < M)[:, None])


class NF4Weight:
    """Triton view of one bitsandbytes Linear4bit weight. Reads bitsandbytes' tensors in place (packed weights and
    the nested 8-bit scales), so it costs no extra VRAM. A first version pre-dequantized the scales to fp16
    (+220 MB), which was enough to push batch-8 decode over the 6 GB edge into system-RAM spill."""

    # Below MIN_M decode is CPU/launch-bound, and a Triton launch from Python costs more CPU time than bitsandbytes'
    # precompiled kernel. do_bench can't see that (it times back-to-back GPU work), so a kernel that "wins" in
    # calibration at M=1 still made batch-1 decode ~1.5 ms/step slower end to end.
    MIN_M, MAX_M = 3, 8

    def __init__(self, param):
        qs = param.quant_state
        assert qs.quant_type == "nf4" and qs.blocksize == 64, (qs.quant_type, qs.blocksize)
        self.N, self.K = qs.shape
        assert self.K % 256 == 0 and self.N % 64 == 0, (self.N, self.K)
        self.packed = param.data.view(-1)
        self.nested = bool(qs.nested)
        if self.nested:
            assert qs.state2.blocksize == 256, qs.state2.blocksize
            self.absmax, self.code2, self.absmax2 = qs.absmax, qs.state2.code, qs.state2.absmax
            self.offset = float(qs.offset)
        else:
            self.absmax = qs.absmax.float()
            self.code2 = self.absmax2 = self.absmax  # unused
            self.offset = 0.0
        self.code = qs.code.float().contiguous()

    def matmul(self, x):
        """x: [..., K] bf16 with at most MAX_M rows -> [..., N] bf16"""
        lead = x.shape[:-1]
        x2 = x.reshape(-1, self.K)
        if x2.stride(-1) != 1:
            x2 = x2.contiguous()
        M = x2.shape[0]
        y = torch.empty((M, self.N), device=x.device, dtype=torch.bfloat16)
        grid = lambda meta: (self.N // meta["BLOCK_N"],)
        _nf4_gemm_kernel[grid](x2, self.packed, self.absmax, self.code2, self.absmax2, self.offset, self.code, y,
                               M, self.N, self.K, x2.stride(0), y.stride(0),
                               NESTED=self.nested, M_PAD=triton.next_power_of_2(M))
        return y.reshape(*lead, self.N)


def calibrate(module, w: NF4Weight, min_m=NF4Weight.MIN_M, max_m=NF4Weight.MAX_M):
    """For each M, time bitsandbytes vs Triton on this layer (L2 flushed) and return the set of M where Triton wins."""
    wins, table = set(), {}
    for M in range(min_m, max_m + 1):
        x = torch.randn(M, 1, w.K, device="cuda", dtype=torch.bfloat16)
        t_bnb = do_bench(lambda: module.forward_bnb(x), warmup=10, rep=40)
        t_tri = do_bench(lambda: w.matmul(x), warmup=10, rep=40)
        table[M] = (round(t_bnb * 1e3, 1), round(t_tri * 1e3, 1))
        if t_tri < 0.90 * t_bnb:  # require a clear (>10%) win; otherwise keep the reference implementation
            wins.add(M)
    return wins, table

"""Flash-decoding attention for one new token per sequence, with grouped-query attention, in Triton.

Replaces SDPA's math fallback for decode. The profile showed that path upcasting the KV cache to fp32, expanding
K/V from 8 to 32 heads (copies), and running small fp32 GEMVs: ~5 ms/step at batch 1, ~41 ms at batch 4.

  q: [B, H, 1, D]   k, v: [B, KVH, S, D]   ->   out: [B, 1, H, D]

Each program owns one (sequence, KV head) pair and one chunk of the sequence, and computes attention for the
H/KVH query heads that share that KV head, so the KV cache is read exactly once, in bf16. Splitting the sequence
across programs ("flash-decoding") keeps all SMs busy even at batch 1, where there are only 8 KV heads. A small
second kernel merges the per-chunk partial softmaxes.
"""
import torch
import triton
import triton.language as tl


@triton.jit
def _partial_kernel(q_ptr, k_ptr, v_ptr, acc_ptr, ml_ptr, S, scale,
                    sq_b, sq_h, sk_b, sk_h, sk_s, sv_b, sv_h, sv_s,
                    KVH: tl.constexpr, GROUP: tl.constexpr, G_PAD: tl.constexpr, D: tl.constexpr,
                    NSPLIT, BLOCK_S: tl.constexpr, CHUNK: tl.constexpr):
    bh = tl.program_id(0)                       # b * KVH + kv_head
    split = tl.program_id(1)
    b, kvh = bh // KVH, bh % KVH
    g = tl.arange(0, G_PAD)
    d = tl.arange(0, D)
    heads = kvh * GROUP + g
    g_ok = g < GROUP

    q = tl.load(q_ptr + b * sq_b + heads[:, None] * sq_h + d[None, :], mask=g_ok[:, None], other=0.0)
    m_i = tl.full((G_PAD,), float("-inf"), tl.float32)
    l_i = tl.zeros((G_PAD,), tl.float32)
    acc = tl.zeros((G_PAD, D), tl.float32)

    start = split * CHUNK
    for s0 in range(start, start + CHUNK, BLOCK_S):
        s = s0 + tl.arange(0, BLOCK_S)
        s_ok = s < S
        k = tl.load(k_ptr + b * sk_b + kvh * sk_h + s[:, None] * sk_s + d[None, :], mask=s_ok[:, None], other=0.0)
        v = tl.load(v_ptr + b * sv_b + kvh * sv_h + s[:, None] * sv_s + d[None, :], mask=s_ok[:, None], other=0.0)
        qk = tl.dot(q, tl.trans(k)) * scale                      # [G_PAD, BLOCK_S]
        qk = tl.where(s_ok[None, :], qk, float("-inf"))
        m_new = tl.maximum(m_i, tl.max(qk, axis=1))
        m_safe = tl.where(m_new == float("-inf"), 0.0, m_new)    # chunk tail entirely past S
        p = tl.exp(qk - m_safe[:, None])
        alpha = tl.exp(m_i - m_safe)
        l_i = l_i * alpha + tl.sum(p, axis=1)
        acc = acc * alpha[:, None] + tl.dot(p.to(tl.bfloat16), v)
        m_i = m_new

    out = (b * KVH * GROUP + heads) * NSPLIT + split               # [B, H, NSPLIT] flattened
    tl.store(acc_ptr + out[:, None] * D + d[None, :], acc, mask=g_ok[:, None])
    tl.store(ml_ptr + out * 2, m_i, mask=g_ok)
    tl.store(ml_ptr + out * 2 + 1, l_i, mask=g_ok)


@triton.jit
def _merge_kernel(acc_ptr, ml_ptr, o_ptr, so_b, so_h,
                  NSPLIT, H: tl.constexpr, D: tl.constexpr, NS_PAD: tl.constexpr):
    bh = tl.program_id(0)                       # b * H + head
    b, h = bh // H, bh % H
    sp = tl.arange(0, NS_PAD)
    d = tl.arange(0, D)
    ok = sp < NSPLIT
    m = tl.load(ml_ptr + (bh * NSPLIT + sp) * 2, mask=ok, other=float("-inf"))
    l = tl.load(ml_ptr + (bh * NSPLIT + sp) * 2 + 1, mask=ok, other=0.0)
    m_max = tl.max(m, axis=0)
    w = tl.where(ok & (m > float("-inf")), tl.exp(m - m_max), 0.0)
    acc = tl.load(acc_ptr + (bh * NSPLIT + sp)[:, None] * D + d[None, :], mask=ok[:, None], other=0.0)
    out = tl.sum(acc * w[:, None], axis=0) / tl.sum(l * w, axis=0)
    tl.store(o_ptr + b * so_b + h * so_h + d, out.to(tl.bfloat16))


def decode_attention(q, k, v, scale, chunk=64, block_s=32):
    """q [B, H, 1, D], k/v [B, KVH, S, D] (bf16, last dim contiguous) -> [B, 1, H, D] bf16."""
    B, H, _, D = q.shape
    KVH, S = k.shape[1], k.shape[2]
    group = H // KVH
    nsplit = triton.cdiv(S, chunk)
    acc = torch.empty((B * H * nsplit, D), device=q.device, dtype=torch.float32)
    ml = torch.empty((B * H * nsplit, 2), device=q.device, dtype=torch.float32)
    out = torch.empty((B, 1, H, D), device=q.device, dtype=q.dtype)
    _partial_kernel[(B * KVH, nsplit)](
        q, k, v, acc, ml, S, scale,
        q.stride(0), q.stride(1), k.stride(0), k.stride(1), k.stride(2), v.stride(0), v.stride(1), v.stride(2),
        KVH=KVH, GROUP=group, G_PAD=max(16, triton.next_power_of_2(group)), D=D,
        NSPLIT=nsplit, BLOCK_S=block_s, CHUNK=chunk, num_warps=4)
    _merge_kernel[(B * H,)](acc, ml, out, out.stride(0), out.stride(2), nsplit,
                            H=H, D=D, NS_PAD=triton.next_power_of_2(nsplit), num_warps=4)
    return out

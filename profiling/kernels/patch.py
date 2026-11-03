"""Swap the custom Triton kernels into a loaded Hugging Face Mistral model.

  patch_attention()      decode steps (q_len == 1, no padding mask) -> Triton flash-decoding; everything else -> SDPA
  patch_linear4bit(m)    every bitsandbytes Linear4bit -> Triton NF4 kernel for the batch sizes where it measured
                         faster on that layer shape, bitsandbytes otherwise
"""
import types

import bitsandbytes as bnb
from transformers import AttentionInterface
from transformers.integrations.sdpa_attention import sdpa_attention_forward

from kernels.decode_attention import decode_attention
from kernels.nf4_gemm import NF4Weight, calibrate

STATS = {"attn_triton": 0, "attn_fallback": 0}


def _attention(module, query, key, value, attention_mask, scaling=None, **kwargs):
    if query.shape[2] == 1 and attention_mask is None and query.dtype == key.dtype:
        STATS["attn_triton"] += 1
        return decode_attention(query, key, value, scaling), None
    STATS["attn_fallback"] += 1
    return sdpa_attention_forward(module, query, key, value, attention_mask, scaling=scaling, **kwargs)


def patch_attention():
    # Re-registering "sdpa" keeps transformers' sdpa mask creation (which returns None when no mask is needed).
    AttentionInterface.register("sdpa", _attention)


def unpatch_attention():
    AttentionInterface.register("sdpa", sdpa_attention_forward)


def _forward(self, x):
    M = x.numel() // x.shape[-1]
    if M in self.triton_m:
        return self.nf4.matmul(x.to(self.compute_dtype)).to(x.dtype)
    return self.forward_bnb(x)


def patch_linear4bit(model, verbose=True):
    """Returns {(N, K): (set of M using Triton, {M: (bnb_us, triton_us)})}."""
    by_shape = {}
    for name, mod in model.named_modules():
        if not isinstance(mod, bnb.nn.Linear4bit):
            continue
        mod.nf4 = NF4Weight(mod.weight)
        mod.forward_bnb = types.MethodType(bnb.nn.Linear4bit.forward, mod)
        shape = (mod.nf4.N, mod.nf4.K)
        if shape not in by_shape:
            by_shape[shape] = calibrate(mod, mod.nf4)
            if verbose:
                wins, table = by_shape[shape]
                print(f"  {name:28s} {shape}: Triton for M in {sorted(wins) or '-'}   (bnb_us, triton_us) {table}")
        mod.triton_m = by_shape[shape][0]
        mod.forward = types.MethodType(_forward, mod)
    return by_shape

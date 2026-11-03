"""Step 4 - Before / after: apply the custom kernels one at a time and measure each step's effect.

Same methodology as 01_baseline (CUDA events, warm-up, median of 64 decode steps at ~550 tokens of context),
in one process so every variant sees the same weights and GPU state. Also checks the optimized model still
generates the same text: greedy tokens vs the stock model, and max logit difference on the first step.

Run:  $env:SDPA_BACKEND="cudnn"; uv run python profiling/04_compare.py
"""
import statistics

import torch

from common import (cuda_timer, decode_step, gpu_info, load_model, make_input_ids, measure_hardware, model_shape,
                    now, prefill, save)
from kernels.patch import STATS, patch_attention, patch_linear4bit, unpatch_attention

BATCH_SIZES = [1, 2, 4, 8]
CONTEXT, STEPS, CHECK_TOKENS = 512, 64, 48


@torch.inference_mode()
def greedy(model, ids, n):
    past, nxt = prefill(model, ids)
    out, first_logits = [nxt], None
    for i in range(n - 1):
        o = model(input_ids=nxt, past_key_values=past, use_cache=True)
        if i == 0:
            first_logits = o.logits[:, -1].float()
        past, nxt = o.past_key_values, o.logits[:, -1:].argmax(-1)
        out.append(nxt)
    return torch.cat(out, 1), first_logits


def bench(model, tok, shape, hw):
    rows = {}
    for bs in BATCH_SIZES:
        ids = make_input_ids(tok, CONTEXT, bs)
        past, nxt = prefill(model, ids)
        for _ in range(5):
            past, nxt = decode_step(model, nxt, past)
        times = []
        for _ in range(STEPS):
            with cuda_timer(times):
                past, nxt = decode_step(model, nxt, past)
        del past
        torch.cuda.empty_cache()
        ms = statistics.median(times)
        floor = (shape["decode_weight_bytes"] + bs * (CONTEXT + 37) * shape["kv_bytes_per_token"]) / (hw["bandwidth_GBps"] * 1e9) * 1e3
        rows[bs] = {"ms_per_step": round(ms, 2), "tokens_per_s": round(bs / ms * 1e3, 1),
                    "roofline_floor_ms": round(floor, 2), "pct_of_roofline": round(100 * floor / ms, 1)}
    return rows


def main():
    hw = measure_hardware()
    tok, model = load_model()
    shape = model_shape(model)
    check_ids = make_input_ids(tok, 64, 4)
    variants = {}

    def run(label):
        print(f"\n=== {label} ===")
        tokens, logits = greedy(model, check_ids, CHECK_TOKENS)
        rows = bench(model, tok, shape, hw)
        for bs, r in rows.items():
            print(f"  decode bs={bs}: {r['ms_per_step']:7.2f} ms/step  {r['tokens_per_s']:6.1f} tok/s  "
                  f"({r['pct_of_roofline']}% of roofline, floor {r['roofline_floor_ms']} ms)")
        variants[label] = {"decode": rows, "tokens": tokens, "logits": logits}

    run("stock")
    patch_attention()
    run("+ triton decode attention")
    print("\nCalibrating Triton NF4 kernel vs bitsandbytes per layer shape ...")
    calibration = patch_linear4bit(model)
    run("+ triton NF4 matmul")
    unpatch_attention()

    ref = variants["stock"]
    print("\n=== correctness vs stock (greedy, 4 sequences x 48 tokens) ===")
    summary = {}
    for label, v in variants.items():
        match = (v["tokens"] == ref["tokens"]).float().mean().item()
        first_div = [(row != rrow).nonzero()[0].item() if (row != rrow).any() else None
                     for row, rrow in zip(v["tokens"], ref["tokens"])]
        dlogit = (v["logits"] - ref["logits"]).abs().max().item()
        print(f"  {label:28s} tokens identical: {100 * match:5.1f}%  first divergence per seq: {first_div}  "
              f"max |logit diff| step 1: {dlogit:.3f}")
        summary[label] = {"decode": v["decode"], "token_match_pct": round(100 * match, 1),
                          "first_divergence": first_div, "max_logit_diff_step1": round(dlogit, 4)}
    print("attention calls:", STATS)
    save("04_compare", {"when": now(), "gpu": gpu_info(), "hardware": hw, "variants": summary,
                        "nf4_calibration": {f"{n}x{k}": {"triton_for_M": sorted(w), "bnb_vs_triton_us": t}
                                            for (n, k), (w, t) in calibration.items()}})


if __name__ == "__main__":
    main()

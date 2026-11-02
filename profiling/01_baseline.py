"""Step 1 - Baseline: how fast is it, and how far from the hardware limit?

Measures:
  * achievable GPU bandwidth / fp16 TFLOPS (the roofline ceilings)
  * prefill latency (TTFT) vs prompt length   -> expected compute-bound
  * decode latency per token vs batch size     -> expected memory-bandwidth-bound
and converts each into "% of roofline" so you know which resource is the bottleneck.

Run:  uv run python profiling/01_baseline.py
"""
import statistics

import torch

from common import (cuda_timer, decode_step, gpu_info, load_model, make_input_ids, measure_hardware,
                    model_shape, now, prefill, save)

PROMPT_LENS = [128, 512, 1024, 2048]
BATCH_SIZES = [1, 2, 4, 8]
DECODE_CONTEXT = 512
DECODE_STEPS = 64


def bench_prefill(model, tok, shape, hw):
    rows = []
    for n in PROMPT_LENS:
        torch.cuda.empty_cache()  # cached blocks from the previous length would otherwise inflate peak VRAM
        torch.cuda.reset_peak_memory_stats()
        ids = make_input_ids(tok, n)
        for _ in range(3):  # warm-up: one call is not enough right after model load
            prefill(model, ids)
        times = []
        for _ in range(5):
            with cuda_timer(times):
                prefill(model, ids)
        ms = statistics.median(times)
        flops = 2 * shape["params_linear"] * n
        tflops = flops / (ms / 1e3) / 1e12
        rows.append({"prompt_tokens": n, "ttft_ms": round(ms, 1), "tokens_per_s": round(n / ms * 1e3),
                     "achieved_TFLOPS": round(tflops, 2),
                     "pct_of_peak_compute": round(100 * tflops / hw["fp16_TFLOPS"], 1),
                     "arith_intensity": round(flops / shape["weight_bytes"], 1),
                     "peak_mem_GB": round(torch.cuda.max_memory_allocated() / 1e9, 2)})
        print(f"prefill {n:5d} tok: {ms:7.1f} ms  {rows[-1]['tokens_per_s']:6d} tok/s  "
              f"{tflops:5.2f} TFLOPS ({rows[-1]['pct_of_peak_compute']}% of peak)  peak {rows[-1]['peak_mem_GB']} GB")
    return rows


def bench_decode(model, tok, shape, hw):
    rows = []
    for bs in BATCH_SIZES:
        try:
            ids = make_input_ids(tok, DECODE_CONTEXT, bs)
            past, nxt = prefill(model, ids)
            for _ in range(5):  # warm-up
                past, nxt = decode_step(model, nxt, past)
            times = []
            for _ in range(DECODE_STEPS):
                with cuda_timer(times):
                    past, nxt = decode_step(model, nxt, past)
        except torch.OutOfMemoryError:
            print(f"decode bs={bs}: OOM, stopping batch sweep")
            break
        finally:
            past = None
            torch.cuda.empty_cache()
        ms = statistics.median(times)
        ctx = DECODE_CONTEXT + 5 + DECODE_STEPS // 2
        bytes_moved = shape["decode_weight_bytes"] + bs * ctx * shape["kv_bytes_per_token"]
        gbps = bytes_moved / (ms / 1e3) / 1e9
        rows.append({"batch": bs, "ms_per_step": round(ms, 2), "tokens_per_s": round(bs / ms * 1e3, 1),
                     "achieved_GBps": round(gbps, 1),
                     "pct_of_peak_bw": round(100 * gbps / hw["bandwidth_GBps"], 1),
                     "roofline_min_ms": round(bytes_moved / (hw["bandwidth_GBps"] * 1e9) * 1e3, 2),
                     "arith_intensity": round(2 * shape["params_linear"] * bs / bytes_moved, 2),
                     "peak_mem_GB": round(torch.cuda.max_memory_allocated() / 1e9, 2)})
        torch.cuda.reset_peak_memory_stats()
        print(f"decode bs={bs}: {ms:6.2f} ms/step  {rows[-1]['tokens_per_s']:7.1f} tok/s  "
              f"{gbps:6.1f} GB/s ({rows[-1]['pct_of_peak_bw']}% of peak bw, "
              f"roofline floor {rows[-1]['roofline_min_ms']} ms)")
    return rows


def main():
    print("Measuring hardware ceilings ...")
    hw = measure_hardware()
    print(hw)
    tok, model = load_model()
    shape = model_shape(model)
    print({k: f"{v:,}" for k, v in shape.items()})
    print(f"VRAM after load: {torch.cuda.memory_allocated() / 1e9:.2f} GB\n")
    result = {"when": now(), "gpu": gpu_info(), "hardware": hw, "model_shape": shape,
              "prefill": bench_prefill(model, tok, shape, hw),
              "decode": bench_decode(model, tok, shape, hw)}
    save("01_baseline", result)


if __name__ == "__main__":
    main()

"""Step 3 - Workload for Nsight Systems (timeline) and Nsight Compute (per-kernel deep dive).

NVTX ranges ("prefill", "decode_step") label the timeline, and cudaProfilerStart/Stop limit capture
to the measured region so model loading and warm-up are excluded.

Nsight Systems (whole-program timeline: CPU threads, CUDA API, kernels, memcpy, GPU idle gaps):
  nsys profile -t cuda,nvtx,cublas --capture-range=cudaProfilerApi --capture-range-end=stop ^
       --cuda-memory-usage=true -o results/nsys_decode -f true .venv/Scripts/python.exe profiling/03_nsight_target.py
  nsys stats --report cuda_gpu_kern_sum,cuda_api_sum,nvtx_sum results/nsys_decode.nsys-rep

Nsight Compute (hardware counters for individual kernels: DRAM throughput, SM occupancy, roofline):
  ncu --nvtx --nvtx-include "decode_step/" --launch-skip 0 --launch-count 40 --set full ^
      -o results/ncu_decode -f .venv/Scripts/python.exe profiling/03_nsight_target.py --steps 2

Run plain (no profiler) to sanity-check:  uv run python profiling/03_nsight_target.py
"""
import argparse

import torch
import torch.cuda.nvtx as nvtx

from common import decode_step, load_model, make_input_ids, prefill


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--prompt", type=int, default=512)
    ap.add_argument("--steps", type=int, default=32)
    ap.add_argument("--batch", type=int, default=1)
    args = ap.parse_args()

    tok, model = load_model()
    ids = make_input_ids(tok, args.prompt, args.batch)

    past, nxt = prefill(model, ids)
    for _ in range(3):
        past, nxt = decode_step(model, nxt, past)
    del past
    torch.cuda.synchronize()

    torch.cuda.profiler.start()
    nvtx.range_push("prefill")
    past, nxt = prefill(model, ids)
    torch.cuda.synchronize()
    nvtx.range_pop()
    for _ in range(args.steps):
        nvtx.range_push("decode_step")
        past, nxt = decode_step(model, nxt, past)
        torch.cuda.synchronize()
        nvtx.range_pop()
    torch.cuda.profiler.stop()
    print(f"done: prefill {args.prompt} tokens + {args.steps} decode steps, batch {args.batch}")


if __name__ == "__main__":
    main()

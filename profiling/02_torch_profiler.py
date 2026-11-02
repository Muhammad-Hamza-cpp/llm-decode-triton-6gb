"""Step 2 - PyTorch profiler: which operators / kernels take the time?

Profiles one prefill and N decode steps, each wrapped in record_function() so they show up as
named ranges. Produces:
  results/02_trace[_cudnn]_bs1.json   Chrome trace -> open in https://ui.perfetto.dev
  results/02_ops_table_*.txt   operator table sorted by GPU time (prefill and decode separately)
  results/02_breakdown_bs1*.json  GPU idle %, kernels/step, time by kernel category
  results/02_memory_*.pickle   (--memory) allocator snapshot -> https://pytorch.org/memory_viz

Run:  uv run python profiling/02_torch_profiler.py [--prompt 512] [--steps 32] [--batch 1] [--memory] [--stack]
"""
import argparse

import torch
from torch.profiler import ProfilerActivity, profile, record_function

from common import RESULTS, TAG, decode_step, load_model, make_input_ids, prefill, save
from trace_analysis import analyze, print_report


def sort_key(prof):
    evt = prof.key_averages()[0]
    return "self_device_time_total" if hasattr(evt, "self_device_time_total") else "self_cuda_time_total"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--prompt", type=int, default=512)
    ap.add_argument("--steps", type=int, default=32)
    ap.add_argument("--batch", type=int, default=1)
    ap.add_argument("--memory", action="store_true", help="also record a CUDA memory snapshot")
    ap.add_argument("--stack", action="store_true", help="record Python stacks (bigger trace, more overhead)")
    args = ap.parse_args()

    tok, model = load_model()
    ids = make_input_ids(tok, args.prompt, args.batch)

    # Warm-up: first calls pay for cuBLAS heuristics, allocator growth, lazy init.
    past, nxt = prefill(model, ids)
    for _ in range(3):
        past, nxt = decode_step(model, nxt, past)
    del past
    torch.cuda.synchronize()

    if args.memory:
        torch.cuda.memory._record_memory_history(max_entries=200_000)

    with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
                 record_shapes=True, profile_memory=True, with_stack=args.stack) as prof:
        with record_function("prefill"):
            past, nxt = prefill(model, ids)
        for i in range(args.steps):
            with record_function(f"decode_step_{i}"):
                past, nxt = decode_step(model, nxt, past)
        torch.cuda.synchronize()

    if args.memory:
        path = RESULTS / f"02_memory{TAG}_bs{args.batch}.pickle"
        torch.cuda.memory._dump_snapshot(str(path))
        torch.cuda.memory._record_memory_history(enabled=None)
        print(f"Memory snapshot: {path}  (drag into https://pytorch.org/memory_viz)")

    key = sort_key(prof)
    table = prof.key_averages().table(sort_by=key, row_limit=30, max_name_column_width=60)
    shapes = prof.key_averages(group_by_input_shape=True).table(sort_by=key, row_limit=20,
                                                                max_name_column_width=40)
    (RESULTS / f"02_ops_table{TAG}_bs{args.batch}.txt").write_text(
        f"== by operator ==\n{table}\n\n== by input shape ==\n{shapes}\n", encoding="utf-8")
    print(table)

    trace = RESULTS / f"02_trace{TAG}_bs{args.batch}.json"
    prof.export_chrome_trace(str(trace))
    print(f"Chrome trace: {trace}  (open in https://ui.perfetto.dev)")

    report = analyze(trace)
    print_report(report)
    save(f"02_breakdown_bs{args.batch}", {"args": vars(args), **report})


if __name__ == "__main__":
    main()

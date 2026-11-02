"""Bottleneck breakdown from a torch.profiler Chrome trace (also usable standalone).

For each phase marked with record_function (prefill, decode_step) it reports:
  * wall time on the GPU timeline vs time the GPU was actually running kernels
    -> GPU idle % = time lost to CPU-side overhead (Python, dispatcher, kernel launches)
  * kernel launches per step and their average duration
  * GPU time split by kernel category (matmul, 4-bit dequant, attention, elementwise ...)

Run:  uv run python profiling/trace_analysis.py results/02_trace.json
"""
import json
import re
import sys
from collections import defaultdict

CATEGORIES = [
    ("4bit_gemv_dequant", r"dequant|gemv_4bit|kgemm_4bit|kQuantize|bnb"),
    ("matmul_gemm", r"gemm|gemv|cutlass|xmma|s16816|s1688|splitK|cublas"),
    ("attention", r"flash|fmha|attn|attention|softmax"),
    ("memcpy_memset", r"memcpy|memset"),
    ("norm_act_elementwise", r"elementwise|vectorized|reduce|rms|silu|mul|add|pow|mean|rsqrt|neg|cat|unrolled|index|copy"),
]


def categorize(name):
    for cat, pattern in CATEGORIES:
        if re.search(pattern, name, re.IGNORECASE):
            return cat
    return "other"


def union_length(intervals):
    total, cur_s, cur_e = 0.0, None, None
    for s, e in sorted(intervals):
        if cur_e is None or s > cur_e:
            if cur_e is not None:
                total += cur_e - cur_s
            cur_s, cur_e = s, e
        else:
            cur_e = max(cur_e, e)
    if cur_e is not None:
        total += cur_e - cur_s
    return total


def analyze(trace_path, phases=("prefill", "decode_step")):
    events = json.load(open(trace_path, encoding="utf-8"))["traceEvents"]
    kernels = [e for e in events if e.get("cat") in ("kernel", "gpu_memcpy", "gpu_memset") and "dur" in e]
    gpu_marks = [e for e in events if e.get("cat") == "gpu_user_annotation"]
    cpu_marks = [e for e in events if e.get("cat") == "user_annotation"]

    report = {}
    for phase in phases:
        windows = [(e["ts"], e["ts"] + e["dur"]) for e in gpu_marks if e["name"].startswith(phase)]
        cpu_durs = [e["dur"] for e in cpu_marks if e["name"].startswith(phase)]
        if not windows:
            continue
        span = sum(e - s for s, e in windows)
        inside = [k for k in kernels if any(s <= k["ts"] < e for s, e in windows)]
        busy = union_length([(k["ts"], k["ts"] + k["dur"]) for k in inside])
        by_cat, top = defaultdict(float), defaultdict(lambda: [0, 0.0])
        for k in inside:
            by_cat[categorize(k["name"])] += k["dur"]
            top[k["name"]][0] += 1
            top[k["name"]][1] += k["dur"]
        kt = sum(by_cat.values()) or 1
        n = len(windows)
        report[phase] = {
            "steps": n,
            "gpu_span_ms_per_step": round(span / n / 1e3, 3),
            "cpu_ms_per_step": round(sum(cpu_durs) / max(len(cpu_durs), 1) / 1e3, 3),
            "gpu_busy_ms_per_step": round(busy / n / 1e3, 3),
            "gpu_idle_pct": round(100 * (1 - busy / span), 1) if span else 0,
            "kernels_per_step": round(len(inside) / n),
            "avg_kernel_us": round(kt / max(len(inside), 1), 1),
            "time_by_category_pct": {c: round(100 * t / kt, 1)
                                     for c, t in sorted(by_cat.items(), key=lambda x: -x[1])},
            "top_kernels": [{"name": name[:110], "calls": c, "total_ms": round(t / 1e3, 2),
                             "pct": round(100 * t / kt, 1), "category": categorize(name)}
                            for name, (c, t) in sorted(top.items(), key=lambda x: -x[1][1])[:12]],
        }
    return report


def print_report(report):
    for phase, r in report.items():
        print(f"\n=== {phase} ({r['steps']} step(s)) ===")
        print(f"  GPU-timeline span / step : {r['gpu_span_ms_per_step']} ms")
        print(f"  CPU time / step          : {r['cpu_ms_per_step']} ms")
        print(f"  GPU busy / step          : {r['gpu_busy_ms_per_step']} ms   -> GPU idle {r['gpu_idle_pct']}%")
        print(f"  kernels / step           : {r['kernels_per_step']}  (avg {r['avg_kernel_us']} us each)")
        print("  GPU time by category     :", r["time_by_category_pct"])
        print("  top kernels:")
        for k in r["top_kernels"]:
            print(f"    {k['pct']:5.1f}%  {k['total_ms']:8.2f} ms  x{k['calls']:<5d} [{k['category']}] {k['name']}")


if __name__ == "__main__":
    print_report(analyze(sys.argv[1]))

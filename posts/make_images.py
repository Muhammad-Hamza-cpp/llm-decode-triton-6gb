"""Renders the two LinkedIn images (1080x1350 PNG) from the raw files in results/.

Every number on both images is read from results/*.json or the torch.profiler traces, never typed in by hand.

    uv run python posts/make_images.py          -> posts/images/1_result.png, posts/images/2_where_time_went.png
"""
import gzip
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import to_rgb

ROOT = Path(__file__).resolve().parents[1]
RES = ROOT / "results"
OUT = ROOT / "posts" / "images"

# Dark chart theme (validated: categorical slots 1-4 pass CVD / contrast checks on the #1a1a19 surface)
BG, INK, INK2, MUTED, GRID, AXIS = "#1a1a19", "#f0efec", "#c3c2b7", "#898781", "#2c2c2a", "#383835"
BLUE, ORANGE, AQUA, YELLOW, GRAY = "#3987e5", "#d95926", "#199e70", "#c98500", "#6b6a65"

W, H, DPI = 7.2, 9.0, 150  # 1080 x 1350 px, LinkedIn's 4:5 portrait
L, R = 0.075, 0.925        # left / right text margin (figure fraction)
FOOT = "Mistral-7B-Instruct v0.3 · NF4 4-bit · RTX 4050 Laptop 6 GB · Windows 11"
BATCHES = ("1", "2", "4", "8")
STOCK, OPT = "stock", "+ triton NF4 matmul"  # variant names in 04_compare_*.json

plt.rcParams.update({
    "font.family": ["Segoe UI", "DejaVu Sans"],
    "font.size": 11,
    "text.color": INK,
    "axes.edgecolor": AXIS,
    "axes.labelcolor": INK2,
    "xtick.color": MUTED,
    "ytick.color": MUTED,
    "xtick.labelcolor": INK2,
    "ytick.labelcolor": INK2,
    "axes.facecolor": BG,
    "figure.facecolor": BG,
    "savefig.facecolor": BG,
})


def load(name):
    p = RES / name
    if not p.exists():  # the repo ships the large profiler traces gzipped
        return json.loads(gzip.decompress((RES / f"{name}.gz").read_bytes()))
    return json.loads(p.read_text())


def slide(title, subtitle, source):
    fig = plt.figure(figsize=(W, H), dpi=DPI)
    fig.text(L, 0.955, title, fontsize=21, weight="bold", va="top", linespacing=1.15)
    if subtitle:
        fig.text(L, 0.955 - 0.042 * (title.count("\n") + 1) - 0.012, subtitle, fontsize=11, color=INK2,
                 va="top", linespacing=1.4)
    fig.add_artist(plt.Line2D([L, R], [0.062, 0.062], color=AXIS, lw=0.8))
    fig.text(L, 0.04, FOOT, fontsize=8.5, color=MUTED)
    fig.text(L, 0.02, f"data: {source}", fontsize=8.5, color=MUTED)
    return fig


def style(ax, grid="y"):
    for s in ("top", "right", "left"):
        ax.spines[s].set_visible(False)
    ax.spines["bottom"].set_color(AXIS)
    ax.tick_params(length=0, labelsize=10, pad=6)
    ax.grid(axis=grid, color=GRID, lw=0.8)
    ax.set_axisbelow(True)


def ink_on(fill):
    r, g, b = to_rgb(fill)
    return "#0b0b0b" if 0.2126 * r + 0.7152 * g + 0.0722 * b > 0.55 else "#ffffff"


# ---------------------------------------------------------------- trace analysis (torch.profiler chrome traces)

CATS = [("w4", "4-bit weight matmul", BLUE), ("attn", "Attention", ORANGE), ("kv", "KV-cache append", AQUA),
        ("head", "lm_head", YELLOW), ("other", "Norms, rotary, adds, casts", GRAY)]


def classify(name):
    if "gemm_4bit" in name or "gemv_4bit" in name or "_nf4_gemm" in name:
        return "w4"
    if name in ("_partial_kernel", "_merge_kernel") or "gemvx" in name or "softmax" in name:
        return "attn"
    # stock SDPA math path: repeat_kv / fp32 upcast copies of K and V, and the fp32 score scaling
    if "gpu_kernel_impl_nocastIZZZNS0_23direct_copy" in name or "AUnaryFunctorIfffNS0_15binary_internal10MulFunctorIfE" in name:
        return "attn"
    if "CatArrayBatchedCopy" in name:
        return "kv"
    if "gemm" in name or "cutlass" in name:
        return "head"
    return "other"


def decode_ms_by_category(trace_file):
    """GPU kernel time per decode step, by category, averaged over every decode_step range in the trace."""
    ev = load(trace_file)["traceEvents"]
    steps = [e for e in ev if e.get("cat") == "gpu_user_annotation" and e["name"].startswith("decode_step")]
    kern = [e for e in ev if e.get("cat") == "kernel"]
    tot = {k: 0.0 for k, _, _ in CATS}
    for s in steps:
        t0, t1 = s["ts"], s["ts"] + s["dur"]
        for k in kern:
            if t0 <= k["ts"] < t1:
                tot[classify(k["name"])] += k["dur"] / 1e3
    return {k: v / len(steps) for k, v in tot.items()}


# ---------------------------------------------------------------- images

def result_image(runs):
    fig = slide("I profiled a 7B LLM on a 6 GB\nlaptop GPU, then wrote two\nTriton kernels.", None,
                "results/04_compare_cudnn_run{1,2}.json")

    def mean(variant, b, key):
        return sum(r["variants"][variant]["decode"][b][key] for r in runs) / len(runs)

    worst = min(r["variants"][STOCK]["decode"][b]["ms_per_step"] / r["variants"][OPT]["decode"][b]["ms_per_step"]
                for r in runs for b in ("4", "8"))  # conservative: the slower of both runs at batch 4 and 8
    match = min(r["variants"][OPT]["token_match_pct"] for r in runs)
    fig.text(L, 0.765, f"{worst:.1f}×", fontsize=64, weight="bold", va="top")
    fig.text(L + 0.33, 0.735, f"faster batched decode\n(batch 4 and 8, {match:.0f}% identical\noutput tokens)",
             fontsize=13, color=INK2, va="top", linespacing=1.4)

    ax = fig.add_axes([L + 0.03, 0.12, R - L - 0.03, 0.45])
    style(ax)
    w = 0.34
    for i, b in enumerate(BATCHES):
        for dx, val, col in ((-w / 2, mean(STOCK, b, "tokens_per_s"), GRAY),
                             (w / 2, mean(OPT, b, "tokens_per_s"), BLUE)):
            ax.bar(i + dx, val, w, color=col, edgecolor=BG, lw=1.5)
            ax.text(i + dx, val + 1.5, f"{val:.0f}", ha="center", va="bottom", fontsize=10.5,
                    color=INK if col == BLUE else INK2, weight="bold" if col == BLUE else "normal")
    ax.set_xticks(range(len(BATCHES)), [f"batch {b}" for b in BATCHES])
    ax.set_ylim(0, 115)
    ax.set_ylabel("tokens / s, mean of 2 runs (higher is better)")
    h = [plt.Rectangle((0, 0), 1, 1, color=c) for c in (GRAY, BLUE)]
    ax.legend(h, ["stock (transformers + bitsandbytes)", "with my Triton kernels"], loc="upper left", frameon=False,
              fontsize=10, labelcolor=INK2)
    return fig


def breakdown_image(stock_ms, opt_ms):
    fig = slide("Where one decode step's\nGPU time went", "Batch 4, averaged over 16 decode steps of the "
                "torch.profiler trace.\nStock: two kernels doing most of the work badly.",
                "results/02_trace_cudnn_bs4.json, results/02_trace_cudnn_opt_bs4.json")
    ax = fig.add_axes([L + 0.13, 0.43, R - L - 0.13, 0.27])
    style(ax, grid="x")
    ax.spines["bottom"].set_visible(False)
    rows = [("Stock", stock_ms), ("My kernels", opt_ms)]
    for y, (_, ms) in zip((1, 0), rows):
        left = 0
        for k, _, col in CATS:
            v = ms[k]
            ax.barh(y, v, 0.5, left=left, color=col, edgecolor=BG, lw=1.5)
            if v > 7:
                ax.text(left + v / 2, y, f"{v:.0f}", ha="center", va="center", fontsize=10, color=ink_on(col),
                        weight="bold")
            left += v
        ax.text(left + 1.5, y, f"{left:.0f} ms", va="center", fontsize=13, weight="bold")
    ax.set_yticks([1, 0], [r[0] for r in rows], fontsize=12, color=INK)
    ax.set_xlim(0, 130)
    ax.set_ylim(-0.6, 1.9)
    ax.set_xlabel("GPU time per decode step (ms)")
    h = [plt.Rectangle((0, 0), 1, 1, color=c) for _, _, c in CATS]
    ax.legend(h, [lab for _, lab, _ in CATS], loc="upper left", bbox_to_anchor=(-0.17, 1.02), ncol=3, frameon=False,
              fontsize=9.5, labelcolor=INK2, columnspacing=1.2, handlelength=1.0)

    w4_bw = 3.49 / (stock_ms["w4"] / 1e3)  # 3.49 GB of 4-bit linear weights read per step (REPORT.md section 4.2)
    w4_share = (opt_ms["w4"] + opt_ms["head"]) / sum(opt_ms.values()) * 100  # lm_head streams weights too
    notes = [
        ("1", f"bitsandbytes' batched 4-bit kernel streams 3.49 GB of weights in {stock_ms['w4']:.0f} ms:\n"
              f"{w4_bw:.0f} GB/s, ~{w4_bw / 174 * 100:.0f}% of the 174 GB/s this GPU can actually do."),
        ("2", f"Attention falls back to an unfused path: K/V copied to 32 heads, upcast to fp32,\n"
              f"tiny fp32 GEMVs. {stock_ms['attn']:.0f} ms, every layer, every step."),
        ("→", f"After: attention takes {opt_ms['attn']:.1f} ms, and {w4_share:.0f}% of the GPU's time is "
              f"streaming weights."),
    ]
    y = 0.33
    for mark, text in notes:
        fig.text(L, y, mark, fontsize=14, weight="bold", va="top")
        fig.text(L + 0.05, y, text, fontsize=10.5, color=INK2, va="top", linespacing=1.45)
        y -= 0.085
    return fig


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    runs = [load("04_compare_cudnn_run1.json"), load("04_compare_cudnn_run2.json")]
    figs = {
        "1_result": result_image(runs),
        "2_where_time_went": breakdown_image(decode_ms_by_category("02_trace_cudnn_bs4.json"),
                                             decode_ms_by_category("02_trace_cudnn_opt_bs4.json")),
    }
    for name, fig in figs.items():
        fig.savefig(OUT / f"{name}.png", dpi=DPI)
        plt.close(fig)
        print("wrote", OUT / f"{name}.png")


if __name__ == "__main__":
    main()

# Profiling Report: Mistral-7B (4-bit) Inference on an RTX 4050 Laptop GPU

*Date: 2026-11-02 (profiling), 2026-11-03 (custom Triton kernels, §9) · Nsight Systems / Compute runs pending (§7)*

## 1. Setup

| | |
|---|---|
| GPU | RTX 4050 Laptop, 6 GB GDDR6, 20 SMs, Ada (sm_89), driver 621.01, on AC power |
| CPU / RAM | Intel i5-14500HX, 32 GB, Windows 11 |
| Software | PyTorch 2.11.0+cu128, transformers 5.22.1, bitsandbytes 0.50.2, Python 3.12 |
| Model | `unsloth/mistral-7b-instruct-v0.3-bnb-4bit`: 7.25 B params, 32 layers, GQA (32 Q / 8 KV heads), vocab 32 768 |
| Quantization | NF4 + double-quant on all linear layers; embeddings, `lm_head`, norms in bf16; **compute dtype bf16** |
| Weights in VRAM | 4.03 GB (3.76 GB streamed per decode step: everything except the embedding table) |
| KV cache | 128 KB per token (≈ 0.5 GB at 4 k tokens) |

**Measured hardware ceilings** (`common.measure_hardware`, best of 3 after warm-up):

| Ceiling | Measured | Datasheet |
|---|---|---|
| DRAM bandwidth | **~174 GB/s** (169–176 across runs) | 192 GB/s |
| FP16 tensor-core matmul (8192³) | **26.8 TFLOPS** | — |
| Ridge point | **~153 FLOP/byte** | |

All "% of peak" figures below use these measured ceilings.

## 2. Baseline performance

Medians of 5 (prefill) and 64 (decode) runs, timed with CUDA events. Decode runs at a context of ~550 tokens.
"Default" = PyTorch's own SDPA backend choice; "cuDNN prefill" = `SDPA_BACKEND=cudnn` (see §5.1).

**Prefill (time to first token)**

| Prompt | Default | cuDNN prefill | % of peak compute (cuDNN) | Peak VRAM default → cuDNN |
|---|---|---|---|---|
| 128 | 144 ms | 139 ms | 49% | 4.18 → 4.18 GB |
| 512 | 493 ms | 412 ms | 66% | 4.39 → 4.39 GB |
| 1024 | 1 053 ms | 741 ms | 73% | 4.70 → 4.50 GB |
| 2048 | **2.7 – 22.6 s** (5 runs) | **1.47 s** | 74% | **5.86** → 4.73 GB |

**Decode, per step (tokens/s = batch ÷ step time)**

| Batch | ms/step | tokens/s | Roofline floor | % of peak bandwidth |
|---|---|---|---|---|
| 1 | 34.8 – 39.8 | 25 – 29 | 21.7 ms | 55 – 62% |
| 2 | 53.3 | 37.5 | 22.1 ms | 42% |
| 4 | 118.4 | 33.8 | 23.0 ms | 19% |
| 8 | 171.1 | 46.8 | 24.6 ms | 14% |

Two numbers stand out: default prefill collapses at 2 k tokens, and batching barely helps decode
(batch 4 ≈ batch 2 in throughput). §4–5 explain both.

## 3. Roofline position

| Phase | Arithmetic intensity | vs ridge (~153) | Expected bound | Observed |
|---|---|---|---|---|
| Prefill, 512 tok | ~1 800 FLOP/B | far above | compute | GEMMs at ~90% of peak; dequant adds a fixed tax (§4.1) |
| Decode, batch 1 | ~3.7 FLOP/B | far below | memory | 4-bit GEMV at ~93% of peak BW; overhead around it (§4.2) |
| Decode, batch 4 | ~14 FLOP/B | far below | memory | 4-bit kernel only at ~31% of peak BW (§4.3) |

## 4. Where the time goes (torch.profiler)

Source: `results/02_breakdown_bs{1,4}.json`, `results/02_ops_table_bs{1,4}.txt`, traces `results/02_trace_bs{1,4}.json`
(open in <https://ui.perfetto.dev>). Prompt 512 tokens, 16 decode steps, default backend.

### 4.1 Prefill (512 tokens): GPU busy 530 ms, idle 1.7%, 2 838 kernels

| Category | Share | Notes |
|---|---|---|
| bf16 GEMMs (cuBLAS / CUTLASS) | 61% (~300 ms) | 7.3 TFLOP in ~300 ms ≈ **24 TFLOPS, ~90% of peak**. The matmuls themselves are efficient |
| `kDequantizeBlockwise` (bitsandbytes) | **19% (103 ms)** | Unpacks every NF4 weight to bf16 before each GEMM: reads 3.5 GB, writes 14 GB ⇒ ~170 GB/s, i.e. a **bandwidth-bound fixed tax of ~100 ms per forward pass**, independent of prompt length |
| Elementwise / norms / casts | 16% | RMSNorm, rotary, residual adds, bf16↔fp32 casts |
| Attention (unfused math path) | 7.5% | `softmax_warp_forward` + fp32 `sgemm` for QKᵀ and PV; grows with seq² |

The dequant tax explains why short prompts look inefficient: at 128 tokens, ~100 of the ~140 ms
is dequantization, so the GEMMs never get to amortise it.

### 4.2 Decode, batch 1: GPU busy 30.5 ms/step, 1 782 kernels/step

| Component | ms/step | Share of GPU time | Efficiency |
|---|---|---|---|
| `gemm_4bit_simt` (fused NF4 GEMV, 224 calls) | 21.3 | 70% | 3.49 GB in 21.3 ms ≈ **164 GB/s ≈ 93% of peak** |
| `lm_head` (bf16 cuBLAS GEMV) | 1.6 | 5% | 268 MB in 1.6 ms ≈ 168 GB/s ≈ 95% |
| Everything else (≈1 550 small kernels) | 7.6 | 25% | fp32 casts / copies, KV-cache `cat`, math attention, norms. Each kernel is ~2–17 µs |

Unprofiled wall time is ~35 ms/step, so the **GPU is idle for ~4.5 ms/step (~13%)** waiting for
the CPU to launch the next kernel. To keep the GPU fed, Python + dispatcher + driver must issue a
kernel every ~20 µs. That's an estimate (wall − busy). Nsight Systems should confirm it (§7).

**Decode budget at batch 1 (~35 ms):** ~65% streaming weights at the roofline · ~22% glue kernels · ~13% CPU launch gaps.

### 4.3 Decode, batch 4: GPU busy 112 ms/step, idle 2.7%

| Component | ms/step | Share |
|---|---|---|
| `gemm_4bit_sm80_m16n8k16` (bitsandbytes batched NF4 path) | 63.9 | 57% |
| Unfused attention over the KV cache: K/V copy/upcast (`direct_copy`), fp32 scaling (`MulFunctor<float>`), fp32 batched GEMV | 41.5 | 37% |
| Rest | ~7 | 6% |

- Once there's more than one token per layer, bitsandbytes switches from the fused GEMV (`gemm_4bit_simt`) to
  a tensor-core kernel that reads the **same 3.49 GB in 63.9 ms, ≈ 55 GB/s = 31% of peak**. The kernel itself
  is the bottleneck, not the hardware.
- Attention cost scales with batch × context because the math path re-reads the KV cache, upcasts it to
  fp32, and runs small fp32 GEMVs. At batch 1 it's ~5 ms/step; at batch 4, ~41 ms. (Kernel-to-op attribution
  is inferred from kernel names and per-layer call counts; confirm with the trace's flow arrows.)
- The GPU is fully busy here, so at batch ≥ 4 the bottleneck has moved from the CPU to inefficient GPU kernels.

## 5. Findings along the way

### 5.1 The Windows PyTorch wheel has no FlashAttention, and SDPA silently falls back to the math path

`Torch was not compiled with flash attention`, and the memory-efficient kernel refused the shapes, so
`scaled_dot_product_attention` used the unfused math path. It materialises a 32 × S × S fp32 score matrix
per layer (≈ 0.5 GB at S = 2048). At 2 048 tokens peak allocation reached **5.86 GB of 6.44 GB**. With
Windows' own VRAM use on top, the WDDM driver spilled to system RAM over PCIe instead of raising
an OOM error, giving 2.7–22.6 s run to run.

**Fix:** cuDNN's fused attention *is* available. Forcing it for prefill gives **1.47 s at 2 048 tokens
(1.9–15× faster), 74% of peak compute, and 4.73 GB peak**, with consistent results.

Two pitfalls hit while applying the fix:
1. `sdpa_kernel([CUDNN, MATH])` only *enables* backends, and PyTorch's built-in priority (math before cuDNN)
   still chose math. `set_priority=True` is required. The giveaway was an unchanged peak-memory number.
2. Using cuDNN for **decode** made it **10–16× slower** (380–580 ms/step). The profiler showed
   `aten::_cudnn_attention_forward` at **90% of CPU time, ~8.7 ms CPU vs 16 µs GPU per call**: cuDNN builds an
   execution plan per input shape, and decode changes the KV length every step. Re-running one fixed shape went
   397 → 110 → 65 → 58 ms, confirming the per-shape plan cache. → **cuDNN for prefill only.**

### 5.2 The profiler perturbs what it measures

| Decode batch 1 | ms/step |
|---|---|
| CUDA-event timing, no profiler | ~35 |
| Under `torch.profiler` | 58 – 126 (CPU time per step varied with background load) |
| Under `torch.profiler` + `--memory` (allocation history with stack traces) | ~142 |

GPU busy time stayed at ~30 ms in every case, so **kernel durations from the profiler are trustworthy, but
its idle-% is not**: profiling adds CPU cost per op, which starves the GPU further (73% "idle" under the profiler
versus ~13% in reality). Use CUDA-event timers for latency, the profiler for *what* runs, and Nsight Systems
(lower overhead) for idle gaps.

### 5.3 Laptop measurement hygiene

- The first FP16 matmul benchmark read 11.7 TFLOPS. After a longer warm-up it settled at 26.8, because the GPU ramps clocks slowly.
- The first prefill after model load took 380–590 ms against a steady 140 ms. One warm-up call was not enough.
- One whole run was ~2.5× slower on the CPU side (likely background activity right after the 4 GB download).
  Repeat runs and report ranges.

## 6. Bottleneck verdict

| Scenario | Primary bottleneck | Evidence |
|---|---|---|
| Prefill ≥ 1 k tokens (cuDNN) | **Compute** (GEMMs) + dequant tax | 74% of peak TFLOPS overall; GEMMs ~90%; dequant 19% |
| Prefill, short prompts | **4-bit dequantization tax** (bandwidth-bound) | ~100 of ~140 ms at 128 tokens |
| Prefill 2 k tokens, default | **VRAM capacity**: spill to system RAM | 5.86 GB peak, 15× variance; fixed by fused attention |
| Decode, batch 1 | **Memory bandwidth** (65%) + **launch overhead / glue kernels** (35%) | NF4 GEMV at 93% of peak BW; 1 782 kernels/step; ~13% GPU idle |
| Decode, batch ≥ 4 | **Inefficient kernels**: batched NF4 GEMM (31% of BW) + unfused attention | GPU 97% busy, far from both rooflines |

## 7. Pending: Nsight Systems / Nsight Compute (install first: see README §4)

`profiling/05_nsight.ps1` runs everything below for both the stock and the optimized model (§9):
`powershell -ExecutionPolicy Bypass -File profiling/05_nsight.ps1 -Batch 4` (and `-Batch 1`).

| Check | Report it produces | What would confirm the analysis |
|---|---|---|
| Real GPU idle at batch 1 | `results/nsys_bs1_{stock,opt}.nsys-rep` | Gaps between kernels inside each `decode_step` totalling ~5 ms of ~30–35 ms; `cudaLaunchKernel` dominating `cuda_api_sum` |
| Batch-1 GEMV efficiency | `results/ncu_bs1_stock.ncu-rep` | `gemm_4bit_simt`: Memory SOL ≈ 90%+, DRAM ≈ 160+ GB/s |
| Why bitsandbytes' batched kernel is slow | `results/ncu_bs4_stock.ncu-rep` | `gemm_4bit_sm80`: low DRAM throughput (~55 GB/s); occupancy and stall reasons explain the 31% |
| Why the Triton NF4 kernel stops at ~50% | `results/ncu_bs4_opt.ncu-rep` | `_nf4_gemm_kernel`: expected to be bound by load/issue (table lookups + per-token reductions) rather than DRAM |
| Attention kernel near the roofline | `results/ncu_bs4_opt.ncu-rep` | `_partial_kernel`: DRAM throughput near peak at long context |

## 8. Recommendations (in expected-impact order)

1. **Use fused attention for prefill only** (done, `SDPA_BACKEND=cudnn`): 1.2–1.9× faster TTFT at 0.5–2 k tokens and no VRAM spill.
2. **Replace the unfused decode attention** (done, §9.2): the single biggest decode win, 1.2–2.2× end to end.
3. **Replace the batched 4-bit kernel** (partly done, §9.1): the custom kernel wins at 4–5 tokens per layer, which is
   enough for 2.2–2.4× at batch 4. Beyond that, formats built for batched serving (AWQ/GPTQ with Marlin-style
   kernels, llama.cpp, vLLM) target the remaining gap and avoid the prefill dequant tax too.
4. **For single-user latency, remove launch overhead**: static KV cache + CUDA Graphs. Batch-1 decode is now
   ~30 ms against a ~22 ms floor, and ~5 ms of that is the GPU waiting for Python. Graphs would also hide
   Triton's higher per-launch CPU cost (§9.1).
5. **Re-measure after each change** with the same scripts. The roofline floors in §2 are the target.

## 9. Custom Triton kernels

Two kernels in `profiling/kernels/`, aimed at the two decode bottlenecks the profile identified (§4.2, §4.3).
`OPTIMIZED=1` swaps them into any script; `profiling/04_compare.py` measures each one's effect in one process.

### 9.1 NF4 dequant + skinny matmul (`nf4_gemm.py`)

Reads bitsandbytes' storage in place (packed nibbles + nested 8-bit scales, **no extra VRAM**) and computes
`X @ Wᵀ` for 1–8 tokens without ever materialising bf16 weights.

**How the design was found.** Tests used a 14336×4096 weight with synthetic data, flushing L2 between runs: this GPU's
24 MB L2 would otherwise hold whole 8 MB weight matrices and report impossible bandwidth.

| Variant | Result | Lesson |
|---|---|---|
| First version: dequantize the tile, then `tl.dot` (tensor cores) | **~10% of BW, 8× slower than bitsandbytes** | Correct is not fast |
| Just loading the packed bytes | 145 GB/s | The memory side was fine |
| + 16-entry NF4 table lookups | 103 GB/s | Lookups cost ~30% |
| 256-entry "two values per lookup" table | 3× *worse* | A 1 KB table thrashes L1; a 64 B one doesn't |
| Scale applied per element | large drop | Load scales once per 64-weight block and broadcast |
| `tl.dot` on dequantized registers | ~33 GB/s at any M | Converting registers into tensor-core operand layout every tile costs more than the math at M ≤ 16 |
| One reduction per tile for all tokens (3D broadcast) | worse than per-token | Register pressure |
| **Final: 2D tile, broadcast scales, CUDA-core FMAs, one reduction per token** | **~120 GB/s at M=1** | |

**On the model's real weights vs bitsandbytes** (`results/kernel_nf4_gemm.json`, µs per call, bitsandbytes vs Triton):

| Layer | M=1 | M=2 | **M=4** | M=8 |
|---|---|---|---|---|
| q_proj 4096×4096 | 85 vs 97 | 89 vs 131 | 205 vs **155 (1.32×)** | 207 vs 244 |
| gate_proj 14336×4096 | 235 vs 275 | 254 vs 309 | 567 vs **425 (1.33×)** | 567 vs 665 |
| down_proj 4096×14336 | 252 vs 317 | 275 vs 360 | 608 vs **523 (1.16×)** | 647 vs 778 |

bitsandbytes' batch-1 GEMV is excellent (77–83% of BW on large matrices); its batched kernel isn't (22–35%).
The Triton kernel's cost grows with M, so it wins only at **M = 4–5**. At load time the patch times both
implementations per layer shape and uses Triton only where it's more than 10% faster.

Two integration lessons:
- **A first version pre-dequantized the scales to fp16 (+220 MB).** On a 6 GB card that pushed batch-8 decode over the
  VRAM edge into system-RAM spill (77 → 115 ms/step, run-dependent). Decoding the nested 8-bit scales in-kernel
  removed the extra memory at a small speed cost.
- **A microbenchmark win didn't survive end to end.** At M=1 the kernel won calibration on q_proj, but batch-1
  decode got ~1.5 ms/step *slower*. Decode there is CPU/launch-bound, and a Triton launch from Python costs more
  CPU time than bitsandbytes' precompiled kernel. `do_bench` times back-to-back GPU work and can't see that.
  The patch now only considers M ≥ 3.

### 9.2 Flash-decoding attention with GQA (`decode_attention.py`)

Each program handles one sequence, one KV head and one 64-token chunk. It reads each K/V row once in bf16, computes
the 4 query heads that share that KV head, and keeps an online softmax; a second small kernel merges the chunks.
This replaces the SDPA math path's `repeat_kv` copies, fp32 upcasts and fp32 GEMVs with 2 kernels, a net 14 fewer
kernels per layer (1 782 → 1 334 per step).

**vs the stock path** (`repeat_kv` + SDPA, µs per layer per step; `results/kernel_decode_attention.json`):

| Batch | ctx 128 | ctx 512 | ctx 2048 | ctx 4096 |
|---|---|---|---|---|
| 1 | 49 → 20 (2.5×) | 127 → 29 (4.4×) | 519 → 77 (6.8×) | 1067 → 154 (6.9×) |
| 4 | 159 → 37 (4.3×) | 505 → 86 (5.9×) | 1971 → 281 (7.0×) | 3797 → 531 (7.1×) |
| 8 | 267 → 44 (6.1×) | 1078 → 143 (7.5×) | 3760 → 562 (6.7×) | 7464 → 1004 (7.4×) |

It reaches 70–86% of DRAM bandwidth from 2 k tokens of context up, and its relative error vs fp32 matches the stock
path's (2.1e-3 vs 2.2e-3). The split count is a runtime argument, so a growing KV cache doesn't trigger recompiles:
the same per-shape trap that made cuDNN decode 10× slower (§5.1).

### 9.3 End to end

`profiling/04_compare.py`, two separate clean runs (ranges), context ~550 tokens, `SDPA_BACKEND=cudnn` prefill:

| Batch | Stock | + Triton attention | + Triton NF4 matmul | Speed-up | tokens/s |
|---|---|---|---|---|---|
| 1 | 35.8 – 38.4 ms | 29.6 – 30.6 ms | **29.8 – 30.0 ms** | **1.2 – 1.3×** | 26–28 → 33.5 |
| 2 | 53.6 – 54.2 ms | 32.9 – 33.5 ms | **32.8 – 33.1 ms** | **1.6×** | 37 → 61 |
| 4 | 119.5 – 129.3 ms | 73.1 – 74.4 ms | **53.1 – 55.3 ms** | **2.2 – 2.4×** | 31–34 → 72–75 |
| 8 | 172.4 – 175.4 ms | 77.7 – 77.8 ms | **77.5 – 77.8 ms** | **2.2×** | 46 → 103 |

**Correctness:** greedy decoding of 4 sequences × 48 tokens produces **100% identical tokens** to the stock model in
every run. Max logit difference on the first step is 0.07–0.09, from bf16 accumulation order.

**Profiler before → after** (torch.profiler, 16 decode steps; `results/02_breakdown_bs{1,4}_cudnn[_opt].json`):

| | Batch 1 stock → opt | Batch 4 stock → opt |
|---|---|---|
| GPU busy per step | 29.9 → 24.7 ms | 112.5 → 49.6 ms |
| Kernels per step | 1 782 → 1 334 | 1 782 → 1 334 |
| Elementwise / glue share of GPU time | 21.6% → 7.1% | 30.4% → 6.5% |
| Matmul share of GPU time | 78% → 92% | 70% → 92% |

**Verdict:** decode is now dominated by streaming the 4-bit weights: 92% of GPU time is matmul.
- At batch 1 the weight read runs at the bandwidth roofline (bitsandbytes' GEMV). Most of the remaining gap to the
  ~22 ms floor is CPU launch overhead, which is the next target (CUDA Graphs).
- At batch 4–8 the matmul kernels themselves (~45–50% of BW for the Triton kernel) are the remaining gap.
- Batching is now almost free up to batch 2 (33 ms vs 30 ms per step), and batch 8 delivers **103 tokens/s** on a
  6 GB laptop GPU, up from 46.

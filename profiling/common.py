"""Shared helpers: model loading, prompt construction, timing, hardware microbenchmarks."""
import json
import os
import time
from contextlib import contextmanager, nullcontext
from pathlib import Path

import torch
from torch.nn.attention import SDPBackend, sdpa_kernel

MODEL_ID = os.environ.get("MODEL_ID", "unsloth/mistral-7b-instruct-v0.3-bnb-4bit")
RESULTS = Path(__file__).resolve().parent.parent / "results"
RESULTS.mkdir(exist_ok=True)

# Windows PyTorch wheels ship without FlashAttention, and SDPA silently falls back to the unfused
# "math" path (materialises the full seq x seq score matrix). SDPA_BACKEND=cudnn forces cuDNN's fused
# kernel for prefill, with math as fallback for shapes cuDNN rejects. Unset = PyTorch's default choice.
SDPA_BACKEND = os.environ.get("SDPA_BACKEND")
# OPTIMIZED=1 swaps in the custom Triton kernels (profiling/kernels/) after loading.
OPTIMIZED = os.environ.get("OPTIMIZED") == "1"
TAG = (f"_{SDPA_BACKEND}" if SDPA_BACKEND else "") + ("_opt" if OPTIMIZED else "")


def attention_backend():
    if not SDPA_BACKEND:
        return nullcontext()
    backend = getattr(SDPBackend, {"cudnn": "CUDNN_ATTENTION", "math": "MATH"}[SDPA_BACKEND])
    # set_priority: without it the list only *enables* backends and PyTorch's built-in order
    # (which ranks MATH above CUDNN) still decides, so cuDNN would never be picked.
    return sdpa_kernel([backend, SDPBackend.MATH], set_priority=True)


def load_model(attn_implementation="sdpa"):
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tok = AutoTokenizer.from_pretrained(MODEL_ID)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        device_map="cuda:0",
        dtype=torch.bfloat16,  # matches bnb_4bit_compute_dtype in the checkpoint's quantization_config
        attn_implementation=attn_implementation,
    )
    model.eval()
    if OPTIMIZED:
        from kernels.patch import patch_attention, patch_linear4bit
        patch_attention()
        print("Calibrating Triton NF4 kernel vs bitsandbytes per layer shape ...")
        patch_linear4bit(model)
    return tok, model


def make_input_ids(tok, n_tokens, batch_size=1):
    """A prompt of exactly n_tokens (repeated natural text, so the tokenizer behaves normally)."""
    text = (
        "The history of computing hardware covers the developments from early simple devices "
        "to aid calculation to modern day computers. "
    )
    ids = tok(text * (n_tokens // 10 + 10), return_tensors="pt").input_ids[:, :n_tokens]
    return ids.repeat(batch_size, 1).cuda()


@contextmanager
def cuda_timer(store: list):
    """Time a GPU region with CUDA events; appends milliseconds to `store`."""
    start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    start.record()
    yield
    end.record()
    torch.cuda.synchronize()
    store.append(start.elapsed_time(end))


@torch.inference_mode()
def prefill(model, input_ids):
    with attention_backend():
        out = model(input_ids=input_ids, use_cache=True)
    return out.past_key_values, out.logits[:, -1:].argmax(-1)


@torch.inference_mode()
def decode_step(model, next_ids, past):
    # No attention_backend() here: cuDNN builds an execution plan per KV length (~9 ms of CPU per call),
    # and decode grows the KV cache every step, so it rebuilds every step (~10x slower decode).
    out = model(input_ids=next_ids, past_key_values=past, use_cache=True)
    return out.past_key_values, out.logits[:, -1:].argmax(-1)


def model_shape(model):
    """Parameter count and per-token KV-cache bytes, derived from the config
    (numel() is misleading for 4-bit weights, which are packed two per byte)."""
    c = model.config
    h, inter, L, vocab = c.hidden_size, c.intermediate_size, c.num_hidden_layers, c.vocab_size
    head_dim = getattr(c, "head_dim", None) or h // c.num_attention_heads
    kv_dim = c.num_key_value_heads * head_dim
    per_layer = h * h * 2 + h * kv_dim * 2 + 3 * h * inter  # q,o + k,v + gate,up,down
    return {
        "layers": L,
        "hidden": h,
        "params_linear": per_layer * L + h * vocab,  # includes lm_head (it is a matmul too)
        "params_total": per_layer * L + 2 * h * vocab,
        "kv_bytes_per_token": 2 * L * kv_dim * 2,  # K and V, fp16
        "weight_bytes": model.get_memory_footprint(),
        # what one decode step actually streams: everything except the embedding table (1 row is read)
        "decode_weight_bytes": model.get_memory_footprint() - model.get_input_embeddings().weight.nbytes,
    }


def measure_hardware(size_mb=512, n=8192, iters=20, trials=3):
    """Achievable (not datasheet) DRAM bandwidth and fp16 tensor-core throughput.
    Laptop GPUs ramp clocks slowly, so warm up generously and keep the best trial."""
    x = torch.empty(size_mb * 2**20, dtype=torch.uint8, device="cuda")
    y = torch.empty_like(x)
    for _ in range(10):
        y.copy_(x)
    t = []
    for _ in range(trials):
        with cuda_timer(t):
            for _ in range(iters):
                y.copy_(x)
    bw = 2 * x.numel() * iters / (min(t) / 1e3) / 1e9  # read + write
    del x, y

    a = torch.randn(n, n, dtype=torch.float16, device="cuda")
    b = torch.randn(n, n, dtype=torch.float16, device="cuda")
    for _ in range(30):
        a @ b
    t = []
    for _ in range(trials):
        with cuda_timer(t):
            for _ in range(iters):
                a @ b
    tflops = 2 * n**3 * iters / (min(t) / 1e3) / 1e12
    del a, b
    torch.cuda.empty_cache()
    return {"bandwidth_GBps": round(bw, 1), "fp16_TFLOPS": round(tflops, 1),
            "ridge_point_flops_per_byte": round(tflops * 1e12 / (bw * 1e9), 1)}


def gpu_info():
    p = torch.cuda.get_device_properties(0)
    return {"name": p.name, "vram_GB": round(p.total_memory / 1e9, 2), "sm_count": p.multi_processor_count,
            "torch": torch.__version__, "cuda": torch.version.cuda, "model": MODEL_ID,
            "sdpa_backend": SDPA_BACKEND or "default", "optimized": OPTIMIZED}


def save(name, data):
    path = RESULTS / f"{name}{TAG}.json"
    path.write_text(json.dumps(data, indent=2))
    print(f"\nSaved {path}")
    return path


def now():
    return time.strftime("%Y-%m-%d %H:%M:%S")

# vram-budget

> Will it fit? And how long will it take? — for any transformer, on any GPU.

`vram-budget` is a universal training-VRAM calculator for LLM fine-tuning. Drop in a
YAML describing your model and your hardware; get a per-GPU memory breakdown, a
fits-or-doesn't verdict, and a wall-clock estimate at 1.5× and 2× active-param
token budgets.

The marquee feature is **frontier search**: pick a GPU, pick an architecture
template (e.g. "Llama-style" or "Gemma-style hybrid SWA+global"), pin one knob
as your focal axis, sweep the others, and get back the **largest model that
fits at every focal value**. Great for "what's the biggest 32k-context model I
can train on a single 4090?".

## Why YAML?

So you can edit one file and re-run, instead of writing Python. The architecture,
the hardware, and the training method are all separate YAMLs. Bundled presets
ship in the repo; bring your own and `vram-budget validate ./my_arch.yaml` for
fast feedback.

## Quickstart

```bash
pip install -e .

# Interactive wizard (asks one question at a time — recommended for first use)
vram-budget

# Will Llama-3 8B QLoRA fit on a single 24 GB GPU?
vram-budget fit --gpu rtx_4090 --arch llama3_8b --method qlora_4bit

# What's the biggest Llama-style model that fits on an RTX 3060?
# (FFN auto-scales with hidden_size when you use a *_style template)
vram-budget frontier --gpu rtx_3060_12gb --arch-template llama_style \
    --focal hidden_size --values 1024,2048,4096 \
    --free 'num_hidden_layers=16,24,32 ; ffn.ffn_ratio=3.5,4'

# 3-tab advanced TUI (skip the wizard)
vram-budget tui --advanced
```

### Style templates vs. pinned models

Two flavours of architecture YAML ship out of the box:

- **Pinned models** — `llama3_8b`, `mixtral_8x7b`, `gemma3_27b`, … — locked to the published architecture's exact dimensions. Use these for "will Llama-3 8B fit on my GPU?" questions.
- **Style templates** — `llama_style`, `gemma_hybrid_style`, `mistral_swa_style`, `mixtral_moe_style`, `deepseek_moe_style` — describe a *family* (attention pattern, GQA ratio, FFN style, MoE shape) without pinning sizes. FFN auto-scales with `hidden_size` via `ffn_ratio`. Use these for frontier sweeps where you want to vary dimensions.

## Supported features (v1)

**Architectures** (any combination, via YAML):
- Dense transformer-decoder (Llama, Mistral, Phi, Yi, Qwen)
- GQA / MQA / MHA
- Sliding-window attention (uniform or hybrid SWA+global per Gemma-3/4)
- Hybrid head dimensions (different head_dim for SWA vs. global layers)
- Per-Layer Embeddings (Gemma 3/4 PLE)
- Shared-KV layers (last N layers reuse upstream K, V)
- Mixture-of-Experts (Mixtral, Qwen-MoE, DeepSeek-V2 with shared experts)
- Dual RoPE bases (different θ per attention type)
- Mixed-precision-by-layer schedules (advanced)

**Training methods:**
- Full fine-tuning
- LoRA (any rank, any target modules)
- QLoRA (4-bit / 8-bit frozen base + bf16/fp16 adapters)

**Optimizers:** AdamW (fp32 / bf16 / 8-bit-paged), SGD, SGD+momentum, Lion, Adafactor.

**Parallelism:** single GPU, DDP, FSDP ZeRO-2, FSDP ZeRO-3.

## Accuracy

The calculator models everything that costs measurable VRAM:

- **Weight storage** (parameterized by precision per layer; supports fp32, bf16, fp16, fp8, int8, int4, int3, int2 packed)
- **Gradient accumulator** (only for trainable params — for LoRA / QLoRA, just the adapter rows)
- **Optimizer state** (per-optimizer byte cost: AdamW fp32 = 8 B/p, AdamW 8-bit paged = 2 B/p, SGD = 0, Lion = 4, Adafactor ≈ 2)
- **Master weights** (fp32 master under mixed precision; null under pure bf16)
- **Activations under gradient checkpointing** (none / sqrt(L) / full)
- **KV cache** (training: bf16; inference: bf16 by default. SWA layers cap at the window; KV-shared layers contribute 0)
- **Cross-entropy fp32 logits promotion** (vocab × seq × 4 B, with optional chunking)
- **FSDP ZeRO sharding factors** — ZeRO-2 shards optimizer state, ZeRO-3 shards weights + grads + optim
- **Workspace + safety buffer** — fixed allowance for cuDNN/cuBLAS scratch and allocator fragmentation

**It does not model:** allocator fragmentation drift across runs, custom kernels (FlashAttention savings are baked in by default), tensor / pipeline parallelism, peer-to-peer comm buffers.

Predictions in `tests/calibration/known_configs.yaml` land within their published-number tolerance bands (±15–35% depending on the source). The CI test suite asserts this and fails if the math drifts. Treat the output as a band, not a single number — don't use this to plan to within 200 MB.

If you want to calibrate on your own hardware:

```bash
pip install vram-budget[calibrate]
vram-budget calibrate                     # planned for v0.2 — opens a small probe model
```

## Project status

Alpha. The core math is ported from a private memory-calc tool that's been
running since early 2026; the YAML schema, MoE branch, and TUI are new.

## License

MIT.

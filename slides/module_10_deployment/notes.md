# Module 10: LLM Deployment. Lecture notes

These notes give sources and derivations for every quantitative claim in the
deck, section by section, in slide order.

## The serving problem (slides: serving-metrics, serving-tension)

**Metric definitions.** Time to first token (TTFT), inter-token latency /
per-stream tokens per second, and aggregate throughput are the standard serving
metrics; every serving system paper reports some subset. See Yu et al., "Orca:
A Distributed Serving System for Transformer-Based Generative Models" (OSDI
2022), https://www.usenix.org/conference/osdi22/presentation/yu, and Kwon et
al., "Efficient Memory Management for Large Language Model Serving with
PagedAttention" (SOSP 2023), https://arxiv.org/abs/2309.06180, both of which
frame serving as a throughput-per-dollar problem under latency constraints.

**Cost per token = hardware cost per hour / throughput.** Definitional. The
"goodput" framing (throughput subject to a latency SLO) is used in, e.g.,
Zhong et al., "DistServe: Disaggregating Prefill and Decoding for Goodput-
Optimized Large Language Model Serving" (OSDI 2024),
https://arxiv.org/abs/2401.09670.

## What one token costs (slides: hbm-phases through hbm-bandwidth-wall)

**Two phases.** The prefill/decode split and their different bottlenecks are
analyzed in Pope et al., "Efficiently Scaling Transformer Inference" (2022),
https://arxiv.org/abs/2211.05102, and in kipply's "Transformer Inference
Arithmetic," https://kipp.ly/transformer-inference-arithmetic/, which is the
source of the napkin-math style used in this section.

**~2 FLOPs per parameter per token.** Each parameter participates in one
multiply and one accumulate per forward pass. The same convention gives the
training-side C = 6ND estimate in Kaplan et al., "Scaling Laws for Neural
Language Models" (2020), https://arxiv.org/abs/2001.08361 (2N for the forward
pass, 4N for the backward). Module 5 used the 6ND form.

**7B fp16 = 14 GB, 14 GFLOPs per token.** 7e9 params x 2 bytes = 14 GB read;
7e9 x 2 FLOPs = 14 GFLOPs. Arithmetic intensity of roughly 1 FLOP/byte for
batch-1 decode.

**H100 SXM: ~1,000 TFLOP/s fp16 dense, 3.35 TB/s HBM3.** NVIDIA H100
datasheet: 989.4 TFLOP/s BF16/FP16 dense (1,979 with sparsity, which the deck
does not use) and 3.35 TB/s memory bandwidth for the SXM part.
https://www.nvidia.com/en-us/data-center/h100/. Ratio: 989e12 / 3.35e12 ≈ 295
FLOPs per byte, rounded to "~300 operations per byte" on the slide.

**The speed limit: tokens/sec ≈ bandwidth / bytes per token.** Batch-1 decode
lower bound: every generated token requires streaming all resident weights
once (plus the KV cache; ignored in the headline formula and noted on the
KV-cache slides). 3.35e12 / 14e9 ≈ 239 tokens/sec, quoted as "near 240". The
formula and its empirical accuracy for memory-bound decode are standard; see
kipply (above) and the roofline model: Williams, Waterman, and Patterson,
"Roofline: An Insightful Visual Performance Model for Multicore
Architectures," CACM 52(4), 2009, https://doi.org/10.1145/1498765.1498785.

**Prefill compute bound, decode memory bound.** Prefill processes the whole
prompt in one pass, so each weight read is amortized over all prompt tokens,
pushing arithmetic intensity far above the hardware ratio; decode's intensity
is ~1. Pope et al. (above), section 2.

**Side quest, laptop predictions.** Apple lists memory bandwidth per chip:
M2 (MacBook Air) 100 GB/s, M4 Max up to 546 GB/s (apple.com tech specs).
RTX 4090: 1,008 GB/s (NVIDIA specs). A 7B model at 4-bit is ~3.5 GB plus
scale-factor overhead (GGUF q4_k_m files for 7B models are ~4.1 GB), so the
predictions on the slide (29, 156, 288 tok/s) are upper bounds that llama.cpp
approaches but does not exceed in practice.

**The bandwidth wall chart.** Peak dense fp16 tensor compute and memory
bandwidth by generation, from NVIDIA datasheets: V100 SXM2 125 TFLOP/s and
900 GB/s; A100 SXM 312 TFLOP/s and 2,039 GB/s (80 GB part); H100 SXM 989
TFLOP/s and 3.35 TB/s; B200 ~2,250 TFLOP/s dense FP16 and 8 TB/s HBM3e
(Blackwell architecture materials, 2024; NVIDIA quotes 4.5 PFLOP/s FP16 with
2:4 sparsity, i.e. ~2.25 dense). Growth multiples on the slide are computed
against V100: compute 18x, bandwidth 8.9x. The FLOP-to-byte ratio thus rose
from ~139 (V100) to ~281-296 (B200/H100): the "widening imbalance" claim.

**FlashAttention (figure slide).** Dao et al., "FlashAttention: Fast and
Memory-Efficient Exact Attention with IO-Awareness" (2022),
https://arxiv.org/abs/2205.14135. The paper's premise matches the slide text:
standard attention is bottlenecked on HBM reads/writes of the N x N attention
matrix, and tiling it through on-chip SRAM gives exact attention with far less
memory traffic.

## The KV cache (slides: kvcache-what through kvcache-prefix)

**Why cache.** Without caching, generating token t recomputes keys and values
for all t-1 previous positions in every layer; with the cache each step
computes K and V only for the newest token. Standard in every inference
implementation since GPT-2's `past` tensors.

**Cache size formula.** bytes/token = 2 (K and V) x n_layers x n_kv_heads x
d_head x bytes per value. See the vLLM paper (Kwon et al., above), section 2,
which uses the same accounting.

**Worked example: 0.5 MB/token, 2 GB at 4K.** For 32 layers, 32 heads, d_head
128, fp16: 2 x 32 x 32 x 128 x 2 = 524,288 bytes ≈ 0.5 MB. Times 4,096 tokens
≈ 2.1 GB. The Llama-2-7B generation of models has exactly this attention
shape (MHA, no GQA), which is why the deck calls it "a 7B-class model". Eight
users x 2 GB = 16 GB > the 14 GB of fp16 weights.

**The cache is also read every step.** Attention at step t reads all cached K
and V for the context, so decode traffic = weights + cache; long contexts
lower tokens/sec. Pope et al. treat this as the second memory term.

**MQA.** Shazeer, "Fast Transformer Decoding: One Write-Head is All You Need"
(2019), https://arxiv.org/abs/1911.02150. One shared K/V head; the paper's
motivation is exactly the incremental-decoding memory-bandwidth problem.

**GQA.** Ainslie et al., "GQA: Training Generalized Multi-Query Transformer
Models from Multi-Head Checkpoints" (2023), https://arxiv.org/abs/2305.13245.
Llama 3 8B uses 8 KV heads for 32 query heads (Grattafiori et al., "The Llama
3 Herd of Models," 2024, https://arxiv.org/abs/2407.21783, Table 3).

**60-80% cache waste under contiguous allocation.** Kwon et al. (vLLM paper),
section 3 and Figure 2: existing systems waste 60-80% of KV cache memory to
internal/external fragmentation and reservation; PagedAttention brings waste
under 4%.

**PagedAttention = virtual memory for the cache.** Same paper, sections 4-5.
Block tables map logical to physical cache blocks; the OS analogy is the
paper's own framing.

**Prefix caching / prompt caching.** Shared-prefix KV reuse is implemented in
vLLM ("automatic prefix caching," docs.vllm.ai) and SGLang (RadixAttention:
Zheng et al., "SGLang: Efficient Execution of Structured Language Model
Programs," 2023, https://arxiv.org/abs/2312.07104). Commercial prompt caching
with discounted cached-token pricing: Anthropic and OpenAI API documentation
(2024). The "stable content first" rule follows from prefix matching.

- **Interactive widget (`:::interactive widget="kvBudget"`):** computes cache bytes per token as $2 \cdot n_{\text{layers}} \cdot n_{\text{kv-heads}} \cdot d_{\text{head}} \cdot 2$ with $d_{\text{head}} = 128$ fixed and an fp16 cache, then lays weights, an 8% runtime allowance for activations and fragmentation, and the cache side by side against the card's HBM. Byte counts are decimal (1 GB = 1000 MB). The defaults reproduce the slide's arithmetic exactly: 32 layers, 32 KV heads, 0.52 MB per token, 2.15 GB for a 4K request, 27 concurrent requests on an 80 GB A100 after a 14 GB fp16 7B model. Switching MHA to GQA (8 KV heads) or MQA (1) divides the cache by 4 or 32, which is the serving pressure that changed model architecture. The lower chart crosses the free-memory line at the concurrency ceiling.

## Batching (slides: batching-insight through batching-side-quest-cost)

**Throughput scales ~linearly with batch size until compute or memory binds.**
Batch-1 decode leaves the arithmetic units ~99.7% idle (1 FLOP/byte used of
~300 available), so adding sequences multiplies tokens per weight-read until
arithmetic intensity approaches the hardware ratio or KV caches exhaust HBM.
Pope et al. (above); also the vLLM paper's motivation section, which reports
serving throughput improvements of 2-4x over prior systems purely from
fitting more sequences per batch.

**Static vs continuous batching.** Yu et al., "Orca" (OSDI 2022, above):
iteration-level scheduling admits and retires requests at every decode step.
The convoy/idle-slot failure of request-level (static) batching is the
paper's Figure 3 argument. Continuous batching plus PagedAttention as the
core of modern engines: vLLM, TGI, TensorRT-LLM, SGLang all implement both.

**Prefill/decode interference and chunked prefill.** Agrawal et al.,
"Sarathi-Serve: Taming Throughput-Latency Tradeoff in LLM Inference" (2024),
https://arxiv.org/abs/2403.02310, names and measures the stall ("generation
stalls") and proposes chunking prefills into decode-sized pieces; vLLM ships
chunked prefill. Disaggregating the phases onto separate machines is
DistServe (above).

**Goodput.** DistServe (above) defines goodput as requests per second meeting
latency SLOs; the deck uses the informal version.

**Side quest arithmetic.** H100 rental at ~$2-3.50/hour on commodity clouds
(Lambda, RunPod, Vast pricing pages, 2024-2025; the slide uses $3). Batched
throughput of ~5,000 output tok/s for an 8B model on one H100 is a mid-range
figure for vLLM with continuous batching at moderate context (vLLM benchmark
reports and community benchmarks; order of magnitude, not a lab-grade
number, and the slide presents it as such). $3/hr / 18M tokens/hr ≈ $0.17 per
million. Single-stream: 150 tok/s → 540k tokens/hr → ~$5.50 per million. The
30x ratio is the point, not the third significant figure.

## Quantization and speculative decoding (slides: quant-*)

**Weight-only int8/int4 with per-group scales.** LLM.int8(): Dettmers et al.
(2022), https://arxiv.org/abs/2208.07339. GPTQ: Frantar et al. (2022),
https://arxiv.org/abs/2210.17323 (one-shot 3-4 bit with second-order error
compensation). AWQ: Lin et al. (2023), https://arxiv.org/abs/2306.00978
(activation-aware scaling; MLSys 2024 best paper). Group sizes of 32-128 with
one fp16 scale per group are the standard formats (GPTQ/AWQ/GGUF k-quants).

**4x decode speedup at 4-bit.** Follows from the speed-limit formula: one
quarter the bytes per token raises the memory-bound ceiling 4x. Measured
speedups are somewhat below 4x because of dequantization overhead and the
unquantized KV cache; the slide says "up to 4x".

**Quality loss is uneven; re-evaluate.** GPTQ and AWQ papers both report
task-level regressions concentrated in harder tasks at 3-4 bits; multiple
studies find average perplexity understating capability-specific damage
(e.g., Jaiswal et al., "Compressing LLMs: The Truth is Rarely Pure and Never
Simple," 2023, https://arxiv.org/abs/2310.01382). The prescription (rerun the
Module 9 suite per quantized artifact) is standard MLOps practice.

**GGUF / llama.cpp.** https://github.com/ggml-org/llama.cpp. Gerganov's
llama.cpp first ran LLaMA-7B on a MacBook in March 2023; Ollama is built on
llama.cpp. fp8 serving as emerging default: H100-generation hardware FP8
support plus fp8 checkpoints from major open-weight releases (e.g.,
Llama-3.1-405B-FP8, DeepSeek-V3's FP8 training).

**Distillation.** Hinton, Vinyals, and Dean, "Distilling the Knowledge in a
Neural Network" (2015), https://arxiv.org/abs/1503.02531; modern LLM practice
is supervised finetuning on teacher outputs (Module 6's machinery), as in,
e.g., the Gemma, Phi, and Llama-3.2 small-model reports.

**Speculative decoding.** Leviathan, Kalman, and Matias, "Fast Inference from
Transformers via Speculative Decoding" (2022), https://arxiv.org/abs/2211.17192,
and Chen et al., "Accelerating Large Language Model Decoding with Speculative
Sampling" (2023), https://arxiv.org/abs/2302.01318. Both prove the modified
rejection-sampling rule leaves the target distribution exactly unchanged and
report 2-3x wall-clock speedups. Verification of k drafted tokens in one
forward pass is prefill-shaped (all positions known), which is the source of
the win.

- **Interactive widget (`:::interactive widget="quantExplorer"`):** 600 weights drawn from a fixed Gaussian ($\sigma = 0.25$, deterministic seed) rounded onto an absmax-scaled grid of $2^{\text{bits}}$ levels, with the reported RMS error $\sqrt{\frac{1}{n}\sum_i (q_i - w_i)^2}$ computed over the actual sample. The outlier button appends three weights near $\pm 3$: absmax scaling ties the step size to the largest magnitude in the tensor, so three values stretch the range by more than 3x and every ordinary weight inherits a coarser grid. That is the failure mode LLM.int8 (Dettmers et al., 2022) and NF4 (Dettmers et al., 2023) exist to avoid, by handling outliers separately or choosing a grid matched to the weight distribution rather than to its extremes.

## MoE at serving time (slides: moe-*)

**Mixtral 8x7B: 47B total, 13B active.** Jiang et al., "Mixtral of Experts"
(2024), https://arxiv.org/abs/2401.04088: 46.7B total parameters, 12.9B used
per token (8 experts per layer, top-2 routing).

**All experts must be resident.** Routing is computed per token per layer, so
no expert can be known-cold in advance; the total parameter count sets the
memory footprint while the active count sets per-token compute/traffic. This
asymmetry is discussed in the Switch Transformer paper: Fedus, Zoph, and
Shazeer (2021), https://arxiv.org/abs/2101.03961.

**Hybrid GPU/CPU expert offload.** llama.cpp's MoE offload and Mixtral-class
models running on 64 GB workstations; academic treatments include Eliseev and
Mazur, "Fast Inference of Mixture-of-Experts Language Models with Offloading"
(2023), https://arxiv.org/abs/2312.17238. The trade (interconnect latency for
HBM capacity) is as stated; PCIe 4.0 x16 is ~32 GB/s against HBM's terabytes
per second, which is why it costs speed.

**Expert parallelism and load balancing.** GShard: Lepikhin et al. (2020),
https://arxiv.org/abs/2006.16668 (expert sharding, auxiliary load-balancing
loss); Switch Transformer (above). Hot experts becoming queues is the
load-imbalance problem both papers mitigate at training time.

**DeepSeek-V3.** DeepSeek-AI (2024), https://arxiv.org/abs/2412.19437: 671B
total / 37B active, 256 fine-grained routed experts per MoE layer (8 active)
plus one shared expert, multi-head latent attention (MLA) compressing the KV
cache into a low-rank latent (introduced in DeepSeek-V2,
https://arxiv.org/abs/2405.04434), auxiliary-loss-free load balancing, FP8
training. Reported training cost 2.788M H800 GPU-hours (~$5.6M at their $2/hr
accounting), the "unusually low cost" headline.

## The serving stack (slides: stack-*)

**Engines.** vLLM (Kwon et al., above; https://docs.vllm.ai). SGLang (Zheng
et al., above; structured generation and RadixAttention prefix reuse).
TensorRT-LLM (https://github.com/NVIDIA/TensorRT-LLM). llama.cpp and Ollama
(above). All expose OpenAI-compatible HTTP endpoints.

**The OpenAI-compatible API as de facto standard.** The chat-completions
schema (model, messages, temperature, max_tokens, stream, usage) is
implemented verbatim by vLLM, Ollama, llama.cpp's llama-server, TGI, SGLang,
and most gateways; streaming uses server-sent events with `data:` lines and a
`data: [DONE]` terminator. OpenAI API reference,
https://platform.openai.com/docs/api-reference/chat; vLLM "OpenAI-Compatible
Server" docs.

**Chat template applied server-side.** The server renders `messages` through
the model's chat template (Module 6) before tokenization; see HuggingFace
`apply_chat_template` documentation and vLLM's serving docs.

**Build-or-buy and production hygiene.** The decision framing (steady volume,
privacy, latency, own finetune vs spiky load, frontier access, ops burden) is
standard industry guidance; the cost crossover follows from the side-quest
arithmetic above. Canarying and regression suites before model swaps are
Module 9 section h practices applied to serving.

## Exercise numbers

All terminal output in the exercise walkthrough is captured from real runs of
the bundled solution on an Apple-silicon MacBook against Ollama 0.1.48
serving qwen2.5:0.5b-instruct with OLLAMA_NUM_PARALLEL=8 (protocol printed by
the runner: temperature 0.7, 120-token budget, 80 in the benchmark). The
stage 1 tables are pure arithmetic from `data/models.json` (parameter counts
and attention shapes from the Qwen2.5 and Llama 3 model cards) and
`data/machines.json` (bandwidths as cited above). The stage 4 curve (110 →
269 total tok/s from 1 to 8 concurrent streams, per-stream 115 → 35) is one
representative run; exact numbers vary by machine and load, which the deck
notes by presenting them as measurements, not constants.

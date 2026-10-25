# Benchmarks

Every number hive-ram has measured, with its conditions, plus the projections and comparison figures the other
documents quote. Each section says what is **measured**, what is an **estimate**, and what has **not been run**.

## Contents

- [Test machines](#test-machines)
- [The link](#the-link)
- [GPU speed and the clock problem](#gpu-speed-and-the-clock-problem)
- [Qwen3.8-Flash-Next on TensorFold and tfpp](#qwen38-flash-next-on-tensorfold-and-tfpp)
- [Qwen3.8-Flash-Next on llama.cpp](#qwen38-flash-next-on-llamacpp)
- [GLM-5.3-Flash with tfpp-glm v1](#glm-53-flash-with-tfpp-glm-v1)
- [MiniMax-M3.1-Flash with vLLM pipeline parallelism](#minimax-m31-flash-with-vllm-pipeline-parallelism-2026-10-25)
- [Prompt processing (prefill)](#prompt-processing-prefill)
- [Projections (not measured)](#projections-not-measured)
- [Comparison: dual DGX Spark, tensor parallel over RoCE](#comparison-dual-dgx-spark-tensor-parallel-over-roce)
- [Not measured yet](#not-measured-yet)
- [How to reproduce](#how-to-reproduce)

## Test machines

| | 128 GB node | 64 GB node |
|---|---|---|
| Machine | NVIDIA N1X laptop (GPU `sm_121`, unified memory) | the same model with 64 GB |
| OS | Windows 11, WSL2 Ubuntu (aarch64), NAT networking | same |
| Usable for model weights and caches (our estimate) | about 92–95 GiB (leave 4–6 GB to Windows) | about 29–46 GiB, depending on firmware, WSL's memory limit and other GPU programs |
| Link | one USB4 / Thunderbolt cable straight between the two machines (no dock), negotiated 20 Gbps | |

- Python 3.14 inside WSL; CUDA 13 builds of torch (2.14.1 for Flash-Next, 2.13.0 for tfpp-glm).
- Engines ran one at a time, with every other GPU program on the node stopped.
- Dates: 2026-10-22 (link, llama.cpp, Flash-Next) and 2026-10-23 (GLM).

## The link

One TCP connection, measured with `linktest.py` (Python) and PowerShell; payloads of 64 B for latency.

| Path | Throughput | Round trip p50 |
|---|---|---|
| Windows → Windows over USB4 | 7.1 Gbit/s | – |
| WSL (one node) → Windows (the other node) | 4.8 Gbit/s | 0.15 ms |
| Into WSL through Windows' `netsh portproxy` (iphlpsvc) | **0.42 Gbit/s** | 0.38 ms, p99 20 ms |
| Same machine: Windows → WSL's address | 16.1 Gbit/s | – |
| Same machine: through the user-mode relay (`relay_user.ps1`) | 10.1 Gbit/s | – |
| **WSL → USB4 → user-mode relay → WSL (what every engine here uses)** | **5.2 Gbit/s (615 MiB/s)** | **0.49 ms, p99 1.4 ms** |

- The built-in portproxy is the bottleneck; the user-mode relay is about 12× faster with a 14× lower p99.
- Engine-level cost of one hop, from the per-round timings below: about **2.3–2.5 ms per round** (link, Python and
  bookkeeping at both ends), small next to the 15–100 ms of compute in a round.
- What crosses a hop per decoded row: 20 KB (Flash-Next), 32 KiB plus 8 KiB per DFlash2 tap (GLM), 40,960 B plus small
  extras (DeepSeek-V4.1). A 6-row GLM verify window is 288–336 KiB, about 0.5 ms on this link.

## GPU speed and the clock problem

Raw speed of the two GPUs (micro-benchmarks):

| | 64 GB node | 128 GB node |
|---|---|---|
| Read bandwidth, normal / large tensors | 263 / 236 GB/s | 285 / 258 GB/s |
| bf16 matmul | 93 TFLOPS | 93 TFLOPS |
| Small-kernel CUDA graph | within 6.5% of each other | |

The same work (Flash-Next layers 32–47 plus the head, one rank's decode step) with and without idle time before it:

| Condition | Step time |
|---|---|
| Back to back, 64 GB node / 128 GB node | 8.7 ms / 9.0 ms |
| After 20–40 ms of idle (either node) | 15–16.5 ms |
| Idle gap with the CPU spinning | 15.9 ms |
| Idle gap with one spinning GPU block on a side stream | **9.9 ms** |

The GPU lowers its clocks within tens of milliseconds of idle, and in a pipeline every stage idles while the others
work. **KeepWarm** (`tfpp/tfpp/warm.py`) keeps a ~1 ms `torch.cuda._sleep` running on a side stream (one block) while a
rank waits for the link. Measure it on your GPU with
`python rank_bench.py <model> 32:48 last 1 --modes sleep@0,sleep@30,cpuspin@30,gpuwarm@30`.

## Qwen3.8-Flash-Next on TensorFold and tfpp

**VERIFIED ON HARDWARE.** Checkpoint `Vontra/Qwen3.8-Flash-Next-MLX-4bit-MTP` (48 layers, MLX 4-bit, MTP head).
TensorFold 0.6.1. The 128 GB node ran the leader (embedding, PLE, layers 0–31); the 64 GB node ran layers 32–47, the
head and the MTP head. Greedy decoding, 200 tokens per prompt, three prompts (a short Chinese essay, a Python function
with tests, a three-point explanation). Run with `tfpp_run.py`; the protocol was the first, unversioned one (the
current `tfpp-flashnext/2` has not been re-run on hardware).

| | One node (TensorFold, `--role local`) | 2 nodes, no KeepWarm | **2 nodes, KeepWarm** |
|---|---|---|---|
| Output tokens | reference | identical 3/3 in all three modes, same MTP acceptance | **identical 3/3** |
| Serial, eager | 32.3–32.8 tok/s | 24.5–25.1 tok/s | **33.0–33.8 tok/s** |
| Serial, CUDA graphs | – | 25.1–25.5 tok/s | – |
| MTP depth 6 + graphs: essay / code / explanation | 31.4 / 77.1 / 38.2 tok/s | 23.3 / 60.3 / 29.4 tok/s (≈77%) | **31.0 / 79.3 / 38.6 tok/s** |

Time per round (serial, `TFPP_TIMING=1`, which adds syncs):

| | 128 GB node | 64 GB node | Link and Python | Leader commit |
|---|---|---|---|---|
| No KeepWarm | 19.0 ms (layers 0–31, embedding, PLE) | 15.5 ms (layers 32–47 + head) | ~2 ms | ~1.7 ms |
| KeepWarm | 17.4 ms | 8.8 ms | ~2.5 ms (all overheads) | |

- A single stream gets no overlap: the round is the sum of the ranks plus the hop. With KeepWarm that sum equals one
  node's time for the whole model, because each node reads only its own share of the weights. The limit is the
  weight bytes per token, not the link.
- Memory after loading: the 128 GB node had about 52 GiB of GPU memory free; the 64 GB node 18.5 GiB of the 46 GiB CUDA
  reported.
- **Three ranks on two machines** (128 GB node: layers 0–15 and 16–31 as two processes linked over localhost; 64 GB
  node: layers 32–47; MTP 6 + graphs): identical 3/3, same acceptance; 24.1 / 62.9 / 30.2 tok/s. The speed only shows
  two processes sharing one GPU (Windows switches GPU contexts between them); it says nothing about three machines.

## Qwen3.8-Flash-Next on llama.cpp

**VERIFIED ON HARDWARE.** unsloth GGUF `UD-Q4_K_XL` (103.7 GiB, four files); llama.cpp `9febe63` built with CUDA 13.1
and `GGML_RPC=ON`; temperature 0, thinking off, one slot, 32K context, flash attention; numbers from the server's
`timings`. Spot check: four questions (math, code, Traditional Chinese, reasoning).

| | **2 nodes, RPC** | 1 node (128 GB), PLE table on SSD |
|---|---|---|
| Flags (besides `-c 32768 -np 1 -fa on`) | `--rpc 10.77.0.1:50052 -ngl 999 --fit on --fit-target 6144,4096 -lm dio -cram 2048` | `-ngl all --fit off -lm dio --lazy-mode on` |
| Placement | ≈80 GB on the 128 GB node, ≈20 GB on the 64 GB node's GPU | everything on one GPU except the 26.8 GiB PLE table |
| Decode, one stream: ~530-token Chinese essay / ~700-token code answer | **22.2 / 22.2 tok/s** | 22.8–23.6 tok/s |
| Prompt of 9,014 tokens | 760 tok/s (about 12 s) | 691 tok/s |
| Load time | 1 min 30 s | 24 s |
| `llama-server` resident memory | 32 GB (PLE table loaded) | 4.7 GB |
| Windows memory available after loading | ~20–23 GB (128 GB node), ~23–24 GB (64 GB node) | ~27.5 GB |
| PLE table reads | all in memory | 239 MiB during the first 300 tokens (cold), then almost none |
| Spot check | 4/4 | 4/4 |

- A trap measured on the way: `-lm mmap` without forced lazy mode marks the file `FADV_SEQUENTIAL`; with WSL's
  `read_ahead_kb` of 8192 each 90-byte row read pulled in 16 MB: **21 GB read for 300 tokens, 8 tok/s** under an 8 GB
  cgroup.
- On one node TensorFold ran the same model (a different 4-bit quantization) at 32.3–32.8 tok/s serial and up to
  77.1 tok/s on code with MTP drafts (the two-node split with KeepWarm reached 79.3). llama.cpp's Flash-Next drafters (MTP heads, DFlash2 GGUFs) did not load on
  mainline at the time.

## GLM-5.3-Flash with tfpp-glm v1

**VERIFIED ON HARDWARE, on truncated models.** TensorFold v0.6.0 with MiaAI-Lab's 107-patch recipe (the "MIA
package"); checkpoint `Mia-AiLab/GLM-5.3-Flash-EXL3-4bpw-TensorFold` (revision `76c0b517`) truncated to its first
8 or 26 layers; DFlash2 drafter `incoai/GLM-5.3-Flash-DFlash2` (CC BY-NC-ND 4.0); context 262,144; the 19 prompts of
`tfpp/tfpp/glm/bench_prompts.jsonl`; 2026-10-23. A truncated GLM writes gibberish: only equality and speed count.
Every comparison covers each round's tokens, keep counts, drafts and logits hashes.

### Correctness

| Test | Setup | Result |
|---|---|---|
| T-K | world-size-1 kernels against the two TP=2 halves, layers 0–3, 81 comparisons | passed on both nodes |
| T0a | split compute (`--pure`) against the package's own `_compute`, one process | bit-identical |
| T0b | CUDA graphs against eager | bit-identical |
| T1 | two stages on one GPU, cuts `[0,4)+[4,8)` and `[0,5)+[5,8)`, against one process | bit-identical at both cuts |
| T1-kept | kept prompts under a 0.5 GiB cap | same replies at both cuts |
| T1-fault | injected exception (that request fails, the next matches); killed follower (leader exits 75) | as specified |
| Across machines, 8 layers | 128 GB node `[0,4)` → USB4 → 64 GB node `[4,8)` + head + DFlash2, against one process on the 64 GB node | **19/19 bit-identical** |
| Across machines, 26 layers | 128 GB node `[0,19)` or `[0,20)` → 64 GB node rest + head + DFlash2; cut 19 against cut 20 (no GPU holds a one-process reference) | **19/19 bit-identical** |
| HTTP | the leader's OpenAI-style API | answered |

The 8-layer tests (T-K to the cross-machine run) were all run a second time after the extensions were rebuilt against
CUDA 13, with the same results.

### Speed

8 layers, tracing on:

| | Overall | Per decoded token |
|---|---|---|
| One process (64 GB node) | 65.1 tok/s | 11.7 ms |
| **Two stages across machines** | **53.6 tok/s** | **14.0 ms** |
| Two stages on one GPU (two processes compete; not representative) | 30.9 tok/s | 25.5 ms |

Round time across machines, tracing off:

| Rows per round | 8 layers | 26 layers | Per extra layer | Fixed part |
|---|---|---|---|---|
| R = 1 (one token, no draft) | 14.0 ms | **32.3 ms (31 tok/s)** | **1.02 ms** | 5.9 ms (head, sampling, one hop, Python) |
| R = 2 (one draft) | 21.0 ms | 44.6 ms | 1.31 ms | 10.5 ms (includes 4.2 ms of DFlash2 drafting) |

- **1.02 ms per layer is the bandwidth bound.** A decode row reads about 0.3 GB per layer (8 routed experts, the
  shared expert, 4-bit attention), and 0.3 GB at about 263 GB/s is about 1.1 ms. The engine is already at the limit.
- Each extra row in the verify window adds about **0.29 ms per layer**, because each row reads its own 8 experts.
- One hop costs about **2.3 ms** per round, as with Flash-Next (~2.5 ms).

Memory of the 26-layer stages (measured): stage 0 on the 128 GB node, `[0,19)` 57.3 GiB or `[0,20)` 60.8 GiB; the
leader on the 64 GB node (`[19,26)` or `[20,26)` plus head and DFlash2) 26.5 GiB, with 10.4 GiB of its budget still
free.

## MiniMax-M3.1-Flash with vLLM pipeline parallelism (2026-10-25)

Complete 60-layer model, SAGE-EXL3 2.15 bpw, vLLM `minimax-m31-flash` image, layers 0–41 on the 128 GB machine and
42–59 on the 64 GB machine, through `n1x_tun.py` (4.8 Gbit/s, 0.55 ms median round trip). Single stream, temperature 0,
8,192-token context, KV 400 MB per rank, tonyd2wild's `bench_tp2_night.py` (one repetition). Decode tok/s:

| Prompt type | No draft | DSpark k=3 | DFlash2 k=3 | DFlash2 k=5 | DFlash2 k=7 |
|---|---:|---:|---:|---:|---:|
| count 1…N | 17.8 | 19.2 | 36.6 | 47.4 | 57.4 |
| tool call | 17.8 | 16.1 | 34.2 | 41.5 | 47.1 |
| JSON | 17.8 | 15.2 | 30.9 | 33.1 | 36.3 |
| code | 17.8 | 13.9 | 27.1 | 30.8 | 31.1 |
| math | 17.8 | 14.1 | 29.0 | 33.4 | 31.0 |
| prose | 17.8 | 9.0 | 19.8 | 18.1 | 16.2 |

Other numbers: prefill about 400 tok/s; verification step 58 ms (no draft), about 100 ms (DFlash2), about 200 ms
(DSpark). One layer costs about 0.87 ms on the 128 GB machine. Without the TensorFold kernels the same pair reaches
10 tok/s. The same split with a stray page-cache-dropping loop left running reached 10–12 tok/s. Full discussion:
[engines/minimax-vllm.md](engines/minimax-vllm.md).

**131,072-token window, 4 streams (2026-10-26).** `--max-model-len 131072 --max-num-seqs 4`, 4.5 GB of KV per rank
(pool 152,064 tokens), DFlash2 k=5, 42/18. Total decode tok/s over all streams (per stream in brackets), 2 rounds:

| Streams | No draft (40/20) | DFlash2 k=5 (42/18) |
|---:|---:|---:|
| 1 | 17.4 (17.8) | 31.1 (33.0) |
| 2 | 15.9 (10.8) | 35.8 (26.2) |
| 3 | 21.4 (8.7) | 30.1 (14.0) |
| 4 | 28.0 (8.4) | 24.8 (9.1) |

Prefill: 8K and 32K prompts about 345 tok/s; a 96K-token prompt 217 tok/s on this 4-stream server (443 s to the first
token) and 308 tok/s on a 1-stream server (316 s). Needle in a haystack (three codes at 10 %, 50 %, 90 %): 8K, 32K and
96K tokens, all three codes returned each time, with and without the drafter.

## Prompt processing (prefill)

What we have measured:

| Engine | Setup | Prompt | Rate |
|---|---|---|---|
| llama.cpp RPC | 2 nodes, Flash-Next `UD-Q4_K_XL` | 9,014 tokens | 760 tok/s |
| llama.cpp, one node | 128 GB node, lazy PLE | 9,014 tokens | 691 tok/s |

Not measured: tfpp's prefill rates (the harness prints them; they were not kept), tfpp-glm prefill over the real link
on a full-size stage, and anything for tfpp-ds41. By design, tfpp and tfpp-glm v1 send prompt chunks without waiting
for replies, so stages overlap during prefill; tfpp-glm v2.0's prompt fill is synchronous and is modelled at about
0.43× of v1's for one long prompt. At the measured link rate a 2,048-row GLM chunk (32 KiB per row, 64 MiB) takes
about 0.1 s per hop.

## Projections (not measured)

| What | Estimate | Basis |
|---|---|---|
| Full 45-layer GLM-5.3-Flash on three nodes, serial | about 54 ms per token (~18.6 tok/s) | 45 × 1.02 ms + fixed part + two hops |
| Same, with DFlash2 | about 96 ms per round: **25–30 tok/s prose, 35–40 tok/s code** | DFlash2 accepting as on a dual DGX Spark (58%, about 2.6 tokens per round, ~3.8-row window) |
| GLM v2.0 (several streams, one round in flight), 4 streams | about 1.5–1.7× one stream's throughput in aggregate, each stream at about 40% | the performance model in the [roadmap](roadmap.md#v21-stream-group-pipelining) |
| DeepSeek-V4.1-Flash on three nodes (tfpp-ds41) | about 45–50 ms per round: **25–37 tok/s prose, 40–55 tok/s code** | ~7.0–7.6 GB of weights read per row in sequence, two hops, DSpark |
| GLM-5.3-Flash or DeepSeek-V4.1-Flash on llama.cpp RPC | about 10–15 tok/s | Flash-Next's measured efficiency and ~7 GB read per token |

The model behind the GLM projection reproduces the one 26-layer measurement: it predicts 32.2 ms per serial round
for the 19 + 7 layer chain, against 32.3 ms measured.

## Comparison: dual DGX Spark, tensor parallel over RoCE

For scale, the same models on a pair of NVIDIA DGX Spark (GB10) machines running two-way tensor parallelism over
RoCE, measured with the corresponding public recipes:

| Model and recipe | One stream | Four streams |
|---|---|---|
| GLM-5.3-Flash EXL3 4 bpw, TensorFold v0.6.0 + MiaAI-Lab's patches | **45.9 tok/s prose, 68.5 tok/s code** | **82.9 / 113.5 tok/s** (aggregate, prose / code) |
| DeepSeek-V4.1-Flash EXL3 EAQ 2.9 bpw, BertholomusAI fork, Zig branch deepseek-v41-zig, + MiaAI-Lab's recipe (published, not measured by us) | **61.4 tok/s prose, 89.3 tok/s code** | – |

- Measured with the public recipe's checkpoint layout.
- Tensor parallelism has both GPUs read half of every layer at the same time. A single stream on a pipeline keeps one
  GPU busy at a time, so it reaches roughly 55–65% of such a pair by design. A pipeline catches up only with several
  streams in flight (tfpp-glm v2, and v2.1's overlapping rounds; see the [roadmap](roadmap.md)).
- Tensor parallelism over USB4 is not a way around this: GLM needs about 90 all-gathers per decoded token, and at an
  estimated 0.25–0.5 ms each over this link that is 22–45 ms of communication per token, before compute (see
  [related work](related-work.md#tensor-parallelism-over-usb4)).
- Numbers published by upstream TensorFold for GB10 pairs are listed in [related work](related-work.md#upstream-tensorfold).

## Not measured yet

- The full GLM-5.3-Flash model on three nodes; any three-machine chain.
- tfpp-glm v2 (`PARALLEL` 2–8): every number for it is a model.
- tfpp-ds41: nothing has run on a GPU.
- Long contexts on a full-size stage; prompt throughput over the real link; streaming under load; tool calls,
  grammars and pictures through a chain.
- llama.cpp RPC with lazy PLE combined, with other models, or with more than two nodes.
- The current wire protocols (`tfpp-flashnext/2`, and every tfpp-glm change after 2026-10-23) on hardware.
- Power draw and thermals under hours of load (the laptops draw 45–80 W).
- WSL2 random-read IOPS for DeepSeek's Engram tables.

## How to reproduce

| Measurement | Guide | Command (summary) |
|---|---|---|
| The link | [hardware setup](hardware-setup.md) | `python3 linktest.py server` on one node, `python3 linktest.py client 10.77.0.1` on the other |
| Clock drop and KeepWarm | [tfpp-flashnext](engines/tfpp-flashnext.md#keepwarm-the-gpu-clock-problem) | `python rank_bench.py <model> 32:48 last 1 --modes sleep@0,sleep@30,cpuspin@30,gpuwarm@30` |
| Flash-Next, tfpp | [tfpp-flashnext](engines/tfpp-flashnext.md#run-two-nodes) | `tfpp_run.py --role last ...`, `--role leader ... --out pp.json`, `--role local ... --out ref.json`, `python compare.py ref.json pp.json` |
| Flash-Next, llama.cpp | [llamacpp-rpc](engines/llamacpp-rpc.md) | `ggml-rpc-server -H 0.0.0.0 -p 50052`; `llama-server ... --rpc 10.77.0.1:50052` |
| GLM, one GPU | [tfpp-glm](engines/tfpp-glm.md#testing-on-one-gpu-run_truncatedsh-v1) | `L=8 CONTEXT=262144 scripts/glm/run_truncated.sh T1` |
| GLM, two machines | [tfpp-glm](engines/tfpp-glm.md#testing-across-two-machines-a-shortened-model) | `start_stage.sh` on both nodes with `TRUNCATE=26`, `BENCH`, `TRACE`; `python -m tfpp.glm.compare cut19.jsonl cut20.jsonl` |

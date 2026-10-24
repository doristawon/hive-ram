# Related work

What else can run one model across several machines, and why hive-ram builds a layer pipeline on top of TensorFold
instead. This page only states what our research notes support; where we read code, the version is given. Upstream
projects move quickly: re-check anything here before you rely on it.

## Contents

- [Summary](#summary)
- [Upstream TensorFold](#upstream-tensorfold)
- [Tensor parallelism over USB4](#tensor-parallelism-over-usb4)
- [The TensorFold trees tfpp builds on](#the-tensorfold-trees-tfpp-builds-on)
- [llama.cpp RPC](#llamacpp-rpc)
- [Other engines for DeepSeek-V4.1-Flash](#other-engines-for-deepseek-v41-flash)
- [Request routing is not model splitting](#request-routing-is-not-model-splitting)
- [Not evaluated](#not-evaluated)

## Summary

| Approach | Splits one model's layers across machines? | Transport | Uneven node memory | Our use |
|---|---|---|---|---|
| **hive-ram tfpp / tfpp-glm / tfpp-ds41** | yes, contiguous layer ranges of any size | one TCP connection per hop | yes, cuts chosen by memory | this project |
| Upstream TensorFold (main, v1.1.2) | only in a cluster planner that does not run | tensor/expert parallel over NCCL, RoCE or Thunderbolt RDMA | no in the running engines | base engine (Python 0.6.x trees) |
| llama.cpp RPC | yes, by layers, through remote devices | TCP | yes (`--fit`, `--fit-target`) | zero-code baseline, measured |
| antirez/ds4 | V4.1: no (pipeline mode limited to other model versions) | RDMA or TCP for tensor parallel | – | not used |
| vLLM, SGLang | not documented for DeepSeek-V4.1 | tensor / data / expert parallel | – | not used |

## Upstream TensorFold

Read at upstream `main` (v1.1.2) on 2026-10-22: code at tree `f8d254c5`, README at `da275a36`.

**Running multi-machine modes are all tensor or expert parallel:**

| Mode | What each machine holds |
|---|---|
| Metal "speed-up mode" (Flash-Next; GLM-5.3-Flash on two Macs), over Thunderbolt 5 RDMA | the whole model |
| GLM-5.3 EXLR8 on two Macs | half of every expert and head |
| GLM CUDA tensor parallel (`families/glm/cuda_native.zig`) | a near-even share of every layer |
| DeepSeek-V4.1 CUDA tensor parallel | an even share; world size can only be 1 or 2 for this model's head and vocabulary sizes |
| Nemotron, two ranks | rank 0 the whole model except a share of the experts; rank 1 its share of the experts |
| Flash-Next CUDA | two local ranks |

**A pipeline planner exists but does not run.** `zig/src/cluster/` plans "pipeline × tensor" layouts: contiguous layer
ranges per stage, weighted by each stage's GPU memory, head and final norm on the last stage, a cost model in which
stages run one after another (no micro-batch overlap, the same assumption as tfpp v1) and one residual hop per stage
boundary. But:

- its only transport is Thunderbolt/ConnectX RDMA ("MCDMA"); the config check refuses any other, and the code says it
  never uses TCP;
- `tensorfold serve --cluster` exits with code 3 because the build does not link the RDMA endpoints, and
  `tensorfold node` refuses to join;
- its only loading sink is Metal; there is no CUDA sink; a CUDA link kind is declared but not implemented;
- its model tables cover the full GLM-5.3 and Kimi K3, not GLM-5.3-Flash.

Upstream's own documentation says that spreading one model's layers over several smaller machines needs a pipeline
mode that is not built.

**Uneven memory.** GLM CUDA tensor parallelism splits every axis evenly (at TP=3: 22/21/21 heads, 704/704/640 expert
rows), so the smallest node sets the limit: GLM-5.3-Flash's 163.6 GiB over three ranks is about 54.5 GiB per rank,
more than our 64 GB node can give. DeepSeek-V4.1 refuses any world size but 1 or 2. Nemotron's uneven split is fixed
by design, not tunable.

**DeepSeek-V4.1 on upstream CUDA.** Pieces landed in 1.1.1, but no release serves DeepSeek checkpoints until a
whole-model gate passes (PR #827, open on 2026-10-22). The Python 0.6.x DeepSeek family is V4-Flash on MLX (Mac) only.

**Ideas worth borrowing** (none used yet, except the first in spirit):

- per-stage layer ranges weighted by memory (`plan.zig`'s `splitLayers`);
- partial-layer loading switches: `TF_DSV41_LAYERS=<lo>-<hi>` for DeepSeek, `TF_GLM_LAYERS` (a prefix) for GLM;
- running both TP=2 shares of a layer in one process (`Exchange.local`): a pipeline stage doing that would reproduce
  a TP=2 deployment's bits exactly;
- the per-node partial download pattern of `pull --exlr8 --node R/N`: read each safetensors header by HTTP range
  request, fetch only this node's tensors into a sparse file at the same offsets, and hash each chunk. (The tool itself
  only handles EXLR8, refuses N = 3 and splits experts, not layers.)

**Numbers upstream publishes for GB10 pairs** (not ours; for context):

| Engine / checkpoint | Setup | Decode | Prefill | Source |
|---|---|---|---|---|
| native 1.1.0, NVFP4 GLM-5.3-Flash | 2 × GB10, TP=2, 245,760 context, drafts on | 35.5 / 27.2 / 34.3 tok/s (count / prose / code, greedy); median 45.0 | median 1,402 tok/s; a 199,307-token prompt in 152.2 s | a contributed measurement in issue #1040 |
| Python 0.6.6, MLX 4-bit GLM-5.3-Flash, DFlash2 | 2 × GB10 | 53.8 / 43.4 / 85.0 / 50.0 tok/s (code / chat × sampled / greedy, 64 tokens) | – | `docs/recipes/glm-5.3-flash.md` |
| Python 0.6.6, MLX 4-bit, MTP only, 262,144 context | 2 × GB10 | 50.9 / 47.2 / 31.9 tok/s at 32,770 / 131,074 / 261,906-token prompts | 1,138 / 898 / 849 tok/s | same |

Upstream publishes no tok/s for the EXL3 GLM checkpoint on its native engine. The dual DGX Spark comparison numbers (public
recipes) are in [benchmarks](benchmarks.md#comparison-dual-dgx-spark-tensor-parallel-over-roce).

## Tensor parallelism over USB4

Could the GB10 tensor-parallel setup simply run over the USB4 link? Our reading, not tested:

- **No TCP data plane upstream.** GLM's forward exchanges tensors only through `ncclAllGather` (or in-process copies).
  NCCL's own socket transport is not ruled out by the code.
- **WSL2 NAT gets in the way.** NCCL's bootstrap and socket transport make every rank listen on random ports that
  peers connect to directly; a relay with one fixed port, as tfpp uses, cannot carry that. It would need WSL2's
  mirrored networking on the USB4 interface, `NCCL_SOCKET_IFNAME`, `NCCL_IB_DISABLE=1` and firewall openings for
  random ports. Upstream also met an NCCL bootstrap problem on point-to-point /32 routes (issue #1067).
- **Latency.** GLM's decode loop does two all-gathers per layer, about 90 per token. On GB10's ConnectX-7 RoCE one
  4 KiB gather costs about 10–12 µs, about 1 ms per token. Over our link (0.5 ms round trip through the relay) each
  collective would plausibly cost 0.25–0.5 ms: **about 22–45 ms of communication per token**, which caps decode at
  roughly 20–45 tokens per second before any compute. tfpp's measured pipeline overhead is about 2.5 ms per hop per
  round.
- **Memory.** Tensor parallelism needs equal shares: TP=2 GLM on two 128 GB nodes is 81.8 GiB of weights per rank
  against about 92 GiB usable; a 64 GB node cannot take a third.

The roadmap keeps an experiment to measure NCCL over USB4 with mirrored networking, for the record.

## The TensorFold trees tfpp builds on

| Variant | Tree | Why this tree |
|---|---|---|
| tfpp (Flash-Next) | TensorFold 0.6.1 (Python + CUDA, Apache-2.0) | Its CUDA Flash-Next path has a per-layer `compute()` loop to cut, and a loader with one line naming the layers to load. |
| tfpp-glm | TensorFold v0.6.0 + MiaAI-Lab's 107 GLM patches ("v1.11-pre") | Stock 0.6.x's GLM CUDA loader hard-codes world size 2 and serves one request at a time; MIA's patches add the multi-stream scheduler, the DFlash2 drafting policies and the recipe that tfpp-glm reproduces at world size 1. The GLM engine already accepts an injected communicator, and GLM layers keep no cross-layer state, so cuts are clean. |
| tfpp-ds41 | BertholomusAI's fork, branch `deepseek-v41-tp2` (Apache-2.0) | It has a Python CUDA path for DeepSeek-V4.1 with a real world-size-1 mode (its communicator is the identity at one rank), Engram reads from local NVMe, and DSpark drafting that matches serial decoding. Upstream's Python trees have no DeepSeek-V4.1 CUDA path. |

Porting to the native (Zig) engines was considered and deferred: they are built around tensor parallelism (followers
replaying rank 0's steps, gathers inside the forward, rank-gathered sampling), the GLM module changed several times a
week in the 1.1.x series, and the toolchain is pinned tightly (Zig 0.17.0 exactly, nvcc 13, a minimum NCCL). The
same author's Zig DeepSeek branch only supports world sizes 2 and 4 today.

## llama.cpp RPC

- **Works today, without code.** `ggml-rpc-server` makes a remote GPU a device; `llama-server` places contiguous
  layer ranges on local and remote devices and fits them to each device's memory. Measured here: Qwen3.8-Flash-Next
  Q4 over two nodes at 22.2 tok/s, about what one node does with the PLE table on SSD (22.8–23.6 tok/s); see
  [llamacpp-rpc](engines/llamacpp-rpc.md).
- **Slower than TensorFold on the same node:** 22.8–23.6 against 32.3–32.8 tok/s serial and up to 77.1 tok/s with
  MTP drafts, mostly from drafting and kernel efficiency. At the time, Flash-Next's MTP-head GGUFs did not load on
  mainline and its DFlash2 GGUFs were not supported; only `ngram-mod` speculation applied.
- **DeepSeek-V4.1:** the `deepseek4` architecture at commit `9febe63` has no Engram support. An open pull request adds
  it, with the Engram tables as separate, lazily read files; whether those tables stay on the main node or travel to
  RPC devices is not known.
- **Security:** the RPC server has no authentication or encryption, and a client can read and write its buffers.
- Public reports (not verified by us) describe llama.cpp RPC across two GB10 machines over 200 GbE and across a Mac
  and a Spark over 10 GbE, the latter losing about half of its decode speed. We found no public N1X measurement before
  ours.

## Other engines for DeepSeek-V4.1-Flash

From our engine survey (October 2026):

| Engine | Licence | Multi-machine | Engram | Why not used |
|---|---|---|---|---|
| antirez/ds4 (C, CUDA/Metal/ROCm) | MIT | tensor parallel over RDMA or TCP; a pipeline mode `--layers a:b` validated only for other model versions, and its release QA requires V4.1's pipeline mode to fail explicitly | read row by row from its GGUF, never resident | no V4.1 pipeline; its own Q2 format is lower quality than EXL3 2.9 bpw |
| vLLM | Apache-2.0 | the official recipe lists tensor, data and expert parallel; no V4.1 pipeline documentation | host memory or UVA | no pipeline path for this model |
| MiaAI-Lab's vLLM kit for this model | **AGPL-3.0** | tensor parallel | file-backed rows | licence; not imported, not copied |
| SGLang | Apache-2.0 | tensor / expert parallel; pipeline only on the Miles training side | GPU or host memory, no SSD | no EXL3; the FP8 checkpoint is 475 GiB |
| lxe/dsx (Rust + CUDA) | MIT | one machine, two GPUs plus CPU RAM | reads the original Engram shards | single machine |
| upstream TensorFold 1.1.x (Zig) | Apache-2.0 | tensor parallel | NVMe reads (io_uring, AIO, thread pool) | gated, prefix-only layer loading, no pipeline |

## Request routing is not model splitting

Tools that send different requests to different machines (NVIDIA's PAIR is one) each need a machine that holds the
whole model. They raise throughput for models that fit; they do not help a model that fits no single machine, which is
the case hive-ram is for.

## Not evaluated

We did not study other distributed runtimes (for example exo) for this hardware, so this page makes no claim about
them. Contributions with measurements are welcome.

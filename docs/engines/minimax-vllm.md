# MiniMax-M3.1-Flash on two N1X laptops: vLLM pipeline + TensorFold kernels + DFlash2

**Status: VERIFIED ON HARDWARE (2026-10-25).** The complete 60-layer model, SAGE-EXL3 2.15 bpw (108.6 GiB), served by
two Windows 11 + WSL2 laptops (NVIDIA N1X, 128 GB and 64 GB of unified memory) joined by one USB4 cable:
18 tok/s plain, **26–57 tok/s** on structured text with the DFlash2 drafter, answers checked on 24 verifiable tasks.

This engine is different from the `tfpp` family. hive-ram does not wrap TensorFold's server here: the model runs in
**vLLM** (the MiniMax image and the EXL3 plugin published by the pack's author), split by vLLM's own
`--pipeline-parallel-size 2`. hive-ram contributes what vLLM cannot do on this hardware:

| Piece | File | Why it exists |
|---|---|---|
| A TCP-only IP link between the nodes that changes nothing on either host | [`n1x_tun.py`](../../tfpp/scripts/minimax/n1x_tun.py) | vLLM's NCCL and Gloo need both machines to open connections to each other on random ports. Here only one machine accepts connections, through a userland relay, and the network configuration must stay as it is. |
| TensorFold EXL3 kernels inside vLLM | [`tfpp/vllm_plugin`](../../tfpp/vllm_plugin/README.md) (AGPL-3.0-only, [why](#the-glue-plugin)), [`prepare_tensorfold.sh`](../../tfpp/scripts/minimax/prepare_tensorfold.sh) and [`patches/`](../../tfpp/scripts/minimax/patches) | On this machine the weights outside the GPU's small dedicated segment are read about 9× slower by ExLlamaV3's small-batch kernels than by TensorFold's. |
| Two vLLM / vllm-exl3 fixes | same plugin, `register_fixes` | The pipeline-stage expert offset, and the GPU free-memory rule on integrated GPUs. |
| Launch, readiness and evaluation scripts | [`tfpp/scripts/minimax`](../../tfpp/scripts/minimax) | `m31_serve.sh`, `launch_rank.sh`, `ready_check.sh`, `eval/` (chat, verifiable tasks, needle-in-a-haystack) |

## Results

Single stream, temperature 0, vLLM 0.1.dev23341 (the `minimax-m31-flash` image). Node 2 (128 GB, the leader) runs
layers 0–41 and the API; node 1 (64 GB, behind the relay, the last rank) runs layers 42–59 and the drafter.
Harness: tonyd2wild's `bench_tp2_night.py` (MIT), 11 prompt types.

| Prompt type | No draft | MiniMax DSpark k=3 | **DFlash2 k=3** | **DFlash2 k=5** | DFlash2 k=7 |
|---|---:|---:|---:|---:|---:|
| count 1…N | 17.8 | 19.2 | 36.6 | 47.4 | 57.4 |
| tool call | 17.8 | 16.1 | 34.2 | 41.5 | 47.1 |
| JSON | 17.8 | 15.2 | 30.9 | 33.1 | 36.3 |
| SQL | 17.8 | 14.2 | 29.8 | 31.0 | 32.1 |
| math | 17.8 | 14.1 | 29.0 | 33.4 | 31.0 |
| code | 17.8 | 13.9 | 27.1 | 30.8 | 31.1 |
| structured (YAML) | 17.8 | 14.3 | 26.2 | 25.1 | 30.9 |
| prose | 17.8 | 9.0 | 19.8 | 18.1 | 16.2 |
| narrative | 17.8 | 8.5 | 17.8 | 16.4 | 15.0 |
| summary | 17.8 | 8.8 | 17.4 | 16.2 | 15.5 |

- Prefill is about 400 tok/s (a 7,000-token prompt answers its first token after 17.6 s).
- A verification step costs about 58 ms without a draft, about 100 ms with DFlash2 and about 200 ms with MiniMax's own
  DSpark drafter (which samples its tokens one after another). That, not the acceptance length, is why DFlash2 is
  roughly twice as fast here; the two drafters accept about the same number of tokens per step.
- **k=5 is the default** we recommend: structured text 31–47 tok/s and plain prose no slower than no draft.
  Use k=7 for code and JSON, k=3 for prose.
- For comparison, the published two-DGX-Spark TP2 recipe reaches 19–39 tok/s on single streams.

## Quality

Speculative decoding verifies every drafted token with the target model, so the output distribution is unchanged.
What we measured on top of that:

- A 24-task check with programmatic answers (12 three-digit products, the first 30 primes, squares 1–20, six Python
  functions that must pass unit tests, four JSON documents with a schema): **24/24 without a draft and 24/24 with
  DFlash2 k=5**, in 247 s and 148 s.
- Chat smoke tests (capital of France, 17 × 23, an `is_prime`, a French translation) pass in every configuration.
- Greedy outputs are **not** byte-identical between runs, even for the plain server run twice (4 of 8 prompts
  matched), so byte comparison cannot show that a drafter is lossless. The cause is floating-point summation order
  in the kernels, not the drafter.
- Not measured: top-1 agreement or KL against the unquantized model (we do not have its weights). The pack's author
  reports 96.89 % top-1 and 0.0253 nats mean KL for the 2.15 bpw pack.

## Long context and concurrent requests

Same two nodes, `--max-model-len 131072`, `--max-num-seqs 4`, 4.5 GB of KV cache per rank (a pool of 152,064 tokens
with the 42/18 split and the drafter, 159,744 without), CUDA graphs captured for 1–6, 12, 18 and 24 tokens (1–4
without a draft). The pool, not the context limit, is what the streams share: one 128K prompt fits, or four of about
38K.

Aggregate decode throughput over all streams (per-stream speed in brackets), tonyd2wild's sweep, 2 rounds:

| Concurrent streams | No draft | DFlash2 k=5 |
|---:|---:|---:|
| 1 | 17.4 (17.8) | **31.1** (33.0) |
| 2 | 15.9 (10.8) | **35.8** (26.2) |
| 3 | 21.4 (8.7) | **30.1** (14.0) |
| 4 | **28.0** (8.4) | 24.8 (9.1) |

- No request failed and the cache never preempted one, with four streams at 128K.
- DFlash2 is the better choice up to three streams, and every stream is as fast or faster with it at every level. At
  four streams the plain server delivers slightly more in total (probably because verifying 4 × 6 tokens per step
  costs more than the accepted drafts save; we did not profile it). The no-draft C2 figure is a noisy run (one round fell below C1).
- Prefill at this length stays near the 8K figure: 357 tok/s for an 11.5K-token prompt, 350 tok/s at 48K and
  308 tok/s at 97.5K, so a 97.5K-token document waits about 5 minutes for its first token.
- Single-stream decode with the 4-stream server is within about 10 % of the 8K table above (count 1…N 41–44, tool call
  39, code/JSON/SQL/math 26–31, prose 14–16 tok/s), time to first token 1.2–1.6 s on short prompts.
- The first attempt at this configuration froze node 1's WSL VM (`get_unused_fd_flags failed`, a driver warning, then
  a hang). `m31_serve.sh` now raises `ulimit -n`, and we also set WSL's `autoMemoryReclaim` to `disabled`; the next
  run stayed up for the whole test. We did not isolate which of the two mattered.

## Method

1. **Link** (`n1x_tun.py`). Each end opens the one TCP connection that is allowed (client dials the relay,
   server accepts), then enters a fresh unprivileged user + network namespace and creates a TUN device there. Packets
   are framed over the connection; a heartbeat keeps the relay from timing it out and a broker process re-dials if the
   relay cuts it, while TCP sessions inside the namespace ride out the gap. Nothing on either host (interfaces, routes,
   firewall, `.wslconfig`) is touched, and the device disappears with the process. vLLM runs inside the namespace
   (`nsenter --preserve-credentials -U -n -t <pid>`). Measured: 4.8 Gbit/s, 0.55 ms median TCP round trip.
2. **Kernels.** vllm-exl3's routed-expert and dense linear paths are replaced for decode-sized calls by TensorFold's
   grouped EXL3 kernels, reading the same trellis tensors in place. The expert epilogue gets MiniMax's clamped
   SwiGLU-OAI activation (`patches/0001-exl3-experts-swiglu-oai.patch`, 50 lines against TensorFold v0.6.6). Cosine
   similarity against the original path: MoE ≥ 0.999994, linears ≥ 0.99997.
3. **Split.** `VLLM_PP_LAYER_PARTITION=42,18`. With a bf16 drafter the 64 GB node cannot hold 20 layers: it ran out of
   GPU memory. Without a drafter 40/20 is as good. 45/15 and 36/24 were not faster.
4. **Drafter.** `incoai/MiniMax-M3.1-Flash-DFlash2` (Apache-2.0, 4.2 GB bf16) with
   `{"method":"dflash","model":...,"num_speculative_tokens":5,"attention_backend":"TRITON_ATTN","kv_cache_dtype":"fp8"}`.
   vLLM selects its `DFlash2Speculator` from the architecture name `DFlash2DraftModel`.

## Setup

Prerequisites: both laptops linked as in [hardware-setup](../hardware-setup.md) (USB4 cable, `setup_usb4_link.ps1`,
`relay_user.ps1` on the node that accepts connections), WSL2 with systemd, an NVIDIA driver that exposes CUDA in WSL.

Pins (these are the versions that were measured):

| Item | Pin |
|---|---|
| vLLM engine image | `vllm/vllm-openai:minimax-m31-flash`, digest `sha256:86f582dae25ea92802353727164ae4eb03b37b18baf9119b805ffaec2a7ef07c` (linux/arm64) |
| ExLlamaV3 fork | `vcruz305/exllamav3`, branch `feat/moe-act3-batched`, `fdd77361971144e8d175e09e7b50b7100739fd95` |
| vllm-exl3 plugin | `vcruz305/vllm-exl3`, branch `feat/minimax-q4-prefill`, `41fdb7d99a441d37938c2a1bd600ac963c18a3cb` (AGPL-3.0) |
| Pack | `vcruz305/MiniMax-M3.1-Flash-EXL3-SAGE-2.15bpw`, revision `99bf3a57af5d07245ea9abb5426e7dce6af6356b` |
| Drafter | `incoai/MiniMax-M3.1-Flash-DFlash2` |
| TensorFold | tag `v0.6.6`, `cb2ebf0540f42604e2759b2ddef497861e928248` |

Steps (details and every flag in the scripts):

```bash
# 0. Both nodes: the vLLM image unpacked into a folder and run with `unshare --user --map-root-user --mount` + chroot
#    (no Docker needed); build the two forks into it as the recipe's Dockerfile does. Copy the pack and the drafter to both.
# 1. Both nodes: kernels for the plugin, then the plugin
bash tfpp/scripts/minimax/prepare_tensorfold.sh ~/TensorFold ~/n1x-cluster/tfx     # TensorFold at v0.6.6
pip install --no-deps --no-build-isolation tfpp/vllm_plugin                        # into the image's Python (AGPL-3.0-only)
# 2. Link: node 1 (the node behind relay_user.ps1, which accepts) first, then node 2
python3 tfpp/scripts/minimax/n1x_tun.py server 5201 10.99.0.1/30                    # node 1
python3 tfpp/scripts/minimax/n1x_tun.py client 10.77.0.1:5201 10.99.0.2/30         # node 2 (its pid file names the namespace)
# 3. One rank on each node (rank 0 = node 2 with the API, rank 1 = node 1); the same EXTRA string on both
bash tfpp/scripts/minimax/launch_rank.sh 1 run1 'PP_SPLIT=42,18 DFLASH2=/n1x/models/dflash2-incoai DFLASH2_K=5'   # node 1
bash tfpp/scripts/minimax/launch_rank.sh 0 run1 'PP_SPLIT=42,18 DFLASH2=/n1x/models/dflash2-incoai DFLASH2_K=5'   # node 2
bash tfpp/scripts/minimax/ready_check.sh run1     # on node 2: READY / WAIT / FAIL; loading takes 6–9 minutes
# 4. Test (inside the namespace, where the API listens)
nsenter --preserve-credentials -U -n -t $(cat ~/n1x-cluster/n1x_tun.pid) python3 tfpp/scripts/minimax/eval/chat.py
```

## What mattered (read before you start)

These cost us hours; each is a hardware or Windows property of this class of machine, not a bug in the model.

1. **The GPU's dedicated memory.** The N1X ships with an 8 GB dedicated segment; everything else a GPU allocation uses is
   "shared system memory", read more slowly by some kernels, and counted in the Windows commit charge. In the NVIDIA
   App (System → Advanced → GPU Settings → GPU Memory Allocation, applied at restart) we set **Dedicated 44 GB** on the
   64 GB machine. The 128 GB machine ran with 8 GB.
2. **A `.wslconfig` memory cap does not limit the GPU, but vLLM thinks it does.** On integrated GPUs vLLM replaces the
   GPU's free memory by `psutil` available RAM; with `memory=12GB` it saw 8 GiB free and refused to start. The plugin's
   `register_fixes` makes `current_platform.is_integrated_gpu()` return false, so vLLM uses CUDA's figure instead.
3. **Drop the page cache only while loading.** Weights are mmapped from the pack; WSL's page cache counts as Windows
   commit and filled it to 0 during loads (the WSL VM was shut down once). A root `drop_caches` loop every 5 s fixes the
   load, but **costs about 35 ms per decoding step** if it keeps running: stop it once the API answers.
4. **Windows commit headroom.** Set the page file to a fixed size large enough that the commit limit exceeds the total
   GPU + host use by about 20 GB (we used 32 GB on the 128 GB machine), and watch `Win32_OperatingSystem.FreeVirtualMemory`
   during loads; kill the serve below ~15–20 GB.
5. **Keep WSL running.** With no `wsl.exe` session open the WSL VM stops after about a minute, taking the serve with it.
   Keep one `wsl.exe --exec /bin/sleep infinity` alive (a scheduled task at boot is the usual way) and check it
   after every reboot.
6. **Weights over a small GPU-memory window.** ATS aliasing of the embedding table (`VLLM_EXL3_ATS_ALIAS_EMBED`) gives
   an illegal address on WSL, which has no ATS: leave those switches off.
7. **Use the chat endpoint to test.** MiniMax's raw completion path prints template markers
   (`]<]minimax[>[`) that look like corruption; `/v1/chat/completions` is correct.

## Tried and not worth repeating

| Idea | Result |
|---|---|
| External process spinning the GPU to avoid downclocking | slower (time slicing between processes) |
| K-split tuning of TensorFold's linear kernel per shape | +1 % (inside the noise) |
| MXFP8 indexer projections as bf16 instead of Marlin W8A16 | no change |
| vLLM `enable_adaptive_verification` | rejected by the config check under pipeline parallelism |
| `ngram` drafter | assertion in `prepare_inputs` under pipeline parallelism |
| 45/15, 43/17, 36/24 layer splits | no faster than 40/20 or 42/18 |
| MiniMax DSpark drafter | slower than no draft on prose, code, JSON and math (200 ms per step) |

## Limits

- The results table was taken at 8,192 tokens of context and one request at a time; the section "Long context and
  concurrent requests" covers 131,072 tokens and up to four streams.
- A prompt of about 100K tokens takes about 5 minutes to prefill (about 310 tok/s).
- Not run: the unquantized model, vision input, the 4.26 bpw pack (needs the third node).
- Every number on this page was measured **with** the glue plugin. Without it the stack runs vllm-exl3's own kernels,
  which on this hardware was about 10 tok/s without a draft; we did not measure the drafter on those kernels.
- The vllm-exl3 plugin and the pack are the pack author's work (AGPL-3.0 plugin); nothing of theirs is copied here.

## The glue plugin

[`tfpp/vllm_plugin`](../../tfpp/vllm_plugin) (about 230 lines, three entry points) imports `vllm_exl3.exl3`, and vllm-exl3
is licensed AGPL-3.0. That folder, and only that folder, is therefore licensed **AGPL-3.0-only** with its own `LICENSE`;
the rest of hive-ram stays Apache-2.0, and nothing of vllm-exl3's source is copied
([`CONTRIBUTING.md`](../../CONTRIBUTING.md#third-party-code-and-models)). What it does:

1. **Experts** (`TFMOE=1`). After vllm-exl3 builds a routed layer's experts, attach a TensorFold expert table that
   reads the same trellis tensors in place (no second copy of the weights). For decode-sized calls (up to 16 rows) call
   TensorFold's grouped EXL3 expert kernel; padded ids of `-1` (CUDA-graph filler) are skipped; larger calls and layers
   without a gate keep vllm-exl3's path. The clamped SwiGLU-OAI activation comes from the epilogue patch
   (`patches/0001-exl3-experts-swiglu-oai.patch`).
2. **Dense linears** (`TFLIN=1`). Route vllm-exl3's dense EXL3 forward for 1–128 rows through TensorFold's
   `linear.cu` on the trellis in place.
3. **Pipeline fix.** vllm-exl3 `41fdb7d` takes a routed layer's checkpoint expert offset from `get_ep_group().rank`,
   the global rank, which is wrong on pipeline stage 1. The plugin uses `rank_in_group * num_experts`.
4. **GPU memory.** `is_integrated_gpu` returns false, as above (`N1X_UMA_FREE=psutil` keeps the stock rule).

Checked against the original path: cosine similarity ≥ 0.999994 for the MoE and ≥ 0.99997 for the linears
(`TFMOE_CHECK=N`, `TFLIN_CHECK=N` repeat the comparison on your hardware). The `TFMOE=1 TFLIN=1` settings in
`launch_rank.sh` are read only by this plugin. If you offer a modified plugin over a network, AGPL section 13 applies.

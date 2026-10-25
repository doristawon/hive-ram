# MiniMax-M3.1-Flash on two N1X laptops, _with TensorFold kernels and DFlash2._

> Recipe · 2 NVIDIA N1X laptops (128 GB + 64 GB) · one USB4 cable · vLLM pipeline parallelism · measured 2026-10-26

| One request | Two at once | Window | Prefill |
|---|---|---|---|
| **15.7** tok/s prose<br>**43.7** tok/s counting | **35.8** tok/s combined (best of 1–4) | **131,072** tokens, 4 streams share a 152K pool | **345** tok/s, 8K–32K prompts |

Serve the complete **MiniMax-M3.1-Flash** (SAGE-EXL3 2.15 bpw, 108.6 GiB, 60 layers) from two Windows 11 + WSL2 laptops
joined by one USB4 cable, through vLLM's OpenAI-compatible API, with **4 concurrent streams**, a **131,072-token
window** and the **DFlash2 drafter**. The model does not fit either laptop alone; layers 0–41 run on the 128 GB machine
and 42–59 on the 64 GB machine. Everything the cable path needs runs as an ordinary user: no firewall, route, interface
or `.wslconfig` change on either host for the link.

- Complete 60-layer model across two machines, vLLM `--pipeline-parallel-size 2`
- TensorFold EXL3 kernels for the routed experts and dense linears (about 1.8× the speed of the stock kernels here)
- Inco AI's **DFlash2** block drafter, `k=5`: 1.7× to 2.7× on structured text and code; about plain speed on prose (a little under it on the 4-stream server)
- 4-bit main KV cache, FP8 indexer KV, CUDA graphs for 1–6, 12, 18 and 24 rows
- A cable-only IP link (`n1x_tun.py`) for vLLM's NCCL and Gloo, where only one machine accepts connections
- Verified: 24/24 on programmatic tasks, a needle in a haystack at 96K tokens
- Apache 2.0 recipe (the glue plugin is not part of it, see [below](#the-one-piece-that-is-not-here))

[Engine guide](../engines/minimax-vllm.md) · [Scripts](../../tfpp/scripts/minimax) · [Benchmarks](../benchmarks.md#minimax-m31-flash-with-vllm-pipeline-parallelism-2026-10-25) · [Hardware setup](../hardware-setup.md)

## Install with an AI agent

Paste this into an AI coding agent that can run commands on both laptops (for example Claude Code or Codex). It reads the
recipe, checks your machines, and stops at the step this repository cannot do for you.

```
You are setting up a local AI model server on two NVIDIA N1X laptops joined by a USB4 cable. Serve MiniMax-M3.1-Flash
(SAGE-EXL3 2.15 bpw, vcruz305/MiniMax-M3.1-Flash-EXL3-SAGE-2.15bpw) with the hive-ram recipe:
docs/recipes/MiniMax-M3.1-Flash-SAGE-EXL3-2x-N1X-vLLM.md and docs/engines/minimax-vllm.md in this repository.

The recipe needs two Windows 11 laptops with WSL2 (Ubuntu, systemd) and a CUDA-capable NVIDIA driver, one with 128 GB and one
with 64 GB of unified memory, linked by a USB4 cable, about 120 GiB of free disk on each, and key-based ssh to both WSL
instances. You are running on one of them and can reach the other.

Work through these steps in order:

1. Check my setup before changing anything. Report both machines' memory, free disk, WSL version and memory limit, the
   NVIDIA driver, whether the USB4 link has an address on each end and answers a ping, whether the machine that accepts
   connections runs relay_user.ps1, the Windows page file size and the GPU's dedicated-memory setting. If something
   required is missing, stop and tell me what is needed.
2. Read docs/engines/minimax-vllm.md in full, including "What mattered" and "Limits". Treat its warnings as hard rules.
3. Do not change Windows network, firewall, firmware or .wslconfig settings yourself. If one is needed (page file,
   dedicated GPU memory, WSL memory cap), write me the exact commands and how to verify and undo each, and wait.
4. Download the pack and the drafter on both machines at the pinned revisions in THIRD_PARTY.md, unpack the pinned vLLM
   image, build the TensorFold kernels with tfpp/scripts/minimax/prepare_tensorfold.sh, and start the link with
   n1x_tun.py.
5. This repository does not contain the small vLLM glue plugin that calls the TensorFold kernels (it imports an AGPL-3.0
   package). Stop here and ask me whether I have it or want to write it from the description in the engine guide. Without it the
   server runs at about 10 tok/s and none of the numbers in the recipe apply.
6. Launch rank 1 on the machine that accepts connections, then rank 0 with the API, with the settings in the recipe's quick
   start. Follow the logs until ready_check.sh prints READY; loading takes 6 to 9 minutes. Drop the page cache only
   while loading.
7. Verify: the chat check, the 24-task check and the needle check in tfpp/scripts/minimax/eval, and compare the speeds
   with the recipe's tables.
8. Finish with a summary: the endpoint and model id, every file and setting you changed, and the commands to check
   status, read the logs, stop and restart.

Ground rules: ask me before installing system packages, using sudo, rebooting, or stopping anything that is not part of
this recipe. Never delete model weights or other data. Keep the API on the local link; do not expose it to the internet.
If a step fails, read the logs and the guide's notes before trying fixes, and tell me what you changed.
```

## Measured, _not promised._

Two N1X laptops (128 GB and 64 GB), layers 42/18 with the DFlash2 drafter at `k=5`, `--max-model-len 131072`,
`--max-num-seqs 4`, 4.5 GB of KV cache per rank (a shared pool of 152,064 tokens), 4-bit main KV, FP8 indexer KV.
Temperature 0, thinking off, measured with tonyd2wild's `bench_tp2_night.py` from inside the link's namespace on the
leader node (the API is not reachable from outside it); nothing was measured from a remote client.

### Decode, by number of streams

Total over all streams, per stream, and the 90th-percentile time to the first token. A mixed prompt set, 2 rounds each.

| Streams | No draft: total | per stream | TTFT p90 | DFlash2 k=5: total | per stream | TTFT p90 |
|---:|---:|---:|---:|---:|---:|---:|
| 1 | 17.4 | 17.8 | 1.08 s | **31.1** | 33.0 | 1.53 s |
| 2 | 15.9 | 10.8 | 2.09 s | **35.8** | 26.2 | 2.85 s |
| 3 | 21.4 | 8.7 | 2.10 s | **30.1** | 14.0 | 2.75 s |
| 4 | **28.0** | 8.4 | 1.95 s | 24.8 | 9.1 | 3.60 s |

tok/s. No request failed and the cache never preempted one. The no-draft server ran the 40/20 split (the 64 GB machine
has no room for the bf16 drafter at 20 layers). The no-draft figure for 2 streams comes from a noisy run (one round
fell below the 1-stream figure). With the drafter every stream is as fast or faster at every level; at 4 streams the plain server
delivers 13 % more in total, probably because verifying 4 × 6 tokens a step costs more than the accepted drafts save
(not profiled).

### One request, by kind of text

DFlash2 `k=5`, one request at a time on the 4-stream server.

| Text | Decode | Tokens per step | First token |
|---|---:|---:|---:|
| Counting 1…100 | **43.7** | 5.74 | 1.37 s |
| Counting 1…300 | 41.3 | 5.74 | 1.36 s |
| Tool call | 38.7 | 5.53 | 1.22 s |
| Math | 30.5 | 4.51 | 1.54 s |
| JSON | 30.4 | 4.31 | 1.57 s |
| SQL | 29.8 | 4.40 | 1.55 s |
| Structured (YAML) | 26.8 | 4.57 | 1.50 s |
| Code | 26.4 | 4.08 | 1.47 s |
| Prose | 15.7 | 2.16 | 1.34 s |
| Narrative | 14.4 | 2.03 | 1.43 s |
| Summary | 14.3 | 2.28 | 1.43 s |

tok/s. Tokens per step is the mean number of tokens one verification step produced (6 is the most at `k=5`). Plain
decoding is 17.8 on every kind of text. Use `k=7` for code and JSON (31 tok/s in the 8K runs) and `k=3` for prose
(19.8). A single-stream server (`--max-num-seqs 1`, 8K) was 5–15 % faster: counting 47.4, tool call 41.5, JSON 33.1,
SQL 31.0, math 33.4, code 30.8, prose 18.1.

### Prefill

How fast a prompt is read before the first token.

| Prompt | Server | Prefill | First token |
|---|---|---:|---:|
| 8,251 tokens | 4 streams | 342 tok/s | 24.1 s |
| 11,503 tokens | 1 stream | **357 tok/s** | 32.2 s |
| 32,217 tokens | 4 streams | 345 tok/s | 93.4 s |
| 48,367 tokens | 1 stream | 350 tok/s | 138.3 s |
| 96,309 tokens | 4 streams | 217 tok/s | 442.9 s |
| 97,519 tokens | 1 stream | 308 tok/s | 316.5 s |

A prompt of about 100K tokens waits 5 to 7.5 minutes for its first token. If long documents are the main use, serve with
`--max-num-seqs 1`: its prefill at 97K tokens was about 40 % faster.

### Quality

- **24 of 24** programmatic tasks (12 three-digit products, the first 30 primes, squares 1–20, six Python functions that
  must pass unit tests, four JSON documents with a schema), with the drafter and at 128K, in 133 s; 24/24 without a draft
  at 8K.
- **Needle in a haystack**, three access codes planted at 10 %, 50 % and 90 % of a filler text, asked for together:
  **8K, 32K and 96K tokens all returned all three codes**, with and without the drafter (`eval/needle.py`; the filler is
  non-repeating but simple, so this is an easy test, not a proof of long-context reasoning).
- Chat checks (capital of France, 17 × 23, an `is_prime`, a French translation) pass in every configuration.
- Speculative decoding verifies each drafted token with the model, so the output distribution does not change. Greedy
  outputs are **not** byte-identical between two runs even of the plain server (4 of 8 prompts matched), so byte
  comparison cannot show it; the checks above are what we rely on.
- Not measured: top-1 agreement or KL against the unquantized model (we do not have its weights). The pack's author
  reports 96.89 % top-1 and 0.0253 nats mean KL for this pack.

## What you need

### Two N1X laptops, 128 GB and 64 GB

Windows 11, WSL2 with systemd (we used Ubuntu), an NVIDIA driver that exposes CUDA in WSL. The 128 GB machine runs layers
0–41, the API and the leader rank; the 64 GB machine runs layers 42–59 and the drafter. On the 64 GB machine set the GPU's
**dedicated memory to 44 GB** (NVIDIA App → System → Advanced → GPU Settings → GPU Memory Allocation, applied at
restart): with the default 8 GB, weights outside it are read about 9× slower by some kernels.

### One USB4 cable

Linked as in [hardware-setup](../hardware-setup.md): an address on each end, and `relay_user.ps1` on the machine that
accepts connections. The link between the ranks (`n1x_tun.py`) measured 4.8 Gbit/s with a 0.55 ms median round trip.

### Windows commit headroom

WSL's page cache and GPU allocations in shared memory both count as Windows commit. Set the page file to a fixed size
so the commit limit exceeds the total use by about 20 GB (32 GB on the 128 GB machine), cap the 64 GB machine's WSL at
`memory=12GB`, and watch `FreeVirtualMemory` while loading.

### Disk

About 108.6 GiB for the pack and 4.2 GB for the drafter on each machine, plus the unpacked vLLM image.

### Keep WSL alive

With no `wsl.exe` session open the WSL VM stops after about a minute. A scheduled task running
`wsl.exe --exec /bin/sleep infinity` at boot and logon keeps it up, but it does **not** restart by itself after the VM
stops: start it again after every `wsl --shutdown`.

## Quick start

### The pins

[`THIRD_PARTY.md`](../../THIRD_PARTY.md#4-minimax-m31-flash-on-vllm-docsenginesminimax-vllmmd) lists the vLLM image digest, the
fork commits, the pack revision, the drafter and the TensorFold tag.

### 1. The kernels and the link

```bash
git clone https://github.com/ashhart/TensorFold ~/TensorFold && git -C ~/TensorFold checkout v0.6.6   # cb2ebf0
bash tfpp/scripts/minimax/prepare_tensorfold.sh ~/TensorFold ~/n1x-cluster/tfx                          # on both machines
python3 tfpp/scripts/minimax/n1x_tun.py server 5201 10.99.0.1/30                                        # the machine that accepts
python3 tfpp/scripts/minimax/n1x_tun.py client 10.77.0.1:5201 10.99.0.2/30                              # the other one
```

### 2. One rank on each machine

The same settings on both. Rank 1 first (64 GB machine), then rank 0 (128 GB machine, serves the API):

```bash
EXTRA='PP_SPLIT=42,18 MAXLEN=131072 SEQS=4 KVB=4500000000 DFLASH2=/n1x/models/dflash2-incoai DFLASH2_K=5 DFLASH2_CAP=1,2,3,4,5,6,12,18,24'
bash tfpp/scripts/minimax/launch_rank.sh 1 run1 "$EXTRA"      # 64 GB machine
bash tfpp/scripts/minimax/launch_rank.sh 0 run1 "$EXTRA"      # 128 GB machine
bash tfpp/scripts/minimax/ready_check.sh run1                 # READY / WAIT / FAIL; loading takes 6 to 9 minutes
```

While the weights load, drop the page cache as root every 5 seconds on both machines
(`while sleep 5; do sync; echo 1 > /proc/sys/vm/drop_caches; done`); **stop that loop once `READY` prints** (left
running it costs about 35 ms per decoding step). `launch_rank.sh` also starts `fadv.py`, which drops the pack's files
from the page cache for 20 minutes without root. Then check:

```bash
nsenter --preserve-credentials -U -n -t $(cat ~/n1x-cluster/n1x_tun.pid) python3 tfpp/scripts/minimax/eval/chat.py
```

### 3. Talk to it

The API listens at `http://127.0.0.1:18100/v1` **inside the link's network namespace** on the leader machine; reach it
with `nsenter --preserve-credentials -U -n -t $(cat ~/n1x-cluster/n1x_tun.pid) <command>`. Any OpenAI client works with the
model id `minimax-m31`. Use the chat endpoint: the raw completion path prints template markers (`]<]minimax[>[`) that
look like corruption. The model often reasons before it answers (`reasoning_content`), so give replies enough
`max_tokens`: a 200-token budget returned an empty answer for the `is_prime` check even with
`"chat_template_kwargs": {"enable_thinking": false}`.

```bash
nsenter --preserve-credentials -U -n -t $(cat ~/n1x-cluster/n1x_tun.pid) curl -s localhost:18100/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{"model":"minimax-m31","messages":[{"role":"user","content":"Write a Python fibonacci function."}],"max_tokens":600,"chat_template_kwargs":{"enable_thinking":false}}'
```

### If the start stops

| Symptom | What to do |
|---|---|
| `ready_check.sh` prints `FAIL` | Read `m31-pp-<tag>-r0.log` and `-r1.log`. An out-of-memory on the 64 GB machine at 20 layers means the bf16 drafter does not fit: use `PP_SPLIT=42,18`. |
| vLLM reports about 8 GiB free GPU memory and refuses to start | The `memory=` cap in `.wslconfig` is read as the GPU's free memory on an integrated GPU; the glue plugin's fix (see below) makes it use CUDA's figure. |
| `get_unused_fd_flags failed` in `dmesg`, then the WSL VM hangs | Raise the open-file limit (`m31_serve.sh` does `ulimit -n`) and set `autoMemoryReclaim=disabled` in `.wslconfig`. We did not separate the two. |
| Windows commit reaches 0 during the load and the WSL VM is shut down | Larger fixed page file, and stop the load below about 15–20 GB of free commit. |
| Both ranks wait for each other and nothing loads | The link is down: `tail n1x_tun.log` on both machines; the relay on the accepting machine must be running. |

## Memory and window

All streams draw their per-token caches from **one shared pool per rank**: 152,064 tokens with the drafter and the
42/18 split, 159,744 without. One request can use the whole 131,072-token window; four requests of up to about 38K
tokens each fit side by side. A request the pool cannot place yet waits.

| Setting | Window | Streams | Note |
|---|---:|---:|---|
| This recipe | 131,072 | 4 | `KVB=4500000000` per rank |
| Long documents first | 131,072 | 1 | `SEQS=1`: faster prefill at 97K (308 against 217 tok/s), single stream 5–15 % faster |
| First tests | 8,192 | 1 | `KVB=400000000` (the script's defaults) |

## Settings

`m31_serve.sh` reads these from its environment; `launch_rank.sh` passes its third argument through.

| Variable | Default here | Meaning |
|---|---|---|
| `PP_SPLIT` | `42,18` | Layers on rank 0 and rank 1. `40,20` without a drafter. |
| `MAXLEN` / `SEQS` | 131072 / 4 | `--max-model-len` and `--max-num-seqs` (the script's own defaults are 8192 / 1) |
| `KVB` | 4500000000 | KV cache bytes per rank |
| `DFLASH2` / `DFLASH2_K` | drafter folder / 5 | DFlash2 on; `k=3` favours prose, `k=7` code and JSON |
| `DFLASH2_CAP` | `1,2,3,4,5,6,12,18,24` | CUDA-graph sizes (rows per step: streams × (k+1)); fewer sizes load faster but fall back to eager for the rest |
| `GMU` | 0.80 (set by `launch_rank.sh`) | `--gpu-memory-utilization` |
| `UMA_ATS` | 0 | 0 keeps the embedding in CUDA memory; WSL has no ATS and an alias gives an illegal address |
| `CG` | `FULL_DECODE_ONLY` | CUDA-graph mode |
| `TFMOE` / `TFLIN` | 1 / 1 | Read only by the glue plugin (kernels for the experts and the linears) |

## The one piece that is not here

The glue plugin that connects TensorFold's kernels to the pack author's `vllm-exl3` imports an AGPL-3.0 package, and this
repository takes no AGPL code in any form. The engine guide says [what it does in four points](../engines/minimax-vllm.md#the-glue-plugin-is-not-in-this-repository),
and the kernels' build script and patch are published, so it can be written again. **Every number above was measured with
it; without it the same stack runs at about 10 tok/s plain and the drafter was not measured on those kernels.** A repository
that carries only the recipe's own scripts is not a one-command install, and this page does not pretend it is.

## Licences

- **The recipe** (scripts, patch, docs): Apache License 2.0, as the rest of hive-ram. The patch modifies TensorFold v0.6.6
  (Apache-2.0).
- **vLLM** and the image: Apache-2.0, run unmodified. By pulling or running the image you accept its base layers' terms.
- **vllm-exl3** (the pack author's plugin inside that image): **AGPL-3.0**. You install and run it yourself; nothing here
  imports it.
- **The pack** (`vcruz305/MiniMax-M3.1-Flash-EXL3-SAGE-2.15bpw`) and **the base model** (MiniMax-M3.1-Flash): each keeps its
  own licence; read both model cards before use.
- **The DFlash2 drafter** (`incoai/MiniMax-M3.1-Flash-DFlash2`): Apache-2.0 as stated on its model card. (The GLM-5.3-Flash
  drafter of the same name is CC BY-NC-ND 4.0, a different file.)

## Credits

- The pack and the vllm-exl3 / exllamav3 forks: `vcruz305`
- DFlash2: Inco AI
- TensorFold: Ash Hart and contributors
- The tp2 benchmark harness and its two-DGX-Spark recipe: tonyd2wild
- The EXL3 format and converter: turboderp
- The format of this page follows the model recipes at [mia-ai.net/models](https://mia-ai.net/models)

The full list with licence texts is in [`THIRD_PARTY_NOTICES.md`](../../THIRD_PARTY_NOTICES.md).

# llama.cpp RPC across two nodes

llama.cpp's RPC backend makes a GPU on another machine look like one more local device. `llama-server` then places
contiguous layer ranges on each device, which is the same layer pipeline hive-ram builds for TensorFold. No hive-ram
code runs here: this guide covers the build, the link, the flags we used and what we measured.

> **Status**
>
> - **VERIFIED ON HARDWARE (2026-10-22):** llama.cpp RPC, Qwen3.8-Flash-Next Q4 (`UD-Q4_K_XL`, 103.7 GiB), 2 nodes,
>   **22.2 tok/s** single-stream decode, 4/4 on a small spot check.
> - Also measured: the same model on **one** 128 GB node with its PLE table left on the SSD, 22.8–23.6 tok/s.
> - Not tested: lazy PLE combined with RPC, any other model over RPC, more than two nodes, prompt caching across nodes.

## Contents

- [When to use it](#when-to-use-it)
- [Layout](#layout)
- [1. Build](#1-build)
- [2. Give the second node the binaries](#2-give-the-second-node-the-binaries)
- [3. Link and relay](#3-link-and-relay)
- [4. Start the RPC server](#4-start-the-rpc-server)
- [5. Start llama-server](#5-start-llama-server)
- [6. Results](#6-results)
- [7. One node is enough for Flash-Next Q4: the PLE table on SSD](#7-one-node-is-enough-for-flash-next-q4-the-ple-table-on-ssd)
- [8. Larger models](#8-larger-models)
- [9. Speculative decoding](#9-speculative-decoding)
- [Troubleshooting](#troubleshooting)

## When to use it

- As a **zero-code baseline**: any GGUF model that llama.cpp supports, split by layers, in an afternoon.
- When no tfpp variant exists for your model.
- Not when you need the last bit of speed: on the same N1X node, TensorFold ran Flash-Next at 32.3–32.8 tok/s serial
  and up to 77.1 tok/s with MTP drafts, against 22.8–23.6 tok/s for llama.cpp (different quantizations of the same
  model, see [benchmarks](../benchmarks.md#qwen38-flash-next-on-tensorfold-and-tfpp)).

## Layout

```
 node 2: 10.77.0.2 (128 GB)                               node 1: 10.77.0.1 (64 GB)
 WSL2: llama-server ──TCP──► USB4 ═══ cable ═══► Windows: relay_user.ps1 ──► WSL2: ggml-rpc-server :50052
       holds the GGUF files, the front layers,             (user-mode relay;          receives the back layers'
       the KV cache and the HTTP API                        portproxy is too slow)     weights over the link at load
```

The node running `ggml-rpc-server` needs **no copy of the model**: `llama-server` sends it the weights of its layers
over the link when the model loads, and they live only in its GPU memory.

## 1. Build

On a node with the CUDA 13.1 toolkit, cmake and ninja (inside WSL2 Ubuntu, aarch64). We built commit `9febe63` of
`ggml-org/llama.cpp`; newer commits should work but are untested here.

```bash
git clone https://github.com/ggml-org/llama.cpp.git && cd llama.cpp
git checkout 9febe63                    # the commit we measured
export PATH=/usr/local/cuda-13.1/bin:$PATH
cmake -B build13 -G Ninja -DCMAKE_BUILD_TYPE=Release \
      -DGGML_CUDA=ON -DGGML_RPC=ON -DCMAKE_CUDA_ARCHITECTURES=121 \
      -DCMAKE_CUDA_COMPILER=/usr/local/cuda-13.1/bin/nvcc \
      -DCUDAToolkit_ROOT=/usr/local/cuda-13.1 \
      -DLLAMA_CURL=OFF
cmake --build build13 -j
ldd build13/bin/llama-server | grep cudart     # must show libcudart.so.13
```

- **The CUDA 12 trap.** Our first build, without `-DCUDAToolkit_ROOT`, linked the distribution's CUDA 12.4 runtime that
  was also installed. Pointing CMake at the 13.1 toolkit fixed it. Check with `ldd` as above.
- `sm_121` is the N1X GPU (`-DCMAKE_CUDA_ARCHITECTURES=121`).
- The RPC server binary is `ggml-rpc-server` in recent trees (older trees call it `rpc-server`).

## 2. Give the second node the binaries

Only one node needs the CUDA toolkit. Our 128 GB node had none: we copied the built binaries (about 91 MB) and three
CUDA 13 runtime libraries from the build node, about 0.8 GB, and installed no system packages:

```bash
# on the build node
mkdir -p pack/lib && cp build13/bin/* pack/
cp /usr/local/cuda-13.1/targets/sbsa-linux/lib/{libcudart,libcublas,libcublasLt}.so* pack/lib/
tar -cf llama-bin.tar pack         # copy it over the link, check its sha256 on both ends
# on the other node, after unpacking
export LD_LIBRARY_PATH=$PWD/pack/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}
export PATH=$PWD/pack:$PATH
llama-server --list-devices        # the GPU must appear
```

## 3. Link and relay

Set up the USB4 link first (repository README, "Step 0"): `setup_usb4_link.ps1 -Node 1` on the node that will run the
RPC server (10.77.0.1, firewall opened to the peer for ports 50052 and 5201), `-Node 2` on the other (10.77.0.2).

WSL2 in its default NAT mode cannot be reached from the cable directly. On node 1's Windows side, start the user-mode
relay and keep its window open:

```powershell
powershell -ExecutionPolicy Bypass -File relay_user.ps1      # 10.77.0.1, ports 50052 and 5201, into WSL
```

Do not use `netsh interface portproxy` for this: it carried 0.42 Gbit/s with a 20 ms p99, against 5.2 Gbit/s and
1.4 ms p99 for the relay ([benchmarks](../benchmarks.md#the-link)). Check the whole path with `linktest.py` before
loading a model:

```bash
python3 linktest.py server              # node 1, inside WSL
python3 linktest.py client 10.77.0.1    # node 2, inside WSL: expect ~5 Gbit/s, ~0.5 ms p50
```

## 4. Start the RPC server

On node 1, inside WSL. Stop other GPU workloads on this node first: on a unified-memory laptop they take the same
memory the layers need.

```bash
ggml-rpc-server -H 0.0.0.0 -p 50052
```

- `-H 0.0.0.0` because the relay forwards to WSL's own NAT address. On native Linux, bind the link address instead
  (`-H 10.77.0.1`).
- **Do not pass `-c`** (the server's local tensor cache). It writes cache files into WSL, which costs both disk and
  WSL memory; on a node with a small `C:` drive it is the wrong trade.
- **Security:** the RPC server has no authentication and no encryption, and a client can read and write its memory
  buffers. Run it only while you use it, only on the point-to-point link, with the firewall admitting only the peer
  (which `setup_usb4_link.ps1` does).

## 5. Start llama-server

On node 2, inside WSL, where the GGUF files are. This is the command of the measured run:

```bash
llama-server -m <dir>/<model>-UD-Q4_K_XL-00001-of-00004.gguf \
  --rpc 10.77.0.1:50052 \
  -ngl 999 --fit on --fit-target 6144,4096 \
  -c 32768 -np 1 -fa on \
  -lm dio -cram 2048 \
  --host 127.0.0.1 --port 18080
```

| Flag | Why |
|---|---|
| `--rpc 10.77.0.1:50052` | the remote device, through node 1's relay |
| `-ngl 999` | every layer on a GPU (local or remote) |
| `--fit on --fit-target 6144,4096` | let llama.cpp distribute layers so that each device keeps this much free memory (MiB, in llama.cpp's device order; list it with `llama-server --rpc 10.77.0.1:50052 --list-devices`) |
| `-c 32768 -np 1 -fa on` | 32K context, one slot, flash attention |
| `-lm dio` | load weights with direct I/O, so the page cache does not take the memory the model needs |
| `-cram 2048` | cap llama.cpp's host-memory prompt cache at 2 GiB |
| `--host 127.0.0.1` | the API stays on this node; the API has no authentication |

What happened on our nodes: loading took **1 min 30 s**; about **80 GB** of weights landed on the 128 GB node and
about **20 GB** on the 64 GB node's GPU.

Query it like any OpenAI-style server:

```bash
curl -s http://127.0.0.1:18080/v1/chat/completions -H 'Content-Type: application/json' -d '{
  "messages": [{"role": "user", "content": "Write a haiku about cables."}],
  "max_tokens": 200, "temperature": 0, "chat_template_kwargs": {"enable_thinking": false}}'
```

The response's `timings` block gives `prompt_per_second` and `predicted_per_second`; that is how the numbers below
were read.

## 6. Results

Qwen3.8-Flash-Next, unsloth GGUF `UD-Q4_K_XL` (103.7 GiB), temperature 0, thinking off, one stream, 2026-10-22.

| | 2 nodes, RPC |
|---|---|
| Spot check (math, code, Traditional Chinese, reasoning) | 4/4 correct |
| Decode, ~530-token Chinese essay | **22.2 tok/s** |
| Decode, ~700-token code answer | **22.2 tok/s** |
| Prompt of 9,014 tokens | 760 tok/s (about 12 s) |
| Load time | 1 min 30 s |
| `llama-server` resident memory | 32 GB (the PLE table was loaded into memory, see below) |

The decode rate is what one N1X does on its own (next section): the split cost little. llama.cpp's RPC path probably
also loses some speed to GPU clocks dropping while each node waits for the other, the effect tfpp's KeepWarm fixes
(see [tfpp-flashnext](tfpp-flashnext.md#keepwarm-the-gpu-clock-problem)); we did not measure that for llama.cpp.

## 7. One node is enough for Flash-Next Q4: the PLE table on SSD

Flash-Next has a per-layer token-embedding (PLE) n-gram table, `per_layer_token_embd`, which takes **26.8 GiB** of the
`UD-Q4_K_XL` file. Each token reads only 16 rows of 90 bytes from it, so it does not need to be in memory. TensorFold
has `--ple-on-ssd` for this; llama.cpp's `--lazy-mode` memory-maps such tables, marks them `MADV_RANDOM` and prefetches
the rows the next tokens need.

The flags that worked:

```bash
llama-server -m <dir>/<model>-UD-Q4_K_XL-00001-of-00004.gguf \
  -ngl all --fit off -c 32768 -np 1 -fa on \
  -lm dio --lazy-mode on \
  --host 127.0.0.1 --port 18080
```

GPU weights go through direct I/O, past the page cache; only the PLE table is memory-mapped. We also ran the server
under a transient systemd unit with a memory ceiling, so the page cache it causes stays bounded and the server
survives the shell that started it:

```bash
sudo systemd-run --unit=llama-lazy --uid="$USER" -p MemoryHigh=8G -p WorkingDirectory="$PWD" \
  "$PWD/build13/bin/llama-server" -m <dir>/<model>-UD-Q4_K_XL-00001-of-00004.gguf \
  -ngl all --fit off -c 32768 -np 1 -fa on -lm dio --lazy-mode on --host 127.0.0.1 --port 18080
```

Traps we hit on the way:

- **`--lazy-mode auto` turns itself off on the N1X**: ggml-cuda reports the N1X as an integrated GPU, and auto mode
  disables lazy loading there. Use `--lazy-mode on`.
- **`-lm dio` alone** loads the whole PLE table into the server's memory (32 GB resident).
- **`-lm mmap` without forced lazy mode** marks the whole file `FADV_SEQUENTIAL`. With WSL's `read_ahead_kb` of 8192,
  every 90-byte row read pulled in 16 MB: 21 GB read for 300 tokens, and 8 tok/s under an 8 GB cgroup.
- **`--fit on` right after another process freed memory** can judge before the memory is really free and leave some
  expert layers on the CPU. Use `-ngl all --fit off` when you know the model fits.

| UD-Q4_K_XL | 2 nodes, RPC | **1 node (128 GB), lazy PLE** |
|---|---|---|
| Decode, one stream | 22.2 tok/s | **22.8–23.6 tok/s** |
| Prompt of 9,014 tokens | 760 tok/s | 691 tok/s |
| Load time | 1 min 30 s | **24 s** |
| `llama-server` resident memory | 32 GB | **4.7 GB** |
| Spot check | 4/4 | 4/4 |
| PLE reads | whole table in memory | 239 MiB during the first 300 tokens (cold), then almost none |

So Flash-Next Q4 does not need a cluster. The same arithmetic, from the GGUF headers (computed, not run):

| Quantization | File | PLE table (can stay on SSD) | Must be in GPU memory | Fits |
|---|---|---|---|---|
| Q4 `UD-Q4_K_XL` | 103.7 GiB | 26.8 GiB | ~77 GiB | one 128 GB node (measured above) |
| Q5_K_S | 119.3 GiB | 35.8 GiB | 83.5 GiB | one 128 GB node |
| Q5_K_M | — | — | 89.7 GiB | one 128 GB node, tight |
| Q6_K | 156.6 GiB | 50.7 GiB | 105.9 GiB | **two nodes** (lazy PLE + RPC, untested) |

## 8. Larger models

Sizes of public GGUF quantizations, from the file headers (computed, nothing below was run):

| Model | Quantization | Size | Can stay on SSD | GPU part |
|---|---|---|---|---|
| GLM-5.3-Flash | unsloth `UD-IQ3_XXS` | 112.1 GiB | nothing | ~112 GiB |
| | unsloth `UD-IQ4_XS` | 146.1 GiB | nothing | ~146 GiB |
| | unsloth `UD-Q4_K_XL` | 186.0 GiB | nothing | ~186 GiB |
| DeepSeek-V4.1-Flash-1017 | a Q2 GGUF (IQ2_XXS / Q2_K mix) | 340.6 GiB | Engram tables 188.8 GiB | ~152 GiB |

- With a 128 GB node (~92 GiB usable) and a 64 GB node (~30–38 GiB usable), only `UD-IQ3_XXS` of GLM-5.3-Flash fits,
  below the 4-bit EXL3 quality the tfpp-glm path targets. Two 128 GB nodes reach `UD-IQ4_XS`; three nodes
  (128 + 128 + 64) reach `UD-Q4_K_XL`.
- **DeepSeek-V4.1-Flash** needs Engram support. llama.cpp's `deepseek4` architecture at commit `9febe63` does not
  implement Engram; an open pull request adds it (with the Engram tables as separate lazily-read shards). Whether the
  tables then stay on the main node's SSD or travel to RPC devices is unknown.
- **Expected speed** for GLM-5.3-Flash or DeepSeek-V4.1-Flash through llama.cpp on these nodes: about 10–15 tok/s
  (an estimate from Flash-Next's measured efficiency and the ~7 GB of weights each token reads). tfpp-glm targets
  more (see [tfpp-glm](tfpp-glm.md)).

## 9. Speculative decoding

llama.cpp mainline has `--spec-type draft-mtp|draft-dflash|ngram-mod`, but at the time of our runs none of the model
drafters worked for Flash-Next:

- the public MTP-head GGUFs for Flash-Next failed to load on mainline (they needed a patch from a fork);
- DFlash2 GGUF drafters were not supported for Flash-Next's hyper-connection averaging;
- only `ngram-mod` applies, which helps on repetitive work such as code edits.

TensorFold's MTP path is a large part of why it is faster on this model.

## Troubleshooting

| Symptom | Cause and fix |
|---|---|
| p99 latency in the tens of ms, ~0.4 Gbit/s | traffic goes through `netsh portproxy`; use `relay_user.ps1` (or `start_n1x_relay.ps1` once, as administrator, to remove old portproxy entries) |
| `llama-server` cannot reach the RPC server after WSL restarted | WSL's NAT address changed: restart the relay |
| load fails or Windows runs out of memory on the 64 GB node | another GPU program took the unified memory; stop it and retry. Re-check free memory after firmware or driver updates, which can change how much memory the GPU gets |
| memory climbs during load on the main node | page cache: use `-lm dio`, and drop WSL's caches after big downloads (`echo 1 \| sudo tee /proc/sys/vm/drop_caches`) |
| a node reboots mid-run | automatic driver or OS updates; pause them while you experiment |
| `--lazy-mode` seems to do nothing | it is `auto` on an integrated GPU; force `--lazy-mode on` |

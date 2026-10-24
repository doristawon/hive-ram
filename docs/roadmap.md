# Roadmap

What comes next, in the order we expect to do it, and what limits it. Nothing on this page has run on hardware unless
it says so. Measured facts are in [benchmarks](benchmarks.md); the engines' current state is in their guides.

## Contents

- [Where things stand](#where-things-stand)
- [1. Housekeeping before more hardware runs](#1-housekeeping-before-more-hardware-runs)
- [2. tfpp-glm v2 on hardware](#2-tfpp-glm-v2-on-hardware)
- [v2.1: stream-group pipelining](#v21-stream-group-pipelining)
- [3. Three nodes, and four](#3-three-nodes-and-four)
- [4. DeepSeek-V4.1-Flash on three nodes](#4-deepseek-v41-flash-on-three-nodes)
- [5. Quality validation against a reference deployment](#5-quality-validation-against-a-reference-deployment)
- [Hardware limits](#hardware-limits)
- [Later and experiments](#later-and-experiments)

## Where things stand

| Component | Status |
|---|---|
| Flash-Next `tfpp` | **VERIFIED ON HARDWARE**: 2-machine split token-identical to a single node and as fast (KeepWarm); 3-rank chain on 2 machines |
| llama.cpp RPC | **VERIFIED ON HARDWARE**: Flash-Next Q4, 2 nodes, 22.2 tok/s |
| `tfpp-glm` v1 | **VERIFIED ON HARDWARE** on truncated models: bit-exact against one process on 8 layers (pure / eager / graphs, 2 cut points, kept-prompt pressure, fault abort and kill, cross-machine 19/19); 26 layers across machines, two cuts bit-identical; HTTP path answered |
| Full 45-layer GLM-5.3-Flash | **NOT RUN.** About 160 GiB of stages: does NOT fit 128 + 64 GB; needs a third node (planned chain 128 → 64 → 128) |
| `tfpp-glm` v2 (`PARALLEL` 2–8) | **VERIFIED ON ONE GPU (2026-10-24), 8-layer truncated model:** batched-row invariance (T0e); pure and eager against graphs on three lockstep schedules; 2-stage chains at 2 cut points x 3 schedules bit-identical, every stage applying the leader's messages; replies equal to P=1 serial under free timing at 1, 2 and 4 streams. **Not yet run across machines.** |
| `tfpp-ds41` | **NOT RUN ON HARDWARE**: static review and CPU tests only |

## 1. Housekeeping before more hardware runs

- **Re-run what changed since the measured runs.** The Flash-Next wire protocol moved to `tfpp-flashnext/2` and
  tfpp-glm received fixes after 2026-10-23; all of it has passed CPU tests only. Re-run the two-node Flash-Next
  comparison and the tfpp-glm truncated tests (`run_truncated.sh all`, the 26-layer two-cut run) before building on
  them. Every stage of a chain must run the same tree: HELLO compares a digest of the code.
- **A fetch-and-verify script for the third-party trees.** None of TensorFold 0.6.1, the MIA package or the
  BertholomusAI fork ship in this repository. A script will fetch each at its pinned version (for the MIA package:
  TensorFold v0.6.0 plus the patches at `b5808ac`), check the digests (`python -m tfpp.glm.pins --check`,
  `python -m tfpp.ds41.pins --check`), and keep the upstream licence files next to them. A tree rebuilt from v0.6.0 +
  the patches already reproduces the pinned digest offline; [`THIRD_PARTY.md`](../THIRD_PARTY.md) has the manual steps.
- **An MTP-only GLM mode in `start_stage.sh`.** The engine can draft with the checkpoint's own MTP head at
  `PARALLEL=1`, but `start_stage.sh` only knows DFlash2 or `NO_DRAFTS=1`. Wiring it in gives a drafting option without
  the non-commercial DFlash2 licence (MIA reports this mode roughly 5–10% slower, single stream only).

## 2. tfpp-glm v2 on hardware

v2.0 (`--parallel 2..8`) serves several streams at once: the leader runs MIA's multi-stream scheduler unchanged, the
followers mirror it, and each round batches every active stream's rows into **one** pass through the chain. The
bring-up order, each step a pass before the next:

1. **One GPU** (`scripts/glm/run_v2.sh`, the 128 GB node): first `T0e` (is a stream's row in a batched window
   bit-identical to its window alone? If not, the stricter comparisons cannot hold), then `all`, `T2m`, the three-stage
   `L=12` run, `T1-v2-fault`, `T1-v2-http`. Keep a v1 run's traces first: the v1 regression check compares against
   them.
2. **Two nodes**: the 26-layer chain of the v1 hardware run with `PARALLEL=4` and a lockstep schedule (`MBENCH`),
   plus `TFPP_GLM_TIMING=1` and `TFPP_GLM_TIME_ROWS=1` to measure round times and the per-row cost for 1–32 rows.
3. **Three nodes** with the full model (section 3).

Known v2.0 regressions, by design until v2.1: no per-request abort (any error stops the chain with 75), and
synchronous prompt fill (one long prompt runs at about 0.43× of v1's prefetching).

## v2.1: stream-group pipelining

v2.0 still has **one round in flight**: while the leader samples and drafts, stages 0 and 1 wait, and so on down the
chain. v2.1 splits the active streams into **groups** and keeps one round per group in flight, so that different
stages work on different groups at the same time:

```
v2.0, 4 streams in one group:          v2.1, 4 streams in two groups (A, B):
stage 0  [AAAA]............[AAAA]       stage 0  [AA][BB]......[AA][BB]
stage 1  ......[AAAA]..............     stage 1  ....[AA][BB]......[AA]
leader   ............[AAAA+tail]....    leader   ........[AA+t][BB+t]..
         one round = sum of stages              rounds overlap; the busiest stage sets the pace
```

v2.1 also brings back pipelined prompt fill ("FILL trains": several chunks in flight, as v1 did). The protocol already
carries what it needs: every v2 frame has a group field `g` (always 0 in v2.0), and AGREE carries `groups: 1`.

**Performance model** (design-time, not measured). Inputs measured on hardware: 1.02 ms per layer at one row, +0.29 ms
per layer for each extra row, about 2.3 ms per hop and round, about 3.6 ms of tail work (head, sampling, Python) at
one stream. Assumed, to be measured: 4 verify rows and 2.6 accepted tokens per stream and round (DFlash2's acceptance
on a dual DGX Spark), a DFlash2 pass of 2.0 ms + 0.4 ms per extra stream, 0.5 ms more tail work per extra stream,
2.0 ms of fixed cost per link and round, and the link's ~615 MiB/s. Two variants of the per-row cost: linear, and
"saturating" (rows share experts as uniform top-8-of-288 routing would predict). A cycle is the longer of one group's
round latency and the busiest stage's total work.

Three nodes, layers 19 / 7 / 19 (GLM-5.3-Flash), aggregate tok/s (linear – saturating):

| Streams | v2.0 (one round in flight) | v2.1, best number of groups | Per stream (v2.1) |
|---|---|---|---|
| 1 | 27.3 – 27.9 | same | 27.3 – 27.9 |
| 2 | 34.8 – 37.2 | 54.5 – 55.8 (2 groups) | 27.3 – 27.9 |
| 4 | 40.5 – 47.6 | 69.7 – 74.5 (2 groups) | 17.4 – 18.6 |
| 8 | 44.0 – 62.3 | 86.2 – 95.1 (2–3 groups) | 10.8 – 11.9 |

- v2.0's gain from batching is modest (about 1.5–1.7× one stream at 4 streams, each stream at about 40%). v2.1 is
  what could bring 4–8 streams near a dual DGX Spark tensor-parallel pair (82.9 / 113.5 tok/s at 4 streams on prose /
  code, measured).
- With 4 groups of one stream, the leader becomes the bottleneck (its tail work runs once per group): 2 groups is the
  model's best at 4 streams.
- Re-balancing the cuts (20 / 9 / 16, 21 / 10 / 14) changes these figures by a few percent only.
- **Checking the model first.** On the 26-layer two-node chain (19 + 7 layers) the model predicts 32.2 ms per serial
  round (measured: 32.3 ms), and with drafts 45.5–46.5 tok/s for one stream, 68.6–80.3 tok/s for 4 streams in one
  group and 89.0–95.6 tok/s in two groups. Those runs fit on two nodes today: v2.0 has run on one GPU only (8 layers), and v2.1 is not written yet.

## 3. Three nodes, and four

**GLM-5.3-Flash, three nodes.** The full model needs about 160 GiB of stages. Two orders are written down:

| Chain | Cuts | Notes |
|---|---|---|
| 128 → 64 → 128 | `[0,19) / [19,26) / [26,45)` | the order the status table names; its first two stages are exactly the measured 26-layer run; the 64 GB node in the middle needs two USB4 ports and a relay; v2 memory estimate at 4 streams and 256K context 69.8 / 33.6 / 84.0 GiB |
| 64 → 128 → 128 | `[0,7) / [7,26) / [26,45)` | shipped in `tfpp/scripts/glm/node.env.example`: the 64 GB node first, so it only connects out (no relay) and holds no head, drafter or HTTP; design estimates 21.2–23.3 / 72.4–76.3 / 77.0–81.1 GiB |

Plan: measure each node's budget (`start_stage.sh --dry-run`), wire one cable and one subnet per hop, run the chain,
and verify it the way the 26-layer run was verified: no single GPU holds a 45-layer one-process reference, so compare
**two different cut placements** with each other (bit-identical), then quality against a reference deployment
(section 5). Measure prefill over the real link, long contexts and streaming on the way.

**Four nodes.** The code accepts any number of stages (each tfpp-glm stage needs at least one DSA and one KDA layer,
so four or more layers), and the one-GPU test scripts include three-stage chains (tfpp-glm with `L=12`) and a
four-stage chain (tfpp-ds41's `T0c`) on truncated models; on hardware, tfpp-glm has run two-stage chains only. A
fourth node would not make one stream faster: it buys memory for longer contexts, more streams and the leader's
drafter, and for v2.1 more stages to overlap. Wiring stays a line: every inner node needs two USB4 ports, one subnet
per cable (`10.77.0.x`, `10.78.0.x`, `10.79.0.x`, ...), one relay per receiving node. No four-node plan has been
computed yet.

## 4. DeepSeek-V4.1-Flash on three nodes

Order of work for `tfpp-ds41` ([guide](engines/tfpp-ds41.md)):

1. **One GPU, the 11-layer test model S11** (`run_shortened.sh`): `T-ref` first. It is also the fork's first
   world-size-1 run, so fork problems may surface here. Then the kernel checks `T-K'` and `T-K`, then `T0a`–`T0c` and the
   `T1-*` modes, each IDENTICAL before the next.
2. **Two nodes**: `T-X0` on both (do the two machines compute alike?), then `T1-c6`, `T1-c3` and the two-hop `T1-3`.
3. **Measure what the design only estimates**: Engram random-read latency and IOPS on WSL2's ext4 (the fork's
   `tools/dsv41/engram_read_bench.py`; whether `O_DIRECT` and AIO work there at all), the work space per chunk
   (`TF_DS_CHUNK_LOG=1`), and the growth of lazily captured round graphs at long contexts. Then decide whether
   `TFPP_DS41_WARM_ALL_BUCKETS=1` becomes the default.
4. **The full model on three nodes**, once memory and disk allow:
   - the weights (196.1 GiB) fit 128 + 128 + 64 GB, but estimates of what is left for caches range from about 2 to
     7.9 GiB on the large stages, down to negative under conservative overheads; three 128 GB nodes, or a quantization
     of about 170 GiB or less, would give real room;
   - a stage owning both Engram layers needs about 273 GiB of disk; a cut before layer 14 (a clean group boundary)
     splits the two tables between two stages;
   - `python -m tfpp.ds41.plan fit` ranks the placements from measured budgets.
5. **v2 items already identified**: a TOUCH message so that the leader can wake an idle Engram owner's disk before a
   round, vision on the leader, and a more accurate work-space term in admission.

## 5. Quality validation against a reference deployment

What "correct" means so far: **bit-exact against one process of the same engine at world size 1**, at several cut
points. That proves the pipeline adds no error of its own. It does not compare with the tensor-parallel deployments
people actually run: TP=2 sums partial results in a different order, so its bits differ, and the recipes' lossy
storage choices (GLM's 4-bit dense projections and fp8 KV, `DENSE=q4` and `KV=fp8`) are inherited by the pipeline.

Planned, for the full models (none of it run):

- **Teacher-forced agreement**: feed the same token sequences to the pipeline and to a TP=2 reference run of the same
  public recipe and checkpoint (for example on a dual DGX Spark), and compare the top-1 prediction at every position. Target: 0.99 or higher.
- **Task accuracy side by side**: the same question set, greedy, on both deployments.
- **Long context**: the same comparisons at 128K and 256K tokens.
- **Speed with the same prompt sets** (`tfpp/tfpp/glm/bench_prompts.jsonl`, `v2bench http`), so that the numbers
  compare with the pair's.
- Optionally, a stage that runs both TP=2 shares of its layers in one process and sums them in rank order would give
  the TP=2 bits exactly; it is an idea from upstream, not planned work.

## Hardware limits

- **Bandwidth bounds a single stream.** A decode row reads each layer's weights once; on an N1X that is 1.02 ms per
  GLM layer, already at the limit (~263 GB/s). A pipeline cannot make one stream faster than one node holding the
  whole model would be; it only makes the model fit. Speed-ups need several streams (v2, v2.1).
- **The link.** About 5.2 Gbit/s and 0.5 ms round trip through the Windows user-mode relay, and about 2.3–2.5 ms per
  hop and round in the engines. Fine for decode; prompt fill of a 2,048-row GLM chunk takes about 0.1 s per hop, and a
  128K-token DeepSeek prompt cut right before layer 20 would move about 5 GB (about 8 s, overlappable with compute).
- **Topology.** USB4 is point-to-point here: no switch, one cable and one subnet per hop, two ports on every inner node.
- **Unified memory.** The GPU shares memory with Windows and WSL. A 64 GB laptop gave anywhere from about 29 to 46 GiB,
  and firmware updates changed the split; other GPU programs take the same memory; WSL's page cache counts against
  Windows' commit. One stage per GPU: two processes on one GPU ran far slower (Windows switches GPU contexts).
- **Disk.** WSL's ext4 disk is a vhdx growing on `C:`; stage shards and especially DeepSeek's Engram tables (189 GiB)
  need room on both.
- **WSL2 NAT.** Every inbound hop needs the user-mode relay on the Windows side; the relay's address inside WSL can
  change when WSL restarts. Mirrored networking or native Linux would remove the relay (untested).
- **Laptops.** 45–80 W, hot under hours of load; automatic driver and OS updates can reboot a node mid-run.
- **Fail-stop.** Any node's failure ends the chain; more nodes mean more chances of that.

## Later and experiments

- **Measure NCCL tensor parallelism over USB4** with WSL2 mirrored networking, for the record. Our estimate is
  22–45 ms of communication per GLM token ([related work](related-work.md#tensor-parallelism-over-usb4)).
- **Ports to TensorFold's native (Zig) engines** once the Python paths are proven, for their kernel speed.
- **Per-node partial downloads**: fetch only a stage's tensors by HTTP range requests into a sparse file (an upstream
  pattern), instead of whole shards.
- **Other machines.** Only N1X laptops under WSL2 have been used. Native Linux nodes, other GPUs or other link types
  (10/25 GbE) should work with the same TCP links, but the pinned trees and the `sm_121` checks in the scripts assume
  this hardware today.

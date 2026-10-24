# tfpp-glm implementation notes

tfpp-glm runs GLM-5.3-Flash on the MIA package (TensorFold v0.6.0 + 107 patches, world=1) split into contiguous layer
ranges over several N1X laptops. The leader is the chain's last stage (MIA's rank 0): it holds the head, the drafters
and the HTTP front.

## Status

| Part | State |
|---|---|
| v1 (`PARALLEL=1`) | **Verified on hardware** (2026-10-23), on a truncated model: 8 layers bit-exact against one process (`--pure`, eager and CUDA graphs; two cut points; kept-prompt pressure; fault abort and a killed follower; across two machines, 19 of 19 bench requests identical); 26 layers across two machines with two different cuts, bit-identical; the HTTP path answered. |
| The full 45-layer model | **Not run.** It needs about 160 GiB of stages, which the two laptops (128 GB + 64 GB) do not hold; it needs a third node (planned chain: 128 GB -> 64 GB -> 128 GB). |
| v2 (`--parallel 2..8`) | **VERIFIED ON ONE GPU (2026-10-24), 8-layer truncated model:** batched-row invariance (T0e); pure and eager against graphs on three lockstep schedules; 2-stage chains at 2 cut points x 3 schedules bit-identical, every stage applying the leader's messages; replies equal to P=1 serial under free timing at 1, 2 and 4 streams. **Not yet run across machines.** |
| Fixes made after the hardware runs (section "After the hardware runs") | **CPU tests only**; not yet re-run on hardware. Every stage must be re-synced (the HELLO `tfpp_sha` changed). |

The rest of this file: v1 (what was built and how it differs from the spec), v2.0, then the post-hardware fixes.

---

# v1 (2026-10-23)

Built from the merged final tfpp-glm spec: `PARALLEL=1`, N1X nodes each running a layer range of GLM-5.3-Flash. The
Flash-Next pipeline's files (`tfpp/load.py`, `part.py`, `driver.py`, `wire.py`, `warm.py`, `tfpp_run.py`,
`compare.py`) are unchanged; `tfpp.glm` only borrows `KeepWarm` from `tfpp/warm.py`.

## Files

| File | Contents |
|---|---|
| `__init__.py` | `PROTO = 1`, the version. Imports neither torch nor MIA modules |
| `proto.py` | Frame header encoding, the op table, `StageInfo`, the spec 2.3 chain checks, tap routing, staging sizes, `PromptView` (no torch) |
| `env.py` | The spec 12.1 settings (filled in when unset, refused when different), import-order guard, the Triton 3.7 check, the spec 11.2 settings vector, the env digest, versions |
| `pins.py` | `EXPECT_PKG_SHA` (the package tree digest `739ce991...`), AST pins of 56 seams, `--check/--print/--diff` |
| `load.py` | The five loader anchors, `load_range` (with the world=2 share T-K uses), `mine`, `needed_shards` |
| `part.py` | `Stage`, `_compute_pp`, `install_common`, scrub, `devhash` / `devhash_rows` |
| `wire.py` | `Link` (LINK handshake, sequence numbers, pinned staging, `send_raw` forwarding unchanged, scatter-gather `sendmsg`, timeouts, `accept_linked`), `ActIO`, `wait_any` |
| `startup.py` | `connect_down` (connect + LINK, retried until the deadline), HELLO -> GO -> CAP -> AGREE -> READY, the BUILT comparison (with `free_after`), ERROR reports |
| `capacity.py` | `stage_geometry` (MIA's `mla_geometry` narrowed to a stage), `measure_budget`, `agree` (window, kept-prompt budget), `kept_room` (tightened again after the build, from the memory really left) |
| `feed.py` | `PipeVisionFeed`, `LocalFeed`, `NullFeed` |
| `tail.py` | `TailEngine`, `LeaderLink`, `PipeGlmEngine` (in the order of G/engine.py:503-891), the v1 seams, `warmup`, fail-stop, the link watch between requests (`_watch`) |
| `follower.py` | `Follower`: one thread; a control op is forwarded upstream first, then handled |
| `trace.py` / `compare.py` | Per request, per chunk and per window (pos, R, window = pending + drafts, keep, every logits row's hash, tap hashes) traces and their comparison; `--tokens-only` compares the reply tokens only |
| `bench.py`, `bench_prompts.jsonl` | The test set without HTTP (the 19 prompts of spec 13.6) |
| `kernel_check.py` | T-K: world=1 against the two TP2 halves, layer by layer; K2, K4 |
| `serve.py` | Entry point `python -m tfpp.glm.serve --role leader|follower ...` (spec 12.3) |
| `run.py` | A tfpp_run-style mode front end: `python -m tfpp.glm.run local|first|middle|leader|last ...` |
| `tests/test_static.py` | T-S, CPU only, no torch (`tests/data/`: see its README) |

`scripts/glm/`: `cluster.env`, `node.env.example`, `toolchain.sh`, `start_stage.sh`, `stop_stage.sh`, `relay_hop.ps1`,
`run_truncated.sh`, `tfpp-glm@.service`.

## How to run

**T-S (any CPU)**
```
cd tfpp
python -m pytest -q tfpp/glm/tests tfpp/ds41/tests
python -m tfpp.glm.pins --check <MIA tree>/tensorfold
```

**The venv (in WSL on each node; Flash-Next's tf-venv-061 is not touched)**: the one-time steps at the top of
`scripts/glm/toolchain.sh` (`$HOME/tf-venv-mia`, torch 2.13.0, triton 3.7.1, pip's nvcc 13.0, the MIA tree copied into
site-packages). Without a second torch install: `TFPP_GLM_TF_ROOT=$HOME/tf-mia` (holding `tensorfold/`) goes first on
PYTHONPATH; the tree digest must match the same way.

**T0 / T1 (one GPU, the 128 GB node, a truncated model)**
```
MODEL_DIR=$HOME/models/glm53 DRAFTER_DIR=$HOME/models/glm53-dflash2 L=8 CONTEXT=2048 \
  scripts/glm/run_truncated.sh all        # T0a, T0b, T1 (N=2, two cuts), T1-kept, T1-fault
L=12 CONTEXT=262144 scripts/glm/run_truncated.sh T1     # long context; L=12 gives N=3
scripts/glm/run_truncated.sh T1-http
```
The criterion: `python -m tfpp.glm.compare ref.jsonl got.jsonl` says IDENTICAL throughout. On a difference, rerun both
sides with `--no-graphs`, then use the followers' traces (the `act` records' x hashes) to find the stage where it starts.
Defaults `BUDGET_GIB=24` (declared, not allocated) and `KEPT_GIB=8` (exported as `TF_GLM_CACHE_GIB`) keep the kept-prompt
byte cap out of play, so only the entry count (32) evicts; each run checks that the leader's log really says
`kept in 8.00 GiB` and fails otherwise (review fixes F2 / F8 below). `T1-kept` is the 0.5 GiB pressure test and compares
reply tokens only (`--tokens-only`).

**T-K**: `python -m tfpp.glm.kernel_check --model DIR --layers 0:4 --out tk.json`

**Three nodes (C -> A -> B)**: on each node copy `node.env.example` to `node.env` and keep that node's block; run
`relay_hop.ps1` on the Windows side of A and B; then `start_stage.sh` on each node (any order; through a relay a follower
retries until the next stage really answers LINK); the leader prints that it serves and passes its smoke test. Stop the
leader only (`stop_stage.sh`): QUIT travels down the chain.
- `start_stage.sh` refuses to start while a port listed in `GPU_GUARD_PORTS` (node.env; other GPU servers on that node)
  is in use, unless `ALLOW_GPU_SHARING=1`.
- With another tfpp-glm stage already running on the node, `BUDGET_GIB` is required.
- `VERIFY_SHARDS=1`: check this stage's shards against the Hub's sha256 first (`.cache/huggingface/download/*.metadata`),
  cached by inode / size / mtime.

**T2** (the 64 GB and 128 GB nodes, L=24): the leader's node.env adds `TRUNCATE=24 TAP_LAYERS=5,9,14,18,22 BENCH=...
TRACE=...`; afterwards compare with a one-process trace on the 128 GB node. s1 and the leader on the 128 GB node both
need `BUDGET_GIB`; both sides of the comparison use `TF_GLM_CACHE_GIB=8` with enough room that the byte cap stays out of
play.

## Differences from the spec

1. **CLI front end**: the task text suggested `glm_run.py`, but the spec puts new code under `tfpp/glm/` only, so it is
   `python -m tfpp.glm.run`. Modes local / first / middle / leader; `last` is an alias of `leader` (the GLM chain's
   leader is its last stage).
2. **END carries `live`**: the snapshot ids the leader still holds, so a follower drops orphan states (a SNAP the leader
   never `_remember`ed after a request stopped midway). The spec has no such field.
3. **ERROR also travels the control direction**: when startup or the handshake fails, the leader sends ERROR upstream;
   followers forward it and exit 75. The spec lists the data direction only.
4. **`self.live = []` after an abort**: conservative; snapshots relying on the live cache rows are dropped later in
   `_take_over`.
5. **Settings vector**: a follower computes HELLO before GO, so the vector holds only env values and what modules read:
   prefill rows from the env, plus the multi settings. P, the CLI's prefill_rows and the context come from GO. Also
   `dflash_policy`.
6. **The env digest excludes more than the spec**: also `TF_GLM_CACHE_GIB` (the leader's only), `TENSORFOLD_MEMORY_RESERVE_GIB`,
   `TFPP_FAULT`, `TFPP_ALLOW_TRITON`, `TFPP_RESERVE_GIB`, `TFPP_GLM_*` (the scripts' own variables).
7. **`env.check_env`**: a required setting that is unset is filled in; only a different value is refused.
8. **`--budget-gib`**: taken as net; the pinned staging is not subtracted again.
9. **Test hook `TFPP_FAULT=REQ:STEP`**: after the warm-up, the leader raises in its REQ-th request after sending the STEP-th
   window, to test the drain and END abort.
10. **BUILT**: also compares each stage's main + sparse graph counts.
11. **v1 refuses `--parallel` != 1**: the P > 1 check of spec 9.2 step 3 is left to v2.
12. **T-K comparison targets**: the EXL3 experts' activations compare `b.exl3.xd` (the EXL3 path does not use `b.eact`).
    The indexer's selected tokens are not kept in a buffer, so its inputs `b.qi` and `b.ikr` are compared. `b.qi` is
    written only for windows with sparse rows (G/forward.py:686, 708), so it is compared only where
    `host + R - 1 >= dense_limit`.
13. **Scripts**: `cluster.env` also holds the leader's serve flags (CONTEXT and so on), outside the digest; node.env's ROLE
    is leader / follower; `start_stage.sh` supports `TRUNCATE` / `TAP_LAYERS` / `BENCH` for T2.
14. **bench**: tokens come from ChatTemplate + `thinking_off`, close to GlmApp's rendering only. Pictures and grammars go
    through HTTP only (T1-http).
15. **The `--pure` reference** is still `PipeGlmEngine`, without `_compute_pp` and the seams. The original `GlmEngine`
    needs NCCL and world >= 2, so it cannot be the reference directly.
16. **v2 was not part of v1**: spec section 14 (multi-stream) came later (below); `pins.py` already listed the seams v2
    had to re-review.
17. **Socket buffers**: spec 7 sets SO_SNDBUF / SO_RCVBUF to 8 MiB; the code leaves them to TCP autotuning (F5) and logs
    the sizes at connect.
18. **Small frames**: spec 7 joins frames under 1 MiB into one `sendall`; on Linux it is a scatter-gather `sendmsg` (F7),
    Windows (no `sendmsg`) keeps the old way.
19. **A LINK of another protocol version** raises `ProtocolError` (not `LinkError`), so connect retries do not retry it;
    the exit code is still 75.
20. **A memory check after the build** (F9): spec 11.3 checks once, at CAP; READY now tightens the kept-prompt budget
    with `free_after` and refuses to start when it is too small.
21. **The settings vector has `revision`** (F10); **the env digest covers the Triton / cuBLAS / TF32 numerics knobs**
    (F12).

## Review fixes (2026-10-23, second round, static)

| # | Problem | Fix |
|---|---|---|
| F1 | T-K compared `b.qi` in the first prefill window (2,048 rows from pos 0, all dense), but MIA writes it only with `sparse_rows` (G/forward.py:686, 708; `torch.empty` at G/forward.py:97): both copies were uninitialised memory, a false failure | `kernel_check.py`: `qi` only in windows with `host + R - 1 >= c.dense_limit`; `ikr` always (written whenever there is an index, G/forward.py:691) |
| F2 / F8 | The kept-prompt byte cap is not cut-invariant: the leader counts only its own layers' snapshot bytes (G/engine.py:1234, 1278, 1335; G/decode.py:515) and `cache_bytes` takes the least room of the stages, so solo and chain runs evict different snapshots and later PREFILL / chunk records differ | Test settings keep the byte cap out of play: `run_truncated.sh` defaults `BUDGET_GIB=24`, `KEPT_GIB=8` (exported as `TF_GLM_CACHE_GIB`) and checks that the leader's log says `kept in 8.00 GiB` (else rc 3). Added `T1-kept` (0.5 GiB pressure) and `compare --tokens-only`. Serving is unchanged: inside a chain every stage follows the leader's decisions (SAVE / DROP), so they agree. T3 invariant 1 (cut invariance) and the 0.5 GiB round either keep the byte cap out of play or compare tokens only |
| F3 | When the relay accepted an upstream connection while nothing listened in WSL, the failed `TcpClient` was only logged and the upstream socket stayed open; the upstream believed it was connected and sat in `handshake` for up to an hour without retrying | `relay_hop.ps1` closes what it opened (`$a` / `$b`) on a failure. `startup.connect_down`: a connect + LINK that fails (EOF, or no reply within 15 s) is closed and retried a second later, until `--connect-timeout`; another protocol version is refused at once. Tested locally through a relay (127.0.0.1 -> 127.0.0.2): with WSL's side listening 4 s late, linked after 4.1 s |
| F4 | The leader read the link only during a request; a follower that saw its upstream break while idle sent ERROR and exited, but the leader found out only at the next request, which then answered 500 | `PipeGlmEngine._watch` (a daemon thread): with no request running (`run_lock` held by `_run` and `quit`), anything readable on the link is read; ERROR, EOF or any unexpected frame goes to `_fatal` (exit 75). `quit()` sets `closing` first, so the EOF after QUIT is not a break |
| F5 | `setsockopt(SO_SNDBUF/SO_RCVBUF, 8 MiB)` is capped by `net.core.[wr]mem_max` on Linux and turns autotuning off. Measured in WSL: 8 MiB set, 425,984 read back; `tcp_wmem` max 4 MiB | Not set any more: autotuning; the real buffer sizes are logged at connect |
| F6 | The relay's USB4-side sockets had no keepalive; when the far node lost power the connection stayed half-open and the chain noticed only at the next request | `relay_hop.ps1`: keepalive on `$a` and `$b` (60 s idle, 10 s probes, best effort). Once detected the relay closes the pair, the stage behind it sees EOF while idle and sends ERROR, and the leader ends through F4's watch. No application-level heartbeat (PING / PONG) |
| F7 | ACT frames under 1 MiB (decode windows) were `tobytes`-ed and `join`ed: two more copies per hop | Linux sends straight from the pinned staging with `sendmsg`, partial writes handled; Windows keeps the old path. Tested in WSL from 0 B to 64 MiB (partial writes included) and for raw forwarding |
| F9 | Memory was checked once, at CAP; graph pools, kernel images and fragmentation could only eat the reserve | BUILT carries `free_after` (measured after the build, the captures and `empty_cache`); CAP carries `declared`. `capacity.kept_room`: on a stage without `--budget-gib` the kept-prompt budget is tightened to `(free_after - reserve/2) / scale`; less than half a reserve left refuses the start and names the stage. The free memory is logged again after the warm-up |
| F10 | Stages compared only the sha256 of config.json and the index; two different checkpoints of one model can have identical copies of both | The settings vector gains `revision` (the first line of `.cache/huggingface/download/config.json.metadata` that `hf download --local-dir` leaves, or the Hub cache's `snapshots/<commit>`; `unknown` without either). `start_stage.sh` gains the optional `VERIFY_SHARDS=1` (`load.verify_shards`). The preflight prints the revision and warns on `unknown` |
| F11 | `start_stage.sh` only warned about other GPU servers on the node, checked the leader's own port only on the leader, and did not require `BUDGET_GIB` with two stages on one GPU | It refuses to start beside a listening GPU server (now the ports in `GPU_GUARD_PORTS`) unless `ALLOW_GPU_SHARING=1`, and refuses without `BUDGET_GIB` when another `tfpp.glm.serve` process runs on the node |
| F12 | The env digest did not cover the Triton / cuBLAS / TF32 numerics knobs | `DIGEST_PREFIXES` gains `TRITON_`, `CUBLAS`, `NVIDIA_TF32_`, `TORCH_ALLOW_TF32`; `TRITON_*` names ending in `_DIR` / `_PATH` / `_HOME` are each node's own. Both scripts clear these prefixes before starting |
| F13 | torch from the default index may not be a cu13 build; the text suggested tf-venv-061 could be used as is | The toolchain's install line uses `--index-url .../whl/cu130` (still 2.13.0 per the spec). The preflight requires `torch.version.cuda` 13.x and runs a small kernel; the toolchain refuses tf-venv-061 |
| F14 | Stage 0 moved a whole request's picture rows (up to ~49k rows, 384 MiB) to the GPU at once, which stage 0's geometry did not count | `LocalFeed` keeps the rows on the host; each chunk uploads only its own (`feed._chunk` adds `.to("cuda")`; on the leader the rows are already on the GPU, a no-op) |

**Not adopted**: F8's "whole-chain byte conversion" (the leader counting kept-prompt bytes for the whole chain): serving
already decides consistently inside a chain, and the difference matters only when comparing different cuts, which the
test settings handle. F13's suggestion of torch 2.14.1: the spec (12.4) fixes 2.13.0 with Triton 3.7.1; 2.14.1 was
verified only on Flash-Next and its Triton 3.7 compatibility is unconfirmed. Other side suggestions not taken: F6's
application heartbeat (a protocol and follower change), F11's `systemctl is-active` check (unit names unknown; the port
check covers a running service) and `HOST_HEADROOM_GIB`, F9's fixed 1 GiB floor (half of each stage's reserve instead:
1.25 GiB on the 64 GB node).

## What the hardware runs did not cover (v1)

Verified on hardware are exactly the items of the status table at the top. Not covered by those runs, so unverified:
- The full 45-layer model and the three-node chain (no third node yet), long contexts on the full model.
- T-K (`kernel_check`, world=1 against TP2 halves) is a diagnostic, not part of the status above. The run log records
  it passing on both nodes (layers 0-3, 81 comparisons), before and after the CUDA 13 rebuild.
- Prefill throughput over real USB4 after autotuning (F5 expects about iperf's 615 MiB/s), how long the relay's keepalive
  takes to close after the far node loses power (Windows defaults: 10 probes, about 160 s), whether `free_after` is
  stable under WSL2's unified memory (F9's tightening and refusal rely on it), `VERIFY_SHARDS` on 80 GB, whether every
  node has the HF metadata (otherwise the revision is `unknown`).
- Admission: `mem_get_info` semantics under WSL2 with unified memory, how far `stage_geometry` is from the real
  `max_memory_allocated`, the kept-prompt conversion ratios.
- Beyond the answered HTTP requests: streaming under load, tool calls, grammars and pictures through the chain, StopVote,
  releasing a client that disconnects.
- The warm-up's self-check (drafted equals serial) rests on MIA's invariant holding on a truncated model too.

---

# v2.0 (2026-10-24, `--parallel 2..8`): verified on one GPU

**VERIFIED ON ONE GPU (2026-10-24), 8-layer truncated model:** batched-row invariance (T0e); pure and eager against graphs on three lockstep schedules; 2-stage chains at 2 cut points x 3 schedules bit-identical, every stage applying the leader's messages; replies equal to P=1 serial under free timing at 1, 2 and 4 streams. **Not yet run across machines.**
Two test-script defects were found on hardware and fixed in `run_v2.sh`: a `local` that expanded `$run` before
assigning it, and `sched_plan` writing the caller's `sp` (bash locals are dynamically scoped).

Built from the tfpp-glm v2 final spec. The leader runs MIA's `GlmScheduler` / `MultiDecoder` as they are; followers
override ADMIT, `_fill`, `_fill_many` and `_round`; a round makes one forward pass through the whole chain. v2.1 (FILL
trains, grouped pipelining) is **not done**; the protocol is final already (every v2 frame carries `g`, always 0 in
v2.0), and `v2part.GROUP` and AGREE's `groups: 1` are the v2.1 hooks. **Statically verified only (CPU, no torch, no
GPU).**

## v1 isolation and its proof

- All v2 code lives in new files `tfpp/glm/v2*.py` (`tfpp_sha` hashes `glm/*.py`, so v2 code is always covered).
- v1 files were left byte-identical to the hardware run (`tail.py part.py wire.py proto.py capacity.py startup.py
  trace.py compare.py bench.py feed.py env.py load.py run.py pins.py kernel_check.py __init__.py bench_prompts.jsonl
  tests/test_static.py`, `scripts/glm/run_truncated.sh`, `scripts/glm/cluster.env` with `PARALLEL=1` until M6.3 signs
  off), apart from 10 whitelisted insertions (`tests/data/v1_frozen/WHITELIST.json`): in `serve.py` S1-S3 (the three v2
  flags `--pool-rows` / `--kept-entries` / `--mbench`), S4 (a follower given one of them is refused), S5 / S6
  (`--parallel > 1` goes to `v2serve.check_args_v2`; P=1 with a v2 flag is refused), S7 (`_leader`'s first statement:
  `--parallel > 1` goes to `v2serve.leader_v2`); in `follower.py` F1 (class attribute `v2 = None`), F2 (GO's `P > 1` goes
  to `v2follower.start`), F3 (`run`'s first statement). At P=1 none of these conditions holds: `check_args`, `_leader`,
  `_start` and `run` execute v1's bytecode, and frames, traces and numerics are unchanged.
- Since then, documented post-hardware fixes changed some v1 files (section "After the hardware runs"); the manifest
  keeps the hardware hashes and pins each changed file with what changed.
- The proof is in `tests/test_v2_static.py`: `test_v1_files_are_byte_identical` (the manifest's sha256, or a documented
  post-v1 pin), `test_post_v1_changes_are_documented`, `test_serve_and_follower_differ_from_v1_only_by_the_whitelist`
  (AST: with the 10 whitelisted statements removed, the module's `ast.dump` equals the frozen copy under
  `tests/data/v1_frozen/`; each statement exactly once; the whitelist's own hash is fixed in the test),
  `test_v1_tfpp_sha_baseline` (the v1 file set's recorded hashes give the hardware run's `tfpp_sha`, `7a8e0a2e...`).
- Note: `tfpp_sha` differs from that baseline since v2 files were added (HELLO compares it: every stage must run the
  same tree). To use the 2026-10-23 hardware run as the v1 baseline, use the manifest's value, not today's HELLO.
- `start_stage.sh` gained only lines that act at P > 1 (`POOL_ROWS` / `KEPT_ENTRIES` / `MBENCH` inside the
  `PARALLEL > 1` block; `MBENCH` also keeps the leader from serving HTTP or checking the HTTP port; `TFPP_GLM_TIMING` /
  `TIME_ROWS` / `FAULT_V2` / `FAULT_FOLLOWER` are exported when set in node.env). With these keys unset the command line
  is v1's.

## Files

| File | Contents |
|---|---|
| `v2proto.py` | The v2 ops (`MULTI` / `WINDOWS` / `KEEPS`), ACT fields (`chunk` / `pieces` / `round`), pure checks of WINDOWS / KEEPS / ACT, `Fault` (`POINT:N` fault injection), `RoundStats`, `describe_ops` (no torch) |
| `v2part.py` | `_compute_pp_multi` (`part._compute_pp` with the prompt ACT under v2 headers; `prefill_pieces` called with `st=None`, fields from `mixer.pieces`), `install` |
| `v2wire.py` | `recv_rows` / `send_rows`: each field and the parts checked against what the stage computed itself |
| `v2capacity.py` | `pool_plan` (pool rows, kept entries, `limit`; the strictest stage wins, a refusal names it), `stage_states`, `kept_room_v2` (refuses after the build, never lowers the count) |
| `v2startup.py` | AGREE's v2 values (`pool_rows`, `entries`, plus `multi_window`, `limit`, `groups`), `check_agree`, `built_info_v2` / `compare_built_v2` |
| `v2tail.py` | Leader: `PipeMultiEngine` (a `PipeGlmEngine` subclass; `__init__` follows tail.py with v2's steps), `PipeBatchedVerify` (WINDOWS / ACT / KEEPS), `PipeMultiDecoder` (`round` / `finish` hold `run_lock`; any exception fail-stops the chain), `send_multi` (MULTI + picture rows), `install_stream_tags`, `build_multi`, `warmup_multi` (lockstep) |
| `v2follower.py` | Followers: `FollowerGlm`, `FollowerMulti` (`apply`'s ADMIT, `_fill`, `_fill_many`, `_round`), `FollowerBatchedVerify`, `start` |
| `v2serve.py` | `check_args_v2`, `leader_v2` (tracer; mbench / bench / HTTP dispatch) |
| `v2trace.py` / `v2compare.py` | Record kinds `meta2 iter msg admit fill pieces round keeps pool idle` (written by the worker), `req done fail` (by the client), the followers' `msg idle act`; five comparisons `strict` / `stages` / `streams [--rows]` / `tokens` / `bench` (exit 0 equal, 1 different, 2 refused) |
| `v2bench.py` | `Lockstep` (deterministic scheduling), schedule loading and expansion, `--mbench` lockstep / free runs, an HTTP client (`python -m tfpp.glm.v2bench http ...`) |
| `v2check.py` | T0e: bit invariance of batched rows at world=1 (forward three ways, two commit sequences, multi-prompt chunk pieces) |
| `v2pins.py` | AST pins of the 55 MIA seams v2 depends on (`--check TREE` / `--print TREE`; not a start condition) |
| `schedules/*.json` | AS1-AS5 (lockstep), FREE1 / FREE2 / FREE4, HTTP8 |
| `tests/test_v2_static.py` | T-S v2; `tests/data/v1_frozen/`: MANIFEST, WHITELIST, the frozen `serve.py` / `follower.py` / `start_stage.sh` |
| `scripts/glm/run_v2.sh` | Below; `node.env.example` documents v2's keys (commented) |

## How to run

**M6.0 (any CPU)**
```
cd tfpp
python -m py_compile tfpp/glm/*.py
python -m pytest -q tfpp/glm/tests
python -m tfpp.glm.pins --check <MIA tree>/tensorfold
python -m tfpp.glm.v2pins --check <MIA tree>/tensorfold
```
Some tests in `test_v2_static.py` import v2part, v2wire, v2follower, v2tail, v2trace, v2check and v2serve under stand-in
`torch` / `tensorfold` modules (MIA's pure `seal` / `parse` and others are taken from the vendored source by AST and run
for real): imports, the followers' frame order (forward upstream first, then handle), the `V2Link` / `send_multi` frames,
`install_stream_tags`. `Lockstep` is tested on a fake scheduler that imitates `GlmScheduler`'s loop (50 client threads
with random delays release the same sequence every time; idle gaps are skipped; `_Flush` wakes the worker).

**M6.1 (one GPU, the 128 GB node, several processes; `scripts/glm/run_v2.sh`; `run_truncated.sh` is not used)**
```
MODEL_DIR=$HOME/models/glm53 DRAFTER_DIR=$HOME/models/glm53-dflash2 L=8 CONTEXT=2048 POOL_ROWS=16384 KEPT_ENTRIES=32 \
  scripts/glm/run_v2.sh all          # T0a-v2 T0b-v2 T1-v2 T1-v2-p1 T1-v2-free
scripts/glm/run_v2.sh T2m            # the smallest end to end: T2m-solo (one process) against T2m-chain (cut at 4), strict + stages
L=8 CONTEXT=262144 POOL_ROWS=65536 scripts/glm/run_v2.sh all
L=8 CONTEXT=2048 POOL_ROWS=6144 SCHEDULES=AS3 scripts/glm/run_v2.sh T1-v2
L=8 CONTEXT=2048 KEPT_ENTRIES=4 SCHEDULES=AS4 scripts/glm/run_v2.sh T1-v2
L=12 CONTEXT=262144 scripts/glm/run_v2.sh T1-v2          # N=3
L=8 CONTEXT=262144 scripts/glm/run_v2.sh T0e              # first: the bit invariance of batched rows
scripts/glm/run_v2.sh T1-v2-fault; scripts/glm/run_v2.sh T1-v2-http      # one mode each
```
Criteria: `python -m tfpp.glm.v2compare strict ref got` (last line `IDENTICAL`); `stages` (the leader's message sequence
and the idle pools are the same on every follower); level A with `v2compare bench` (P=4 against P=1 `--no-drafts`
`--bench-out`). **Before starting**: back up the traces of the 2026-10-23 v1 hardware run (spec 16.4's v1 regression
compares against them).

**M6.2 / M6.3 (two, three nodes)**: the leader's node.env adds `PARALLEL=4 POOL_ROWS=... KEPT_ENTRIES=...
MBENCH=.../schedules/AS2.json` (followers need no new key); the rest as v1. Measurements: `TFPP_GLM_TIMING=1` (a line
every 100 rounds: ms a round, ms of it waiting for the stage before), `TFPP_GLM_TIME_ROWS=1` (ms for R=1..32 after the
graph captures; a non-first stage fills `b.x` with random values first), `FREE1/2/4`.

## Differences from the spec

1. **Scripts**: the task text said "run_truncated.sh's v2 modes"; spec 6.1 and 16.2 keep `run_truncated.sh` unchanged
   and want a separate `run_v2.sh`. Followed the spec: modes `T0a-v2 T0b-v2 T1-v2 T1-v2-p1 T1-v2-free T0e T1-v2-fault
   T1-v2-http`, plus `T2m` (`T2m-solo` against `T2m-chain`) and `ref-v2`; `all` is the spec's five of M6.1.
2. **`GuardedEngine`**: the chain's `decode.Engine` (`graphs=False`) is replaced by its subclass, whose `forward` raises
   with a link (F-14: nobody should call `e.forward` at P > 1; if someone did, a non-first stage would compute silently
   wrong from an x it never received). The spec says `D.Engine(...)`.
3. **`--no-graphs` also turns `MultiDrafter`'s graphs off** (`MultiDecoder(draft_graphs=False)`). Not in the spec; MIA's
   `draft_graphs=None` captures regardless of the `graphs` flag, and T0b-v2 would not compare eager then.
4. **Traces**: the `done` record is written client-side by the `generate` wrapper (the spec says "wrap
   `scheduler._reply`"; not done); one more record kind, `idle` (when the leader sends IDLE and when a follower applies
   it, each writes a pool summary, for `stages`); `msg` records carry a `sha` (the whole sealed message).
5. **`Lockstep`** lives in `v2bench.py` (with `Landings`, `Req`), shared by `v2tail`'s warm-up and `--mbench`;
   `RoundStats` / `log_rows` live in `v2proto.py` (leader and followers need them; followers should not import `v2tail`).
6. **T0e's commit part** is an interpretation: the same random windows and keeps through `SerialVerify` and through
   `BatchedVerify` (forward + commit), comparing the logits hashes of every later window.
7. **`compare bench`**: the rounds of a free run are told apart as `rep + 1000 x round` and compared with the reference's
   `rep`; both sides refused (e.g. a 6k prompt past the context, each error with its own limit) counts as equal; one
   refused and one answered counts as different.
8. **v2pins** pins two more seams than the spec lists (`GlmScheduler._Flush`, `forward.py::compute_steps`): 55.
9. "Pinning the pool and the kept count" (spec 10.2 / M6.1): `--pool-rows` / `--kept-entries` override independently;
   given one, the other follows the formula.
10. A draft of `v2proto.py` existed before the work started (consistent with spec section 5); it was kept and extended
    with the helpers above.

## Review fixes (2026-10-24, second round, static)

Twelve findings checked against the source; the real defects were fixed, and the P=1 path did not change (`serve.py` /
`follower.py` kept exactly the 10 whitelisted statements).

**Fixed**
1. **A cancelled lane still filling was never reaped** (`v2tail.PipeMultiDecoder._iterate`). MIA polls
   `emit.cancelled` only with a `fill_budget` (G/multi.py:1696-1705), and `TF_GLM_FILL_BUDGET_MS=0` is required. The
   override does the same reaping when `fill_budget` is 0 (no I/O), returns the cancelled streams, and `finish` sends
   FINISH (CANCELLED) as usual and every stage runs `_finish`. `partial` is always None in v2, so sliced fills need no
   handling.
2. **`stop_after`'s cap was lost after a replay** (`install_stream_tags`). A give-back or a background yield makes a new
   Stream with `Stream.continued()` and the same `glm` (G/multi.py:2642, 2651); the cap used to be attached only at the
   first put (no tag yet). Now every put looks the tag up in `eng.v2cancel`, and `stream._tfpp_capped` prevents a double
   wrap.
3. **`Landings` grew without bound** (HTTP gives each request a `reqN` tag whose Event was never removed). `Landings`
   now keeps Events only for registered tags: `register` (called by `Lockstep.__init__`), `signal` (used by
   `tagged_put` and `_client`; an unregistered tag does nothing), `forget` (at the end of `Lockstep.run`). HTTP serving
   accumulates nothing; `eng.v2cancel` is written only by `--mbench`.
4. **The leader's `TFPP_GLM_TIME_ROWS` timed an all-zero x**: `build_multi` fills `b.x` with `normal_()` on a non-first
   stage first, as the followers do, so M-1's columns compare.
5. **The leader's kept state was underestimated by about 20 MiB** (`v2capacity`). MIA's 22 MiB is one TP2 rank's figure
   (G/engine.py:747); v2's leader is world=1, and `draft_window` (G/multi.py:311-317, G/decode.py:450-453) copies the
   whole ring window: 2 x layers x KV heads x sliding_window x head_dim x 2 B, with DFlash2's settings 5 x 8 x 2048 x 128
   -> 40 MiB, of which a TP2 rank holds 20. Added `draft_window_extra(drafter_cfg)`; `stage_states(..., drafter_cfg=)`
   adds it on the last stage (the leader) only; 0 without a drafter (`--no-drafts`). As a result the 19/7/19 leader's
   kept state goes from 78 to 98 MiB, the pool and entries come out slightly smaller, and `kept_room_v2`'s half-reserve
   guarantee holds.
6. **T0e checked only half the claim** (`v2check.py` part 3): a piece of a multi-prompt chunk gets the bits of the same
   rows as a chunk of their own (G/multi_prefill.py:27-28). Added the pieces part: mixes of 2, 3 and P prompts cut at
   grid point 64 (a piece that stops midway, one that ends its prompt, a single-piece prompt), comparing head rows,
   every piece's tap rows, the logits of one window afterwards, and `pos`; results in `result["pieces"]`, any
   difference fails T0e (only then does level B become a criterion).
7. **`run_v2.sh`**
   - `T1-v2-fault follower-stopped`: no more `sleep 25` guessing when to SIGSTOP; followers gained
     `TFPP_GLM_FAULT_FOLLOWER=windows:40:stop` (`Fault`'s `:stop` suffix SIGSTOPs itself) at the protocol's 40th
     WINDOWS. The script checks the s0 log for `SIGSTOP` and says the case proved nothing without it.
   - `AS3` / `AS4`'s `suggest` (`pool_rows` 6144 / `kept_entries` 4) is now read: `sched_plan` decides each schedule's
     pool plan (the caller's own `POOL_ROWS` / `KEPT_ENTRIES` win), the reference and every cut use the same one, and
     the pinned-plan check uses that schedule's values.
   - `L=26` allows only the one-process modes (`T0a-v2 T0b-v2 T0e ref-v2`); the other modes would put both 26-layer
     stages on one GPU (66.7 + 34.8 GiB) and run out of memory, so they are refused, pointing at `start_stage.sh`.
8. **`start_stage.sh`'s P=1 invariance had only a substring test.** The v1 file had not been kept, but removing v2's
   three additions (the export list, the `MBENCH` checks, the `PARALLEL>1` block, plus the BENCH/MBENCH condition of the
   port check) gave exactly the sha256 the manifest recorded for v1 (`f75165d9...`). That file is now
   `tests/data/v1_frozen/start_stage.sh`, and `test_start_stage_minus_v2_is_v1_byte_for_byte` requires "the live file
   without its v2 lines == the frozen file".

**Rejected / unchanged**
- **The follower's `_round` does not check ROUND's depths (PLAUSIBLE)**: depth and window rows have no fixed relation
  (copy drafts and `trim` change the rows: G/multi.py:2070-2095, and grammars cut them), so "rows == depth + 1" is a
  wrong invariant; positions are checked through WINDOWS' `[slot, pos, rows]` and KEEPS' `end`, and no follower code
  reads `lane.depth` / `s.out` / `s.done` (`_keep`, `_finish`, `_fill`, `_admitted`, `_stop_bound` checked). By design
  the followers' lanes carry no depth state (the reason is a comment at the top of `FollowerMulti._round`).
- **info: HELLO's `tfpp_sha` differs from the 2026-10-23 HELLO log**: the new `v2*.py` files are inside `glm/*.py`, so
  the P=1 HELLO's `tfpp_sha` changed; by design (v2 code must be covered by the digest, spec 6.1), P=1 frames, traces
  and numerics are unchanged. For a v1 regression use the manifest's `tfpp_sha_v1` (`7a8e0a2e...`). Every stage of a
  chain must run the same tree.

## Not tested (all of it needs a GPU or MIA's full environment)

- **Every CUDA path**: `_compute_pp_multi` and `BatchedVerify.capture` under scrub (32 graphs for R=1..32 at P>1, one set
  per stage), ACT over pinned staging (R <= 32 and 2,048 rows), KeepWarm's power and clocks while s0 / s1 idle long,
  the memory of `decode.Engine(graphs=False)`.
- **T0e has no verdict yet** (forward, commit, pieces): batched-row bit invariance at world=1 is MIA's claim only
  (G/verify.py:5-12). If T0e fails, level B (equal logits across schedules) does not hold, and level A (P=4 against
  P=1 tokens) can flip on near-tie argmaxes; the warm-up's "drafted equals serial" check would then fail the start
  (`--no-warmup` skips it).
- **The hooks into MIA's scheduler**: `sch._iteration` overridden as an instance attribute, `waiting.put` wrapped as an
  instance attribute, `(_Flush, Event)` waking an idle worker: all read from the code (G/multi.py:2586-2587, 2662;
  TF/cuda/scheduler.py:23-24) and run on the fake scheduler only; `v2pins` pins that source, so a new package version
  needs a review.
- **The whole chain**: handshake, the order and back-pressure of MULTI / WINDOWS / KEEPS / ACT, fault injection (the five
  `round.windows`-style points, a follower's self-kill and SIGSTOP), `_watch` with `run_lock`, client disconnects,
  background yields, give-back replays, grammar / tools / thinking / vision (picture rows forwarded with MULTI at every
  hop, up to about 384 MiB).
- **Memory estimates** (19/7/19, P=4, 256K: 69.8 / 33.6 / 84.0 GiB) and `pool_plan` against a real `mem_get_info`;
  `pool_plan`'s arithmetic was checked on CPU with MIA's geometry (per stage per token 2800 / 1672 / 3364 B, kept states
  83 / 42 / 78 MiB without a drafter (98 MiB on the leader with DFlash2, above), a pool of 1,452,032 rows, limit
  262,144, matching the design's memory model).
- **`run_v2.sh`** and the new lines of `start_stage.sh` were only checked with `bash -n` (and the fault-case verdict with
  a stubbed chain, below).
- **Throughput**: v2.0's aggregate is modelled at about 1.5-1.7x a single stream with each stream at about 40% (spec 0.3,
  17; M-1, M-4, M-6 must be measured first). Whether to serve with PARALLEL=4 or 2 waits for the M6.3 measurements;
  matching the dual DGX Spark (GB10, tensor parallel over RoCE) deployment needs v2.1b.
- **Known regressions**: v2 has no per-request abort protocol (an exception in any iteration exits the chain with 75);
  FILL is synchronous (one long prompt runs at about 0.43x of v1's prefetching); `stop_after`-style limits are
  replaced in lockstep by a worker-side cap (applied again after a replay).
- New in the second round and not run: `_iterate` reaping cancelled fill lanes (logic checked on a fake decoder, no real
  disconnected HTTP request), `v2check`'s pieces part, `TFPP_GLM_FAULT_FOLLOWER=windows:40:stop` (`Fault`'s unit test
  mocks `os.kill`), `run_v2.sh`'s `sched_plan` (checked with local bash for AS1 / AS3 / AS4) and the L=26 refusal. The
  effect of the leader's extra 20 MiB on `pool_plan` was not checked against a real `mem_get_info`.

---

# After the hardware runs (2026-10-24): fixes and publish hygiene, CPU tests only

A review of v1 and v2 after the hardware runs. Each defect was checked against the code first; the fixes below change
behaviour only where the defect was. **None of this has run on hardware yet**; CPU tests:
`tests/test_glm_fixes.py` (sockets under a stand-in torch, the tools, the shell scripts under bash with /proc) and the
updated `tests/test_v2_static.py`. The v1 files changed here are pinned in `tests/data/v1_frozen/MANIFEST.json` under
`post_v1`, next to their hardware hashes.

**v1 (verified code, so the smallest change that fixes the defect)**
- **A stray connection could take a listening stage's one link** (`wire.Link.accept_linked`, used by
  `startup.leader_listen` and `follower_connect`). `Link.listen` accepted once and closed the listener before LINK, so a
  port probe, a health check or a scanner killed the startup (exit 75), and a silent one held it for up to an hour.
  Now the listener stays open until a connection exchanges LINK: anything else is closed and logged, each accepted
  connection has 5 s for its LINK (under connect_down's 15 s), a peer that sent LINK and already closed is passed over,
  and a LINK of another protocol version still fails at once.
- **A non-tfpp peer was retried for the whole connect timeout** (`wire.Link.recv(link_frame=True)`): during the LINK
  exchange a first-frame header over 64 KiB, or a first frame with parts, is a `ProtocolError`, so `connect_down`
  refuses an SSH or TLS server at once, as its docstring said.
- **A follower stopped by the leader's startup ERROR left through `sys.exit`** (`startup.forward_or_fail`), past
  serve's `os._exit` and into the interpreter teardown with CUDA state and a KeepWarm thread. It raises `LinkError` now:
  `follower.run` returns 75 and the process ends through `os._exit`.
- **A stage with a DSA layer and no KDA layer passed HELLO** (`proto.check_chain`) and failed only at READY, after all
  weights were loaded (one graph parity against two). HELLO now requires a KDA layer in every stage too.
- **An error after every prompt chunk was committed killed the chain** (`tail.py`'s prefill seam): a grammar or
  sampling error at the first token is the request's alone when no chunk is in flight and the link is fine; it now goes
  through `_run`'s END abort, as the same errors in decode do. Chunks in flight or a link error stay fatal.
- **`compare` passed on nothing** (`compare.py`): two traces without requests, or two runs that both stopped mid-request
  (no `done` / `fail`), returned 0. Both are failures now (`INCOMPLETE`, "nothing was compared").
- **An MLX checkpoint's embedding** (`load.mine`): `embed_tokens.scales` / `.biases` are the embedding's, like `.weight`.
- **A refused admission did not say which stage binds** (`capacity.agree`): the package's message ("across the ranks")
  now gets each stage's budget, what it needs at the requested window, the window it fits, and the least one.
- **Scripts**: `start_stage.sh`'s CUDA-12 extension probe ended the script silently (status 1, `set -euo pipefail`)
  when the extensions folder did not exist yet, e.g. on a first start or `--dry-run`; it tolerates that now. The pid
  file was trusted with `kill -0` alone: after a crash or a WSL restart its number could belong to another process,
  which `start_stage.sh` took for the stage ("already running") and `stop_stage.sh` (also systemd's ExecStop) signalled.
  Both now accept the pid only while its command line is `-m tfpp.glm.serve --role ROLE`. `stop_stage.sh`'s log
  rotation failed the stop (status 2) when no rotated log existed yet, e.g. every systemd stop.
- **`relay_hop.ps1`**: a synchronous connect to a stale WSL address held the relay's only loop for ~21 s; it is bounded
  now (`-ConnectTimeoutMs`, 5 s; a refusal takes ~2 s on Windows and still counts as "not listening yet"). A connect
  that times out or finds no route looks WSL's address up again (at most every 10 s, never with `-WslIp`). The address
  comes from `-WslIp`, else `wsl.exe -e hostname -I`; `-PortproxyPort N` reads it from an existing portproxy rule (for a
  relay run from another Windows account than the WSL distro's owner, which earlier versions read from a fixed port).

**v2**
- **T0e part 3 compared the pieces' state with itself** (`v2check.py`): the solo run and the pieces run share their
  slots and extents (`State` views), so the solo window was computed on the memory the pieces run had left. The solo
  window is now taken before the pieces run, and the extents' cache rows are zeroed in between, so rows the pieces fail
  to write cannot pass as the solo run's.
- **T0e passed without a batched window** when `TF_GLM_MULTI_WINDOW` is under the mixes' 32 rows: the mixes are scaled
  down to the verify window now, and no batched mix (or no committed window) is a failure.
- **`Fault` fired at every hit from the N-th on** (`v2proto.py`): a `:stop` fault stopped the process again after
  SIGCONT. It fires at the N-th hit only.
- **Two trace races** (`v2bench.Lockstep`, `v2tail.PipeMultiDecoder`): `Lockstep.run` now returns only after the worker
  wrote its last `iter` record and came to rest, and the warm-up's eviction event is set after that iteration's
  `finish` (its IDLE), so two runs' traces start and end alike.
- **`run_v2.sh`**: a fault case passes only when the leader and every follower exit 75 by themselves (137 means reap had
  to kill it; stage 0 of the self-kill case must exit 137 by its own SIGKILL); the HTTP idle-kill case judges the
  leader's exit (75) and any middle stage's; a killed reference run is not reused (the bench file marks a finished
  one); signals and the verdicts are scoped to the run's own processes instead of every `tfpp.glm.serve` on the node.

**Docs and defaults**: role names in comments; `start_stage.sh`'s GPU guard reads `GPU_GUARD_PORTS` from node.env, default none); repository-relative paths and `$HOME` defaults
(`toolchain.sh` finds the checkout it lies in, or the calling script's); the DFlash2 config in `tests/data` is a
four-field stand-in (that drafter is CC BY-NC-ND 4.0); `cluster.env` says plainly that `HTTP_HOST=0.0.0.0` serves the
API unauthenticated on every interface.

**Operators of existing nodes**: re-sync every stage (the `tfpp_sha` changed). Put the ports of other GPU servers on a
node into its node.env as `GPU_GUARD_PORTS="..."`. A relay started from a Windows
account that does not own the WSL distro needs `-WslIp` or `-PortproxyPort N`. The systemd unit's paths assume the
repository at `~/hive-ram`; an installed copy of the unit is not affected until it is replaced.

**Not changed (deliberately)**
- `follower.py`'s busy-mode wait for the next control frame keeps `--io-timeout`: without it a hung leader would hold
  every follower (and its GPU memory) forever; a leader blocked longer than that on one slow streaming client is the
  rarer case. Revisit with a separate `--busy-timeout` if it shows up.
- HELLO does not compare the nvcc / ptxas patch level and the kernel cache is not keyed by it: adding it changes the
  versions every stage must agree on and moves every node's kernel cache (a full JIT rebuild), so it waits for the next
  planned re-sync of the toolchain.

# tfpp-ds41 v1 implementation notes (2026-10-23)

v1 implements the merged final spec of 2026-10-23: DeepSeek-V4.1-Flash on the bertholomus fork
(deepseek-v41-tp2 @ af26d42b1, world=1), split into contiguous layer ranges and run across N1X laptops.
The last stage is the leader: it holds the head, DSpark, the Scheduler and the HTTP front.

**Status: everything has been verified statically only; nothing has run on a GPU.** None of the following was changed:

- `vendor/`
- `tfpp/tfpp/glm/`, `tfpp/scripts/glm/`
- Flash-Next's `tfpp/tfpp/*.py` (`warm.py` included)

This package only uses (reads) these modules:
- `tfpp.warm.KeepWarm` (subclassed)
- `tfpp.glm.wire`: `Link`, `LinkError`, `wait_any`
- `tfpp.glm.proto`: the codec
- `tfpp.glm.compare`: `load`, `first_difference`
- `tfpp.glm.pins`: `tree_sha`, `installed_root`, `_find`, `SUFFIXES`
- `tfpp.glm.env`: `model_revision`, `file_sha`
- `tfpp.glm.capacity.measure_budget`
- `tfpp.glm.load.verify_shards` (only in `start_stage.sh`'s preflight)

## Files

Under `tfpp/tfpp/ds41/`:

| File | Contents |
|---|---|
| `__init__.py` | `PROTO = "tfpp-ds41/1"`. Imports neither torch nor the fork at import time |
| `plan.py` | Pure functions. `ModelView` (reads the config), `carry_into` (§5.1), `StagePlan`, `plan_chain` (§5.2), `chunk_in` (§6.2), the ACT part lists (§6.4, §7.1, §9.3), `cand_form`, `buckets`. CLI: `rules`, `chain`, `fit` (§16.1) |
| `patch.py` | 6 anchors, 4 F1 anchors and `patched()` (§3.1). Also `function_source` and `check_tree`, which check the source tree without importing it |
| `proto.py` | The op table, `DsStageInfo` (one more field, `extra`, for each stage's own values), `compare_stages`, `check_engram` (§5.2 item 3: one owner a layer, exactly its own tables), `digest` |
| `env.py` | §13.1's enforced values, the variables that must be unset, the import-order guard, the Triton 3.7 check, §13.2's settings vector and env digest, `tfpp_sha`, `versions`, `check_device` |
| `pins.py` | `EXPECT_COMMIT`, `EXPECT_PKG_SHA` (`c64d3e0a…`, LF), `FAMILY_SHA`, the AST hashes of 90 seams. CLI: `--check`, `--print`, `--diff` |
| `load.py` | `StageLayers` (§4.1), `AliasShards` (§4.2), `load_stage` (§4.3), `load_full` (swapped in for `weights.load` by ref), `token_map`. Pure functions too: `mine`, `needed_shards`, `stage_files` (an extracted `embed.weight` file can stand in for shard 1), `embed_only_file`, `selection_config`, `selection_namer`, `engram_rows`. The `token_map` cache is written to a temporary file, then `os.replace`d; a corrupt file is rebuilt |
| `testmodel.py` | S11's `build` (`--revision`; brings each shard's `.metadata` into the S11 folder; `link_shard` replaces a stale or dangling shard link), the Engram test tables `table` (Mode D / Z), `extract`, `selftest`, `downloads` (only prints a script whose `REV` line must be edited first; `--need-bytes`), `table_identity` |
| `part.py` | `StageModel` (`new_cache`, `forward`, `_forward_k` with M1 / M2, the F1 hooks), `StageEngram` |
| `decode.py` | `StageBuffers` (the static round views), `StageRoundDecoder` (`run` with R1 / R2, `_enter` / `_exit`, `_l2_next`, N1's `capture`, the F1 hooks), `StageRoundRunner` (`_forward` with R3, SHARED_OUT only on the last stage, the leader sends STEP) |
| `wire.py` | `DsLink` (its own LINK protocol string), `StageIO` (sends, receives and checks the three ACT kinds chunk, kv and round, and the local mode) |
| `multi.py` | `StageMulti` (MU1, `_shape_pp`, `_broken`, `_drafts` with the gate, `_fill_follower`), `PipeOps` (the fork's `_send` as frames, outstanding, drain, `TFPP_FAULT`), `PipeWatch` |
| `engine.py` | `StageEngine`, `PipeDsEngine`, `FollowerEngine` (F/engine.py:46-167's order in two phases: the weights, then the build after AGREE), `stage_warm` (§11.3), `self_check` (§11.4) |
| `follower.py` | The follower's main loop (§8.2): ADMIT, FILL, STEP, FINISH, DROP, QUIT. Every op checks the digest; idle, it watches upstream |
| `startup.py` | HELLO → GO → CAP → AGREE → READY (G/startup.py's shape, on `DsLink`), `stage_extra` (the Engram tables' identity and the token map) |
| `capacity.py` | §12's pool formula, staging, `check_caps` (staging not counted twice; the stages sharing one GPU on one machine summed as well), `check_built`, `switches_sha`, `warm_buckets` / `lazy_buckets` (the round graphs the warm-up leaves to serving) |
| `warm_gate.py` | `GatedKeepWarm` (§11.1) |
| `trace.py` | Instance hooks recording `req`, `admit`, `chunk`, `step`, `keep`, `done`, `fail`; followers also record `act`. The first `meta` record carries `run` (`run_meta`: the settings vector, the env digest, versions, context / parallel / drafts / graphs). Holds `devhash` (copied from G/part.py:117-138). A round's taps are recorded only on an engine with a drafter |
| `compare.py` | Three comparisons: record by record (default), `--tokens-only`, `--by-position`. The two traces' `run` metas are compared first: different settings return 2 (not comparable); `--allow KEY,…` lets through the keys a test changes on purpose, `--ignore-meta` skips the check |
| `bench.py` | Runs a prompt set without HTTP. Supports `rand_len`, `ids`, `text`, `group` (concurrent within a group), `--bench-max-prompt` |
| `oracle.py` | T1-oracle's draft replacement (§15.6) |
| `ref.py` | The reference: the unmodified `DsEngine(world=1, parallel=P)`, only `weights.load` swapped for `load_full` |
| `kernel_check.py` | `--hc` (T-K'), `--layer i` (T-K) |
| `prebuild.py` | Clears build locks nobody holds, builds the four extensions, the readelf check, the Triton PDL check |
| `serve.py`, `run.py` | Entry points (§14.1). run's modes are local, first, middle, last (leader is an alias of last) |
| `bench_prompts.jsonl`, `bench_multi.jsonl`, `bench_long.jsonl` | §15.6's prompt set, the T1-multi set, the T1-long set |
| `tests/test_ds41_static.py`, `tests/data/ds41_config.json` | T-S: 39 tests, no torch needed (2 symlink tests skip on a Windows account without symlink rights). The data file is the public checkpoint's config.json. The file name differs from tfpp-glm's `tests/test_static.py`, so one pytest run collects both suites |

Under `tfpp/scripts/ds41/`:

- `toolchain.sh`, `cluster.env`, `node.env.example`
- `start_stage.sh`, `stop_stage.sh`, `tfpp-ds41@.service`
- `make_testmodel.sh`, `run_shortened.sh`, `download_list.sh`
- the relay is `scripts/glm/relay_hop.ps1`, unchanged

## Seams

Fork line numbers are af26d42b1's (LF). Seam hashes show their first 16 hex digits; the full values are in `pins.PINS`.

| Change | Where | Seam hash |
|---|---|---|
| M1: `floor` starts at the stage's replay floor | `Model._forward_k`, model.py:1325 | `558bea5ad1a3feb4` |
| M2: a stage without the head returns its exit state | same, model.py:1419-1420 | same |
| F1c, F1d (carry mode only) | same, model.py:1331-1332, 1416-1419 | same |
| R1: read the round ACT | `RoundDecoder.run`, rounds.py:600 | `5a1f6e4fcf19238a` |
| R2: send the round ACT | same, rounds.py:614 | same |
| R3: the decoder class comes from the runner | `RoundRunner._forward`, rounds.py:703 | `3579fda3ebd1d007` |
| MU1: the runner class comes from the MultiDecoder | `MultiDecoder.__init__`, multi.py:401 | `07cb794e7f65df4f` |
| F1a, F1b (carry mode only) | `RoundDecoder._layers_fused`, rounds.py:486, 549-551 | `5871bd71f4d3e508` |
| Subclass: `StageModel.__init__`, `new_cache`, `forward` | model.py:727-741, 744-769, 1241-1293 | `561aa7ed…`, `bce6a796…`, `fe1d4f83…` |
| `StageEngram`: `__init__`, `prefetch`, `touch`, `rows` | model.py:496, 599, 648, 677 | `aee9e664…`, `fa843dd9…`, `16f99e6e…`, `98e800cc…` |
| `StageRoundDecoder`: `__init__`, `_stretch`, `_l2_next`, `capture` | rounds.py:76, 396, 469, 553 | `e858e457…`, `95a84f45…`, `c679f1be…`, `9443dcb8…` |
| `StageRoundRunner`: `__init__` (no super call), `forward` | rounds.py:620, 657 | `502aad40…`, `8eaf2dfd…` |
| `StageMulti`: `_shape_pp`, `_broken`, `_drafts`, `_fill_follower` | after multi.py:535, 502, 466, 889 | `4377f8f4…`, `15d159c3…`, `d0dfa6fb…`, `010603b8…` |
| `StageEngine` follows `DsEngine.__init__`'s steps; `stage_warm` follows `warm` | engine.py:46-167, 191-225 | `0b89d0a8…`, `0e0c152e…` |
| `AliasShards`; `load_stage` follows `_load`; `load_full` replaces `load` | weights.py:35-45, 442-487, 422 | `c02b277f…`, `ee433ead…`, `6a98935e…`; `load_block` is `e71d1f5a…` |

How the anchors were verified:
- `patch.check_tree` against the vendor tree: each of the 10 anchors occurs exactly once, and the replaced sources compile.
- T-S includes this check.

## How to run

### T-S (any CPU, runnable now)

```
cd tfpp      # this checkout's tfpp folder
PYTHONDONTWRITEBYTECODE=1 python -m pytest -q -p no:cacheprovider tfpp/ds41/tests
python -m tfpp.ds41.plan rules --config <S11's or the original config.json>
```

### One-time set-up in each node's WSL

Follow the steps at the top of `scripts/ds41/toolchain.sh`:
1. Create the venv `~/tf-venv-dsv41`.
2. Extract the fork's LF bytes with `git archive` from a clone of the fork (commit af26d42b1) into the folder whose
   `src` is `TFPP_DS41_TF_ROOT`. A Windows work tree is CRLF: `pins --check` refuses it and says how to fix it.
3. Run `python -m tfpp.ds41.pins --check …`; it must report `c64d3e0a…`.
4. Run `python -m tfpp.ds41.prebuild`.

### The test model (downloaded by the operator)

1. `scripts/ds41/download_list.sh` only prints the hf commands. Check §15.3's disk conditions before the download.
2. Run `SRC=… scripts/ds41/make_testmodel.sh` on each node. It makes S11 and `~/engram-d/{L1,L4,both}` and prints
   sha256 values, which must be the same on both nodes.
3. When the source folder is copied to the other node over USB4, copy `.cache/huggingface` with it, or the revision
   check cannot hold.

### One GPU (the 128 GB node)

In §15.6's order; every step must be IDENTICAL before the next:

```
MODEL_DIR=~/models/ds41-s11 scripts/ds41/run_shortened.sh T-ref     # the reference twice, then with graphs=0
python -m tfpp.ds41.kernel_check --hc --model ~/models/ds41-s11 --out tk.json      # T-K'
python -m tfpp.ds41.kernel_check --model ~/models/ds41-s11 --layer 5               # T-K (diagnostic)
scripts/ds41/run_shortened.sh T0a; … T0b; … T0c
scripts/ds41/run_shortened.sh T1-oracle; … T1-exact; … T1-long; … T1-multi; … T1-fault; … T1-http
```

- The judge: `python -m tfpp.ds41.compare ref.jsonl got.jsonl`. T1-multi adds `--tokens-only` and `--by-position`.
- If T-K' fails: set `TFPP_DS41_CUT_POST=carry` (F1) on every stage, then run everything again.

### Two nodes

1. Run `run_shortened.sh T-X0` on both nodes, then compare the two `TX0.jsonl` with `compare`.
2. T1-c6, T1-c3, T-X, T1-3: each node's `node.env` takes the matching test block of `node.env.example`.
3. On the receiving node's Windows side, run `relay_hop.ps1 -ListenAddress <10.77.0.x> -Ports 5321`.
4. Run `start_stage.sh` on each node (in any order). A leader given `BENCH` sends QUIT by itself when it is done.
5. Compare the leader's trace with the 128 GB node's `ref.jsonl` using `compare`.

## Departures from the spec

1. **weights.py is not recompiled.**
   - The task text mentions a "loader patch module (anchors recompiled and registered in sys.modules)". The final
     spec's §0.1 item 4 and §4 call the public `load_block` directly instead, so the spec is followed.
   - `patch.py` only recompiles four functions inside the fork modules' own globals. It makes no new module and
     defines no dataclass, so nothing has to be registered in sys.modules (lesson 1 does not apply).
2. **HELLO's stage description is `DsStageInfo`.**
   - Defined in `proto.py`, with one more field, `extra`, for each stage's own values: the Engram tables' identity,
     the token map's sha.
   - `compare_stages` is written anew, because the GLM version's messages and eos comparison are GLM-specific.
   - `startup.py` is copied from G/startup.py and changed, because that module is not on the importable list.
3. **What `tfpp_sha` covers.**
   - `ds41/*.py`, `warm.py`, and the 7 imported glm files: `__init__`, `wire`, `proto`, `compare`, `pins`, `env`,
     `capacity`.
   - CRLF is turned into LF before hashing.
   - Not the tests and not the jsonl files.
4. **Trace and self-check.**
   - The trace has one more record kind, `admit` (sid, slot, extent).
   - `ref.py` runs the same self-check (§11.4), so the two runs' sids, slots and cache histories match. The trace
     starts recording after the self-check.
5. **The `--by-position` key.** A row's whole input is the key: stream, P, row index, the window's token prefix up to
   the row. The position alone does not work, because after a rollback the same position is computed again.
6. **`--no-drafts` relaxes the tap rule (new).** When the leader starts after a tap layer, `_enter` sets up the taps
   buffer with the upstream taps already counted, or the assert at F/rounds.py:461-463 fails. Those upstream blocks
   stay zeros (no drafter reads them), so a trace records a round's taps only on a drafting engine.
7. **Where `TFPP_FAULT` starts counting.**
   - The leader counts from the end of the self-check (`ops.armed`).
   - A follower cannot know when the self-check ends, so `run_shortened.sh` uses `follower:on-step:40`.
8. **The round's cand form.**
   - Decided by `plan.cand_form`, whose rule copies F/rounds.py:339-375; the ACT header carries it as `ck`.
   - `StageBuffers` is sized by the widest of every bucket the pool can have, not a fixed 16·2048.
9. **What AGREE carries.** cap, slots, context, drafts. Each stage computes its staging itself from the plan; it is
   not in AGREE.
10. **`check_built`'s 1 GiB floor.** Only stages without `--budget-gib` are checked: when T0c's stages share one GPU,
    the free memory is the whole GPU's.
11. **`mine()`.** Vision tensors belong to the leader (the v2 design). v1 has no vision.
12. **`--experts K` (T-X0).** The DSpark blocks are cut to `min(K, dspark_routed)` experts as well.
13. **90 SEAMS.** Beyond the spec's list: `weights.load`, `Cfg.from_dict`, `fp32_gemms`, `DsApp`, `Scheduler`,
    `Stream`.
14. **The env digest's exclusions.** Adds `TF_DS_SPILL_DIR`, `TF_DS_VISION_EXTRA`, `TFPP_DS41_WORKSPACE_GIB`,
    `TFPP_DS41_ALLOW_TRITON`. A variable whose name holds KEY, TOKEN, SECRET or PASSWORD is always left out.
15. **What T-K compares.**
    - Only the MoE, and wo_a plus wo_b (the same random rows as input).
    - Attention itself is computed head by head, independently, and is not compared.
    - T-K' takes the real hc parameters with `--model`, random ones without.
16. **T1-http.**
    - Sends 4 HTTP requests and runs the fork's `parallel_check.py` (with synthetic prompt ids).
    - No grammar test: the DS engine offers no structured output.
17. **bench.** `ignore_eos=true` by default; adds `rand_len`, `group`, `--bench-max-prompt`.
18. **`start_stage.sh`'s start conditions.**
    - `TFPP_DS41_TF_ROOT` must be set.
    - When a guarded port (`GPU_GUARD_PORTS`) listens or another tfpp process runs, it starts only with both
      `ALLOW_GPU_SHARING=1` and `BUDGET_GIB`.
19. **A follower's extra checks.**
    - STEP round numbers must follow each other.
    - The handlers run inside `torch.no_grad()`, as the fork's `admit` / `round` do.
20. **What `tfpp_selection.json` holds.** Only `orig_layers`, `engram_vocab`, `revision`; no paths, or the two nodes'
    sha would differ.
21. **The download's revision.** The full commit is not known (only `45ee5d0f…`). `download_list.sh` prints a
    script: `REV='<FULL_40_HEX_COMMIT>'` is the one line the operator edits, and the script runs `hf download` only when
    it is 40 hex digits starting with `45ee5d0f`; `--revision` can pass it as well.

## Not tested

All of these need a GPU:

- No CUDA path has run, including:
  - the recompiled anchors running
  - the graph capture of StageModel and StageRoundDecoder (with `_enter` / `_exit` captured into the graphs)
  - N1's reset
  - GatedKeepWarm's interplay with the global capture mode
  - the static views' H2D / D2H copies
  - `index_copy_` writing the replica rows
- The fork itself has never run world=1 full-width kernels or world=1 round graphs on hardware (T-ref is the first).
- The whole chain has not run, including:
  - the handshake and seq numbers
  - the digest mirroring (ADMIT, FILL, FINISH)
  - drain and DROP
  - PipeWatch's fatal
  - the exit with 75 when a follower is killed mid-run
  - QUIT passed all the way up
- T-K' (fused equals unfused at world=1) is not verified. The 4 F1 anchors were only confirmed to occur once and to
  compile; no F1 path has run.
- The two nodes run different drivers (616.41 vs 621.01): whether cuBLAS fp32 GEMMs agree is T-X0's question.
- Admission:
  - `workspace_est`'s default of 4 GiB is an estimate, to be measured with `TF_DS_CHUNK_LOG=1` and written back.
  - What `mem_get_info` means under WSL2 is not confirmed.
  - The round graphs of buckets past 8192 are captured while serving, in memory CAP and READY never saw (see the third
    review round).
- Whether Engram AIO / O_DIRECT is refused on WSL2 ext4. When it is, the log says
  `Engram decode reads stay on the reader pool`.
- HTTP (DsApp over PipeDsEngine): streaming, tool calls, client disconnects.
- The scripts were only checked with `bash -n`, for CRLF, and in snippets; no shellcheck.
- The Python's static checks: py_compile, pytest (37 passed and 2 symlink tests skipped on Windows after the third
  review round), and a home-made AST lint (undefined names, unused imports), which came out clean (the lint ran in the
  first round only; later changes had py_compile and pytest).

## Second review round (2026-10-24)

Each of the 15 review findings was checked against the code and the fork.

### Fixed

1. **The bench prompt set lacked "a last chunk of 2..16 rows whose window reaches below the replay floor".** Added
   `r2050`, `r2064`, `r4098`, `r4112` after `r6000` (`bench_prompts.jsonl`; `test_bench_prompts_file` checks the
   lengths and their order). It is the one case where `TF_DS_REPLAY_FLOOR` chooses between the ring and the gather
   (F/model.py:37-41), and M1's `_tfpp_floor` is verified by it too.
2. **A leader with an extracted embedding could not start.** `needed_shards` maps `embed.weight` to shard 1, and the
   extracted file has another name, so the preflight reported a missing file. Added `load.stage_files` /
   `embed_only_file`: when the folder holds a file the index does not name whose header holds `embed.weight`, and
   shard 1 gives this stage nothing but the embedding, that file stands in for shard 1. `start_stage.sh`'s preflight
   uses `stage_files`. With shard 1 and the extracted file both there, `AliasShards`' same-name check still refuses
   (load.py). `extract`'s usage is in `testmodel.py`'s help and in `node.env.example`.
3. **The token map cache was not written atomically.** It goes to `.<name>.<pid>.tmp` first, then `os.replace`; a
   corrupt file (`ValueError`) is rebuilt from `tokenizer.json`. A valid file of the wrong size still raises (as
   before). (The "truncated but valid" JSON the review described does not exist; what can happen is a JSON parse
   error.)
4. **The leader's readiness probe always failed.** With `TF_API_KEY(_FILE)` set, the fork's server wants the key on
   every route but `/health` (cuda/http.py:60-90). `start_stage.sh`'s `up()` probes `/health`, which is answered only
   after `eng.start()` and the self-check.
5. **T1-fault's kill came after a fixed 60 s.** A background watcher now follows `kill.leader.log` and `pkill`s
   stage 0 once `bench r300#` shows (the bench is under way); when it did not kill (a timeout, or the bench ended
   first) the test fails instead of passing wrongly.
6. **The S11 folder had no `.metadata`, so `VERIFY_SHARDS=1` checked nothing, and the revision depended on how the
   folder was copied.** `testmodel build` copies each shard's `.cache/huggingface/download/<shard>.metadata` into S11;
   added `build --revision FULL_COMMIT` (`make_testmodel.sh`'s variable `REVISION`), which fails when it contradicts the
   .metadata commit, so both nodes write a byte-identical `tfpp_selection.json`. Without .metadata and without
   `--revision` it warns.
7. **Traces carried no settings: a comparison across sessions or machines could not tell a real difference from
   different settings.** The leader's, a follower's and the reference's first `meta` record carries `run`
   (`trace.run_meta`: `settings_vector` with the env digest, `versions`, context / parallel / drafts / graphs). The
   `compare` command compares `run` first and returns 2, naming the keys, when they differ. The keys a test changes on
   purpose are let through by `run_shortened.sh` with `--allow`: T-ref-graphs (`graphs,settings.graphs_env,
   settings.env`), T0b (`graphs`), T1-oracle-tokens (`settings.depth,settings.env`). Old traces (without `run`) only
   warn.
8. **node.env's `TFPP_*` / `TF_DS_*` never reached the stage.** After sourcing `node.env`, `start_stage.sh` exports
   every assignment in it whose name starts with `TFPP_`, `TF_DS_`, `TF_EXL3_` or `TENSORFOLD_`
   (`TFPP_DS41_WORKSPACE_GIB`, `TFPP_FAULT`, `TFPP_DS41_ALLOW_TRITON`, the Engram I/O knobs). `node.env.example` says
   so.
9. **The stages sharing one GPU were not summed.**
   - `capacity.measure` adds `host` to CAP; for a group of 2 or more stages on one machine, `check_caps` checks that
     "weights + need + 512 MiB of CUDA context each" fits in the GPU's total less the reserve, and otherwise refuses,
     listing each stage's numbers and the remedies (fewer stages, a lower `TFPP_DS41_WORKSPACE_GIB` / context /
     parallel).
   - `run_shortened.sh` got `start_stage.sh`'s guard (a guarded port listening, or a `tfpp.(glm|ds41).(serve|run|ref)`
     process running, is refused unless `ALLOW_GPU_SHARING=1`), so ref and T-X0 are covered too.
   - `run_shortened.sh`'s environment clearing keeps `TFPP_DS41_WORKSPACE_GIB`, so a 4-stage T0c can lower the work
     space.
10. **Staging was counted twice.** `measure_budget`'s budget already has the staging taken out, and `need` added it
    again. `check_caps` uses `effective_need`: a measured budget (`declared` false) takes the staging out; a
    `--budget-gib` is outright and still counts it.
11. **`TFPP_DIR` was cleared, and in `start_stage.sh` cluster.env overrode the caller's `CONTEXT` and the like.** Both
    scripts' clearing keeps `TFPP_DIR`. `start_stage.sh` saves the caller's `CONTEXT PARALLEL MTP_DRAFTS HTTP_PORT
    HTTP_HOST MAX_TOKENS THINKING TEMPERATURE SERVED_NAME NO_DRAFTS` before it reads cluster.env, puts them back after
    and logs a line (lesson 4). node.env can still override them, being this machine's explicit setting.
12. **The commands `download_list.sh` printed were a syntax error when pasted.** It prints a script now: a `REV='…'`
    line with a regex check, then `--revision "$REV"` (`test_download_commands` updated).
13. **`download_list.sh` skipped §15.3's disk check.** It now checks that the WSL disk under `--dir` has 75 GB or more
    free, then that `/mnt/c` has 75 GB (the vhdx may have to grow), and otherwise exits 1 printing nothing.
    `--vhdx-ok` when the operator knows the vhdx is already that large; `--no-disk-check` skips both.

### Rejected

- **A follower wakes the Engram disk while idle (review #5).** It affects speed only, and was not measured: the fork's
  comment says about 1 ms for the first read after idling and about 0.3 ms warm (F/model.py:648); at 45–50 ms a round
  that is 1–2% at most. It is a known v1 limitation (no TOUCH frame); v2's TOUCH op (§16.3 item 2) is the real fix (the
  leader sends the tails to the owner before drafting). An untested wait loop in v1 would change `recv_ctl`'s timeout
  semantics: not worth it. Before v2's TOUCH, measure `rows(1)`'s wait with `TF_DS_ROUND_STATS=1` when stage 0 owns
  Engram layers.
- **`prebuild` deletes a live build lock (review #11).** The premise is "FileBaton closes its fd right after it takes
  the lock". torch's `FileBaton.try_acquire` is `self.fd = os.open(path, O_CREAT | O_EXCL)`, and the fd stays open
  until `release()` closes it, which is the behaviour `clear_stale_locks` relies on (the reviewer said this was from
  memory too). This environment has no torch to check the source; before the first prebuild in WSL, confirm it with
  `python -c "import inspect, torch.utils.file_baton as f; print(inspect.getsource(f.FileBaton))"`.

### New untested items

- How often `compare`'s `run` meta check refuses real traces by mistake. If some value of `settings_vector` changes on
  every run (none seen so far), put it in `--allow` or take it out of the vector.
- Four T0c stages on one GPU with the default 4 GiB work space: `check_caps`' sum (about 78 GiB of weights, plus
  4×(4+pool), plus 4×0.5) may just pass the 128 GB node's CUDA capacity less 3 GiB; if it refuses, lower
  `TFPP_DS41_WORKSPACE_GIB` (to the value measured with `TF_DS_CHUNK_LOG=1`).
- `download_list.sh`'s `/mnt/c` check: that `df /mnt/c` in WSL2 reports Windows C:'s free space follows WSL's general
  behaviour and was not confirmed on these nodes.
- `start_stage.sh`'s new parts (restoring the caller's variables, exporting node.env's assignments, the `/health`
  probe) and `run_shortened.sh`'s watcher and guard: `bash -n` and snippet tests only, never a whole run.
- The leader with an extracted embedding (the `stage_files` stand-in path) end to end: `load_stage` reading the
  extracted file, `AliasShards`' same-name check, `VERIFY_SHARDS` reporting "no Hub sha256" for the extracted file.

## Third review round (2026-10-24)

### Fixed

1. **Two stages on one node shared every state file.** `start_stage.sh` kept `.ss`, `.pg`, `preflight.json`, `.sos`,
   `.elf` and one pid file and log per role in one folder, so T1-3's stage 0 and leader (or two systemd units started
   together at boot) could read each other's preflight, and two followers on one node shared `follower.pid` (the
   second start said "already running" and started nothing). Each stage now has its own folder,
   `$TFPP_DS41_STATE/<role>-<A>-<B>` (default root `~/.local/state/tfpp-ds41`), computed after node.env is read;
   `stop_stage.sh` derives the same folder. The shard hash cache stays in the root (keyed by path, inode, size and
   mtime: a torn read only means hashing again).
2. **`run_shortened.sh all` ran every test after T1-exact with `TF_DS_REPLAY=1`.** T1-exact and T1-oracle hard-coded
   the value they put back (`TF_DS_REPLAY=1`, `TF_DS_DEPTH_POLICY=conf`); with `TF_DS_REPLAY=0 run_shortened.sh all`
   the later chains then ran in another mode than the cached `ref.jsonl`, and `compare` refused them (NOT COMPARABLE).
   `set_env` / `restore_env` now put back the caller's value, set or unset.
3. **T1-http failed when the shell held an API key.** The environment clearing never removed `TF_API_KEY` /
   `TF_API_KEY_FILE`, and the readiness probe was `/v1/models`, which a key closes. Both are unset now, and the probe is
   `/health` (as `start_stage.sh`'s).
4. **A missing candidate pool went out as an all-zero mask.** `StageIO.send_chunk` sent zeros ("no block is a
   candidate") when the cut carries the pool but the stage computed none; the stages after the cut then computed with
   an empty pool where the one-process run fails. It raises now (the warm-up's local mode, which sends nothing, keeps a
   placeholder). Not reachable with S11 or the full config (the candidate source has ratio 1).
5. **A `--no-drafts` leader past a tap layer could never match the reference.** Its upstream taps are zeros
   (`decode._seed_taps`), the reference computes them, and the trace hashed them into every `step` record. The trace
   records a round's taps only when the engine has a drafter (nothing else reads them); the reference does the same,
   so both runs' records stay comparable.
6. **HELLO accepted an owner given more tables than it owns.** `proto.check_engram` checked only that an owner's tables
   hold its layers; a stage given the `both` folder while owning L1 alone passed HELLO and failed in `StageEngram` after
   every stage had loaded its weights. Tables of layers the stage does not own are refused at HELLO now.
7. **The preflight refused tables whose rows end before the AIO self-test's row.** The runtime only notes it (the fork
   falls back to the reader pool, the same bytes); `start_stage.sh` prints it as a warning now (`notes`) and keeps
   refusing real mismatches (layers, row counts).
8. **A rebuild of S11 from another source kept the old shard links.** `testmodel.build` created a link only when none
   "existed", and `Path.exists` follows links: a link to the old source stayed (the new config and .metadata then
   described one checkpoint while the shards came from another), and a dangling one made `os.symlink` fail.
   `link_shard` replaces a link that points anywhere else or nowhere; a regular file is never deleted (a note).
9. **`stop_stage.sh` exited 2 when there was nothing to stop.** The log pruning's `ls` fails with no saved log and
   `pipefail` ended the script with its status; it does not now.
10. **The round graphs a long context captures while serving are outside admission.** The fork's warm-up captures the
    buckets up to 8192; a longer conversation captures the larger ones mid-request, in memory CAP and READY never saw.
    Opt-in: `TFPP_DS41_WARM_ALL_BUCKETS=1` (cluster.env, every stage alike: it is digested) makes the build capture
    every bucket of the pool, so READY's free-memory check covers them; every stage logs the buckets left to serving.
    The default is unchanged (see Remaining).
11. **Comments and defaults.** Role names in comments; the GPU guard reads `GPU_GUARD_PORTS` (node.env; default 18891, a tfpp-ds41 leader's HTTP), the leader's
    own HTTP port is always checked. `cluster.env` points to the fork's public recipe and says what `HTTP_HOST=0.0.0.0`
    needs (a key file). These notes are in English.
12. **The T-S file is `tests/test_ds41_static.py`.** Its old name was tfpp-glm's too, so pytest could not collect the
    two suites in one run. New tests cover items 1-3 (the scripts' text), 4 (the module loaded with a torch stand-in),
    5, 6, 8 and 10.

### Remaining

- By default the lazily captured round graphs are still not counted at CAP or READY; a stage that is tight on memory
  can run out at a long context's first round in a new bucket (a follower's failure ends the chain with 75). Measure
  the graph pool's growth on hardware, then either make `TFPP_DS41_WARM_ALL_BUCKETS=1` the default or add the measured
  term to `stage_need`.
- `GPU_GUARD_PORTS` knows no other server by default: a node that also runs other GPU servers must list their
  ports in its node.env to keep the guard.
- Everything in this round is static too: no GPU run, no whole script run (the scripts' new parts had `bash -n` and
  snippet tests with stubs).

# hive-ram documentation

hive-ram splits one large quantized LLM into contiguous **layer ranges** across 2–4 machines linked by a Thunderbolt /
USB4 cable. The [top-level README](../README.md) has the short version: what it is, headline results, the status table
and a quick start per engine ([繁體中文](../README.zh-TW.md)). The pages below go into detail. They are in English.

Paths in these pages are relative to the repository root unless a page says otherwise.

## Where to start

| You want to... | Read |
|---|---|
| Know what works today before you spend a weekend on it | [Verification status](#verification-status) below, then the status table in the [README](../README.md#engines-and-verification-status) |
| Cable two or three machines and get traffic into WSL2 | [hardware-setup.md](hardware-setup.md) |
| Run a model that already works across two nodes, with no hive-ram code | [engines/llamacpp-rpc.md](engines/llamacpp-rpc.md) |
| Run the TensorFold pipeline that is verified on hardware | [engines/tfpp-flashnext.md](engines/tfpp-flashnext.md) |
| Run or test GLM-5.3-Flash across machines | [engines/tfpp-glm.md](engines/tfpp-glm.md) |
| Bring up DeepSeek-V4.1-Flash (never run on a GPU yet) | [engines/tfpp-ds41.md](engines/tfpp-ds41.md) |
| Understand why a pipeline and how a round works | [architecture.md](architecture.md) |
| Fix something that broke | [troubleshooting.md](troubleshooting.md) |

A suggested order for a first try: [hardware-setup.md](hardware-setup.md) (link test passes) →
[engines/llamacpp-rpc.md](engines/llamacpp-rpc.md) or [engines/tfpp-flashnext.md](engines/tfpp-flashnext.md) (a model
answers across the cable) → [engines/tfpp-glm.md](engines/tfpp-glm.md) (the truncated-model tests on one GPU, then
across machines).

## Guides

| Page | What it covers |
|---|---|
| [hardware-setup.md](hardware-setup.md) | What you need, topology and cabling (one subnet per cable), the Windows side of the link (`setup_usb4_link.ps1`, `undo_usb4_link.ps1`), the user-mode relay into WSL2 NAT (`relay_user.ps1`, `relay_hop.ps1`), WSL2 settings, GPU memory on unified-memory laptops, disk planning, expected link numbers (`linktest.py`), and a checklist before every run. |
| [troubleshooting.md](troubleshooting.md) | Symptom → cause → fix tables: link and cabling, Windows networking and firewall, processes that die with a remote session, WSL memory and disk, GPU memory and clocks, updates and reboots, toolchain, shell scripts, llama.cpp on N1X, bit-exact comparisons. Each fix is tagged with how far it has been proven. |

## Engines

| Page | Engine | Status |
|---|---|---|
| [engines/llamacpp-rpc.md](engines/llamacpp-rpc.md) | llama.cpp RPC: build, binaries for the second node, link and relay, `ggml-rpc-server` and `llama-server` flags, results, the PLE-on-SSD finding that puts Flash-Next Q4 on one node, larger models, speculative decoding | **Verified on hardware**: Flash-Next Q4, 2 nodes, 22.2 tok/s |
| [engines/tfpp-flashnext.md](engines/tfpp-flashnext.md) | `tfpp` for Qwen3.8-Flash-Next on TensorFold 0.6.1: setup, two nodes, more ranks, options, reading the output, KeepWarm, results | **Verified on hardware**: 2-machine split token-identical to one node and as fast (KeepWarm); 3-rank chain on 2 machines |
| [engines/tfpp-glm.md](engines/tfpp-glm.md) | `tfpp-glm` for GLM-5.3-Flash on the MIA-patched TensorFold v0.6.0: install, weights and the optional (non-commercial) DFlash2 drafter, layer sizes and cuts, wiring three nodes, `cluster.env` / `node.env`, start / use / stop, the one-GPU and two-machine tests, v2, diagnostics | **v1 verified on hardware on a truncated model** (8 layers bit-exact, 26 layers across machines bit-identical at two cuts, HTTP answered). Full 45-layer model **not run** (does not fit 128 + 64 GB). v2 (`PARALLEL` 2–8) **not run on hardware** |
| [engines/tfpp-ds41.md](engines/tfpp-ds41.md) | `tfpp-ds41` for DeepSeek-V4.1-Flash on the BertholomusAI TensorFold fork: toolchain, the 11-layer test model, the test ladder, two-node blocks | **Not run on hardware**: static review and CPU tests only |

## Design and protocol

| Page | What it covers |
|---|---|
| [architecture.md](architecture.md) | The idea, the engines, what crosses a cut, a round step by step, commit and rollback, why the leader is the last stage, KeepWarm (the idle-GPU downclock), CUDA graphs per stage, chains of N stages, memory planning, speed, tfpp-ds41's design, known limits. |
| [protocol.md](protocol.md) | The three wires (Flash-Next `tfpp-flashnext/2`, tfpp-glm, `tfpp-ds41/1`): transport, frame format, linking and the startup handshake, the parity checks that refuse mismatched stages, runtime and multi-stream ops, failure semantics (fail-stop, exit code 75), tracing and comparison, environment knobs. |

## Evidence

| Page | What it covers |
|---|---|
| [benchmarks.md](benchmarks.md) | Every measured number with its conditions (machines, checkpoint, prompts, flags), kept apart from estimates and projections. |
| [testing.md](testing.md) | The CPU test suites, the one-GPU equivalence tests, the cross-machine tests, what "bit-exact" is compared against, and which tests have run on hardware. |

## Project

| Page | What it covers |
|---|---|
| [roadmap.md](roadmap.md) | What comes next and in which order: a third node for full GLM-5.3-Flash, tfpp-glm v2 on hardware, tfpp-ds41 bring-up, re-running post-hardware fixes, experiments. |
| [../THIRD_PARTY.md](../THIRD_PARTY.md) | How to fetch TensorFold 0.6.1, the MIA package and the BertholomusAI fork at their pins, and check their digests. |
| [../THIRD_PARTY_NOTICES.md](../THIRD_PARTY_NOTICES.md) | Credits and licences of the third-party code tfpp adapts or runs on, and of the models (weights never ship here). |
| [../SECURITY.md](../SECURITY.md) | Threat model: the stage links and HTTP APIs are meant for a private point-to-point cable, not a shared network. |
| [../tfpp/tfpp/glm/IMPLEMENTATION_NOTES.md](../tfpp/tfpp/glm/IMPLEMENTATION_NOTES.md) | tfpp-glm developer notes: status, files, how to run each test, differences from the design spec, review fixes made after the hardware runs, what the hardware runs did not cover. |
| [../tfpp/tfpp/ds41/IMPLEMENTATION_NOTES.md](../tfpp/tfpp/ds41/IMPLEMENTATION_NOTES.md) | tfpp-ds41 developer notes: files, the seams it patches in the fork, how to run, departures from the design spec, what is not tested, review rounds. |

## Verification status

The same labels are used on every page.

| Component | Status |
|---|---|
| llama.cpp RPC, Qwen3.8-Flash-Next Q4 (`UD-Q4_K_XL`) | **VERIFIED ON HARDWARE**: 2 nodes, 22.2 tok/s |
| `tfpp`, Qwen3.8-Flash-Next on TensorFold 0.6.1 | **VERIFIED ON HARDWARE**: the 2-machine split is token-identical to a single node and as fast (KeepWarm); a 3-rank chain ran on 2 machines |
| `tfpp-glm` v1 (`PARALLEL=1`), truncated GLM-5.3-Flash | **VERIFIED ON HARDWARE**: bit-exact against one process on the 8-layer model (pure, eager and graphs, 2 cut points, kept-prompt pressure, fault abort and kill, cross-machine 19/19); 26 layers across machines bit-identical at two cuts; the HTTP path answered |
| `tfpp-glm`, full 45-layer GLM-5.3-Flash | **NOT RUN**: needs about 160 GiB of stages and does NOT fit 128 + 64 GB; needs a third node (planned chain 128 GB → 64 GB → 128 GB) |
| `tfpp-glm` v2 (`PARALLEL` 2–8) | **VERIFIED ON ONE GPU (2026-10-24), 8-layer truncated model:** batched-row invariance (T0e); pure and eager against graphs on three lockstep schedules; 2-stage chains at 2 cut points x 3 schedules bit-identical, every stage applying the leader's messages; replies equal to P=1 serial under free timing at 1, 2 and 4 streams. **Not yet run across machines.** |
| `tfpp-ds41`, DeepSeek-V4.1-Flash | **NOT RUN ON HARDWARE**: static review only, plus its CPU tests |

"Verified on hardware" means two NVIDIA N1X laptops (a 128 GB node and a 64 GB node, Windows 11 + WSL2 in NAT mode)
joined by one USB4 cable. Nothing has been run on other GPUs, on native Linux nodes, or on three physical machines.
Fixes made after the hardware runs have passed CPU tests only; the engine pages say which.

# Third-party notices

hive-ram's own work is licensed under the Apache License 2.0 ([`LICENSE`](LICENSE), [`NOTICE`](NOTICE)). This
repository ships **no third-party source tree, binary or model weight**. tfpp runs on top of engines you install
yourself at pinned versions ([`THIRD_PARTY.md`](THIRD_PARTY.md) says how), and part of tfpp is adapted from their
source. This file lists what comes from where and under which licence.

## Summary

| Component | What hive-ram does with it | Licence | In this repository |
|---|---|---|---|
| [TensorFold](https://github.com/ashhart/TensorFold) 0.6.1 | Engine of the Flash-Next pipeline (`tfpp/tfpp/*.py`); adapted functions | Apache-2.0 (code written before 0.6.0 also keeps an MIT notice) | No (adapted code only) |
| TensorFold v0.6.0 + the [MIA recipe](https://github.com/MiaAI-Lab/GLM-5.3-Flash-EXL3-2x-DGX-Sparks-TensorFold) patches | Engine of tfpp-glm (`tfpp/tfpp/glm/`); adapted functions | Apache-2.0 (TensorFold, MiaAI-Lab), with the credits below | No (adapted code only) |
| [BertholomusAI's TensorFold fork](https://github.com/bertholomus/TensorFold) (`deepseek-v41-tp2`) | Engine of tfpp-ds41 (`tfpp/tfpp/ds41/`); adapted functions | Apache-2.0 | No (adapted code only) |
| [ExLlamaV3](https://github.com/turboderp-org/exllamav3) | Its EXL3 weight format, read by the engines above | MIT | No |
| [llama.cpp / ggml](https://github.com/ggml-org/llama.cpp) | Separate program for the llama.cpp RPC baseline | MIT | No |
| [vLLM](https://github.com/vllm-project/vllm), image `vllm/vllm-openai:minimax-m31-flash` | The MiniMax recipe runs it as a separate program, unmodified, from the image you pull | Apache-2.0 | No |
| [vllm-exl3 and exllamav3 forks](https://github.com/vcruz305/vllm-exl3) (`vcruz305`), inside that image | Run by vLLM; only `tfpp/vllm_plugin/` imports vllm-exl3 | vllm-exl3: **AGPL-3.0**; exllamav3: MIT | No |
| hive-ram's own `tfpp/vllm_plugin/` (MiniMax glue) | Imports vllm-exl3 and calls TensorFold's kernels | **AGPL-3.0-only** (because of that import) | Yes |
| Model weights and drafters | Loaded at run time from folders you download | Each checkpoint's own (see [Models](#models)) | No |

No AGPL source is copied into this repository. One folder, `tfpp/vllm_plugin/`, is licensed AGPL-3.0-only because it
imports `vllm-exl3` (the MiniMax pack author's vLLM plugin, AGPL-3.0, shipped inside their image) at run time; the rest of
hive-ram stays Apache-2.0.


## TensorFold

<https://github.com/ashhart/TensorFold>, Copyright 2026 TensorFold contributors. Apache License 2.0 from 0.6.0;
releases up to 0.5.0 were MIT-licensed, and code written before 0.6.0 keeps its MIT notice (text at the end of this
file). TensorFold's own `THIRD_PARTY_NOTICES.md` credits what it builds on (MLX and mlx-lm, Hugging Face
transformers, mlx-vlm, ExLlamaV3, z-lab DFlash, flash-linear-attention, DeepSeek's encoding code, vLLM's stream
convention, MiaAI-Lab's Flash Next vision patch, PyTorch and Triton); it travels with every TensorFold tree you
install.

**How tfpp uses it.** tfpp imports TensorFold at run time from the tree you installed and never modifies that tree.
Where it needs a different behaviour it either subclasses TensorFold classes or reads the source of one installed
module or function, replaces a few anchored lines (each must occur exactly once, or the stage refuses to start),
compiles that copy and runs it in the original module's namespace. Every tensor is still built by the upstream code.
tfpp-glm and tfpp-ds41 also check the tree digest and the source of every function they rely on before they start
(`python -m tfpp.glm.pins`, `python -m tfpp.ds41.pins`).

**Files adapted from TensorFold source** (modified by the hive-ram contributors; Apache-2.0 section 4(b)):

| Variant | Files | What they adapt |
|---|---|---|
| Flash Next on TensorFold 0.6.1 | `tfpp/tfpp/load.py`, `part.py`, `driver.py` | The Flash Next weight loader with its layer list replaced by a range; the CUDA `compute` cut at a layer boundary; the decode, MTP and commit steps split between ranks |
| tfpp-glm on TensorFold v0.6.0 + MIA patches | `tfpp/tfpp/glm/part.py`, `tail.py`, `v2tail.py`, `follower.py`, `v2follower.py`, `capacity.py`, `v2capacity.py`, `load.py`, `env.py`, `v2part.py` | `forward._compute` cut at a layer range; a pipeline engine that follows `GlmEngine.__init__` step by step; the package's capacity plan and `mla_geometry` restricted to one stage; the weight loader with five anchors replaced; the multi-stream decoder and batched verify subclassed |
| tfpp-ds41 on the BertholomusAI fork | `tfpp/tfpp/ds41/part.py`, `decode.py`, `multi.py`, `engine.py`, `load.py`, `patch.py`, `plan.py`, `capacity.py`, `env.py`, `ref.py`, `trace.py` | The fork's `Model`, `RoundDecoder` / `RoundRunner`, `MultiDecoder` and `DsEngine` on a layer range; a few lines of the fork's `weights._load` copied into `load.py`; functions recompiled with anchors by `patch.py` |

Everything else (the wire framing and link handshake, the startup and HELLO chain checks, the pins, the comparison,
trace and benchmark tools, KeepWarm, the Windows link scripts, the shell scripts and every test) is original work of
the hive-ram contributors.

## The MIA recipe patches (GLM-5.3-Flash)

<https://github.com/MiaAI-Lab/GLM-5.3-Flash-EXL3-2x-DGX-Sparks-TensorFold>, commit
`b5808ac620c795ec9c0640f3ca2a2f7be21aaa63` (v1.11-pre), Copyright 2026 MiaAI-Lab, Apache License 2.0. Its 107 patch
files modify TensorFold v0.6.0; the TensorFold code they modify stays under TensorFold's licences.

tfpp-glm adapts functions of the patched package, which can contain the third-party work that the recipe's `NOTICE`
and `CREDITS.md` credit. As listed there:

- **glm53-tensorfold-spark**, <https://github.com/jayleaton/glm53-tensorfold-spark>, Copyright 2026 Jay Leaton,
  Apache-2.0: parts of the tool-call handling and kept reasoning (patch 0036), the L2 prefetch (0046) and the EXL3
  decode loads (0047) are adapted from it; patch 0058 is that author's TensorFold pull request #218.
- The NVMe spill tier for kept prompt states (patch 0088), authored by Robert Wojciechowski
  (<https://github.com/wojo>).
- **b12x**, <https://github.com/local-inference-lab/b12x>, Apache-2.0: the RoCE transport (patch 0006). tfpp-glm does
  not use it (a pipeline has no communicator).
- Patches contributed through the recipe's issues and pull requests, and upstream TensorFold commits backported by
  it (for example 0059, TensorFold #181).

Keep TensorFold's `LICENSE`, `NOTICE` and `LICENSES/`, and the recipe's `NOTICE`, with any copy of a patched tree you
make. hive-ram does not redistribute one.

## The deepseek_v41 TensorFold fork (DeepSeek-V4.1-Flash)

<https://github.com/bertholomus/TensorFold>, branch `deepseek-v41-tp2`, commit
`af26d42b1f618e9e653903a6c043448b52729c64`: a fork of TensorFold 0.6.6 by BertholomusAI (Albert Lee,
<https://github.com/bertholomus>), Copyright 2026, Apache License 2.0. Its `NOTICE` lists the upstream files it
changes; its `tools/dsv41/ATTRIBUTION.md` lists every outside source. In short:

- The model math is re-implemented from DeepSeek's inference code and tech report for DeepSeek-V4.1-Flash
  (MIT License, Copyright (c) 2023 DeepSeek), without including that source.
- vLLM's `deepseek_v41` code (Apache-2.0) was read only as a cross-check; ExLlamaV3 (MIT) is the EXL3 format
  reference.
- A vLLM kit licensed AGPL-3.0 was only run as a black box over HTTP to measure baseline numbers; none of its code was
  read or included. Nothing of it is in the fork or in hive-ram.

## MiniMax recipe: vLLM, the pack and its drafter

- **vLLM**, <https://github.com/vllm-project/vllm>, Apache-2.0. The recipe runs the `vllm/vllm-openai:minimax-m31-flash` image
  (digest in [`THIRD_PARTY.md`](THIRD_PARTY.md)) without modifying it.
- **The pack and its plugin**, by `vcruz305`: `MiniMax-M3.1-Flash-EXL3-SAGE-2.15bpw` (the weights) and `vllm-exl3` (the vLLM
  plugin that reads them, AGPL-3.0). hive-ram copies neither. Its glue plugin imports `vllm_exl3` and is therefore AGPL-3.0-only
  ([`tfpp/vllm_plugin`](tfpp/vllm_plugin), [what it does](docs/engines/minimax-vllm.md#the-glue-plugin)).
- **TensorFold v0.6.6**, <https://github.com/ashhart/TensorFold>, Apache-2.0. `tfpp/scripts/minimax/prepare_tensorfold.sh` copies
  twelve of its `cuda/exl3` files out of your own checkout at the pinned tag and applies
  `tfpp/scripts/minimax/patches/0001-exl3-experts-swiglu-oai.patch` (50 lines, a modification of `experts.cu` and
  `experts.py`, Apache-2.0 section 4(b)); the copy is built on your machine and is not part of this repository.
- **The DFlash2 drafter**, `incoai/MiniMax-M3.1-Flash-DFlash2` by Inco AI, Apache-2.0 as stated on its model card. Not included.
- **tonyd2wild's benchmark harness**, <https://github.com/tonyd2wild/MiniMax-M3.1-Flash-EXL3-TP2-2x-DGX-Spark>, MIT. It produced the
  tables and is not copied here. The recipe's settings were compared with that repository's and with the pack author's.

## ExLlamaV3

<https://github.com/turboderp-org/exllamav3>, MIT License, Copyright (c) 2025 Turboderp. The GLM-5.3-Flash and
DeepSeek-V4.1-Flash checkpoints store their routed experts in its EXL3 format; TensorFold's decoders and kernels read
it. hive-ram contains none of its code.

## llama.cpp / ggml

<https://github.com/ggml-org/llama.cpp>, MIT License, Copyright (c) the ggml authors. The llama.cpp baseline in the
documentation runs `llama-server` with the RPC backend as separate programs built from llama.cpp's own source.
hive-ram contains none of its code and does not link to it.

## Test data

The CPU tests read three small JSON files of model hyper-parameters. They contain no weights.

| File | What it is |
|---|---|
| `tfpp/tfpp/glm/tests/data/glm53_config.json` | The `config.json` of `Mia-AiLab/GLM-5.3-Flash-EXL3-4bpw-TensorFold`, revision `76c0b517` |
| `tfpp/tfpp/ds41/tests/data/ds41_config.json` | The `config.json` of `Mia-AiLab/DeepSeek-V4.1-Flash-1017-EXL3-EAQ-v1-2.9-bpw` (see that folder's README) |
| `tfpp/tfpp/glm/tests/data/dflash2_config.json` | A stand-in written for the tests: only the four values one function reads, set to those of `incoai/GLM-5.3-Flash-DFlash2` (revision `bf582e4e`). The drafter's own files are not copied (CC BY-NC-ND 4.0) |

## Models

No model file is part of this repository, and nothing downloads automatically: the scripts set `HF_HUB_OFFLINE=1`
and load only folders you downloaded. **Each checkpoint keeps its own licence. Read the model card before you
download, and again before any use that is not personal.** What the sources we have state:

| Checkpoint | Used by | Licence as stated, and where |
|---|---|---|
| `vcruz305/MiniMax-M3.1-Flash-EXL3-SAGE-2.15bpw` (revision `99bf3a57`) | MiniMax recipe weights | Not stated in any source we have for the pack itself; the base model is MiniMax's. Check both model cards |
| `incoai/MiniMax-M3.1-Flash-DFlash2` | MiniMax recipe drafter, optional | Apache-2.0 as stated on the model card (verify it before use) |
| `incoai/GLM-5.3-Flash-DFlash2` (revision `bf582e4e`) | tfpp-glm drafter, **optional** | **CC BY-NC-ND 4.0: non-commercial use only, no derivatives** (TensorFold's third-party notices; the MIA recipe's NOTICE). Do not redistribute it or any converted or quantized copy. Without it: `NO_DRAFTS=1`, or MTP-only drafting at `PARALLEL=1` (MIA reports it as about 5-10% slower; not run on hardware by us) |
| `Mia-AiLab/GLM-5.3-Flash-EXL3-4bpw-TensorFold` | tfpp-glm weights | The MIA recipe's NOTICE states Apache-2.0, derived from `zai-org/GLM-5.3-Flash` (MIT per the same NOTICE). Check both model cards |
| `deepseek-ai/DeepSeek-V4.1-Flash` | Origin of the DeepSeek weights | MIT, per the fork's ATTRIBUTION.md |
| `Mia-AiLab/DeepSeek-V4.1-Flash-1017-EXL3-EAQ-v1-2.9-bpw` | tfpp-ds41 weights | Not stated in any source we have. The fork's ATTRIBUTION.md lists a related quantization (`Mia-AiLab/DeepSeek-V4.1-Flash-EXL3-2.9bpw`) as MIT, inherited from the base model. Check the model card |
| `Vontra/Qwen3.8-Flash-Next-MLX-4bit-MTP` | Flash-Next tfpp | Not stated in any source we have. Check the model card |
| unsloth Qwen3.8-Flash-Next GGUF (`UD-Q4_K_XL`) | llama.cpp baseline | Not stated in any source we have. Check the model card |

Commercial use is not a supported scenario of this project; the drafter above forbids it outright.

## Trademarks

NVIDIA, N1X, DGX Spark and GB10 (NVIDIA), Thunderbolt (Intel), USB4 (USB-IF), DeepSeek, Qwen and
GLM are trademarks of their owners. hive-ram uses them only to describe the hardware and models it runs on, and is not
affiliated with or endorsed by them or by the authors of the projects above.

## License texts

The Apache License 2.0 text is in [`LICENSE`](LICENSE); TensorFold, the MIA recipe and the fork use the same text.

### TensorFold: MIT notice for code written before 0.6.0

```
MIT License

Copyright (c) 2026 TensorFold contributors

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
```

### MiaAI-Lab: Flash Next vision patch 0008 (MIT), included in TensorFold 0.6.1

TensorFold 0.6.1's Flash Next CUDA image integration is adapted from this patch; the Flash-Next files of tfpp adapt
TensorFold 0.6.1 functions that may contain it.

```
MIT License

Copyright (c) 2026 MiaAI-Lab

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
```

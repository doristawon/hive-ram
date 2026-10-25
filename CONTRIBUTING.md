# Contributing to hive-ram

Thanks for helping. hive-ram is small, research-grade software that splits one model by layer ranges across a few
machines. Every claim in it is backed by a measurement or a test, and every component says how far it has been
verified. Contributions that keep it that way are very welcome: hardware reports, fixes, tests and documentation.

## Verification status comes first

Each component is labelled with exactly what was run, and nothing more:

| Component | Status |
|---|---|
| Flash-Next tfpp (TensorFold 0.6.1) | **Verified on hardware**: a 2-machine split is token-identical to one node and as fast (with KeepWarm); a 3-rank chain on 2 machines |
| llama.cpp RPC baseline | **Verified on hardware**: Flash-Next Q4 on 2 nodes, 22.2 tok/s |
| tfpp-glm v1 (GLM-5.3-Flash, MIA package) | **Verified on hardware** on shortened models: bit-exact against one process on the 8-layer model (pure, eager and CUDA graphs, 2 cut points, kept-prompt pressure, fault abort/kill, cross-machine 19/19); 26 layers across machines with two cuts bit-identical; the HTTP path answered. The full 45-layer model (~160 GiB) does **not** fit 128 + 64 GB and has not run: it needs a third node |
| tfpp-glm v2 (`PARALLEL` 2-8 streams) | **Not run on hardware.** CPU tests only |
| tfpp-ds41 (DeepSeek-V4.1-Flash, fork) | **Not run on hardware.** Written and statically reviewed; CPU tests only |

If your change touches documentation, keep these labels. If you run something on hardware, tell us (see
[Reporting hardware results](#reporting-hardware-results)) and we will move the label with your evidence.

## Development setup

You do not need a GPU, torch or TensorFold for the CPU tests.

```bash
git clone https://github.com/doristawon/hive-ram.git && cd hive-ram
python3 -m venv .venv && . .venv/bin/activate      # Python 3.11 or newer (CI: 3.12; the hardware runs: 3.14)
pip install -e ".[test]"                            # the tfpp package (no dependencies) + pytest + numpy

cd tfpp
python -m pytest -q -p no:cacheprovider tfpp/glm/tests tfpp/ds41/tests    # tfpp-glm and tfpp-ds41
python -m pytest -q -p no:cacheprovider tests scripts/tests               # Flash-Next pipeline, link tools, scripts
```

Some tests read a third-party engine tree to check the seams tfpp patches. They skip when the tree is absent, which
is the normal case (CI never has one). To run them, fetch the trees as [`THIRD_PARTY.md`](THIRD_PARTY.md) describes
and either put them under `vendor/` next to `tfpp/` (`vendor/tensorfold-mia-v111pre/tensorfold`,
`vendor/tensorfold-dsv41-tp2/src/tensorfold`; `vendor/` is git-ignored) or point the tests at them:

```bash
TFPP_GLM_VENDOR=$HOME/tf-mia/tensorfold TFPP_DS41_VENDOR=$HOME/tf-dsv41/src/tensorfold \
  python -m pytest -q -p no:cacheprovider -rs tfpp/glm/tests tfpp/ds41/tests
```

Some link-tool tests need Windows PowerShell or bash and skip elsewhere.

## Rules for code changes

- **Do not change what a stage computes without saying so.** tfpp-glm and tfpp-ds41 promise that an N-stage chain
  equals one process bit for bit, and Flash-Next tfpp that it is token-identical to one node. A change that can move
  a single bit of a forward pass (kernels, dtypes, the order of reductions, what crosses the link) needs a hardware
  run of the variant's equivalence tests (`tfpp/scripts/glm/run_truncated.sh`, `tfpp/scripts/ds41/run_shortened.sh`,
  `tfpp/compare.py`) before it can be called verified.
- **One engine tree per variant.** Each variant patches named functions of one exact TensorFold tree, pinned by
  digest (`python -m tfpp.glm.pins`, `python -m tfpp.ds41.pins`). Moving to another tree means running
  `python -m tfpp.glm.pins --diff <tree>` (or the ds41 one), re-reading every seam it lists, and updating the pins in
  the same change.
- **Keep the CPU/GPU split.** Importing `tfpp.glm` or `tfpp.ds41` imports neither torch nor TensorFold, and the
  CPU-side modules (`proto`, `pins`, `capacity`, `env`, `compare`, `startup` and the like) must stay importable
  without them, so the CPU tests and the pin tools run anywhere. Only the modules that move or compute tensors
  (`part`, `tail`, `follower`, `wire` and their v2 / ds41 counterparts) import torch or TensorFold at the top.
- **No new runtime dependencies** in `pyproject.toml`. The engines need specific torch and Triton builds that the
  toolchain scripts install; pip must not choose them.
- **Tests with the change.** New behaviour comes with a CPU test where one is possible. Keep `pytest` green.
- **Style.** English comments and messages. LF line ends everywhere (`.gitattributes` enforces it). PowerShell
  scripts stay ASCII without a BOM (Windows PowerShell 5.1 reads them as ANSI); the tests check this.
- **Nothing personal in the tree.** No absolute paths of your machines (use `$HOME`, relative paths, or an
  environment variable with a default), no host names, user names, VPN or Tailscale addresses, tokens or keys. The
  documented link addresses `10.77.0.x` / `10.78.0.x` are the defaults to use in examples.

## Third-party code and models

- Never commit a third-party tree (`vendor/`), a wheel, a built extension or any model file.
- Code adapted from TensorFold, the MIA recipe patches or the BertholomusAI fork is a modification of
  Apache-2.0 code: keep the upstream copyright, say in the file that it was modified, and add the file to the
  table in [`THIRD_PARTY_NOTICES.md`](THIRD_PARTY_NOTICES.md).
- **AGPL.** No AGPL source may be copied into this repository, and nothing outside `tfpp/vllm_plugin/` may import an AGPL
  package or depend on one. That folder holds the MiniMax glue plugin, which imports the pack author's AGPL-3.0
  `vllm-exl3`; it is licensed AGPL-3.0-only and has its own `LICENSE`. Write it against vllm-exl3's interface and do not
  paste from its source. The AGPL-3.0 vLLM kit the DeepSeek fork's attribution mentions stays off limits: do not add,
  import or copy it, and keep both AGPL packages out of every `tfpp` venv.
- Models keep their own licences. The optional DFlash2 drafter for GLM-5.3-Flash (`incoai/GLM-5.3-Flash-DFlash2`) is
  CC BY-NC-ND 4.0: non-commercial use only and no derivatives. Do not add it, a converted or quantized copy, or its
  files to the repository, and keep every path that uses it optional (`NO_DRAFTS=1` must keep working).
- Test data may contain public model hyper-parameters (`config.json` values), never weights. Name the source
  repository and revision in the folder's `README.md`.

## Reporting hardware results

Open an issue with:

- the variant and the commit of hive-ram you ran;
- the engine tree digest (`python -m tfpp.glm.pins --check`, `python -m tfpp.ds41.pins --check`, or for Flash-Next
  `python -c "from tfpp.glm.pins import tree_sha, installed_root; print(tree_sha(installed_root()))"` in the
  Flash-Next venv);
- the machines (GPU, memory, OS / WSL version), the link and its measured throughput and round trip
  (`python3 linktest.py server` on one node, `python3 linktest.py client <peer address>` on the other);
- the model checkpoint and revision, the layer ranges, and the exact command lines with `node.env` (remove secrets
  such as `TF_API_KEY_FILE` paths you consider private, and every token; replace host names, user names and non-link
  IP addresses in logs and traces: stages log each node's hostname at HELLO);
- what you compared against (one process / one node) and the result: the comparison tool's output, tokens per
  second, and the conditions (prompt length, context size, eager or CUDA graphs, drafting on or off).

Negative results are just as useful: a run that does not fit, a mismatch, or a crash with its log.

## Pull requests

1. Fork, branch from `main`, keep the change focused.
2. Run the CPU tests above. CI runs the same on Python 3.12 without a GPU.
3. Describe what changed, why, and how it was verified (CPU tests only, or which hardware run).
4. Update the documentation, including the verification labels, when behaviour or status changes.

Security problems: please do not open a public issue, see [`SECURITY.md`](SECURITY.md).

## Licence of contributions

hive-ram is licensed under the Apache License 2.0. Unless you state otherwise, any contribution you submit is
licensed under the same terms (section 5 of the licence), without additional terms or conditions.

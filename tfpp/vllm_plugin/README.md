# n1x-tfmoe: vLLM plugins for MiniMax-M3.1-Flash EXL3 on WSL2 N1X

Three small vLLM general plugins used by the two-node MiniMax recipe ([docs/engines/minimax-vllm.md](../../docs/engines/minimax-vllm.md)).

| Entry point | Switch | What it does |
|---|---|---|
| `register` | `TFMOE=1` | Routes decode-sized MoE calls (up to `TFMOE_MAX_ROWS`, default 16 rows) through TensorFold's grouped EXL3 expert kernels, reading vllm-exl3's trellis tensors in place. |
| `register_linear` | `TFLIN=1` | Routes vllm-exl3's dense EXL3 linears (1–128 rows) through TensorFold's `linear.cu`. `TFLIN_SPLITS=<json>` picks a K-split per shape. |
| `register_fixes` | always | (1) vllm-exl3 41fdb7d takes the checkpoint expert offset from the **global** rank, which breaks pipeline stage 1; (2) vLLM replaces the GPU's free memory by `psutil` available RAM on integrated GPUs, which a WSL `memory=` cap makes wrong; `N1X_UMA_FREE=psutil` keeps the stock rule. |

Install into the Python that runs vLLM (inside the unpacked image):

```bash
bash tfpp/scripts/minimax/prepare_tensorfold.sh /path/to/TensorFold-v0.6.6 /opt/tfx   # kernels + one patch
pip install --no-deps --no-build-isolation tfpp/vllm_plugin
export TFMOE_ROOT=/opt/tfx TFMOE=1 TFLIN=1
```

`TFMOE_CHECK=N` / `TFLIN_CHECK=N` compare the first N eager calls with the original path and log the cosine similarity.

## Licence

**AGPL-3.0-only** (see [`LICENSE`](LICENSE)), unlike the rest of hive-ram, which is Apache-2.0. These plugins `import vllm_exl3`,
the pack author's vLLM plugin, which is licensed AGPL-3.0, so they are licensed the same way. They contain none of its code and no
code of TensorFold; the kernels come from your own TensorFold checkout through `prepare_tensorfold.sh`. This folder is the one
place in the repository where code that depends on an AGPL package may live ([`CONTRIBUTING.md`](../../CONTRIBUTING.md#third-party-code-and-models)).
If you offer a modified version of the plugin over a network, AGPL section 13 applies to you.

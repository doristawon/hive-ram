# Test data

| File | What it is |
|---|---|
| `ds41_config.json` | The `config.json` of the DeepSeek-V4.1-Flash EXL3 checkpoint tfpp-ds41 loads (Hugging Face `Mia-AiLab/DeepSeek-V4.1-Flash-1017-EXL3-EAQ-v1-2.9-bpw`), copied unchanged. It holds the layer layout, sizes, Engram and quantization settings. The tests read its geometry only; no weights are involved. The revision it was taken from was not recorded; the download script (`scripts/ds41/download_list.sh`) expects a commit starting with `45ee5d0f`. The checkpoint's model card gives the license. |

# Test data

| File | What it is |
|---|---|
| `glm53_config.json` | The `config.json` of the GLM-5.3-Flash EXL3 4-bit checkpoint the stages load (Hugging Face `Mia-AiLab/GLM-5.3-Flash-EXL3-4bpw-TensorFold`, revision `76c0b517`). It holds the layer types, sizes and quantization settings. The tests read its geometry only; no weights are involved. The checkpoint's model card gives the license. |
| `dflash2_config.json` | A stand-in, not the drafter's config: only the four fields `v2capacity.draft_window_extra` reads, set to the values of `incoai/GLM-5.3-Flash-DFlash2` (revision `bf582e4e`). That drafter is licensed CC BY-NC-ND 4.0, so its files are not copied here; it is optional at run time (`--no-drafts`). |
| `v1_frozen/` | tfpp-glm v1 as it ran on hardware: `MANIFEST.json` (sha256 of every v1 file, the HELLO digest of that run, and the `post_v1` pins of the files fixed since, each with what changed), `WHITELIST.json` (the ten statements v2 inserted into `serve.py` / `follower.py`), and the frozen `serve.py`, `follower.py`, `start_stage.sh` the v2 tests compare against. The frozen files carry the same post-v1 fixes and wording changes as the live ones; their hardware hashes stay in the manifest. |

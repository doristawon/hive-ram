#!/bin/bash
# inside the unpacked vLLM MiniMax rootfs (see docs/engines/minimax-vllm.md): serve the MiniMax SAGE 2.15 pack (or a truncated view) with the
# recipe's measured q4-32k settings (q4 main KV, fp8 indexer KV + MXFP8 indexer, UMA load hygiene, full decode graphs)
set -euo pipefail
MODEL=${MODEL:-/n1x/models/minimax-m31-trunc50}
PORT=${PORT:-18100}
MAXLEN=${MAXLEN:-8192}
KVB=${KVB:-400000000}
export VLLM_EXL3_Q4KV=${VLLM_EXL3_Q4KV:-1} VLLM_EXL3_INDEXER_MXFP8=${VLLM_EXL3_INDEXER_MXFP8:-1}
# UMA/ATS=0 keeps the embedding in CUDA memory: WSL2 has no ATS, so a GPU read of an mmap alias is a slow host-mapped read
if [ "${UMA_ATS:-1}" = 1 ]; then
  export VLLM_EXL3_UMA_POSTLOAD=1 VLLM_EXL3_ATS_ALIAS=1 VLLM_EXL3_ATS_ALIAS_EMBED=1
fi
export VLLM_EXL3_TRIM_AFTER_LAYER=1
export VLLM_EXL3_DRAFT_VOCAB_META=1 VLLM_EXL3_TRIM_BEFORE_KV=1 VLLM_EXL3_MOE_TEMP_ROWS=512 VLLM_EXL3_PERIODIC_TRIM_S=30
export MALLOC_ARENA_MAX=2 MIMALLOC_PURGE_DELAY=0 MIMALLOC_RESET_DELAY=0
export VLLM_TRITON_JIT_WARMUP_NUM_THREADS=1 TORCHINDUCTOR_COMPILE_THREADS=1
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export TRITON_CACHE_DIR=/n1x/m31-jit/triton CUDA_CACHE_PATH=/n1x/m31-jit/nv-compute CUDA_CACHE_MAXSIZE=4294967296
mkdir -p /n1x/m31-jit
# CUDA graphs and GPU-memory sharing open many descriptors; the WSL GPU driver logs get_unused_fd_flags failed (EMFILE) otherwise
ulimit -n 1048576 2>/dev/null || ulimit -n "$(ulimit -Hn)" 2>/dev/null || true
SPEC=()
# DSPARK_ADAPT=1: per-request draft budget from DSpark's confidence head; DSPARK_TOPK=k: Markov head on top-k only
if [ "${DSPARK_ADAPT:-0}" = 1 ]; then DSPARK_EXTRA="${DSPARK_EXTRA:+$DSPARK_EXTRA,}\"enable_adaptive_verification\":true"; fi
if [ -n "${DSPARK_TOPK:-}" ]; then DSPARK_EXTRA="${DSPARK_EXTRA:+$DSPARK_EXTRA,}\"dspark_draft_topk\":$DSPARK_TOPK"; fi
if [ -n "${DFLASH2:-}" ]; then
  # DFlash2 block-diffusion drafter (Inco AI, Apache-2.0): method dflash + the DFlash2DraftModel architecture picks DFlash2Speculator
  SPEC=(--speculative-config "{\"method\":\"dflash\",\"model\":\"$DFLASH2\",\"num_speculative_tokens\":${DFLASH2_K:-3},\"attention_backend\":\"TRITON_ATTN\",\"kv_cache_dtype\":\"fp8\"${DFLASH2_EXTRA:+,$DFLASH2_EXTRA}}")
  CAP=${DFLASH2_CAP:-$(seq -s, 1 $(( ${DFLASH2_K:-3} + 1 )))}
elif [ "${NGRAM:-0}" = 1 ]; then
  # prompt-lookup drafting: no draft model, and a step with no match verifies only the sampled token
  SPEC=(--speculative-config "{\"method\":\"ngram\",\"num_speculative_tokens\":${NGRAM_K:-4},\"prompt_lookup_max\":${NGRAM_MAX:-5},\"prompt_lookup_min\":${NGRAM_MIN:-2}}")
  CAP=$(seq -s, 1 $(( ${NGRAM_K:-4} + 1 )))
elif [ "${DSPARK:-0}" = 1 ]; then
  SPEC=(--speculative-config '{"method":"dspark","num_speculative_tokens":'"${DSPARK_K:-3}"',"attention_backend":"TRITON_ATTN","kv_cache_dtype":"fp8","draft_sample_method":"probabilistic","rejection_sample_method":"block"'"${DSPARK_EXTRA:+,$DSPARK_EXTRA}"'}')
  CAP=$(seq -s, 1 $(( ${DSPARK_K:-3} + 1 )))
else
  CAP=${CAP:-1}
fi
# two-node pipeline (inside the n1x_tun namespace): PP=2 NNODES=2 NODE_RANK=0|1 MASTER=10.99.0.2 PP_SPLIT=45,15
DIST=()
if [ "${NNODES:-1}" -gt 1 ]; then
  export NCCL_SOCKET_IFNAME=${N1X_IF:-n1x0} GLOO_SOCKET_IFNAME=${N1X_IF:-n1x0} NCCL_IB_DISABLE=1 VLLM_HOST_IP=$HOST_IP
  if [ -n "${PP_SPLIT:-}" ]; then export VLLM_PP_LAYER_PARTITION=$PP_SPLIT; fi
  DIST=(--nnodes "$NNODES" --node-rank "$NODE_RANK" --master-addr "$MASTER" --master-port "${MPORT:-29511}")
  if [ "$NODE_RANK" != 0 ]; then DIST+=(--headless); fi
fi
PROF=()
if [ -n "${PROF_DIR:-}" ]; then
  mkdir -p "$PROF_DIR"
  PROF=(--profiler-config "{\"profiler\":\"torch\",\"torch_profiler_dir\":\"$PROF_DIR\",\"torch_profiler_with_stack\":false,\"torch_profiler_use_gzip\":false}")
fi
exec vllm serve "$MODEL" --served-model-name minimax-m31 --host 0.0.0.0 --port "$PORT" \
  --tensor-parallel-size 1 --pipeline-parallel-size "${PP:-1}" "${DIST[@]}" --language-model-only --block-size 128 --disable-hybrid-kv-cache-manager \
  --max-model-len "$MAXLEN" --max-num-seqs "${SEQS:-1}" --max-num-batched-tokens 1024 \
  --gpu-memory-utilization "${GMU:-0.92}" --kv-cache-memory-bytes "$KVB" --kv-cache-dtype nvfp4 --attention-config '{"indexer_kv_dtype":"fp8"}' \
  "${SPEC[@]}" "${PROF[@]}" \
  --compilation-config "{\"cudagraph_mode\":\"${CG:-FULL_DECODE_ONLY}\",\"cudagraph_capture_sizes\":[$CAP],\"mode\":0}" \
  --max-cudagraph-capture-size "${CAP##*,}" \
  --kernel-config '{"enable_flashinfer_autotune":false}' \
  --reasoning-parser minimax_m3 --tool-call-parser minimax_m3 --enable-auto-tool-choice

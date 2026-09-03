#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$REPO_ROOT"

PYTHON_BIN="${PYTHON_BIN:-python}"
CHECKPOINT_DIR="${CHECKPOINT_DIR:-Graph-COM/Parallel-Synthesis-qwen3-14B}"
MODEL_NAME="${MODEL_NAME:-Qwen/Qwen3-14B}"
TASKS="${TASKS:-aime2024,gpqa,humanevalplus}"
SPLIT="${SPLIT:-test}"
MAX_SAMPLES="${MAX_SAMPLES:--1}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-4096}"
OUTPUT_DIR="${OUTPUT_DIR:-results/single_turn_direct}"

"$PYTHON_BIN" -m parallel_synthesis.cli.post_eval_parallel_kv \
  --checkpoint_dir "$CHECKPOINT_DIR" \
  --method parallel_kv \
  --model_name "$MODEL_NAME" \
  --tasks "$TASKS" \
  --split "$SPLIT" \
  --eval_samples_per_task "$MAX_SAMPLES" \
  --generate_bs 1 \
  --max_new_tokens "$MAX_NEW_TOKENS" \
  --temperature 0 \
  --top_p 1 \
  --seed 42 \
  --output_dir "$OUTPUT_DIR" \
  "$@"

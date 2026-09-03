#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$REPO_ROOT"

PYTHON_BIN="${PYTHON_BIN:-python}"
CHECKPOINT_DIR="${CHECKPOINT_DIR:-Graph-COM/Parallel-Synthesis-qwen3-14B}"
MODEL_NAME="${MODEL_NAME:-Qwen/Qwen3-14B}"
GAIA_CONFIG="${GAIA_CONFIG:-2023_level1}"
SPLIT="${SPLIT:-validation}"
MAX_SAMPLES="${MAX_SAMPLES:-10}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-10000}"
MAX_TOOL_STEPS="${MAX_TOOL_STEPS:-5}"
OUTPUT_DIR="${OUTPUT_DIR:-results/gaia_direct}"
SAFETY_ARGS=()
if [[ "${ENABLE_PYTHON_INTERPRETER:-0}" != "1" ]]; then
  SAFETY_ARGS+=(--disable_python_interpreter)
fi

"$PYTHON_BIN" -m parallel_synthesis.cli.run_gaia_eval \
  --method parallel_kv \
  --model_name "$MODEL_NAME" \
  --parallel_kv_load_dir "$CHECKPOINT_DIR" \
  --gaia_config "$GAIA_CONFIG" \
  --split "$SPLIT" \
  --max_samples "$MAX_SAMPLES" \
  --max_new_tokens "$MAX_NEW_TOKENS" \
  --max_tool_steps "$MAX_TOOL_STEPS" \
  --temperature 0 \
  --top_p 1 \
  --seed 42 \
  --attn_implementation sdpa \
  --output_dir "$OUTPUT_DIR" \
  "${SAFETY_ARGS[@]}" \
  "$@"

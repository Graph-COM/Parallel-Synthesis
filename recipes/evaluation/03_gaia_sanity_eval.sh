#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$REPO_ROOT"

: "${SOURCE_RUN_DIR:?Set SOURCE_RUN_DIR to a previous GAIA run.}"

PYTHON_BIN="${PYTHON_BIN:-python}"
CHECKPOINT_DIR="${CHECKPOINT_DIR:-Graph-COM/Parallel-Synthesis-qwen3-14B}"
MODEL_NAME="${MODEL_NAME:-Qwen/Qwen3-14B}"
MAX_SAMPLES="${MAX_SAMPLES:-10}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-4096}"
MAX_TOOL_STEPS="${MAX_TOOL_STEPS:-5}"
MEMORY_MODE="${MEMORY_MODE:-final_output}"
OUTPUT_DIR="${OUTPUT_DIR:-results/gaia_sanity_eval}"
SAFETY_ARGS=()
if [[ "${ENABLE_PYTHON_INTERPRETER:-0}" != "1" ]]; then
  SAFETY_ARGS+=(--disable_python_interpreter)
fi

"$PYTHON_BIN" -m parallel_synthesis.cli.run_gaia_sanity_eval \
  --source_run_dir "$SOURCE_RUN_DIR" \
  --method parallel_kv \
  --model_name "$MODEL_NAME" \
  --parallel_kv_load_dir "$CHECKPOINT_DIR" \
  --source_worker_memory_mode "$MEMORY_MODE" \
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

#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$REPO_ROOT"

: "${BROWSECOMP_CHECKPOINT:?Set BROWSECOMP_CHECKPOINT to the completed BrowseComp SFT checkpoint.}"
: "${GENERAL_EARLY_CHECKPOINT:?Set GENERAL_EARLY_CHECKPOINT to the general checkpoint at global step 95060.}"

PYTHON_BIN="${PYTHON_BIN:-python}"
OUTPUT_DIR="${OUTPUT_DIR:-checkpoints/merged_train_fixed_parallel_kv_browsecomp_SFT_valid_Qwen-Qwen3-14B_050749}"

"$PYTHON_BIN" -m parallel_synthesis.utils.merge_model_checkpoints parallel-kv \
  --checkpoint_a "$BROWSECOMP_CHECKPOINT" \
  --checkpoint_b "$GENERAL_EARLY_CHECKPOINT" \
  --output_dir "$OUTPUT_DIR" \
  --weight_a 0.5 \
  --weight_b 0.5 \
  --cache_mapper_source b \
  --missing_policy error \
  "$@"

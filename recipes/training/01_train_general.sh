#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$REPO_ROOT"

NPROC_PER_NODE="${NPROC_PER_NODE:-4}"
MODEL_NAME="${MODEL_NAME:-Qwen/Qwen3-14B}"
OUTPUT_DIR="${OUTPUT_DIR:-checkpoints/parallel_synthesis_general_qwen3_14b}"
STEPS_PER_EPOCH="${STEPS_PER_EPOCH:--1}"
SAVE_EVERY_STEPS="${SAVE_EVERY_STEPS:-5}"
ATTN_IMPLEMENTATION="${ATTN_IMPLEMENTATION:-auto}"

torchrun --standalone --nproc_per_node="$NPROC_PER_NODE" -m parallel_synthesis.cli.train_mixed_tasks \
  --method fixed_parallel_kv \
  --model_name "$MODEL_NAME" \
  --tasks wildchat,ultrachat,lmsys_chat,toucan_single_parallel,toucan_multi_parallel,dta_tool,flan,2wiki_multihopqa \
  --train_split train \
  --seed 42 \
  --generate_bs 1 \
  --max_new_tokens 4096 \
  --temperature 0.6 \
  --top_p 0.95 \
  --attn_implementation "$ATTN_IMPLEMENTATION" \
  --num_epochs 1 \
  --steps_per_epoch "$STEPS_PER_EPOCH" \
  --learning_rate 1e-4 \
  --weight_decay 0 \
  --parallel_kv_train_components both \
  --parallel_kv_mapper_hidden_dim 32 \
  --parallel_kv_mapper_metadata_features both \
  --parallel_kv_num_parallel_agents 3 \
  --parallel_kv_lora_r 16 \
  --parallel_kv_lora_alpha 32 \
  --parallel_kv_lora_dropout 0 \
  --parallel_kv_skip_train_if_total_tokens_exceed 20000 \
  --parallel_kv_skip_train_if_live_tokens_exceed 1800 \
  --parallel_kv_skip_train_if_attention_area_exceed 20000000 \
  --fixed_parallel_kv_cache_max_tokens_per_text -1 \
  --fixed_parallel_kv_skip_train_if_cache_total_prefill_tokens_exceed 13000 \
  --parallel_kv_save_every_steps "$SAVE_EVERY_STEPS" \
  --parallel_kv_save_dir "$OUTPUT_DIR" \
  "$@"

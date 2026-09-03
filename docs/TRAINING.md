# Training

`parallel-synthesis-train` trains the cache mapper and synthesizer LoRA while
keeping the Qwen3 backbone frozen. Two data routes are supported:

- `fixed_parallel_kv` encodes saved or reference worker texts. This is the
  route used for the release checkpoint.
- `parallel_kv` generates worker outputs during training.

Prepare all inputs in [DATA.md](DATA.md) before launching the recipes.

## Saved checkpoint files

With `--parallel_kv_train_components both`, a training directory contains:

```text
cache_mapper.pt
judger_lora/
training_state.pt
checkpoint_meta.json
run_args.json
train.log
train_mixed_summary.json
```

`train.log` records step loss, perplexity, skip decisions, and memory
diagnostics. `train_mixed_summary.json` records aggregate progress and
per-task coverage.

Periodic saves use `step_checkpoints/latest/`. That directory is replaced at
each save, so copy any milestone that must be retained to a separate path.

## Release training sequence

### 1. General training

```bash
NPROC_PER_NODE=4 \
OUTPUT_DIR=checkpoints/parallel_synthesis_general_qwen3_14b \
  recipes/training/01_train_general.sh
```

The recipe specifies the eight-task mix, seed, optimizer settings, mapper
configuration, rank-16 LoRA, and cache-size guards used for the checkpoint.
Preserve these two milestones:

- global step 95,060: final merge input and cache-mapper source;
- global step 176,175: initialization for BrowseComp trajectory SFT.

For a quick pipeline check, cap the run:

```bash
STEPS_PER_EPOCH=2 \
SAVE_EVERY_STEPS=0 \
OUTPUT_DIR=checkpoints/smoke_general \
  recipes/training/01_train_general.sh
```

### 2. BrowseComp trajectory SFT

The repository bundles a 1,211-row format-filtered trajectory dataset at
`data/browsecomp-textmas/processed/filtered_train.jsonl.gz`. Point
`GENERAL_CHECKPOINT` to the step-176,175 checkpoint and run:

```bash
GENERAL_CHECKPOINT=/path/to/general_step_176175 \
OUTPUT_DIR=checkpoints/parallel_synthesis_browsecomp_sft_qwen3_14b \
  recipes/training/02_train_browsecomp.sh
```

The recipe reads the compressed file directly. `DATA_FILE` may override it
with another processed `.jsonl` or `.jsonl.gz` file.

The published checkpoint used the unfiltered 1,266-row dataset recorded in
the manifests. Its reference four-rank run processed 302 synchronized global
steps, of which 166 produced optimizer updates after the cache and attention
guards. Training with the bundled filtered dataset exercises the same
end-to-end workflow but does not reproduce those weights exactly.

### 3. Final merge

The released LoRA is a 50/50 parameter average of the BrowseComp SFT adapter
and the step-95,060 general adapter. Its cache mapper comes unchanged from the
step-95,060 general checkpoint:

```bash
BROWSECOMP_CHECKPOINT=/path/to/browsecomp_sft \
GENERAL_EARLY_CHECKPOINT=/path/to/general_step_95060 \
OUTPUT_DIR=checkpoints/merged_train_fixed_parallel_kv_browsecomp_SFT_valid_Qwen-Qwen3-14B_050749 \
  recipes/training/03_merge_release_checkpoint.sh
```

The merge utility validates adapter compatibility and records the merge
metadata. Component hashes for the expected result are in
[artifacts/release_checkpoint.json](../artifacts/release_checkpoint.json).

## Distributed training and resume

The release recipes default to four processes with NCCL and a per-rank batch
size of one. Skip decisions are synchronized so every rank either updates or
skips together. Keep the world size, task order, processed data, and seed fixed
when comparing training coverage.

Resume the general recipe from the same shuffled batch order by supplying the
source checkpoint and completed global step:

```bash
OUTPUT_DIR=checkpoints/resumed_run \
  recipes/training/01_train_general.sh \
  --parallel_kv_load_dir /path/to/checkpoint \
  --resume_from_global_step 95060
```

Exact weight-level retraining also depends on preserved optimizer/RNG state,
distributed ordering, and the software and hardware stack. For reproducing the
reported evaluation results, use the released checkpoint and verify its
manifest hashes.

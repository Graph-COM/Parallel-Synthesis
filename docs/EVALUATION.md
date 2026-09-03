# Evaluation

Run commands from the repository root after installing the package. The
installed commands and equivalent Python modules are:

| Dataset family | Command | Python module | Checkpoint flag |
| --- | --- | --- | --- |
| Single-turn | `parallel-synthesis-single` | `parallel_synthesis.cli.post_eval_parallel_kv` | `--checkpoint_dir` |
| GAIA | `parallel-synthesis-gaia` | `parallel_synthesis.cli.run_gaia_eval` | `--parallel_kv_load_dir` |
| GAIA fixed-trajectory | `parallel-synthesis-gaia-sanity` | `parallel_synthesis.cli.run_gaia_sanity_eval` | `--parallel_kv_load_dir` |
| MARBLE DB | `parallel-synthesis-marble` | `parallel_synthesis.cli.run_marble_db_eval` | `--parallel_kv_load_dir` |

Use `--help` on any command for all generation, quantization, resume, and
backend options.

Checkpoint flags accept either a local directory or the Hugging Face repository
ID `Graph-COM/Parallel-Synthesis-qwen3-14B`. The runners download Hub artifacts
to the Hugging Face cache automatically. The repository contains only the
Parallel Synthesis cache mapper and synthesizer LoRA, so continue to pass
`--model_name Qwen/Qwen3-14B` for the frozen backbone.

## Single-turn benchmarks

Supported task names are `gsm8k`, `aime2024`, `aime2025`, `gpqa`,
`medqa`, `humanevalplus`, and `mbppplus`.

```bash
parallel-synthesis-single \
  --checkpoint_dir "$CHECKPOINT_DIR" \
  --method parallel_kv \
  --model_name Qwen/Qwen3-14B \
  --tasks gsm8k,aime2024,aime2025,gpqa,medqa,humanevalplus,mbppplus \
  --split test \
  --eval_samples_per_task -1 \
  --generate_bs 1 \
  --max_new_tokens 4096 \
  --temperature 0 \
  --top_p 1 \
  --seed 42 \
  --output_dir results/single_turn
```

The requested split must exist for each selected dataset. Use a shorter task
list or dataset-appropriate split when necessary.

## GAIA end-to-end evaluation

The paper evaluates the public GAIA validation split at all three difficulty
levels:

| Level | Hugging Face config | Validation samples |
| --- | --- | ---: |
| 1 | `2023_level1` | 53 |
| 2 | `2023_level2` | 86 |
| 3 | `2023_level3` | 26 |

All reported GAIA results use `--split validation`. The test split has no
public answers and must be evaluated through an official GAIA submission;
there is no train evaluation split.

The runner defaults to Level 1, but exact experiment commands should always
specify `--gaia_config`. For example:

```bash
parallel-synthesis-gaia \
  --method parallel_kv \
  --model_name Qwen/Qwen3-14B \
  --parallel_kv_load_dir "$CHECKPOINT_DIR" \
  --gaia_config 2023_level1 \
  --split validation \
  --max_samples -1 \
  --max_tool_steps 5 \
  --temperature 0 \
  --top_p 1 \
  --seed 42 \
  --output_dir results/gaia
```

Repeat with `2023_level2` and `2023_level3`, changing `--output_dir` for
each level.

The Hugging Face dataset is gated. Accept its access conditions and
authenticate with `HF_TOKEN` or `huggingface-cli login`. The loader then
resolves and downloads attachments from each row automatically. Downloads are
reused from `~/.cache/huggingface/hub/` by default; processed dataset data is
stored under `~/.cache/huggingface/datasets/`. Set `HF_HOME` before running if
the cache root should live elsewhere.

The prompt lists any resolved attachment paths, and agents read them through
the `parse_file` tool. Text, office documents, spreadsheets, and common
structured formats are supported. Images require OCR for their contents;
audio/video parsing currently exposes metadata rather than transcription.

GAIA uses the prompt reported in the paper appendix. The web tools can run
without paid credentials, but `SERPER_KEY_ID` and `JINA_API_KEYS` generally
improve search and page retrieval. See
[the GAIA package guide](../parallel_synthesis/toolcall/README.md) for tool
configuration and safety details.

### Fixed-trajectory sanity evaluation

Search results can be costly and can change between runs. To test synthesis
while holding worker search trajectories fixed, point the sanity runner at a
completed GAIA output directory:

```bash
parallel-synthesis-gaia-sanity \
  --source_run_dir results/gaia \
  --method parallel_kv \
  --model_name Qwen/Qwen3-14B \
  --parallel_kv_load_dir "$CHECKPOINT_DIR" \
  --source_worker_memory_mode final_output \
  --temperature 0 \
  --top_p 1 \
  --seed 42 \
  --output_dir results/gaia_sanity
```

`final_output` is the standard setting. Other accepted memory modes are
controlled synthesis variants; none recollects web-search trajectories.
The sanity runner does not take `--gaia_config`: it inherits the level from
the source run's `run_args.json`.

## MARBLE DB end-to-end evaluation

Install `requirements-tools.txt` and Docker with Compose. The default
`marble_docker` backend starts PostgreSQL 15 from the environment bundled in
this repository and resets its disposable state between samples.

```bash
parallel-synthesis-marble \
  --method parallel_kv \
  --model_name Qwen/Qwen3-14B \
  --parallel_kv_load_dir "$CHECKPOINT_DIR" \
  --db_backend marble_docker \
  --max_samples -1 \
  --max_tool_steps 5 \
  --temperature 0 \
  --top_p 1 \
  --seed 42 \
  --output_dir results/marble_db
```

Use `--db_port` if port 5432 is occupied. MARBLE DB always uses the prompt
reported in the paper appendix. The benchmark JSONL is downloaded directly
from the public MARBLE repository when the runner starts.

## Text-synthesis baseline

For a single-turn task:

```bash
parallel-synthesis-text-baseline \
  --method text_mas \
  --task aime2024 \
  --model_name Qwen/Qwen3-14B \
  --max_samples 50 \
  --temperature 0 \
  --top_p 1 \
  --log_dir results/text_mas
```

For GAIA or MARBLE DB, use the corresponding main command with
`--method text_mas` and omit the Parallel Synthesis checkpoint flag. The
baseline concatenates worker text before synthesis and requires no additional
adapter.

## Sharding

For every main runner, add `--num_shards N --shard_id I`, where
`0 <= I < N`, and launch each shard with the same base `--output_dir`.
Each shard writes under `<output_dir>/shards/shard<I>of<N>/`.

Merge a single-turn run:

```bash
python -m parallel_synthesis.cli.merge_post_eval_shards \
  --run_dir results/single_turn \
  --num_shards 4 \
  --strict
```

Merge a GAIA or MARBLE DB run:

```bash
python -m parallel_synthesis.toolcall.merge_toolcall_shards \
  --run_dir results/gaia \
  --num_shards 4 \
  --strict
```

The merge commands create run-level prediction and summary files without
modifying the shard outputs.

## Results

GAIA, GAIA sanity, and MARBLE DB stream `preds.jsonl` and finish with
`summary.json`. Single-turn evaluation writes one
`post_eval_<split>_<task>_preds.jsonl` per task and
`post_eval_summary.json`. All main runs also record their arguments in
`run_args.json`.

For reproducibility checks, use `--temperature 0 --top_p 1 --seed 42`.
Different GPU kernels and dependency versions may still prevent bitwise
identity, so compare answers and metrics rather than timing fields.

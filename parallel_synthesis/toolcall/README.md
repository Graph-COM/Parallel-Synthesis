# GAIA tool-calling evaluation

This package implements the end-to-end GAIA worker and synthesizer workflow.
It supports Parallel Synthesis and the prompt-only text-synthesis baseline.
Both use the GAIA prompt reported in the paper appendix.

## Data and tools

The runner loads the gated `gaia-benchmark/GAIA` dataset and defaults to
`2023_level1`. Select a paper evaluation level with
`--gaia_config 2023_level1`, `2023_level2`, or `2023_level3`.
All reported results use `--split validation`; test answers require official
submission, and GAIA has no train evaluation split.

After Hugging Face authentication, the runner automatically downloads
attachments referenced by `file_path` or `file_name`. Downloads are reused
from `~/.cache/huggingface/hub/` by default; processed dataset data is stored
under `~/.cache/huggingface/datasets/`. Set `HF_HOME` before running to use a
different cache root. No dataset-path or source-code configuration is required.

Agents can call `search`, `visit`, `parse_file`, and
`PythonInterpreter`. Install `requirements-tools.txt` for attachment
parsing. The file paths are placed in the agent prompt and consumed through
`parse_file`; binary files are not passed directly into the language model.
Office documents, spreadsheets, text, and common structured files are parsed
locally. Image contents require OCR, while audio/video transcription is not
currently implemented.

Optional environment variables:

- `SERPER_KEY_ID`: enable Serper-backed search; otherwise the runner uses its
  DuckDuckGo fallback.
- `JINA_API_KEYS`: comma-separated Jina Reader keys for page retrieval.
- `JINA_VISIT_TIMEOUT`, `JINA_VISIT_RETRIES`,
  `DIRECT_VISIT_TIMEOUT`, and `VISIT_MAX_CHARS`: retrieval limits.
- `TOOL_PY_TIMEOUT`: local Python execution timeout.

The runner loads a repository-local `.env` without overriding variables
already exported in the shell.

## End-to-end run

```bash
parallel-synthesis-gaia \
  --method parallel_kv \
  --model_name Qwen/Qwen3-14B \
  --parallel_kv_load_dir "$CHECKPOINT_DIR" \
  --gaia_config 2023_level1 \
  --split validation \
  --max_tool_steps 5 \
  --output_dir results/gaia
```

Use `--method text_mas` and omit `--parallel_kv_load_dir` for the
text-synthesis baseline.

## Fixed-trajectory sanity run

After an end-to-end run, reuse its worker search trajectories and rerun
synthesis:

```bash
parallel-synthesis-gaia-sanity \
  --source_run_dir results/gaia \
  --method parallel_kv \
  --model_name Qwen/Qwen3-14B \
  --parallel_kv_load_dir "$CHECKPOINT_DIR" \
  --source_worker_memory_mode final_output \
  --output_dir results/gaia_sanity
```

This keeps web retrieval fixed and does not call the search agents again.
The GAIA level is inherited from the source run, so the sanity command does
not need a separate `--gaia_config`.

## HTML report

```bash
python -m parallel_synthesis.toolcall.analyze_logs \
  --run_dir results/gaia \
  --include_prompts
```

The renderer reads `preds.jsonl` and `summary.json` and writes
`toolcall_analysis.html` beside them. It does not modify the JSON results.
Scoring uses normalized exact match against the reference answer.

## Safety

`PythonInterpreter` executes model-produced Python on the local machine. Run
GAIA in an isolated environment, or pass
`--disable_python_interpreter` to remove that tool.

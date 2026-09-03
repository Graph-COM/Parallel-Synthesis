# MARBLE DB evaluation

This package contains the MARBLE DB worker, synthesizer, scorer, and local
Docker environment. The runner uses the MARBLE prompt reported in the paper
appendix.

## Requirements

- Docker with `docker compose`
- The dependencies in `requirements-tools.txt`, including
  `psycopg2-binary`

The default `marble_docker` backend starts PostgreSQL 15 and resets its
disposable state between samples. The benchmark JSONL is downloaded from the
public MARBLE repository when the runner starts.

## Run

```bash
parallel-synthesis-marble \
  --method parallel_kv \
  --model_name Qwen/Qwen3-14B \
  --parallel_kv_load_dir "$CHECKPOINT_DIR" \
  --db_backend marble_docker \
  --max_tool_steps 5 \
  --output_dir results/marble_db
```

Use `--method text_mas` and omit `--parallel_kv_load_dir` for the
text-synthesis baseline. For a one-sample environment check, add
`--max_samples 1`.

If Docker Compose requires elevated privileges, pass:

```bash
--db_docker_compose_cmd "sudo docker compose"
```

Use `--db_port` if the default PostgreSQL port is occupied.

## HTML report

```bash
python -m parallel_synthesis.marble_db.analyze_logs \
  --run_dir results/marble_db \
  --include_prompts
```

The renderer reads `preds.jsonl`, `summary.json`, and `run_args.json`
when available, then writes `marble_db_analysis.html` beside them. It does
not modify the JSON results.

Only use the bundled environment or another disposable database. The
benchmark executes SQL and anomaly triggers and should never target valuable
data.

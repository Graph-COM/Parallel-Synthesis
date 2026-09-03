import argparse
import json
import time
from pathlib import Path
from typing import Dict, List


def evaluate(preds: List[Dict]) -> Dict[str, float]:
    total = len(preds)
    correct = sum(1 for p in preds if bool(p.get("correct", False)))
    ttfts = [
        float(p["ttft_sec"])
        for p in preds
        if p.get("ttft_sec") is not None
    ]
    cache_prepare_times = [
        float(p["cache_prepare_sec"])
        for p in preds
        if p.get("cache_prepare_sec") is not None
    ]
    return {
        "samples": total,
        "correct": correct,
        "accuracy": (correct / total) if total > 0 else 0.0,
        "avg_ttft_sec": round(sum(ttfts) / len(ttfts), 4) if ttfts else None,
        "ttft_measured_samples": len(ttfts),
        "avg_cache_prepare_sec": (
            round(sum(cache_prepare_times) / len(cache_prepare_times), 4)
            if cache_prepare_times
            else None
        ),
        "cache_prepare_measured_samples": len(cache_prepare_times),
    }


def read_jsonl(path: Path) -> List[Dict]:
    rows: List[Dict] = []
    with path.open("r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description="Merge sharded GAIA tool-calling runs.")
    parser.add_argument("--run_dir", type=str, required=True, help="Base run dir (contains shards/).")
    parser.add_argument("--num_shards", type=int, required=True)
    parser.add_argument(
        "--strict",
        action="store_true",
        help="Fail if any shard preds file is missing.",
    )
    args = parser.parse_args()

    run_dir = Path(args.run_dir)
    num_shards = max(1, int(args.num_shards))
    shards_root = run_dir / "shards"
    if num_shards <= 1:
        raise ValueError("num_shards must be > 1 for merging.")
    if not shards_root.exists():
        raise FileNotFoundError(f"Missing shards directory: {shards_root}")

    merged_preds: List[Dict] = []
    shard_summaries: List[Dict] = []
    missing: List[Path] = []

    for shard_id in range(num_shards):
        shard_dir = shards_root / f"shard{shard_id:02d}of{num_shards:02d}"
        preds_path = shard_dir / "preds.jsonl"
        summary_path = shard_dir / "summary.json"

        if not preds_path.exists():
            missing.append(preds_path)
            continue
        merged_preds.extend(read_jsonl(preds_path))

        if summary_path.exists():
            with summary_path.open("r", encoding="utf-8") as fh:
                shard_summaries.append(json.load(fh))

    if missing and args.strict:
        missing_str = "\n".join(str(p) for p in missing)
        raise FileNotFoundError(f"Missing shard prediction files:\n{missing_str}")

    out_preds = run_dir / "preds.jsonl"
    out_summary = run_dir / "summary.json"

    with out_preds.open("w", encoding="utf-8") as fh:
        for row in merged_preds:
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")

    metrics = evaluate(merged_preds)
    if shard_summaries:
        first = shard_summaries[0]
        for key in ("benchmark", "method", "model_name", "scoring"):
            if key in first:
                metrics[key] = first[key]
    metrics["timestamp"] = time.strftime("%Y-%m-%d %H:%M:%S")
    metrics["num_shards"] = num_shards
    metrics["merged_shards_found"] = len(shard_summaries)
    metrics["missing_shards"] = [str(p) for p in missing]

    with out_summary.open("w", encoding="utf-8") as fh:
        json.dump(metrics, fh, ensure_ascii=False, indent=2)

    print(f"Merged predictions: {out_preds}")
    print(f"Merged summary: {out_summary}")
    print(json.dumps(metrics, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

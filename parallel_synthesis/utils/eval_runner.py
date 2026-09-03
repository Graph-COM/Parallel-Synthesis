import json
from typing import Dict, List, Optional, TextIO, Tuple

from tqdm import tqdm


def evaluate(preds: List[Dict]) -> Tuple[float, int]:
    total = len(preds)
    correct = sum(1 for p in preds if p.get("correct", False))
    acc = correct / total if total > 0 else 0.0
    return acc, correct


def summarize_latency_metrics(preds: List[Dict]) -> Dict[str, Optional[float]]:
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
        "avg_ttft_sec": round(sum(ttfts) / len(ttfts), 4) if ttfts else None,
        "ttft_measured_samples": len(ttfts),
        "avg_cache_prepare_sec": (
            round(sum(cache_prepare_times) / len(cache_prepare_times), 4)
            if cache_prepare_times
            else None
        ),
        "cache_prepare_measured_samples": len(cache_prepare_times),
    }


def process_batch(
    method,
    batch: List[Dict],
    processed: int,
    preds: List[Dict],
    progress,
    max_samples: int,
    args,
    log_fh: Optional[TextIO] = None,
) -> Tuple[int, List[Dict]]:
    remaining = max_samples - processed
    if remaining <= 0:
        return processed, preds
    current_batch = batch[:remaining]
    if not getattr(args, "use_vllm", False):
        results = method.run_batch(current_batch)
    elif getattr(args, "method", "") in {"baseline", "text_mas"}:
        results = method.run_batch_vllm(current_batch)
    else:
        raise ValueError("--use_vllm is currently only supported for baseline and text_mas inference.")
    if len(results) > remaining:
        results = results[:remaining]
    batch_start = processed
    for offset, res in enumerate(results):
        preds.append(res)
        if log_fh is not None:
            log_fh.write(json.dumps(res, ensure_ascii=False) + "\n")
            log_fh.flush()
        problem_idx = batch_start + offset + 1
        print(f"\n==================== Problem #{problem_idx} ====================")
        print("Question:")
        print(res.get("question", "").strip())
        agents = res.get("agents", [])
        for agent in agents:
            name = agent.get("name", "Agent")
            role = agent.get("role", "")
            print(f"----- Agent: {name} ({role}) -----")
            agent_input = agent.get("input", "").rstrip()
            agent_output = agent.get("output", "").rstrip()
            latent_steps = agent.get("latent_steps", None)
            print("[To Tokenize]")
            print(agent_input)
            if latent_steps is not None:
                print("[Latent Steps]")
                print(latent_steps)
            print("[Output]")
            print(agent_output)
            print("----------------------------------------------")
        print(
            f"Result: Pred={res.get('prediction')} | Gold={res.get('gold')} | OK={res.get('correct')}"
        )

    processed += len(results)
    if progress is not None:
        progress.update(len(results))
    return processed, preds


def run_eval(
    method,
    dataset_iter,
    args,
    *,
    log_fh: Optional[TextIO] = None,
    desc: str = "",
) -> Tuple[List[Dict], int]:
    preds: List[Dict] = []
    processed = 0
    batch: List[Dict] = []

    max_samples = args.max_samples
    if max_samples == -1:
        dataset_iter = list(dataset_iter)
        max_samples = len(dataset_iter)

    progress = tqdm(total=max_samples, desc=desc or None)
    for item in dataset_iter:
        if processed >= max_samples:
            break
        batch.append(item)
        if len(batch) == args.generate_bs or processed + len(batch) == max_samples:
            processed, preds = process_batch(
                method,
                batch,
                processed,
                preds,
                progress,
                max_samples,
                args,
                log_fh=log_fh,
            )
            batch = []
            if processed >= max_samples:
                break

    if batch and processed < max_samples:
        processed, preds = process_batch(
            method,
            batch,
            processed,
            preds,
            progress,
            max_samples=max_samples,
            args=args,
            log_fh=log_fh,
        )
    progress.close()
    return preds, max_samples

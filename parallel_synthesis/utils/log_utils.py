from typing import Any, Dict, Sequence


def shorten_for_log(text: Any, limit: int = 120) -> str:
    rendered = str(text).strip()
    if limit <= 0 or len(rendered) <= limit:
        return rendered
    return rendered[:limit] + "... [truncated]"

def log_fixed_parallel_kv_auto_cap_prepared(
    method: Any,
    item: Dict[str, Any],
    *,
    total_prefill_tokens: int,
    capped_prepared: Sequence[Dict[str, Any]],
) -> None:
    print(
        "[fixed_parallel_kv] auto-capped prepared cache sample due to potential OOM: "
        f"total_prefill_tokens={total_prefill_tokens} -> "
        f"{sum(int(entry['prefill_tokens']) for entry in capped_prepared)} "
        f"threshold={getattr(method, 'fixed_parallel_kv_auto_cap_total_tokens_threshold', -1)} "
        f"per_text_cap={getattr(method, 'fixed_parallel_kv_auto_cap_tokens_per_text', -1)} "
        f"query={shorten_for_log(method._item_query_text(item))}"
    )


def _remaining_fixed_cache_debug_slots(method: Any, batch_size: int) -> int:
    debug_limit = int(getattr(method, "fixed_parallel_kv_debug_print_limit", -1))
    debug_printed = int(getattr(method, "_fixed_parallel_kv_debug_printed", 0))
    if debug_limit == 0:
        return 0
    if debug_limit > 0:
        return max(debug_limit - debug_printed, 0)
    return batch_size


def debug_print_fixed_cache_inputs(
    method: Any,
    items: Sequence[Dict[str, Any]],
    cache_entries_by_item: Sequence[Sequence[Dict[str, Any]]],
) -> None:
    if not getattr(method, "fixed_parallel_kv_debug_print_cache_inputs", False):
        return

    remaining = _remaining_fixed_cache_debug_slots(method, len(items))
    if remaining <= 0:
        return

    debug_printed = int(getattr(method, "_fixed_parallel_kv_debug_printed", 0))
    for item, entries in zip(items, cache_entries_by_item):
        if remaining <= 0:
            break
        debug_printed += 1
        remaining -= 1
        print(
            "[fixed_parallel_kv][debug] "
            f"sample={debug_printed} "
            f"task={getattr(method, 'task', '')} "
            f"query={shorten_for_log(method._item_query_text(item), limit=240)}"
        )
        for cache_idx, entry in enumerate(entries, start=1):
            print(
                "[fixed_parallel_kv][debug] "
                f"sample={debug_printed} cache={cache_idx} "
                f"prefill_tokens={int(entry.get('prefill_tokens', 0) or 0)} "
                f"extract_tokens={int(entry.get('extract_tokens', 0) or 0)}"
            )
            print("[fixed_parallel_kv][debug] prefill_text:")
            print(str(entry.get("prefill_text", "")).rstrip())
            print("[fixed_parallel_kv][debug] extract_text:")
            print(str(entry.get("extract_text", "")).rstrip())

    setattr(method, "_fixed_parallel_kv_debug_printed", debug_printed)


__all__ = [
    "debug_print_fixed_cache_inputs",
    "log_fixed_parallel_kv_auto_cap_prepared",
    "shorten_for_log",
]

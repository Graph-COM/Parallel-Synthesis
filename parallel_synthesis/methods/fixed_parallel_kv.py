import argparse
from typing import Any, Dict, List, Optional, Tuple

import torch

from parallel_synthesis.models import _past_length
from parallel_synthesis.prompts import format_fixed_parallel_cache_text
from parallel_synthesis.utils.log_utils import (
    debug_print_fixed_cache_inputs,
    log_fixed_parallel_kv_auto_cap_prepared,
    shorten_for_log,
)

from .parallel_kv import ParallelKV


class FixedParallelKV(ParallelKV):
    """
    Shared FixedParallelKV implementation used across the repo.

    Fixed caches are encoded directly from dataset-provided
    `agent_reference_contexts` / `agent_extract_contexts`.
    """

    def __init__(
        self,
        model,
        *,
        judger_max_new_tokens: int = 256,
        temperature: float = 0.6,
        top_p: float = 0.95,
        generate_bs: int = 1,
        args: argparse.Namespace = None,
    ) -> None:
        super().__init__(
            model,
            judger_max_new_tokens=judger_max_new_tokens,
            temperature=temperature,
            top_p=top_p,
            generate_bs=generate_bs,
            args=args,
        )
        self.fixed_parallel_kv_cache_max_tokens_per_text = int(
            getattr(args, "fixed_parallel_kv_cache_max_tokens_per_text", -1) if args else -1
        )
        self.fixed_parallel_kv_auto_cap_on_potential_oom = bool(
            getattr(args, "fixed_parallel_kv_auto_cap_on_potential_oom", False) if args else False
        )
        self.fixed_parallel_kv_auto_cap_total_tokens_threshold = int(
            getattr(args, "fixed_parallel_kv_auto_cap_total_tokens_threshold", -1) if args else -1
        )
        self.fixed_parallel_kv_auto_cap_tokens_per_text = int(
            getattr(args, "fixed_parallel_kv_auto_cap_tokens_per_text", 256) if args else 256
        )
        self.fixed_parallel_kv_skip_train_if_cache_total_prefill_tokens_exceed = int(
            getattr(args, "fixed_parallel_kv_skip_train_if_cache_total_prefill_tokens_exceed", -1)
            if args
            else -1
        )
        self.fixed_parallel_kv_debug_print_cache_inputs = bool(
            getattr(args, "fixed_parallel_kv_debug_print_cache_inputs", False) if args else False
        )
        self.fixed_parallel_kv_debug_print_limit = int(
            getattr(args, "fixed_parallel_kv_debug_print_limit", 4) if args else 4
        )
        self._fixed_parallel_kv_debug_printed = 0

    def _num_parallel_agents_for_items(self, items: List[Dict[str, Any]]) -> int:
        max_num_caches = 0
        for item in items:
            num_caches = len(self._fixed_cache_entries_for_item(item))
            if num_caches > max_num_caches:
                max_num_caches = num_caches
        return max(1, max_num_caches)

    def _extract_dataset_context_entries(self, item: Dict[str, Any]) -> List[Dict[str, Any]]:
        source_field = "agent_reference_contexts"
        contexts = item.get(source_field)
        if not isinstance(contexts, list):
            return []
        extract_contexts = item.get("agent_extract_contexts")
        use_extract_field = isinstance(extract_contexts, list)

        entries: List[Dict[str, Any]] = []
        for idx, ctx in enumerate(contexts):
            text = str(ctx).strip()
            if not text:
                continue
            extract_text = text
            if use_extract_field and idx < len(extract_contexts):
                candidate = str(extract_contexts[idx]).strip()
                if candidate:
                    extract_text = candidate
            entries.append(
                {
                    "text": text,
                    "prefill_text": text,
                    "extract_text": extract_text,
                    "source_field": source_field,
                    "context_index": idx,
                }
            )
        return entries

    def _fixed_cache_entries_for_item(self, item: Dict[str, Any]) -> List[Dict[str, Any]]:
        return self._extract_dataset_context_entries(item)

    def _apply_dataset_template(
        self,
        item: Dict[str, Any],
        cache_entry: Dict[str, Any],
        cache_idx: int,
    ) -> str:
        cache_text = str(
            cache_entry.get("prefill_text", cache_entry.get("text", ""))
        ).strip()
        if not cache_text:
            return cache_text
        return format_fixed_parallel_cache_text(
            task=self.task,
            item=item,
            cache_text=cache_text,
            cache_idx=cache_idx,
            cache_meta=cache_entry,
        )

    def _token_len(self, text: str) -> int:
        ids = self.model.tokenizer(
            str(text),
            add_special_tokens=False,
            return_tensors="pt",
        )["input_ids"]
        return int(ids.shape[-1])

    def _truncate_text_to_last_tokens(self, text: str, max_tokens: int) -> str:
        if max_tokens <= 0:
            return str(text)
        encoded = self.model.tokenizer(
            str(text),
            add_special_tokens=False,
            return_tensors="pt",
        )["input_ids"][0]
        if int(encoded.shape[0]) <= max_tokens:
            return str(text)
        clipped = encoded[-max_tokens:].tolist()
        return self.model.tokenizer.decode(
            clipped,
            skip_special_tokens=False,
            clean_up_tokenization_spaces=False,
        ).strip()

    def _suffix_token_count_in_text(self, full_text: str, extract_text: str) -> int:
        rendered_full = str(full_text or "")
        rendered_extract = str(extract_text or "")
        if not rendered_full:
            return 0
        if not rendered_extract or rendered_full == rendered_extract:
            return self._token_len(rendered_full)
        if not rendered_full.endswith(rendered_extract):
            return self._token_len(rendered_extract)

        start_char = len(rendered_full) - len(rendered_extract)
        try:
            encoded = self.model.tokenizer(
                rendered_full,
                add_special_tokens=False,
                return_offsets_mapping=True,
            )
            offsets = encoded.get("offset_mapping", [])
            if hasattr(offsets, "tolist"):
                offsets = offsets.tolist()
            extract_tokens = sum(1 for start, end in offsets if int(end) > start_char)
            if extract_tokens > 0:
                return int(extract_tokens)
        except Exception:
            pass
        return self._token_len(rendered_extract)

    def _cache_entries_by_item(self, items: List[Dict[str, Any]]) -> List[List[Dict[str, Any]]]:
        all_entries: List[List[Dict[str, Any]]] = []
        for item in items:
            entries = self._fixed_cache_entries_for_item(item)
            prepared: List[Dict[str, Any]] = []
            for idx, cache_entry in enumerate(entries, start=1):
                prefill_text = self._apply_dataset_template(item, cache_entry, idx)
                extract_text = str(cache_entry.get("extract_text", "")).strip() or prefill_text
                prefill_tokens = self._token_len(prefill_text)
                extract_tokens = self._suffix_token_count_in_text(prefill_text, extract_text)
                prepared.append(
                    {
                        **cache_entry,
                        "prefill_text": prefill_text,
                        "extract_text": extract_text,
                        "prefill_tokens": prefill_tokens,
                        "extract_tokens": extract_tokens,
                        "prefix_tokens": max(prefill_tokens - extract_tokens, 0),
                    }
                )

            if (
                self.fixed_parallel_kv_auto_cap_on_potential_oom
                and self.fixed_parallel_kv_auto_cap_total_tokens_threshold > 0
                and self.fixed_parallel_kv_auto_cap_tokens_per_text > 0
                and prepared
            ):
                total_prefill_tokens = sum(int(entry["prefill_tokens"]) for entry in prepared)
                if total_prefill_tokens > self.fixed_parallel_kv_auto_cap_total_tokens_threshold:
                    capped_prepared: List[Dict[str, Any]] = []
                    for entry in prepared:
                        capped_prefill = self._truncate_text_to_last_tokens(
                            str(entry.get("prefill_text", "")),
                            self.fixed_parallel_kv_auto_cap_tokens_per_text,
                        )
                        raw_extract = str(entry.get("extract_text", ""))
                        capped_extract = self._truncate_text_to_last_tokens(
                            raw_extract,
                            self.fixed_parallel_kv_auto_cap_tokens_per_text,
                        )
                        if not capped_extract:
                            capped_extract = capped_prefill
                        capped_prefill_tokens = self._token_len(capped_prefill)
                        capped_extract_tokens = self._suffix_token_count_in_text(capped_prefill, capped_extract)
                        capped_entry = dict(entry)
                        capped_entry.update(
                            {
                                "prefill_text": capped_prefill,
                                "extract_text": capped_extract,
                                "prefill_tokens": capped_prefill_tokens,
                                "extract_tokens": capped_extract_tokens,
                                "prefix_tokens": max(capped_prefill_tokens - capped_extract_tokens, 0),
                            }
                        )
                        capped_prepared.append(capped_entry)
                    log_fixed_parallel_kv_auto_cap_prepared(
                        self,
                        item,
                        total_prefill_tokens=total_prefill_tokens,
                        capped_prepared=capped_prepared,
                    )
                    prepared = capped_prepared

            if entries and not prepared:
                raise ValueError(
                    "fixed_parallel_kv failed to prepare cache texts for "
                    f"task={self.task} question={shorten_for_log(self._item_query_text(item))}"
                )
            all_entries.append(prepared)
        return all_entries

    def _group_batch_indices(self, items: List[Dict[str, Any]]) -> List[List[int]]:
        cache_entries_by_item = self._cache_entries_by_item(items)
        grouped: Dict[Tuple[Any, ...], List[int]] = {}
        for idx, entries in enumerate(cache_entries_by_item):
            signature = (
                len(entries),
                tuple(
                    (
                        int(entry.get("prefill_tokens", 0) or 0),
                        int(entry.get("extract_tokens", 0) or 0),
                    )
                    for entry in entries
                ),
            )
            grouped.setdefault(signature, []).append(idx)
        return list(grouped.values())

    def _maybe_skip_train_for_cache_encode(
        self,
        items: List[Dict[str, Any]],
        cache_entries_by_item: List[List[Dict[str, Any]]],
    ) -> Optional[Dict[str, Any]]:
        unique_counts = {len(entries) for entries in cache_entries_by_item}
        if len(unique_counts) > 1:
            raise ValueError(
                "fixed_parallel_kv received mixed cache counts in one sub-batch. "
                "This batch should be grouped before cache encoding."
            )

        max_num_caches = max((len(row) for row in cache_entries_by_item), default=0)
        all_cache_prefill_tokens_per_sample = [
            sum(int(entry.get("prefill_tokens", 0) or 0) for entry in entries)
            for entries in cache_entries_by_item
        ]
        all_cache_extract_tokens_per_sample = [
            sum(int(entry.get("extract_tokens", 0) or 0) for entry in entries)
            for entries in cache_entries_by_item
        ]
        all_cache_total_prefill_tokens = int(sum(all_cache_prefill_tokens_per_sample))
        total_prefill_limit = int(self.fixed_parallel_kv_skip_train_if_cache_total_prefill_tokens_exceed)
        if total_prefill_limit > 0 and all_cache_total_prefill_tokens > total_prefill_limit:
            return self._build_skip_train_result(
                reason="fixed_cache_total_prefill_tokens_exceeded",
                message=(
                    "Skipping batch before fixed-cache encoding because total cache prefill tokens "
                    "exceed the configured limit."
                ),
                batch_size=len(items),
                num_total_caches=int(max_num_caches),
                all_cache_prefill_tokens_per_sample=all_cache_prefill_tokens_per_sample,
                all_cache_extract_tokens_per_sample=all_cache_extract_tokens_per_sample,
                all_cache_total_tokens=all_cache_total_prefill_tokens,
                all_cache_total_extract_tokens=int(sum(all_cache_extract_tokens_per_sample)),
                total_tokens=all_cache_total_prefill_tokens,
                token_limit=int(total_prefill_limit),
                example_ids=[str(item.get("id", "")).strip() for item in items],
            )

        return None

    def _collect_parallel_agent_past(
        self,
        items: List[Dict[str, Any]],
        *,
        include_traces: bool = False,
        cache_entries_by_item: Optional[List[List[Dict[str, Any]]]] = None,
    ) -> Tuple[List[Dict[str, Any]], Optional[List[List[Dict[str, Any]]]]]:
        self._ensure_hf_backend()
        if cache_entries_by_item is None:
            cache_entries_by_item = self._cache_entries_by_item(items)
        debug_print_fixed_cache_inputs(self, items, cache_entries_by_item)
        all_cache_prefill_tokens_per_sample = [
            sum(int(entry.get("prefill_tokens", 0) or 0) for entry in entries)
            for entries in cache_entries_by_item
        ]
        all_cache_extract_tokens_per_sample = [
            sum(int(entry.get("extract_tokens", 0) or 0) for entry in entries)
            for entries in cache_entries_by_item
        ]
        unique_counts = {len(entries) for entries in cache_entries_by_item}
        if len(unique_counts) > 1:
            raise ValueError(
                "fixed_parallel_kv received mixed cache counts in one sub-batch. "
                "This batch should be grouped before cache encoding."
            )

        traces: Optional[List[List[Dict[str, Any]]]] = None
        if include_traces:
            traces = [[] for _ in range(len(items))]
            for cache_idx, slot_entries in enumerate(zip(*cache_entries_by_item), start=1):
                for row_idx, cache_entry in enumerate(slot_entries):
                    traces[row_idx].append(
                        {
                            "name": f"Cache{cache_idx}",
                            "role": "fixed_cache",
                            "input": str(cache_entry.get("prefill_text", "")),
                            "extract": str(cache_entry.get("extract_text", "")),
                            "output": "",
                        }
                    )

        max_num_caches = max((len(row) for row in cache_entries_by_item), default=0)
        parallel_agent_past: List[Dict[str, Any]] = []
        model_obj = self.model.model
        get_base_model = getattr(model_obj, "get_base_model", None)
        causal_lm = get_base_model() if callable(get_base_model) else model_obj
        causal_backbone = getattr(causal_lm, "model", None)
        cache_encoder = causal_backbone if callable(causal_backbone) else model_obj
        was_training = bool(model_obj.training)
        with self._disable_lora_for_non_judger():
            model_obj.eval()
            try:
                for cache_idx in range(max_num_caches):
                    slot_entries = [item_cache_entries[cache_idx] for item_cache_entries in cache_entries_by_item]
                    slot_texts = [str(entry.get("prefill_text", "")) for entry in slot_entries]
                    prefill_token_counts = {int(entry.get("prefill_tokens", 0) or 0) for entry in slot_entries}
                    extract_token_counts = {int(entry.get("extract_tokens", 0) or 0) for entry in slot_entries}
                    if len(prefill_token_counts) > 1 or len(extract_token_counts) > 1:
                        raise ValueError(
                            "fixed_parallel_kv received mixed prefill/extract lengths in one sub-batch. "
                            "This batch should be grouped before cache encoding."
                        )
                    encoded = self.model.tokenizer(
                        slot_texts,
                        return_tensors="pt",
                        padding=True,
                        add_special_tokens=False,
                        truncation=(self.fixed_parallel_kv_cache_max_tokens_per_text > 0),
                        max_length=(
                            self.fixed_parallel_kv_cache_max_tokens_per_text
                            if self.fixed_parallel_kv_cache_max_tokens_per_text > 0
                            else None
                        ),
                    )
                    cache_ids = encoded["input_ids"].to(self.model.device)
                    cache_mask = encoded["attention_mask"].to(self.model.device)
                    if int(cache_mask.sum().item()) <= 0:
                        continue

                    try:
                        with torch.no_grad():
                            outputs = cache_encoder(
                                input_ids=cache_ids,
                                attention_mask=cache_mask,
                                use_cache=True,
                                return_dict=True,
                            )
                    except torch.OutOfMemoryError as exc:
                        slot_prefill_tokens_per_sample = [int(x) for x in cache_mask.sum(dim=1).tolist()]
                        slot_extract_tokens_per_sample = [
                            min(int(entry.get("extract_tokens", 0) or 0), slot_prefill_tokens_per_sample[row_idx])
                            for row_idx, entry in enumerate(slot_entries)
                        ]
                        self._raise_augmented_oom(
                            exc,
                            tag="fixed_parallel_kv_cache_encode_oom",
                            details={
                                "task": self.task,
                                "batch_size": len(items),
                                "cache_slot_index": int(cache_idx + 1),
                                "num_total_caches": int(max_num_caches),
                                "slot_prefill_tokens_max": max(slot_prefill_tokens_per_sample) if slot_prefill_tokens_per_sample else 0,
                                "slot_prefill_tokens_per_sample": slot_prefill_tokens_per_sample,
                                "slot_extract_tokens_max": max(slot_extract_tokens_per_sample) if slot_extract_tokens_per_sample else 0,
                                "slot_extract_tokens_per_sample": slot_extract_tokens_per_sample,
                                "slot_total_tokens": int(sum(slot_prefill_tokens_per_sample)),
                                "all_cache_prefill_tokens_per_sample": all_cache_prefill_tokens_per_sample,
                                "all_cache_extract_tokens_per_sample": all_cache_extract_tokens_per_sample,
                                "all_cache_total_tokens": int(sum(all_cache_prefill_tokens_per_sample)),
                                "all_cache_total_extract_tokens": int(sum(all_cache_extract_tokens_per_sample)),
                                "example_ids": [str(item.get("id", "")).strip() for item in items],
                            },
                        )
                    cache_past = self._ensure_cache_object(outputs.past_key_values)
                    if cache_past is None or _past_length(cache_past) <= 0:
                        continue
                    extract_token_count = max(extract_token_counts) if extract_token_counts else 0
                    if extract_token_count > 0:
                        cache_past = self._truncate_past(cache_past, extract_token_count)
                    if cache_past is None or _past_length(cache_past) <= 0:
                        continue
                    prefix_lens = (cache_mask.sum(dim=1) - int(extract_token_count)).clamp_min(0).to(self.model.device)

                    parallel_agent_past.append(
                        {
                            "past": cache_past,
                            "role": f"fixed_cache_{cache_idx + 1}",
                            "prompt_lens": prefix_lens,
                        }
                    )
            finally:
                if was_training:
                    model_obj.train()
                else:
                    model_obj.eval()
        return parallel_agent_past, traces

    def _train_batch_impl(self, items: List[Dict[str, Any]]) -> Dict[str, Any]:
        cache_entries_by_item = self._cache_entries_by_item(items)
        skip_result = self._maybe_skip_train_for_cache_encode(items, cache_entries_by_item)
        if skip_result is not None:
            return skip_result
        parallel_agent_past, _ = self._collect_parallel_agent_past(
            items,
            include_traces=False,
            cache_entries_by_item=cache_entries_by_item,
        )
        return self._compute_train_loss(items, parallel_agent_past)


__all__ = ["FixedParallelKV"]

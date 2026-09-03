import argparse
import json
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import torch

from parallel_synthesis.methods.parallel_kv import _past_length
from parallel_synthesis.models import ModelWrapper

try:
    from transformers.cache_utils import Cache
except ImportError:
    Cache = None

try:
    from transformers.cache_utils import DynamicCache
except ImportError:
    DynamicCache = None

from . import prompts
from .methods import _ToolCallingMethodBase, ParallelKVToolCallingMethod
from .scoring import extract_answer_text, free_form_exact_match
from .tools import ToolRegistry


FILE_SECTION_HEADERS = (
    "Attached files you can parse with parse_file:",
    "Attached files (use parse_file if needed):",
)

SOURCE_WORKER_MEMORY_MODES = (
    "final_output",
    "full_prompt_output",
    "each_turn_output",
    "full_prompt_output_no_system",
)


def _source_worker_memory_mode(args: argparse.Namespace) -> str:
    mode = str(getattr(args, "source_worker_memory_mode", "final_output") or "final_output").strip().lower()
    if mode not in SOURCE_WORKER_MEMORY_MODES:
        raise ValueError(
            f"Unsupported source_worker_memory_mode={mode!r}. "
            f"Expected one of: {', '.join(SOURCE_WORKER_MEMORY_MODES)}"
        )
    return mode


def _load_jsonl_rows(path: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as fh:
        for line in fh:
            text = line.strip()
            if not text:
                continue
            rows.append(json.loads(text))
    return rows


def _normalize_files(files: List[str]) -> List[str]:
    seen = set()
    ordered: List[str] = []
    for value in files:
        item = str(value).strip()
        if not item or item in seen:
            continue
        seen.add(item)
        ordered.append(item)
    return ordered


def _extract_files_from_prompt(prompt: str) -> List[str]:
    text = str(prompt or "")
    lines = text.splitlines()
    for idx, line in enumerate(lines):
        stripped = line.strip()
        if stripped not in FILE_SECTION_HEADERS:
            continue
        files: List[str] = []
        for next_line in lines[idx + 1 :]:
            candidate = str(next_line).strip()
            if not candidate:
                if files:
                    break
                continue
            if candidate.endswith(":") and not candidate.startswith("/"):
                break
            if candidate == "Your response:":
                break
            files.append(candidate)
        return _normalize_files(files)
    return []


def _clone_tool_calls(tool_calls: List[Any]) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    for call in tool_calls or []:
        out.append(
            {
                "name": str(call.get("name", "")),
                "arguments": dict(call.get("arguments", {})),
                "result_preview": str(call.get("result_preview", "")),
            }
        )
    return out


def _clone_messages(messages: List[Any]) -> List[Dict[str, str]]:
    out: List[Dict[str, str]] = []
    for msg in messages or []:
        out.append(
            {
                "role": str(msg.get("role", "")),
                "content": str(msg.get("content", "")),
            }
        )
    return out


def _clone_text_list(values: List[Any]) -> List[str]:
    return [str(v) for v in (values or [])]


def clone_source_agent_trace(agent: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "name": str(agent.get("name", "")),
        "role": str(agent.get("role", "")),
        "input": str(agent.get("input", "")),
        "output": str(agent.get("output", "")),
        "final_answer": str(agent.get("final_answer", "")),
        "tool_calls": _clone_tool_calls(agent.get("tool_calls", [])),
        "messages": _clone_messages(agent.get("messages", [])),
        "step_prompts": _clone_text_list(agent.get("step_prompts", [])),
        "source_trace": True,
    }


def _worker_output_text(agent: Dict[str, Any]) -> str:
    output = str(agent.get("output", ""))
    if output:
        return output
    messages = agent.get("messages", []) or []
    for msg in reversed(messages):
        if str(msg.get("role", "")) != "assistant":
            continue
        content = str(msg.get("content", ""))
        if content:
            return content
    return str(agent.get("final_answer", ""))


def _worker_last_prompt(agent: Dict[str, Any]) -> str:
    step_prompts = agent.get("step_prompts", []) or []
    if step_prompts:
        return str(step_prompts[-1])
    return ""


def _worker_messages_before_final_turn(agent: Dict[str, Any]) -> List[Dict[str, str]]:
    messages = _clone_messages(agent.get("messages", []))
    last_assistant_idx: Optional[int] = None
    for idx, msg in enumerate(messages):
        if str(msg.get("role", "")).strip().lower() == "assistant":
            last_assistant_idx = idx
    if last_assistant_idx is None:
        return messages
    return messages[:last_assistant_idx]


def _token_length(model: ModelWrapper, text: str) -> int:
    encoded = model.tokenizer(
        str(text),
        return_tensors="pt",
        add_special_tokens=False,
    )
    return int(encoded["attention_mask"].sum().item())


def _to_legacy_past(past_kv: Optional[Tuple]) -> Optional[Tuple]:
    if past_kv is None:
        return None
    if Cache is not None and isinstance(past_kv, Cache):
        return past_kv.to_legacy_cache()
    return past_kv


def _slice_tensor_span(tensor: torch.Tensor, start: int, end: int) -> torch.Tensor:
    start = max(0, int(start))
    end = max(start, min(int(end), tensor.shape[-2]))
    return tensor[..., start:end, :].contiguous()


def _slice_past_key_values(
    past_kv: Optional[Tuple],
    start: int,
    end: int,
) -> Optional[Tuple]:
    legacy = _to_legacy_past(past_kv)
    if legacy is None or end <= start:
        return None
    sliced_layers = []
    for layer in legacy:
        if isinstance(layer, tuple):
            sliced_layers.append(tuple(_slice_tensor_span(t, start, end) for t in layer))
        elif torch.is_tensor(layer):
            sliced_layers.append(_slice_tensor_span(layer, start, end))
        else:
            sliced_layers.append(layer)
    return tuple(sliced_layers)


def _strip_system_block_from_prompt(prompt: str) -> str:
    text = str(prompt or "")
    patterns = [
        r"^\s*<\|system\|>\n[\s\S]*?\n</\|system\|>\n*",
        r"^\s*<\|im_start\|>system\n[\s\S]*?<\|im_end\|>\n*",
        r"^\s*<\|start_header_id\|>system<\|end_header_id\|>\s*[\s\S]*?<\|eot_id\|>\s*",
    ]
    for pattern in patterns:
        stripped = re.sub(pattern, "", text, count=1, flags=re.IGNORECASE)
        if stripped != text:
            return stripped
    return text


def _render_worker_prompt_from_messages(
    model: ModelWrapper,
    worker: Dict[str, Any],
    *,
    include_system: bool,
) -> str:
    messages = _worker_messages_before_final_turn(worker)
    if not include_system:
        messages = [
            msg for msg in messages
            if str(msg.get("role", "")).strip().lower() != "system"
        ]
    if not messages:
        return ""
    return model.render_chat(messages, add_generation_prompt=True)


def _worker_final_prompt_text(
    model: ModelWrapper,
    worker: Dict[str, Any],
    *,
    include_system: bool,
) -> str:
    if include_system:
        prompt = _worker_last_prompt(worker)
        if prompt:
            return prompt
    prompt = _render_worker_prompt_from_messages(
        model,
        worker,
        include_system=include_system,
    )
    if prompt:
        return prompt
    if include_system:
        return _worker_last_prompt(worker)
    return _strip_system_block_from_prompt(_worker_last_prompt(worker))


def _worker_turn_prompt_outputs(
    model: ModelWrapper,
    worker: Dict[str, Any],
) -> List[Dict[str, str]]:
    step_prompts = [str(x) for x in (worker.get("step_prompts", []) or [])]
    assistant_texts = _worker_turn_outputs(worker)
    if not assistant_texts:
        return []

    prompt_outputs: List[Dict[str, str]] = []
    for idx, output_text in enumerate(assistant_texts):
        prompt_text = step_prompts[idx] if idx < len(step_prompts) else ""
        if not prompt_text and idx == len(assistant_texts) - 1:
            prompt_text = _worker_last_prompt(worker)
        prompt_outputs.append(
            {
                "prompt_text": prompt_text,
                "output_text": output_text,
            }
        )
    return prompt_outputs


def _worker_full_prompt_output_without_system_spec(
    model: ModelWrapper,
    worker: Dict[str, Any],
    *,
    output_text: str,
) -> Optional[Dict[str, Any]]:
    full_prompt = _worker_final_prompt_text(model, worker, include_system=True)
    if not full_prompt and not output_text.strip():
        return None

    stripped_prompt = _strip_system_block_from_prompt(full_prompt)
    system_prefix_text = ""
    if stripped_prompt != full_prompt and full_prompt.endswith(stripped_prompt):
        system_prefix_text = full_prompt[: len(full_prompt) - len(stripped_prompt)]
    else:
        rendered_without_system = _worker_final_prompt_text(model, worker, include_system=False)
        stripped_prompt = rendered_without_system
        if rendered_without_system and full_prompt.endswith(rendered_without_system):
            system_prefix_text = full_prompt[: len(full_prompt) - len(rendered_without_system)]
        else:
            prefix_messages = _worker_messages_before_final_turn(worker)
            if prefix_messages and str(prefix_messages[0].get("role", "")).strip().lower() == "system":
                system_prefix_text = model.render_chat([prefix_messages[0]], add_generation_prompt=False)

    combined_full_text = f"{full_prompt}{output_text}"
    combined_display_text = f"{stripped_prompt}{output_text}"
    strip_prefix_tokens = _token_length(model, system_prefix_text) if system_prefix_text else 0
    return {
        "kind": "full_text_strip_prefix",
        "text": combined_full_text,
        "display_text": combined_display_text,
        "strip_prefix_tokens": strip_prefix_tokens,
    }


def _worker_turn_outputs(worker: Dict[str, Any]) -> List[str]:
    outputs: List[str] = []
    for msg in worker.get("messages", []) or []:
        if str(msg.get("role", "")).strip().lower() != "assistant":
            continue
        content = str(msg.get("content", ""))
        if content.strip():
            outputs.append(content)
    if outputs:
        return outputs
    fallback = _worker_output_text(worker)
    return [fallback] if fallback.strip() else []


def _worker_memory_specs(
    model: ModelWrapper,
    worker: Dict[str, Any],
    *,
    mode: str,
) -> List[Dict[str, Any]]:
    resolved_mode = str(mode or "final_output").strip().lower()
    output_text = _worker_output_text(worker)

    if resolved_mode == "final_output":
        if not output_text.strip():
            return []
        prompt_text = _worker_final_prompt_text(model, worker, include_system=True)
        if prompt_text.strip():
            return [
                {
                    "kind": "output_only",
                    "prompt_text": prompt_text,
                    "output_text": output_text,
                }
            ]
        return [{"kind": "full_text", "text": output_text}]

    if resolved_mode == "full_prompt_output":
        prompt_text = _worker_final_prompt_text(model, worker, include_system=True)
        combined = f"{prompt_text}{output_text}"
        return [{"kind": "full_text", "text": combined}] if combined.strip() else []

    if resolved_mode == "each_turn_output":
        return [
            {
                "kind": "output_only",
                "prompt_text": str(turn.get("prompt_text", "")),
                "output_text": str(turn.get("output_text", "")),
            }
            for turn in _worker_turn_prompt_outputs(model, worker)
            if str(turn.get("output_text", "")).strip()
        ]

    if resolved_mode == "full_prompt_output_no_system":
        spec = _worker_full_prompt_output_without_system_spec(
            model,
            worker,
            output_text=output_text,
        )
        return [spec] if spec is not None and str(spec.get("display_text", "")).strip() else []

    raise ValueError(
        f"Unsupported worker memory mode {resolved_mode!r}. "
        f"Expected one of: {', '.join(SOURCE_WORKER_MEMORY_MODES)}"
    )


def _worker_memory_text_blocks(
    model: ModelWrapper,
    worker: Dict[str, Any],
    *,
    mode: str,
) -> List[str]:
    blocks: List[str] = []
    for spec in _worker_memory_specs(model, worker, mode=mode):
        if spec.get("kind") == "output_only":
            text = str(spec.get("output_text", ""))
        else:
            text = str(spec.get("display_text", spec.get("text", "")))
        if text.strip():
            blocks.append(text)
    return blocks


def _extract_source_workers(row: Dict[str, Any]) -> List[Dict[str, Any]]:
    workers: List[Dict[str, Any]] = []
    for agent in row.get("agents", []) or []:
        role = str(agent.get("role", "")).strip().lower()
        if role == "judger":
            continue
        workers.append(clone_source_agent_trace(agent))
    return workers


def build_sanity_items_from_preds(
    source_preds_path: Path,
    *,
    max_samples: int = -1,
) -> List[Dict[str, Any]]:
    rows = _load_jsonl_rows(source_preds_path)
    if max_samples >= 0:
        rows = rows[:max_samples]

    items: List[Dict[str, Any]] = []
    for row_idx, row in enumerate(rows, start=1):
        workers = _extract_source_workers(row)
        if not workers:
            raise ValueError(
                f"Source row {row_idx} in {source_preds_path} has no non-judger worker traces."
            )

        row_files_raw = row.get("files", [])
        files: List[str] = []
        if isinstance(row_files_raw, list):
            files.extend(str(x) for x in row_files_raw)
        elif isinstance(row_files_raw, str) and row_files_raw.strip():
            files.append(row_files_raw)
        if not files:
            for worker in workers:
                files.extend(_extract_files_from_prompt(str(worker.get("input", ""))))
        items.append(
            {
                "question": row.get("question", ""),
                "gold": str(row.get("gold", "")),
                "solution": row.get("solution", row.get("gold", "")),
                "files": _normalize_files(files),
                "source_workers": workers,
                "source_row_index": row_idx,
                "source_prediction": row.get("prediction", ""),
                "source_correct": bool(row.get("correct", False)),
            }
        )
    return items


class FixedWorkerTextMASToolCallingMethod(_ToolCallingMethodBase):
    method_name = "sanity_text_mas_toolcall"

    def _system_prompt(self) -> str:
        return prompts.build_tool_system_prompt(self.tools.schemas())

    @torch.no_grad()
    def run_batch(self, items: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        if len(items) > self.generate_bs:
            raise ValueError("Batch size exceeds configured generate_bs")

        results: List[Dict[str, Any]] = []
        system_prompt = self._system_prompt()
        prompt_mod = prompts

        for sample_idx, item in enumerate(items, start=1):
            traces: List[Dict[str, Any]] = []
            context_parts: List[str] = []
            memory_mode = _source_worker_memory_mode(self.args)
            for worker in item.get("source_workers", []) or []:
                traces.append(clone_source_agent_trace(worker))
                for block_text in _worker_memory_text_blocks(
                    self.model,
                    worker,
                    mode=memory_mode,
                ):
                    context_parts.append(f"[{worker.get('name', 'Agent')}]:\n{block_text}\n\n")

            prompt = prompt_mod.build_textmas_role_prompt(
                role="judger",
                question=str(item.get("question", "")),
                context="".join(context_parts),
                files=list(item.get("files", [])),
            )
            run = self._new_react_agent(
                verbose_prefix=f"[sanity_text_mas][sample {sample_idx}][judger] "
            ).run(
                system_prompt=system_prompt,
                user_prompt=prompt,
                measure_first_step_ttft=True,
            )

            traces.append(
                {
                    "name": "Judger",
                    "role": "judger",
                    "input": prompt,
                    "output": run.final_output,
                    "final_answer": run.final_answer,
                    "tool_calls": self._serialize_tool_calls(run.tool_calls),
                    "messages": self._serialize_messages(run.messages),
                    "step_prompts": self._serialize_text_list(run.step_prompts),
                }
            )

            pred = extract_answer_text(run.final_answer or run.final_output)
            results.append(
                self._build_result(
                    item=item,
                    pred_text=pred,
                    raw_text=run.final_output,
                    agents=traces,
                    ttft_sec=run.first_step_ttft_sec,
                )
            )

        return results

    def run_item(self, item: Dict[str, Any]) -> Dict[str, Any]:
        return self.run_batch([item])[0]


class FixedWorkerParallelKVToolCallingMethod(ParallelKVToolCallingMethod):
    method_name = "sanity_parallel_kv_toolcall"

    def _system_prompt(self) -> str:
        return prompts.build_tool_system_prompt(self.tools.schemas())

    def _encode_exact_text(self, text: str) -> Tuple[torch.Tensor, torch.Tensor]:
        encoded = self.model.tokenizer(
            str(text),
            return_tensors="pt",
            add_special_tokens=False,
        )
        return encoded["input_ids"].to(self.model.device), encoded["attention_mask"].to(self.model.device)

    def _merge_worker_reconstructed_entries(
        self,
        worker: Dict[str, Any],
        entries: List[Dict[str, Any]],
    ) -> List[Dict[str, Any]]:
        valid_pasts: List[Optional[Tuple]] = []
        first_prompt_len: Optional[int] = None
        max_position_end = 0
        for entry in entries:
            past = self._ensure_cache_object(entry.get("past"))
            if past is None or _past_length(past) <= 0:
                continue
            prompt_lens = entry.get("prompt_lens")
            if prompt_lens is not None:
                prompt_len = int(prompt_lens.view(-1)[0].item())
                if first_prompt_len is None:
                    first_prompt_len = prompt_len
                # Preserve the original sequential turn spacing within one worker:
                # [prompt1][output1][prompt2][output2] -> shift every retained output
                # cache by the same first-turn prompt offset, so output1 starts at 0
                # while later outputs keep their original prompt/tool-response gaps.
                position_shift = -int(first_prompt_len)
                if position_shift != 0:
                    past = self.model.shift_rope_past_key_values(
                        past,
                        torch.tensor([position_shift], device=self.model.device, dtype=torch.long),
                    )
                output_len = int(entry.get("output_len", _past_length(past)) or 0)
                max_position_end = max(
                    max_position_end,
                    max(0, prompt_len - int(first_prompt_len)) + output_len,
                )
            else:
                if first_prompt_len is None:
                    first_prompt_len = 0
                max_position_end = max(
                    max_position_end,
                    int(entry.get("position_span", _past_length(past)) or 0),
                )
            past = self._ensure_cache_object(past)
            if past is None or _past_length(past) <= 0:
                continue
            valid_pasts.append(past)

        merged = self._merge_parallel_past(valid_pasts)
        merged = self._ensure_cache_object(merged)
        if merged is None or _past_length(merged) <= 0:
            return []
        position_span = max(max_position_end, _past_length(merged))
        return [
            {
                "past": merged,
                "role": str(worker.get("role", "")),
                "prompt_lens": None,
                "position_span": position_span,
            }
        ]

    def _reconstruct_worker_past_entries(self, worker: Dict[str, Any]) -> List[Dict[str, Any]]:
        entries: List[Dict[str, Any]] = []
        memory_mode = _source_worker_memory_mode(self.args)
        for spec in _worker_memory_specs(
            self.model,
            worker,
            mode=memory_mode,
        ):
            if spec.get("kind") == "output_only":
                prompt_text = str(spec.get("prompt_text", ""))
                output_text = str(spec.get("output_text", ""))
                if not prompt_text:
                    raise ValueError(
                        f"Missing step_prompts for source worker {worker.get('role', '<unknown>')} "
                        "while reconstructing output-only cache."
                    )
                if not output_text.strip():
                    continue

                prompt_ids, prompt_mask = self._encode_exact_text(prompt_text)
                output_ids, _ = self._encode_exact_text(output_text)
                generated_len = int(output_ids.shape[-1])
                if generated_len <= 0:
                    continue

                full_ids = torch.cat([prompt_ids, output_ids], dim=-1)
                full_mask = torch.cat(
                    [
                        prompt_mask,
                        torch.ones(
                            (prompt_mask.shape[0], generated_len),
                            dtype=prompt_mask.dtype,
                            device=prompt_mask.device,
                        ),
                    ],
                    dim=-1,
                )
                outputs = self.model.model(
                    input_ids=full_ids,
                    attention_mask=full_mask,
                    use_cache=True,
                    return_dict=True,
                )
                past = self._ensure_cache_object(outputs.past_key_values)
                past = self._truncate_past(past, generated_len)
                past = self._ensure_cache_object(past)
                if past is None or _past_length(past) <= 0:
                    continue

                entries.append(
                    {
                        "past": past,
                        "role": str(worker.get("role", "")),
                        "prompt_lens": prompt_mask.sum(dim=1).to(self.model.device),
                        "output_len": generated_len,
                    }
                )
                continue

            if spec.get("kind") == "full_text":
                text = str(spec.get("text", ""))
                if not text.strip():
                    continue
                full_ids, full_mask = self._encode_exact_text(text)
                outputs = self.model.model(
                    input_ids=full_ids,
                    attention_mask=full_mask,
                    use_cache=True,
                    return_dict=True,
                )
                past = self._ensure_cache_object(outputs.past_key_values)
                past = self._ensure_cache_object(past)
                if past is None or _past_length(past) <= 0:
                    continue
                entries.append(
                    {
                        "past": past,
                        "role": str(worker.get("role", "")),
                        "prompt_lens": None,
                    }
                )
                continue

            if spec.get("kind") == "full_text_strip_prefix":
                text = str(spec.get("text", ""))
                if not text.strip():
                    continue
                full_ids, full_mask = self._encode_exact_text(text)
                outputs = self.model.model(
                    input_ids=full_ids,
                    attention_mask=full_mask,
                    use_cache=True,
                    return_dict=True,
                )
                past = _slice_past_key_values(
                    outputs.past_key_values,
                    int(spec.get("strip_prefix_tokens", 0) or 0),
                    int(full_mask.sum().item()),
                )
                past = self._ensure_cache_object(past)
                strip_prefix_tokens = int(spec.get("strip_prefix_tokens", 0) or 0)
                if past is not None and strip_prefix_tokens > 0:
                    past = self.model.shift_rope_past_key_values(
                        past,
                        torch.tensor([-strip_prefix_tokens], device=self.model.device, dtype=torch.long),
                    )
                    past = self._ensure_cache_object(past)
                if past is None or _past_length(past) <= 0:
                    continue
                entries.append(
                    {
                        "past": past,
                        "role": str(worker.get("role", "")),
                        "prompt_lens": None,
                    }
                )
        return entries

    def _collect_parallel_agent_past(
        self,
        items: List[Dict[str, Any]],
        *,
        include_traces: bool = False,
    ):
        self._ensure_hf_backend()
        if len(items) != 1:
            raise ValueError(
                "Sanity parallel_kv reconstructs worker caches per sample. "
                "run_batch should route this method through singleton items."
            )

        item = items[0]
        traces: Optional[List[List[Dict[str, Any]]]] = [[]] if include_traces else None
        parallel_agent_past: List[Dict[str, Any]] = []
        model_obj = self.model.model
        was_training = bool(model_obj.training)

        with self._disable_lora_for_non_judger():
            model_obj.eval()
            try:
                for worker in item.get("source_workers", []) or []:
                    if traces is not None:
                        traces[0].append(clone_source_agent_trace(worker))
                    reconstructed_entries = self._reconstruct_worker_past_entries(worker)
                    if reconstructed_entries:
                        if (
                            _source_worker_memory_mode(self.args) == "each_turn_output"
                            and len(reconstructed_entries) > 1
                        ):
                            reconstructed_entries = self._merge_worker_reconstructed_entries(
                                worker,
                                reconstructed_entries,
                            )
                        parallel_agent_past.extend(reconstructed_entries)
            finally:
                if was_training:
                    model_obj.train()
                else:
                    model_obj.eval()

        return parallel_agent_past, traces

    @torch.no_grad()
    def run_batch(self, items: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        if len(items) > self.generate_bs:
            raise ValueError("Batch size exceeds configured generate_bs")
        results: List[Dict[str, Any]] = []
        system_prompt = self._system_prompt()
        prompt_mod = prompts

        for sample_idx, item in enumerate(items, start=1):
            parallel_agent_past, traces = self._collect_parallel_agent_past(
                [item], include_traces=True
            )
            if traces is None:
                traces = [[]]

            dummy_mask = torch.ones((1, 1), dtype=torch.long, device=self.model.device)
            past_for_decoding, position_offset, cache_prepare_sec = self._prepare_parallel_judger_past_with_timing(
                parallel_agent_past, judger_mask=dummy_mask
            )

            user_prompt = prompt_mod.build_parallel_kv_judger_prompt(
                str(item.get("question", "")),
                list(item.get("files", [])),
            )
            run = self._new_react_agent(
                verbose_prefix=f"[sanity_parallel_kv][sample {sample_idx}][judger] "
            ).run(
                system_prompt=system_prompt,
                user_prompt=user_prompt,
                persistent_prefix_past_key_values=past_for_decoding,
                persistent_prefix_position_offset=position_offset,
                measure_first_step_ttft=True,
            )
            total_ttft_sec = self._add_prepare_time_to_ttft(
                run.first_step_ttft_sec,
                cache_prepare_sec,
            )

            pred = extract_answer_text(run.final_answer)
            gold = str(item.get("gold", ""))
            ok = free_form_exact_match(pred, gold)

            traces[0].append(
                {
                    "name": "Judger",
                    "role": "judger",
                    "input": user_prompt,
                    "output": run.final_output,
                    "final_answer": run.final_answer,
                    "tool_calls": _ToolCallingMethodBase._serialize_tool_calls(run.tool_calls),
                    "messages": _ToolCallingMethodBase._serialize_messages(run.messages),
                    "step_prompts": _ToolCallingMethodBase._serialize_text_list(run.step_prompts),
                }
            )

            results.append(
                {
                    "question": item.get("question", ""),
                    "gold": gold,
                    "solution": item.get("solution", gold),
                    "prediction": pred,
                    "raw_prediction": run.final_output,
                    "agents": traces[0],
                    "ttft_sec": total_ttft_sec,
                    "cache_prepare_sec": cache_prepare_sec,
                    "correct": ok,
                }
            )
        return results


def build_sanity_method(
    model: ModelWrapper,
    args: argparse.Namespace,
    tools: ToolRegistry,
):
    common = dict(
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        top_p=args.top_p,
        generate_bs=args.generate_bs,
        max_tool_steps=args.max_tool_steps,
        tools=tools,
        args=args,
    )
    if args.method == "text_mas":
        return FixedWorkerTextMASToolCallingMethod(model, **common)
    if args.method == "parallel_kv":
        return FixedWorkerParallelKVToolCallingMethod(
            model,
            judger_max_new_tokens=args.max_new_tokens,
            temperature=args.temperature,
            top_p=args.top_p,
            generate_bs=args.generate_bs,
            max_tool_steps=args.max_tool_steps,
            tools=tools,
            args=args,
        )
    raise ValueError(f"Unsupported sanity method: {args.method}")

import argparse
import json
import math
import os
import time
from contextlib import contextmanager
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint

from parallel_synthesis.utils.common import canonicalize_task_name
from parallel_synthesis.models import (
    ModelWrapper,
    _past_length,
    _synchronize_timing_device,
    align_model_vocab_to_adapter,
)
from parallel_synthesis.prompts import (
    CONTEXT_QA_TASKS,
    build_parallel_kv_prompt,
    PRETRAINING_TASKS,
)
from parallel_synthesis.utils.utils import (
    extract_after_think,
    extract_gsm8k_answer,
    extract_markdown_python_block,
    normalize_answer,
    run_with_timeout,
)

from . import default_agents

try:
    from peft import LoraConfig, PeftModel, get_peft_model
except ImportError:
    LoraConfig = None
    PeftModel = None
    get_peft_model = None

try:
    from transformers.cache_utils import Cache
except ImportError:
    Cache = None
try:
    from transformers.cache_utils import DynamicCache
except ImportError:
    DynamicCache = None


class _ReadOnlyDynamicCache(DynamicCache if DynamicCache is not None else object):
    """DynamicCache view whose updates concatenate without mutating stored past."""

    def update(
        self,
        key_states: torch.Tensor,
        value_states: torch.Tensor,
        layer_idx: int,
        cache_kwargs: Optional[Dict[str, Any]] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        _ = cache_kwargs
        layer = self.layers[layer_idx]
        return (
            torch.cat((layer.keys, key_states), dim=-2),
            torch.cat((layer.values, value_states), dim=-2),
        )


@contextmanager
def _cache_preserving_layer_checkpointing(backbone: torch.nn.Module):
    layers = getattr(backbone, "layers", None)
    if layers is None:
        raise ValueError("Cache-preserving checkpointing requires decoder layers.")
    original_forwards = []
    try:
        for layer in layers:
            original_forward = layer.forward
            original_forwards.append((layer, original_forward))

            def checkpointed_forward(
                *args,
                __original_forward=original_forward,
                **kwargs,
            ):
                if not args:
                    raise RuntimeError("Decoder layer did not receive hidden states.")
                hidden_states, *remaining_args = args

                def run_layer(checkpoint_hidden_states: torch.Tensor):
                    return __original_forward(
                        checkpoint_hidden_states,
                        *remaining_args,
                        **kwargs,
                    )

                return checkpoint(
                    run_layer,
                    hidden_states,
                    use_reentrant=False,
                )

            layer.forward = checkpointed_forward
        yield
    finally:
        for layer, original_forward in original_forwards:
            layer.forward = original_forward


def _single_example_target_position_loss(
    model: torch.nn.Module,
    *,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    position_ids: torch.Tensor,
    past_key_values: Optional[Any],
    labels: torch.Tensor,
    prompt_len: int,
    full_len: int,
    logit_chunk_size: int,
    cache_preserving_backbone_checkpoint: bool = False,
) -> torch.Tensor:
    """Compute exact completion loss with checkpointed vocabulary projections."""

    if input_ids.ndim != 2 or input_ids.shape[0] != 1:
        raise ValueError("Target-position loss requires a single-example batch.")
    if prompt_len <= 0 or full_len <= prompt_len or full_len > input_ids.shape[1]:
        raise ValueError(
            "Invalid completion boundary for target-position loss: "
            f"prompt_len={prompt_len}, full_len={full_len}, "
            f"input_len={input_ids.shape[1]}."
        )
    if logit_chunk_size <= 0:
        raise ValueError("logit_chunk_size must be positive.")

    # A causal logit at live-sequence position i predicts token i + 1.
    prediction_positions = torch.arange(
        prompt_len - 1,
        full_len - 1,
        dtype=torch.long,
        device=input_ids.device,
    )
    get_base_model = getattr(model, "get_base_model", None)
    causal_lm = get_base_model() if callable(get_base_model) else model
    backbone = getattr(causal_lm, "model", None)
    lm_head = getattr(causal_lm, "lm_head", None)
    if not callable(backbone) or not callable(lm_head):
        raise RuntimeError(
            "Target-position loss requires a causal LM with model and lm_head modules."
        )

    if cache_preserving_backbone_checkpoint:
        if past_key_values is None or DynamicCache is None:
            raise ValueError(
                "Cache-preserving backbone checkpointing requires a DynamicCache."
            )
        legacy_cache = (
            past_key_values.to_legacy_cache()
            if hasattr(past_key_values, "to_legacy_cache")
            else past_key_values
        )
        read_only_cache = _ReadOnlyDynamicCache(ddp_cache_data=legacy_cache)
        with _cache_preserving_layer_checkpointing(backbone):
            outputs = backbone(
                input_ids=input_ids[:, :full_len],
                attention_mask=attention_mask,
                position_ids=position_ids[:, :full_len],
                past_key_values=read_only_cache,
                use_cache=False,
                return_dict=True,
            )
        all_hidden_states = outputs.last_hidden_state
    else:
        outputs = backbone(
            input_ids=input_ids[:, :full_len],
            attention_mask=attention_mask,
            position_ids=position_ids[:, :full_len],
            past_key_values=past_key_values,
            use_cache=False,
            return_dict=True,
        )
        all_hidden_states = outputs.last_hidden_state
    prediction_hidden_states = all_hidden_states[:, prediction_positions, :]
    completion_labels = labels[:, prompt_len:full_len]
    expected_target_tokens = full_len - prompt_len
    if prediction_hidden_states.shape[:2] != (1, expected_target_tokens):
        raise RuntimeError(
            "Model returned an unexpected target-position hidden-state shape: "
            f"got={tuple(prediction_hidden_states.shape)}, "
            f"expected=(1, {expected_target_tokens}, hidden_size)."
        )
    if bool((completion_labels == -100).any().item()):
        raise RuntimeError("Supervised completion labels unexpectedly contain -100.")

    def chunk_loss(
        hidden_states: torch.Tensor,
        chunk_labels: torch.Tensor,
    ) -> torch.Tensor:
        logits = lm_head(hidden_states)
        return F.cross_entropy(
            logits.float().reshape(-1, logits.shape[-1]),
            chunk_labels.reshape(-1),
            reduction="sum",
        )

    loss_sum: Optional[torch.Tensor] = None
    for start in range(0, expected_target_tokens, logit_chunk_size):
        end = min(start + logit_chunk_size, expected_target_tokens)
        hidden_chunk = prediction_hidden_states[:, start:end, :]
        label_chunk = completion_labels[:, start:end]
        if torch.is_grad_enabled():
            one_chunk_loss = checkpoint(
                chunk_loss,
                hidden_chunk,
                label_chunk,
                use_reentrant=False,
            )
        else:
            one_chunk_loss = chunk_loss(hidden_chunk, label_chunk)
        loss_sum = one_chunk_loss if loss_sum is None else loss_sum + one_chunk_loss

    if loss_sum is None:
        raise RuntimeError("Target-position loss produced no supervised chunks.")
    return loss_sum / expected_target_tokens


MAPPER_METADATA_FEATURE_MODES = ("both", "length", "num_agents", "none")


def _normalize_mapper_metadata_features(value: str) -> str:
    mode = str(value or "both").strip().lower()
    if mode not in MAPPER_METADATA_FEATURE_MODES:
        raise ValueError(
            f"Unknown mapper metadata feature mode {value!r}; "
            f"expected one of {MAPPER_METADATA_FEATURE_MODES}."
        )
    return mode


class LengthAwareAffineKVMapper(nn.Module):
    """Learns a lightweight, optionally metadata-conditioned affine KV map.

    Layer position is always supplied because it identifies which transformer
    layer is being mapped. ``metadata_features`` controls only the two runtime
    cache descriptors used by the original mapper: branch length and worker
    count. The default ``both`` retains the original three-input architecture
    and is therefore compatible with existing checkpoints.
    """

    def __init__(self, hidden_dim: int = 32, metadata_features: str = "both") -> None:
        super().__init__()
        self.hidden_dim = int(hidden_dim)
        self.metadata_features = _normalize_mapper_metadata_features(metadata_features)
        input_dim = 1
        if self.metadata_features in {"both", "length"}:
            input_dim += 1
        if self.metadata_features in {"both", "num_agents"}:
            input_dim += 1
        self.net = nn.Sequential(
            nn.Linear(input_dim, self.hidden_dim),
            nn.SiLU(),
            nn.Linear(self.hidden_dim, 4),
        )
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)

    def _compute_coeffs(
        self,
        length: int,
        num_agents: int,
        layer_idx: int,
        total_layers: int,
        *,
        device: torch.device,
        dtype: torch.dtype,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        layer_position = 0.0 if total_layers <= 1 else float(layer_idx) / float(total_layers - 1)
        features = []
        if self.metadata_features in {"both", "length"}:
            features.append(math.log1p(float(max(length, 1))))
        if self.metadata_features in {"both", "num_agents"}:
            features.append(math.log1p(float(max(num_agents, 1))))
        features.append(layer_position)
        feature_tensor = torch.tensor(
            features,
            dtype=torch.float32,
            device=device,
        )
        raw = self.net(feature_tensor.unsqueeze(0)).squeeze(0)
        alpha_k = (1.0 + 0.1 * torch.tanh(raw[0])).to(dtype=dtype, device=device)
        beta_k = (0.1 * torch.tanh(raw[1])).to(dtype=dtype, device=device)
        alpha_v = (1.0 + 0.1 * torch.tanh(raw[2])).to(dtype=dtype, device=device)
        beta_v = (0.1 * torch.tanh(raw[3])).to(dtype=dtype, device=device)
        return alpha_k, beta_k, alpha_v, beta_v

    def map_past(self, past_kv: Optional[Tuple], *, length: int, num_agents: int) -> Optional[Tuple]:
        if past_kv is None:
            return None

        if Cache is not None and isinstance(past_kv, Cache):
            total_layers = len(past_kv.layers)
            for layer_idx, layer in enumerate(past_kv.layers):
                if not hasattr(layer, "keys") or layer.keys is None:
                    continue
                key = layer.keys
                value = layer.values
                alpha_k, beta_k, alpha_v, beta_v = self._compute_coeffs(
                    length=length,
                    num_agents=num_agents,
                    layer_idx=layer_idx,
                    total_layers=total_layers,
                    device=key.device,
                    dtype=key.dtype,
                )
                layer.keys = (alpha_k * key + beta_k).contiguous()
                layer.values = (alpha_v * value + beta_v).contiguous()
            return past_kv

        mapped_layers = []
        total_layers = len(past_kv)
        for layer_idx, layer in enumerate(past_kv):
            if not isinstance(layer, tuple):
                mapped_layers.append(layer)
                continue
            key, value = layer
            alpha_k, beta_k, alpha_v, beta_v = self._compute_coeffs(
                length=length,
                num_agents=num_agents,
                layer_idx=layer_idx,
                total_layers=total_layers,
                device=key.device,
                dtype=key.dtype,
            )
            mapped_layers.append((alpha_k * key + beta_k, alpha_v * value + beta_v))
        return tuple(mapped_layers)


class ParallelKV:
    """
    Shared ParallelKV implementation used across the repo.

    The cache pipeline is intentionally simple:
    1. collect per-agent caches,
    2. optionally apply the cache mapper,
    3. concatenate caches,
    4. decode/train the judger against that concatenated cache.
    """

    DEFAULT_LORA_TARGET_MODULES = ["q_proj", "k_proj", "v_proj", "o_proj"]

    def __init__(
        self,
        model: ModelWrapper,
        *,
        judger_max_new_tokens: int = 256,
        temperature: float = 0.6,
        top_p: float = 0.95,
        generate_bs: int = 1,
        args: argparse.Namespace = None,
    ) -> None:
        self.args = args
        self.model = model
        self.judger_max_new_tokens = judger_max_new_tokens
        self.temperature = temperature
        self.top_p = top_p
        self.generate_bs = max(1, int(generate_bs))
        self.default_num_parallel_agents = max(
            1,
            int(getattr(args, "parallel_kv_num_parallel_agents", 3) if args else 3),
        )
        self.agents = default_agents(self.default_num_parallel_agents)
        self.method_name = "parallel_kv"
        self.task = canonicalize_task_name(getattr(args, "task", "")) if args else ""

        self.train_parallel_kv = bool(getattr(args, "train_parallel_kv", False)) if args else False
        self.parallel_kv_enable_affine_map = bool(
            getattr(args, "parallel_kv_enable_affine_map", False)
        ) if args else False
        self.parallel_kv_mapper_hidden_dim = int(
            getattr(args, "parallel_kv_mapper_hidden_dim", 32)
        ) if args else 32
        self.parallel_kv_mapper_metadata_features = _normalize_mapper_metadata_features(
            getattr(args, "parallel_kv_mapper_metadata_features", "both") if args else "both"
        )
        self.parallel_kv_enable_judger_lora = bool(
            getattr(args, "parallel_kv_enable_judger_lora", False)
        ) if args else False
        self.parallel_kv_lora_r = int(getattr(args, "parallel_kv_lora_r", 16)) if args else 16
        self.parallel_kv_lora_alpha = int(getattr(args, "parallel_kv_lora_alpha", 32)) if args else 32
        self.parallel_kv_lora_dropout = float(
            getattr(args, "parallel_kv_lora_dropout", 0.0)
        ) if args else 0.0
        self.parallel_kv_lora_target_modules = list(self.DEFAULT_LORA_TARGET_MODULES)
        self.parallel_kv_target_position_logits = bool(
            getattr(args, "parallel_kv_target_position_logits", False)
        ) if args else False
        self.parallel_kv_target_position_logit_chunk_size = int(
            getattr(args, "parallel_kv_target_position_logit_chunk_size", 64)
        ) if args else 64
        self.parallel_kv_cache_preserving_backbone_checkpoint = bool(
            getattr(args, "parallel_kv_cache_preserving_backbone_checkpoint", False)
        ) if args else False

        self._lora_initialized = False
        self.cache_mapper: Optional[LengthAwareAffineKVMapper] = None
        if self.parallel_kv_enable_affine_map:
            self.cache_mapper = LengthAwareAffineKVMapper(
                hidden_dim=self.parallel_kv_mapper_hidden_dim,
                metadata_features=self.parallel_kv_mapper_metadata_features,
            ).to(self.model.device)

        if (
            self.train_parallel_kv
            and not self.parallel_kv_enable_judger_lora
            and hasattr(self.model, "model")
        ):
            for param in self.model.model.parameters():
                param.requires_grad_(False)

    def _ensure_hf_backend(self) -> None:
        if not hasattr(self.model, "model"):
            raise RuntimeError("This clean ParallelKV path requires the HF backend; disable --use_vllm.")

    def _num_parallel_agents_for_items(self, items: List[Dict[str, Any]]) -> int:
        _ = items
        return self.default_num_parallel_agents

    def _agents_for_items(self, items: List[Dict[str, Any]]) -> List[Any]:
        return default_agents(self._num_parallel_agents_for_items(items))

    @staticmethod
    def _item_query_text(item: Dict[str, Any]) -> str:
        # For dataset-authored pretraining tasks, `query` already contains the
        # full instruction/prompt. For evaluation tasks such as GSM8K and
        # MedQA, `query` is absent, so use `question` and its task prompt.
        query = str(item.get("query", "")).strip()
        if query:
            return query
        return str(item.get("question", "")).strip()

    def _build_role_messages(self, items: List[Dict[str, Any]], role: str) -> List[List[Dict[str, str]]]:
        return [
            build_parallel_kv_prompt(
                role=role,
                question=self._item_query_text(item),
                context="",
                args=self.args,
            )
            for item in items
        ]

    def _render_role_prompts(
        self,
        batch_messages: List[List[Dict[str, str]]],
    ) -> List[str]:
        """Render role messages; raw-prompt add-ons may override this hook."""

        prompts, _, _, _ = self.model.prepare_chat_batch(
            batch_messages,
            add_generation_prompt=True,
        )
        return prompts

    def _worker_generation_cap(self) -> int:
        """Return the worker allowance; add-ons may override this hook."""

        return int(
            getattr(
                self.args,
                "max_new_tokens",
                self.judger_max_new_tokens,
            )
        )

    def _wrap_prompts_for_generation(self, prompts: List[str]) -> List[str]:
        if bool(getattr(self.args, "think", False)):
            return [f"{prompt}<think>" for prompt in prompts]
        return prompts

    def _tokenize_prompt_texts(
        self,
        prompts: List[str],
    ) -> Tuple[torch.Tensor, torch.Tensor, List[List[str]]]:
        encoded = self.model.tokenizer(
            prompts,
            return_tensors="pt",
            padding=True,
            add_special_tokens=False,
        )
        input_ids = encoded["input_ids"].to(self.model.device)
        attention_mask = encoded["attention_mask"].to(self.model.device)
        tokens_batch: List[List[str]] = []
        for ids_row, mask_row in zip(input_ids, attention_mask):
            active_ids = ids_row[mask_row.bool()].tolist()
            tokens_batch.append(self.model.tokenizer.convert_ids_to_tokens(active_ids))
        return input_ids, attention_mask, tokens_batch

    @staticmethod
    def _slice_tensor(tensor: torch.Tensor, tokens_to_keep: int) -> torch.Tensor:
        if tokens_to_keep <= 0:
            return tensor[..., 0:0, :].contiguous()
        keep = min(tokens_to_keep, tensor.shape[-2])
        return tensor[..., -keep:, :].contiguous()

    def _truncate_past(self, past_kv: Optional[Tuple], tokens_to_keep: int) -> Optional[Tuple]:
        if past_kv is None or tokens_to_keep <= 0:
            return None
        if Cache is not None and isinstance(past_kv, Cache):
            for layer in past_kv.layers:
                if not hasattr(layer, "keys") or layer.keys is None:
                    continue
                if layer.keys.shape[-2] > tokens_to_keep:
                    layer.keys = layer.keys[..., -tokens_to_keep:, :].contiguous()
                    layer.values = layer.values[..., -tokens_to_keep:, :].contiguous()
            return past_kv
        trimmed_layers = []
        for layer in past_kv:
            if isinstance(layer, tuple):
                trimmed_layers.append(tuple(self._slice_tensor(t, tokens_to_keep) for t in layer))
            elif torch.is_tensor(layer):
                trimmed_layers.append(self._slice_tensor(layer, tokens_to_keep))
            else:
                trimmed_layers.append(layer)
        return tuple(trimmed_layers)

    @staticmethod
    def _ensure_cache_object(past_kv: Optional[Tuple]) -> Optional[Tuple]:
        if past_kv is None:
            return None
        if Cache is not None and isinstance(past_kv, Cache):
            return past_kv
        if DynamicCache is not None:
            try:
                return DynamicCache.from_legacy_cache(past_kv)
            except Exception:
                return past_kv
        return past_kv

    @staticmethod
    def _merge_parallel_past_legacy(past_list: List[Tuple]) -> Tuple:
        merged_layers = []
        for layers in zip(*past_list):
            first = layers[0]
            if isinstance(first, tuple):
                merged_layers.append(tuple(torch.cat(parts, dim=-2) for parts in zip(*layers)))
            elif torch.is_tensor(first):
                merged_layers.append(torch.cat(list(layers), dim=-2))
            else:
                merged_layers.append(first)
        return tuple(merged_layers)

    def _merge_parallel_past(self, past_list: List[Optional[Tuple]]) -> Optional[Tuple]:
        valid = [p for p in past_list if p is not None and _past_length(p) > 0]
        if not valid:
            return None
        if Cache is not None and isinstance(valid[0], Cache):
            return valid[0].__class__.from_legacy_cache(
                self._merge_parallel_past_legacy([p.to_legacy_cache() for p in valid])
            )
        return self._merge_parallel_past_legacy(valid)

    def _ensure_judger_lora(self) -> None:
        if self._lora_initialized:
            return
        self._ensure_hf_backend()
        if get_peft_model is None or LoraConfig is None:
            raise ImportError("peft is required for judger LoRA. Install it with `pip install peft`.")
        lora_config = LoraConfig(
            r=self.parallel_kv_lora_r,
            lora_alpha=self.parallel_kv_lora_alpha,
            lora_dropout=self.parallel_kv_lora_dropout,
            target_modules=self.parallel_kv_lora_target_modules,
            task_type="CAUSAL_LM",
            bias="none",
        )
        self.model.model = get_peft_model(self.model.model, lora_config)
        self._lora_initialized = True

    @contextmanager
    def _disable_lora_for_non_judger(self):
        model = getattr(self.model, "model", None)
        if (
            not self.parallel_kv_enable_judger_lora
            or not self._lora_initialized
            or PeftModel is None
            or model is None
            or not isinstance(model, PeftModel)
        ):
            yield
            return

        disable_adapter = getattr(model, "disable_adapter", None)
        if not callable(disable_adapter):
            raise RuntimeError(
                "Judger-only LoRA requires a PEFT model that supports disable_adapter()."
            )
        with disable_adapter():
            yield

    @staticmethod
    def _entry_position_span(entry: Dict[str, Any], past_kv: Optional[Tuple]) -> int:
        if past_kv is None:
            return 0
        position_span = entry.get("position_span")
        if position_span is None:
            return _past_length(past_kv)
        return max(0, int(position_span))

    def _map_cache_if_enabled(
        self,
        past_kv: Optional[Tuple],
        *,
        num_agents: int,
        length: Optional[int] = None,
    ) -> Optional[Tuple]:
        if not self.parallel_kv_enable_affine_map or self.cache_mapper is None or past_kv is None:
            return past_kv
        mapped = self.cache_mapper.map_past(
            past_kv,
            length=max(0, int(length if length is not None else _past_length(past_kv))),
            num_agents=max(1, num_agents),
        )
        return mapped

    def _prepare_parallel_judger_past(
        self,
        parallel_agent_past: List[Dict[str, Any]],
        *,
        judger_mask: torch.Tensor,
    ) -> Tuple[Optional[Tuple], torch.Tensor]:
        batch_size = judger_mask.shape[0]
        if not parallel_agent_past:
            return None, torch.zeros(batch_size, dtype=torch.long, device=self.model.device)

        num_agents = len(parallel_agent_past)
        prepared: List[Optional[Tuple]] = []
        max_agent_len = 0
        for entry in parallel_agent_past:
            past = self._ensure_cache_object(entry.get("past"))
            if past is None or _past_length(past) <= 0:
                continue

            prompt_lens = entry.get("prompt_lens")
            if prompt_lens is not None:
                past = self.model.shift_rope_past_key_values(past, -prompt_lens.to(self.model.device))

            position_span = self._entry_position_span(entry, past)
            past = self._map_cache_if_enabled(
                past,
                num_agents=num_agents,
                length=position_span,
            )
            past = self._ensure_cache_object(past)
            prepared.append(past)
            max_agent_len = max(max_agent_len, position_span)

        merged = self._merge_parallel_past(prepared)
        offset = torch.full(
            (batch_size,),
            int(max_agent_len),
            dtype=torch.long,
            device=self.model.device,
        )
        return self._ensure_cache_object(merged), offset

    def _prepare_parallel_judger_past_with_timing(
        self,
        parallel_agent_past: List[Dict[str, Any]],
        *,
        judger_mask: torch.Tensor,
    ) -> Tuple[Optional[Tuple], torch.Tensor, float]:
        _synchronize_timing_device(self.model.device)
        start_time = time.perf_counter()
        past, offset = self._prepare_parallel_judger_past(
            parallel_agent_past,
            judger_mask=judger_mask,
        )
        _synchronize_timing_device(self.model.device)
        return past, offset, time.perf_counter() - start_time

    @staticmethod
    def _add_prepare_time_to_ttft(
        decode_ttft_sec: Optional[float],
        cache_prepare_sec: Optional[float],
    ) -> Optional[float]:
        if decode_ttft_sec is None:
            return None
        return float(decode_ttft_sec) + float(cache_prepare_sec or 0.0)

    def _build_judger_batch(
        self,
        items: List[Dict[str, Any]],
    ) -> Tuple[List[str], torch.Tensor, torch.Tensor, List[List[str]]]:
        batch_messages = [
            build_parallel_kv_prompt(
                role="judger",
                question=self._item_query_text(item),
                context="",
                args=self.args,
            )
            for item in items
        ]
        prompts = self._render_role_prompts(batch_messages)
        judger_prompts = self._wrap_prompts_for_generation(prompts)
        judger_ids, judger_mask, judger_tokens = self._tokenize_prompt_texts(judger_prompts)
        return judger_prompts, judger_ids, judger_mask, judger_tokens

    def _collect_parallel_agent_past(
        self,
        items: List[Dict[str, Any]],
        *,
        include_traces: bool = False,
    ) -> Tuple[List[Dict[str, Any]], Optional[List[List[Dict[str, Any]]]]]:
        self._ensure_hf_backend()
        batch_size = len(items)
        traces: Optional[List[List[Dict[str, Any]]]] = [[] for _ in range(batch_size)] if include_traces else None
        parallel_agent_past: List[Dict[str, Any]] = []
        model_obj = self.model.model
        was_training = bool(model_obj.training)
        with self._disable_lora_for_non_judger():
            model_obj.eval()
            try:
                for agent in self._agents_for_items(items):
                    if agent.role == "judger":
                        continue

                    batch_messages = self._build_role_messages(items, agent.role)
                    prompts = self._render_role_prompts(batch_messages)
                    wrapped_prompts = self._wrap_prompts_for_generation(prompts)
                    wrapped_ids, wrapped_mask, wrapped_tokens_batch = self._tokenize_prompt_texts(wrapped_prompts)
                    prompt_lens = [int(x) for x in wrapped_mask.sum(dim=1).tolist()]
                    generation_cap = self._worker_generation_cap()
                    try:
                        generated_batch, past_kv, generated_ids_list = self.model.generate_text_batch(
                            wrapped_ids,
                            wrapped_mask,
                            max_new_tokens=generation_cap,
                            temperature=self.temperature,
                            top_p=self.top_p,
                            past_key_values=None,
                            return_generated_ids=True,
                        )
                    except torch.OutOfMemoryError as exc:
                        self._raise_augmented_oom(
                            exc,
                            tag="parallel_kv_agent_oom",
                            details={
                                "task": self.task,
                                "role": agent.role,
                                "batch_size": batch_size,
                                "prompt_tokens_max": max(prompt_lens) if prompt_lens else 0,
                                "prompt_tokens_per_sample": prompt_lens,
                                "max_new_tokens": generation_cap,
                            },
                        )

                    generated_len = max((int(ids.shape[0]) for ids in generated_ids_list), default=0)
                    agent_output_past = self._truncate_past(past_kv, generated_len) if generated_len > 0 else None
                    if agent_output_past is not None and _past_length(agent_output_past) > 0:
                        parallel_agent_past.append(
                            {
                                "past": self._ensure_cache_object(agent_output_past),
                                "role": agent.role,
                                "prompt_lens": wrapped_mask.sum(dim=1).to(self.model.device),
                            }
                        )

                    if traces is not None:
                        for idx in range(batch_size):
                            mask = wrapped_mask[idx].bool()
                            traces[idx].append(
                                {
                                    "name": agent.name,
                                    "role": agent.role,
                                    "input": wrapped_prompts[idx],
                                    "input_ids": wrapped_ids[idx][mask].to("cpu").tolist(),
                                    "input_tokens": wrapped_tokens_batch[idx],
                                    "output": generated_batch[idx].strip(),
                                }
                            )
            finally:
                if was_training:
                    model_obj.train()
                else:
                    model_obj.eval()

        return parallel_agent_past, traces

    def _score_final_text(self, item: Dict[str, Any], raw_final_text: str) -> Tuple[Any, Any, Optional[bool]]:
        final_text = extract_after_think(raw_final_text)
        if self.task in {"mbppplus", "humanevalplus"}:
            pred = extract_markdown_python_block(final_text)
            gold = item.get("gold", "")
            if pred is None:
                return pred, gold, False
            ok, _ = run_with_timeout(pred + "\n" + gold, timeout=10)
            return pred, gold, ok

        if self.task in {"aime2024", "aime2025"}:
            pred = normalize_answer(extract_gsm8k_answer(final_text))
            gold = str(item.get("gold", "")).strip()
            if pred is None:
                return pred, gold, False
            try:
                return pred, gold, int(pred) == int(gold)
            except ValueError:
                return pred, gold, False

        if self.task in PRETRAINING_TASKS:
            pred = raw_final_text.strip()
            gold = self._supervised_answer_text(item)
            return pred, gold, None

        pred = normalize_answer(extract_gsm8k_answer(final_text))
        gold = item.get("gold", "")
        ok = (pred == gold) if (pred and gold) else False
        return pred, gold, ok

    def _build_results(
        self,
        items: List[Dict[str, Any]],
        agent_traces: List[List[Dict[str, Any]]],
        final_texts: List[str],
        *,
        batch_ttft_sec: Optional[float] = None,
        cache_prepare_sec: Optional[float] = None,
    ) -> List[Dict[str, Any]]:
        results: List[Dict[str, Any]] = []
        for idx, item in enumerate(items):
            pred, gold, ok = self._score_final_text(item, final_texts[idx])
            results.append(
                {
                    "question": self._item_query_text(item),
                    "gold": gold,
                    "solution": item.get("solution", item.get("gold", "")),
                    "prediction": pred,
                    "raw_prediction": final_texts[idx],
                    "agents": agent_traces[idx],
                    "ttft_sec": batch_ttft_sec,
                    "cache_prepare_sec": cache_prepare_sec,
                    "correct": ok,
                }
            )
        return results

    @torch.no_grad()
    def _run_batch_impl(self, items: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        parallel_agent_past, agent_traces = self._collect_parallel_agent_past(items, include_traces=True)
        if agent_traces is None:
            agent_traces = [[] for _ in range(len(items))]

        judger_prompts, judger_ids, judger_mask, judger_tokens = self._build_judger_batch(items)
        past_for_decoding, position_offset, cache_prepare_sec = self._prepare_parallel_judger_past_with_timing(
            parallel_agent_past,
            judger_mask=judger_mask,
        )
        generated_batch, _, timing_metrics = self.model.generate_text_batch_with_offset(
            judger_ids,
            judger_mask,
            max_new_tokens=self.judger_max_new_tokens,
            temperature=self.temperature,
            top_p=self.top_p,
            past_key_values=past_for_decoding,
            position_offset=position_offset,
            return_timings=True,
        )
        decode_ttft_sec = timing_metrics.get("ttft_sec")
        batch_ttft_sec = self._add_prepare_time_to_ttft(
            decode_ttft_sec,
            cache_prepare_sec,
        )

        final_texts = [text.strip() for text in generated_batch]
        for idx in range(len(items)):
            mask = judger_mask[idx].bool()
            agent_traces[idx].append(
                {
                    "name": "Judger",
                    "role": "judger",
                    "input": judger_prompts[idx],
                    "input_ids": judger_ids[idx][mask].to("cpu").tolist(),
                    "input_tokens": judger_tokens[idx],
                    "output": final_texts[idx],
                }
            )
        return self._build_results(
            items,
            agent_traces,
            final_texts,
            batch_ttft_sec=batch_ttft_sec,
            cache_prepare_sec=cache_prepare_sec,
        )

    @staticmethod
    def _supervised_answer_text(item: Dict[str, Any]) -> str:
        answer = str(item.get("answer", "")).strip()
        if answer:
            return answer
        return str(item.get("gold", "")).strip()

    @staticmethod
    def _final_eval_answer(item: Dict[str, Any]) -> str:
        final_answer = str(item.get("final_answer", "")).strip()
        if final_answer:
            return final_answer
        return str(item.get("gold", "")).strip()

    def _build_supervised_target(self, item: Dict[str, Any], *, prompt_text: str = "") -> str:
        if self.task in {"mbppplus", "humanevalplus"}:
            raise ValueError(
                "Clean ParallelKV training currently targets answer-selection and summarization tasks."
            )
        eos_token = str(getattr(self.model.tokenizer, "eos_token", "") or "")
        gold_text = str(item.get("gold", "")).strip()
        supervised_answer = self._supervised_answer_text(item)
        if eos_token and gold_text.endswith(eos_token):
            gold_text = gold_text[: -len(eos_token)].rstrip()
        if eos_token and supervised_answer.endswith(eos_token):
            supervised_answer = supervised_answer[: -len(eos_token)].rstrip()

        if self.task in PRETRAINING_TASKS:
            return "\n" + supervised_answer + eos_token

        if self.task in CONTEXT_QA_TASKS:
            if supervised_answer:
                return "\n" + supervised_answer + eos_token
            final_answer = self._final_eval_answer(item)
            return f"\n\\boxed{{{final_answer}}}" + eos_token

        answer = gold_text.upper() if self.task in {"gpqa", "medqa"} else gold_text
        return f"\n\\boxed{{{answer}}}" + eos_token

    @staticmethod
    def _raise_augmented_oom(exc: torch.OutOfMemoryError, *, tag: str, details: Dict[str, Any]) -> None:
        payload = {key: value for key, value in details.items() if value is not None}
        base_message = str(exc).strip() or "CUDA out of memory"
        raise torch.OutOfMemoryError(
            f"{base_message}\n[{tag}] {json.dumps(payload, ensure_ascii=False, sort_keys=True)}"
        ) from exc

    @staticmethod
    def _build_skip_train_result(
        *,
        reason: str,
        message: str,
        batch_size: int,
        **extra: Any,
    ) -> Dict[str, Any]:
        payload: Dict[str, Any] = {
            "skip": True,
            "skip_reason": reason,
            "skip_message": message,
            "batch_size": int(batch_size),
            "target_tokens": 0,
        }
        payload.update(extra)
        return payload

    def _compute_train_loss(
        self,
        items: List[Dict[str, Any]],
        parallel_agent_past: List[Dict[str, Any]],
    ) -> Dict[str, Any]:
        judger_prompts, judger_ids, judger_mask, _ = self._build_judger_batch(items)

        # Build full-sequence lengths before the expensive KV-cache merge so
        # that skip guards can fire without ever allocating the merged tensor.
        target_texts = [
            self._build_supervised_target(item, prompt_text=prompt)
            for item, prompt in zip(items, judger_prompts)
        ]
        full_texts = [f"{prompt}{target}" for prompt, target in zip(judger_prompts, target_texts)]
        full_encoded = self.model.tokenizer(
            full_texts,
            return_tensors="pt",
            padding=True,
            add_special_tokens=False,
        )
        prompt_lens_cpu = judger_mask.sum(dim=1).cpu()
        full_lens_cpu = full_encoded["attention_mask"].sum(dim=1)
        max_full_len = int(full_lens_cpu.max().item()) if full_lens_cpu.numel() > 0 else 0

        # Estimate merged past length as the sum of individual agent cache
        # lengths (agents are cat'd along the seq dim). This is exact when the
        # affine map preserves sequence length, which it does.
        estimated_past_len = sum(
            _past_length(entry["past"])
            for entry in parallel_agent_past
            if entry.get("past") is not None and _past_length(entry["past"]) > 0
        )

        live_tokens_limit = int(
            getattr(self.args, "parallel_kv_skip_train_if_live_tokens_exceed", -1)
        ) if self.args is not None else -1
        if live_tokens_limit > 0 and max_full_len > live_tokens_limit:
            return self._build_skip_train_result(
                reason="actual_live_tokens_exceeded",
                message=(
                    "Skipping batch because the live judger sequence exceeds the configured limit."
                ),
                batch_size=len(items),
                past_tokens=int(estimated_past_len),
                live_tokens=int(max_full_len),
                live_token_limit=int(live_tokens_limit),
                judger_prompt_tokens=int(prompt_lens_cpu.max().item()) if prompt_lens_cpu.numel() > 0 else 0,
                max_seq_tokens=int(max_full_len),
            )
        attention_area = int(max_full_len * (estimated_past_len + max_full_len))
        attention_area_limit = int(
            getattr(self.args, "parallel_kv_skip_train_if_attention_area_exceed", -1)
        ) if self.args is not None else -1
        if attention_area_limit > 0 and attention_area > attention_area_limit:
            return self._build_skip_train_result(
                reason="actual_attention_area_exceeded",
                message=(
                    "Skipping batch because the live-by-context attention area exceeds the configured limit."
                ),
                batch_size=len(items),
                past_tokens=int(estimated_past_len),
                live_tokens=int(max_full_len),
                judger_prompt_tokens=int(prompt_lens_cpu.max().item()) if prompt_lens_cpu.numel() > 0 else 0,
                max_seq_tokens=int(max_full_len),
                attention_area=int(attention_area),
                attention_area_limit=int(attention_area_limit),
            )
        estimated_total_tokens = estimated_past_len + max_full_len
        total_tokens_limit = int(
            getattr(self.args, "parallel_kv_skip_train_if_total_tokens_exceed", -1)
        ) if self.args is not None else -1
        if total_tokens_limit > 0 and estimated_total_tokens > total_tokens_limit:
            return self._build_skip_train_result(
                reason="actual_total_tokens_exceeded",
                message=(
                    "Skipping batch because actual training token load exceeds the configured limit."
                ),
                batch_size=len(items),
                past_tokens=int(estimated_past_len),
                max_seq_tokens=int(max_full_len),
                total_tokens=int(estimated_total_tokens),
                token_limit=int(total_tokens_limit),
            )

        # All skip checks passed — now materialise the merged KV cache.
        past_for_training, position_offset = self._prepare_parallel_judger_past(
            parallel_agent_past,
            judger_mask=judger_mask,
        )
        past_len = _past_length(past_for_training) if past_for_training is not None else 0
        total_tokens = past_len + max_full_len

        input_ids = full_encoded["input_ids"].to(self.model.device)
        seq_mask = full_encoded["attention_mask"].to(self.model.device)

        prompt_lens = judger_mask.sum(dim=1)
        full_lens = seq_mask.sum(dim=1)
        labels = torch.full_like(input_ids, -100)
        position_ids = torch.zeros_like(input_ids, dtype=torch.long, device=self.model.device)

        for row_idx in range(input_ids.shape[0]):
            prompt_len = int(prompt_lens[row_idx].item())
            full_len = int(full_lens[row_idx].item())
            target_len = max(full_len - prompt_len, 0)
            prompt_base = int(position_offset[row_idx].item())
            if prompt_len > 0:
                position_ids[row_idx, :prompt_len] = prompt_base + torch.arange(
                    prompt_len,
                    device=self.model.device,
                )
            if target_len > 0:
                base_pos = prompt_base + prompt_len
                position_ids[row_idx, prompt_len:full_len] = base_pos + torch.arange(
                    target_len,
                    device=self.model.device,
                )
                labels[row_idx, prompt_len:full_len] = input_ids[row_idx, prompt_len:full_len]

        target_tokens = int((labels != -100).sum().item())
        if target_tokens <= 0:
            raise RuntimeError("Training loss is empty; no valid supervised target tokens found.")

        attention_mask = seq_mask
        if past_for_training is not None:
            if past_len > 0:
                past_mask = torch.ones(
                    (seq_mask.shape[0], past_len),
                    dtype=seq_mask.dtype,
                    device=seq_mask.device,
                )
                attention_mask = torch.cat([past_mask, seq_mask], dim=-1)

        try:
            if self.parallel_kv_target_position_logits and input_ids.shape[0] == 1:
                train_loss = _single_example_target_position_loss(
                    self.model.model,
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    position_ids=position_ids,
                    past_key_values=past_for_training,
                    labels=labels,
                    prompt_len=int(prompt_lens[0].item()),
                    full_len=int(full_lens[0].item()),
                    logit_chunk_size=self.parallel_kv_target_position_logit_chunk_size,
                    cache_preserving_backbone_checkpoint=(
                        self.parallel_kv_cache_preserving_backbone_checkpoint
                    ),
                )
            else:
                outputs = self.model.model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    position_ids=position_ids,
                    past_key_values=past_for_training,
                    labels=labels,
                    use_cache=False,
                    return_dict=True,
                )
                if outputs.loss is None:
                    raise RuntimeError(
                        "Training loss is None; failed to build valid supervision labels."
                    )
                train_loss = outputs.loss
        except torch.OutOfMemoryError as exc:
            self._raise_augmented_oom(
                exc,
                tag="parallel_kv_train_oom",
                details={
                    "task": self.task,
                    "batch_size": len(items),
                    "num_agent_caches": len(parallel_agent_past),
                    "past_tokens": int(past_len),
                    "judger_prompt_tokens_max": int(prompt_lens_cpu.max().item()) if prompt_lens_cpu.numel() > 0 else 0,
                    "judger_prompt_tokens_per_sample": [int(x) for x in prompt_lens_cpu.tolist()],
                    "full_seq_tokens_max": int(max_full_len),
                    "full_seq_tokens_per_sample": [int(x) for x in full_lens_cpu.tolist()],
                    "target_tokens": int(target_tokens),
                    "total_tokens": int(total_tokens),
                    "token_limit": int(total_tokens_limit) if total_tokens_limit > 0 else None,
                    "example_ids": [str(item.get("id", "")).strip() for item in items],
                },
            )
        return {
            "loss": train_loss,
            "target_tokens": target_tokens,
            "batch_size": len(items),
        }

    def _train_batch_impl(self, items: List[Dict[str, Any]]) -> Dict[str, Any]:
        parallel_agent_past, _ = self._collect_parallel_agent_past(items, include_traces=False)
        return self._compute_train_loss(items, parallel_agent_past)

    def _group_batch_indices(self, items: List[Dict[str, Any]]) -> List[List[int]]:
        return [list(range(len(items)))]

    def trainable_parameters(self) -> List[torch.nn.Parameter]:
        if self.parallel_kv_enable_judger_lora and not self._lora_initialized:
            self._ensure_judger_lora()
        params: List[torch.nn.Parameter] = []
        if self.parallel_kv_enable_affine_map and self.cache_mapper is not None:
            params.extend(list(self.cache_mapper.parameters()))
        if self.parallel_kv_enable_judger_lora and hasattr(self.model, "model"):
            params.extend([p for p in self.model.model.parameters() if p.requires_grad])

        unique: List[torch.nn.Parameter] = []
        seen = set()
        for param in params:
            pid = id(param)
            if pid in seen:
                continue
            seen.add(pid)
            unique.append(param)
        return unique

    def train_batch(self, items: List[Dict[str, Any]]) -> Dict[str, Any]:
        if len(items) > self.generate_bs:
            raise ValueError("Batch size exceeds configured generate_bs.")
        self._ensure_hf_backend()
        if self.parallel_kv_enable_judger_lora:
            self._ensure_judger_lora()

        self.model.model.train()
        if self.cache_mapper is not None:
            self.cache_mapper.train()

        group_indices = self._group_batch_indices(items)
        if len(group_indices) == 1:
            return self._train_batch_impl(items)

        weighted_loss: Optional[torch.Tensor] = None
        total_target_tokens = 0
        for idxs in group_indices:
            group_items = [items[i] for i in idxs]
            stats = self._train_batch_impl(group_items)
            if bool(stats.get("skip", False)):
                return stats
            group_target_tokens = int(stats["target_tokens"])
            if group_target_tokens <= 0:
                continue
            contribution = stats["loss"] * group_target_tokens
            weighted_loss = contribution if weighted_loss is None else weighted_loss + contribution
            total_target_tokens += group_target_tokens

        if weighted_loss is None or total_target_tokens <= 0:
            raise RuntimeError("Grouped training produced no valid target tokens.")

        return {
            "loss": weighted_loss / total_target_tokens,
            "target_tokens": total_target_tokens,
            "batch_size": len(items),
            "num_groups": len(group_indices),
        }

    @torch.no_grad()
    def run_batch(self, items: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        if len(items) > self.generate_bs:
            raise ValueError("Batch size exceeds configured generate_bs.")

        group_indices = self._group_batch_indices(items)
        if len(group_indices) == 1:
            return self._run_batch_impl(items)

        outputs: List[Optional[Dict[str, Any]]] = [None] * len(items)
        for idxs in group_indices:
            group_items = [items[i] for i in idxs]
            group_outputs = self._run_batch_impl(group_items)
            for out_idx, orig_idx in enumerate(idxs):
                outputs[orig_idx] = group_outputs[out_idx]

        finalized: List[Dict[str, Any]] = []
        for row in outputs:
            if row is None:
                raise RuntimeError("Failed to assemble grouped ParallelKV outputs.")
            finalized.append(row)
        return finalized

    def save_trainable_modules(self, save_dir: str) -> None:
        os.makedirs(save_dir, exist_ok=True)
        if self.parallel_kv_enable_affine_map and self.cache_mapper is not None:
            torch.save(self.cache_mapper.state_dict(), os.path.join(save_dir, "cache_mapper.pt"))
            with open(
                os.path.join(save_dir, "cache_mapper_config.json"),
                "w",
                encoding="utf-8",
            ) as handle:
                json.dump(
                    {
                        "hidden_dim": self.parallel_kv_mapper_hidden_dim,
                        "metadata_features": self.parallel_kv_mapper_metadata_features,
                    },
                    handle,
                    ensure_ascii=False,
                    indent=2,
                )
        if self.parallel_kv_enable_judger_lora and hasattr(self.model.model, "save_pretrained"):
            self.model.model.save_pretrained(os.path.join(save_dir, "judger_lora"))

    def load_trainable_modules(self, load_dir: str, *, trainable: bool = False) -> None:
        if self.parallel_kv_enable_affine_map and self.cache_mapper is not None:
            mapper_config_path = os.path.join(load_dir, "cache_mapper_config.json")
            if os.path.exists(mapper_config_path):
                with open(mapper_config_path, "r", encoding="utf-8") as handle:
                    mapper_config = json.load(handle)
                checkpoint_hidden_dim = int(
                    mapper_config.get("hidden_dim", self.parallel_kv_mapper_hidden_dim)
                )
                checkpoint_metadata_features = _normalize_mapper_metadata_features(
                    mapper_config.get("metadata_features", "both")
                )
                if (
                    checkpoint_hidden_dim != self.parallel_kv_mapper_hidden_dim
                    or checkpoint_metadata_features
                    != self.parallel_kv_mapper_metadata_features
                ):
                    self.parallel_kv_mapper_hidden_dim = checkpoint_hidden_dim
                    self.parallel_kv_mapper_metadata_features = checkpoint_metadata_features
                    if self.args is not None:
                        self.args.parallel_kv_mapper_hidden_dim = checkpoint_hidden_dim
                        self.args.parallel_kv_mapper_metadata_features = (
                            checkpoint_metadata_features
                        )
                    self.cache_mapper = LengthAwareAffineKVMapper(
                        hidden_dim=checkpoint_hidden_dim,
                        metadata_features=checkpoint_metadata_features,
                    ).to(self.model.device)
            mapper_path = os.path.join(load_dir, "cache_mapper.pt")
            if not os.path.exists(mapper_path):
                raise FileNotFoundError(f"Missing cache mapper checkpoint: {mapper_path}")
            mapper_state = torch.load(mapper_path, map_location=self.model.device)
            self.cache_mapper.load_state_dict(mapper_state, strict=True)
            if trainable:
                self.cache_mapper.train()
            else:
                self.cache_mapper.eval()

        if self.parallel_kv_enable_judger_lora:
            if PeftModel is None:
                raise ImportError(
                    "peft is required to load judger LoRA checkpoints. Install it with `pip install peft`."
                )
            self._ensure_hf_backend()
            lora_dir = os.path.join(load_dir, "judger_lora")
            if not os.path.isdir(lora_dir):
                raise FileNotFoundError(f"Missing judger LoRA checkpoint directory: {lora_dir}")
            align_model_vocab_to_adapter(
                self.model.model,
                self.model.tokenizer,
                lora_dir,
            )
            self.model.model = PeftModel.from_pretrained(
                self.model.model,
                lora_dir,
                is_trainable=trainable,
            )
            if not bool(getattr(self.model.model, "is_loaded_in_4bit", False)):
                self.model.model.to(self.model.device)
            if trainable:
                self.model.model.train()
            else:
                self.model.model.eval()
            self._lora_initialized = True

    def run_item(self, item: Dict[str, Any]) -> Dict[str, Any]:
        return self.run_batch([item])[0]


__all__ = [
    "LengthAwareAffineKVMapper",
    "MAPPER_METADATA_FEATURE_MODES",
    "ParallelKV",
]

import os
import csv
import time
import importlib.util
import torch
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
from transformers import AutoModelForCausalLM, AutoTokenizer
try:
    from transformers.models.qwen3.modeling_qwen3 import Qwen3RotaryEmbedding
except ImportError:
    Qwen3RotaryEmbedding = None

try:
    from vllm import LLM, SamplingParams
    _HAS_VLLM = True
except ImportError:
    _HAS_VLLM = False
try:
    from transformers.cache_utils import Cache
except ImportError:
    Cache = None


def _ensure_pad_token(tokenizer: AutoTokenizer) -> None:
    if tokenizer.pad_token_id is None:
        if tokenizer.eos_token is not None:
            tokenizer.pad_token = tokenizer.eos_token
        else:
            tokenizer.add_special_tokens({"pad_token": "<pad>"})


def _past_length(past_key_values: Optional[Tuple]) -> int:
    if not past_key_values:
        return 0
    if Cache is not None and isinstance(past_key_values, Cache):
        return past_key_values.get_seq_length()
    k = past_key_values[0][0]
    return k.shape[-2]


def _synchronize_timing_device(device: Optional[torch.device]) -> None:
    if device is None or not torch.cuda.is_available():
        return
    device_obj = device if isinstance(device, torch.device) else torch.device(device)
    if device_obj.type == "cuda":
        torch.cuda.synchronize(device_obj)


def _load_tokenizer(model_name: str) -> AutoTokenizer:
    try:
        return AutoTokenizer.from_pretrained(model_name, use_fast=True)
    except Exception:
        return AutoTokenizer.from_pretrained(model_name, use_fast=False)


def adapter_embedding_vocab_size(adapter_dir: str | Path) -> Optional[int]:
    """Read the vocabulary size of embedding/head tensors saved with a PEFT adapter.

    PEFT automatically saves embedding layers when training resized a model's
    vocabulary, even when ``modules_to_save`` is unset. Inspecting the tensor
    header lets inference resize the base model before PEFT loads those tensors.
    """

    weights_path = Path(adapter_dir).expanduser().resolve() / "adapter_model.safetensors"
    if not weights_path.is_file():
        return None
    try:
        from safetensors import safe_open
    except ImportError as exc:
        raise ImportError(
            "Reading a PEFT adapter vocabulary requires safetensors."
        ) from exc

    sizes = set()
    with safe_open(str(weights_path), framework="pt", device="cpu") as handle:
        for key in handle.keys():
            if key.endswith("embed_tokens.weight") or key.endswith("lm_head.weight"):
                shape = handle.get_slice(key).get_shape()
                if shape:
                    sizes.add(int(shape[0]))
    if not sizes:
        return None
    if len(sizes) != 1:
        raise ValueError(
            f"Adapter embedding/head tensors disagree on vocabulary size: {sorted(sizes)}"
        )
    return next(iter(sizes))


def align_model_vocab_to_adapter(model: Any, tokenizer: Any, adapter_dir: str | Path) -> bool:
    """Resize a base causal LM to match embedding tensors stored in an adapter."""

    target_size = adapter_embedding_vocab_size(adapter_dir)
    if target_size is None:
        return False
    current_size = int(model.get_input_embeddings().weight.shape[0])
    if current_size == target_size:
        return False
    tokenizer_size = len(tokenizer) if tokenizer is not None else None
    if tokenizer_size is not None and tokenizer_size > target_size:
        raise ValueError(
            "Adapter vocabulary is smaller than the active tokenizer: "
            f"adapter={target_size}, tokenizer={tokenizer_size}."
        )
    model.resize_token_embeddings(target_size)
    print(
        "[HF] Resized base-model vocabulary for PEFT adapter: "
        f"{current_size} -> {target_size}."
    )
    return True


def _flash_attn_available() -> bool:
    try:
        return importlib.util.find_spec("flash_attn") is not None
    except Exception:
        return False


def _normalize_attn_implementation(value: Optional[str]) -> str:
    raw = str(value or "auto").strip().lower()
    aliases = {
        "": "auto",
        "default": "auto",
        "none": "auto",
        "flash": "flash_attention_2",
        "flash2": "flash_attention_2",
        "fa2": "flash_attention_2",
    }
    return aliases.get(raw, raw)


def _resolve_attn_implementation(device: torch.device, args: Any = None) -> Tuple[str, str, bool]:
    requested = _normalize_attn_implementation(getattr(args, "attn_implementation", "auto"))
    device_obj = device if isinstance(device, torch.device) else torch.device(device)
    if requested != "auto":
        return requested, "user requested", False

    if device_obj.type == "cuda":
        if _flash_attn_available():
            return "flash_attention_2", "flash_attn package detected", True
        return "sdpa", "flash_attn not installed", False

    return "eager", f"non-CUDA device ({device_obj.type})", False


def _load_hf_causal_lm(
    model_name: str,
    *,
    device: torch.device,
    args: Any = None,
):
    dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32
    requested = _normalize_attn_implementation(getattr(args, "attn_implementation", "auto"))
    attn_impl, reason, chosen_automatically = _resolve_attn_implementation(device, args)
    load_in_4bit = bool(getattr(args, "load_in_4bit", False)) if args else False
    load_kwargs = {
        "torch_dtype": dtype,
        "attn_implementation": attn_impl,
    }
    if load_in_4bit:
        if device.type != "cuda":
            raise ValueError("--load_in_4bit requires a CUDA device.")
        try:
            from transformers import BitsAndBytesConfig
        except ImportError as exc:
            raise ImportError(
                "4-bit model loading requires a Transformers installation with "
                "BitsAndBytesConfig support."
            ) from exc

        compute_dtype_name = str(
            getattr(args, "bnb_4bit_compute_dtype", "bfloat16")
        ).strip().lower()
        compute_dtypes = {
            "bfloat16": torch.bfloat16,
            "bf16": torch.bfloat16,
            "float16": torch.float16,
            "fp16": torch.float16,
            "float32": torch.float32,
            "fp32": torch.float32,
        }
        if compute_dtype_name not in compute_dtypes:
            raise ValueError(
                "--bnb_4bit_compute_dtype must be one of bfloat16, float16, or float32."
            )
        load_kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type=str(
                getattr(args, "bnb_4bit_quant_type", "nf4")
            ).strip().lower(),
            bnb_4bit_compute_dtype=compute_dtypes[compute_dtype_name],
            bnb_4bit_use_double_quant=bool(
                getattr(args, "bnb_4bit_use_double_quant", True)
            ),
        )
    try:
        model = AutoModelForCausalLM.from_pretrained(model_name, **load_kwargs)
    except Exception as exc:
        if chosen_automatically and attn_impl.startswith("flash_attention"):
            fallback_kwargs = dict(load_kwargs)
            fallback_kwargs["attn_implementation"] = "sdpa" if device.type == "cuda" else "eager"
            print(
                f"[HF] Failed to initialize {model_name} with attention backend {attn_impl}; "
                f"falling back to {fallback_kwargs['attn_implementation']}. Error: {exc}"
            )
            model = AutoModelForCausalLM.from_pretrained(model_name, **fallback_kwargs)
        else:
            raise

    resolved_attn = str(getattr(model.config, "_attn_implementation", attn_impl) or attn_impl)
    print(
        f"[HF] Using attention backend {resolved_attn} for model {model_name} "
        f"(requested={requested}, reason={reason})"
    )
    if load_in_4bit:
        print(
            f"[HF] Loaded {model_name} in 4-bit "
            f"(quant_type={getattr(args, 'bnb_4bit_quant_type', 'nf4')}, "
            f"compute_dtype={getattr(args, 'bnb_4bit_compute_dtype', 'bfloat16')}, "
            f"double_quant={bool(getattr(args, 'bnb_4bit_use_double_quant', True))})"
        )
    return model


class ModelWrapper:
    def __init__(self, model_name: str, device: torch.device, use_vllm: bool = False, args = None):
        self.model_name = model_name
        self.device = device
        self.use_vllm = use_vllm and _HAS_VLLM
        self.vllm_engine = None
        self.latent_space_realign = bool(getattr(args, "latent_space_realign", False)) if args else False
        self._latent_realign_matrices: Dict[int, Tuple[torch.Tensor, torch.Tensor]] = {}
        self._rope_shift_rotary_emb: Dict[str, Any] = {}
        self.args = args

        # for ablation
        self.pre_aligned = None

        if self.use_vllm:

            tp_size = max(1, int(getattr(args, "tensor_parallel_size", 1)))
            gpu_util = float(getattr(args, "gpu_memory_utilization", 0.9))
            enable_prefix_caching = bool(getattr(args, "enable_prefix_caching", False)) if args else False
            enable_prompt_embeds = bool(getattr(args, "method", "") == "parallel_kv") if args else False

            print(f"[vLLM] Using vLLM backend for model {model_name}")
            llm_kwargs = dict(
                model=model_name,
                tensor_parallel_size=tp_size,
                gpu_memory_utilization=gpu_util,
            )
            if enable_prefix_caching or enable_prompt_embeds:
                self.vllm_engine = LLM(
                    enable_prefix_caching=enable_prefix_caching,
                    enable_prompt_embeds=enable_prompt_embeds,
                    **llm_kwargs,
                )
            else:
                self.vllm_engine = LLM(**llm_kwargs)
            self.tokenizer = _load_tokenizer(model_name)

            use_second_hf = bool(getattr(args, "use_second_HF_model", False)) if args else False
            if use_second_hf:
                hf_device = torch.device(args.device2)
                self.HF_model = _load_hf_causal_lm(
                    model_name,
                    device=hf_device,
                    args=args,
                ).to(args.device2).eval()
                self.embedding_layer = self.HF_model.get_input_embeddings()
                self.HF_device = args.device2
                # if self.latent_space_realign:
                self._ensure_latent_realign_matrix(self.HF_model, torch.device(self.HF_device), args)
            elif self.latent_space_realign:
                raise ValueError("latent_space_realign requires --use_second_HF_model when using vLLM backend.")
            _ensure_pad_token(self.tokenizer)
            return  # skip loading transformers model

        # fallback: normal transformers path
        self.tokenizer = _load_tokenizer(model_name)
        _ensure_pad_token(self.tokenizer)
        with torch.no_grad():
            self.model = _load_hf_causal_lm(
                model_name,
                device=device,
                args=args,
            )
        aligned_to_adapter = False
        parallel_kv_load_dir = str(
            getattr(args, "parallel_kv_load_dir", "") if args else ""
        ).strip()
        if parallel_kv_load_dir and bool(
            getattr(args, "parallel_kv_enable_judger_lora", False)
        ):
            candidate = Path(parallel_kv_load_dir).expanduser().resolve()
            adapter_dir = candidate / "judger_lora"
            if not adapter_dir.is_dir() and (candidate / "adapter_config.json").is_file():
                adapter_dir = candidate
            if adapter_dir.is_dir():
                aligned_to_adapter = align_model_vocab_to_adapter(
                    self.model,
                    self.tokenizer,
                    adapter_dir,
                )
        embedding_vocab_size = self.model.get_input_embeddings().weight.shape[0]
        if not aligned_to_adapter and len(self.tokenizer) > embedding_vocab_size:
            self.model.resize_token_embeddings(len(self.tokenizer))
        elif not aligned_to_adapter and len(self.tokenizer) < embedding_vocab_size:
            print(
                f"[HF] Keeping padded model vocabulary size {embedding_vocab_size}; "
                f"tokenizer exposes {len(self.tokenizer)} tokens."
            )
        is_loaded_in_4bit = bool(getattr(self.model, "is_loaded_in_4bit", False))
        prepare_kbit_for_training = bool(
            getattr(args, "train_parallel_kv", False) if args else False
        )
        if is_loaded_in_4bit and prepare_kbit_for_training:
            try:
                from peft import prepare_model_for_kbit_training
            except ImportError as exc:
                raise ImportError(
                    "4-bit training requires PEFT's prepare_model_for_kbit_training."
                ) from exc
            self.model = prepare_model_for_kbit_training(
                self.model,
                use_gradient_checkpointing=False,
            )
            if bool(
                getattr(args, "kbit_keep_embeddings_in_compute_dtype", False)
            ):
                compute_dtype_name = str(
                    getattr(args, "bnb_4bit_compute_dtype", "bfloat16")
                ).strip().lower()
                compute_dtype = {
                    "bfloat16": torch.bfloat16,
                    "bf16": torch.bfloat16,
                    "float16": torch.float16,
                    "fp16": torch.float16,
                    "float32": torch.float32,
                    "fp32": torch.float32,
                }[compute_dtype_name]
                embedding_modules = {
                    id(module): module
                    for module in (
                        self.model.get_input_embeddings(),
                        self.model.get_output_embeddings(),
                    )
                    if module is not None
                }
                for module in embedding_modules.values():
                    module.to(dtype=compute_dtype)
                print(
                    "[HF] Kept frozen input/output embeddings in "
                    f"{compute_dtype} for memory-efficient k-bit training."
                )
                torch.cuda.empty_cache()
        elif is_loaded_in_4bit:
            print(
                "[HF] Skipping PEFT k-bit training preparation for inference-only "
                "4-bit model loading."
            )
        else:
            self.model.to(device)
        self.model.eval()
        if hasattr(self.model.config, "use_cache"):
            self.model.config.use_cache = True
        if self.latent_space_realign:
            self._ensure_latent_realign_matrix(self.model, self.device, args)

    def render_chat(self, messages: List[Dict], add_generation_prompt: bool = True) -> str:
        tpl = getattr(self.tokenizer, "chat_template", None)
        if tpl:
            return self.tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=add_generation_prompt
            )
        segments = []
        for message in messages:
            role = message.get("role", "user")
            content = message.get("content", "")
            segments.append(f"<|{role}|>\n{content}\n</|{role}|>")
        if add_generation_prompt:
            segments.append("<|assistant|>")
        return "\n".join(segments)

    def prepare_chat_input(
        self, messages: List[Dict], add_generation_prompt: bool = True
    ) -> Tuple[str, torch.Tensor, torch.Tensor, List[str]]:
        prompt_text = self.render_chat(messages, add_generation_prompt=add_generation_prompt)
        encoded = self.tokenizer(
            prompt_text,
            return_tensors="pt",
            add_special_tokens=False,
        )
        input_ids = encoded["input_ids"].to(self.device)
        attention_mask = encoded["attention_mask"].to(self.device)
        active_ids = input_ids[0][attention_mask[0].bool()].tolist()
        tokens = self.tokenizer.convert_ids_to_tokens(active_ids)
        return prompt_text, input_ids, attention_mask, tokens

    def prepare_chat_batch(
        self,
        batch_messages: List[List[Dict]],
        add_generation_prompt: bool = True,
    ) -> Tuple[List[str], torch.Tensor, torch.Tensor, List[List[str]]]:
        prompts: List[str] = []
        for messages in batch_messages:
            prompts.append(self.render_chat(messages, add_generation_prompt=add_generation_prompt))
        encoded = self.tokenizer(
            prompts,
            return_tensors="pt",
            padding=True,
            add_special_tokens=False,
        )
        input_ids = encoded["input_ids"].to(self.device)
        attention_mask = encoded["attention_mask"].to(self.device)
        tokens_batch: List[List[str]] = []
        for ids_row, mask_row in zip(input_ids, attention_mask):
            active_ids = ids_row[mask_row.bool()].tolist()
            tokens_batch.append(self.tokenizer.convert_ids_to_tokens(active_ids))
        return prompts, input_ids, attention_mask, tokens_batch

    def vllm_generate_text_batch(
        self,
        prompts: List[str],
        *,
        max_new_tokens: int = 256,
        temperature: float = 0.7,
        top_p: float = 0.95,
    ) -> List[str]:
        if not self.vllm_engine:
            raise RuntimeError("vLLM engine not initialized. Pass use_vllm=True to ModelWrapper.")
        sampling_params = SamplingParams(
            temperature=temperature,
            top_p=top_p,
            max_tokens=max_new_tokens,
        )
        outputs = self.vllm_engine.generate(prompts, sampling_params)
        generations = [out.outputs[0].text.strip() for out in outputs]
        return generations

    @staticmethod
    def _rotate_half(x: torch.Tensor) -> torch.Tensor:
        half = x.shape[-1] // 2
        x1 = x[..., :half]
        x2 = x[..., half:]
        return torch.cat([-x2, x1], dim=-1)

    @staticmethod
    def _apply_rotary_pos_emb(
        q: torch.Tensor,
        k: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
        unsqueeze_dim: int = 1,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        cos = cos.unsqueeze(unsqueeze_dim)
        sin = sin.unsqueeze(unsqueeze_dim)
        q_embed = (q * cos) + (ModelWrapper._rotate_half(q) * sin)
        k_embed = (k * cos) + (ModelWrapper._rotate_half(k) * sin)
        return q_embed, k_embed

    def _get_qwen3_rotary_emb(self, device: torch.device) -> Optional[Any]:
        key = str(device)
        if key in self._rope_shift_rotary_emb:
            return self._rope_shift_rotary_emb[key]

        if hasattr(self, "model"):
            source_model = self.model
        elif hasattr(self, "HF_model"):
            source_model = self.HF_model
        else:
            return None

        def _build_rotary_emb(rotary_cls):
            if rotary_cls is None:
                return None
            try:
                emb = rotary_cls(source_model.config, device=device)
            except Exception:
                try:
                    emb = rotary_cls(source_model.config)
                except Exception:
                    return None
            if isinstance(emb, torch.nn.Module):
                emb = emb.to(device)
            return emb

        rotary_emb = None
        try:
            rotary_emb = source_model.model.rotary_emb
        except Exception:
            try:
                rotary_emb = source_model.rotary_emb
            except Exception:
                rotary_emb = None
        if rotary_emb is None:
            rotary_emb = _build_rotary_emb(Qwen3RotaryEmbedding)

        if rotary_emb is None:
            return None
        if isinstance(rotary_emb, torch.nn.Module):
            rotary_emb = rotary_emb.to(device)
        self._rope_shift_rotary_emb[key] = rotary_emb
        return rotary_emb

    def _get_rope_inv_freq(self, device: torch.device, dtype: torch.dtype, head_dim: int) -> torch.Tensor:
        source_model = self.HF_model if hasattr(self, "HF_model") else self.model
        inv_freq = None
        try:
            inv_freq = source_model.model.layers[0].self_attn.rotary_emb.inv_freq
        except Exception:
            try:
                inv_freq = source_model.layers[0].self_attn.rotary_emb.inv_freq
            except Exception:
                inv_freq = None
        if inv_freq is None:
            base = float(getattr(source_model.config, "rope_theta", 10000.0))
            inv_freq = 1.0 / (base ** (torch.arange(0, head_dim, 2, device=device, dtype=torch.float32) / head_dim))
        else:
            inv_freq = inv_freq.detach().to(device=device, dtype=torch.float32)
        rope_scaling = getattr(source_model.config, "rope_scaling", None)
        if isinstance(rope_scaling, dict) and rope_scaling.get("type") == "linear":
            factor = float(rope_scaling.get("factor", 1.0))
            if factor != 0:
                inv_freq = inv_freq / factor
        return inv_freq.to(device=device, dtype=dtype)

    def _apply_rope_shift(self, key: torch.Tensor, offsets: torch.Tensor) -> torch.Tensor:
        if key.shape[-1] % 2 != 0:
            return key
        if offsets.numel() == 1:
            offsets = offsets.expand(key.shape[0])
        offsets = offsets.to(device=key.device)

        # use model rotary embedding class to get cos/sin for key rotation
        rotary_emb = self._get_qwen3_rotary_emb(key.device)
        if rotary_emb is not None:
            position_ids = offsets.to(dtype=torch.long).view(-1, 1)
            cos, sin = rotary_emb(key, position_ids)
            _, k_embed = self._apply_rotary_pos_emb(key, key, cos, sin, unsqueeze_dim=1)
            return k_embed.to(key.dtype)

        offsets = offsets.to(dtype=torch.float32)
        head_dim = key.shape[-1]
        inv_freq = self._get_rope_inv_freq(key.device, torch.float32, head_dim)
        angles = offsets[:, None] * inv_freq[None, :]
        cos = torch.cos(angles)
        sin = torch.sin(angles)
        cos = torch.repeat_interleave(cos, 2, dim=-1)
        sin = torch.repeat_interleave(sin, 2, dim=-1)
        shape = [cos.shape[0]] + [1] * (key.dim() - 2) + [cos.shape[-1]]
        cos = cos.view(*shape)
        sin = sin.view(*shape)
        key_fp32 = key.to(torch.float32)
        shifted = key_fp32 * cos + self._rotate_half(key_fp32) * sin
        return shifted.to(key.dtype)

    def shift_rope_past_key_values(
        self,
        past_key_values: Optional[Tuple],
        offsets: torch.Tensor,
    ) -> Optional[Tuple]:
        if past_key_values is None:
            return None
        if isinstance(offsets, int):
            offsets = torch.tensor([offsets], device=self.device)
        if offsets.dim() == 0:
            offsets = offsets.view(1)
        if Cache is not None and isinstance(past_key_values, Cache):
            for layer in past_key_values.layers:
                if not hasattr(layer, "keys") or layer.keys is None:
                    continue
                layer.keys = self._apply_rope_shift(layer.keys, offsets).contiguous()
            return past_key_values
        shifted_layers = []
        for layer in past_key_values:
            if isinstance(layer, tuple):
                k, v = layer
                k_shifted = self._apply_rope_shift(k, offsets)
                shifted_layers.append((k_shifted, v))
            else:
                shifted_layers.append(layer)
        return tuple(shifted_layers)

    @staticmethod
    def _sample_top_p(logits: torch.Tensor, top_p: float) -> torch.Tensor:
        probs = torch.softmax(logits, dim=-1)
        if top_p >= 1.0:
            return torch.multinomial(probs, 1)
        sorted_probs, sorted_idx = torch.sort(probs, descending=True)
        cum = torch.cumsum(sorted_probs, dim=-1)
        cutoff = cum > top_p
        cutoff[..., 0] = False
        sorted_probs = sorted_probs.masked_fill(cutoff, 0.0)
        sorted_probs = sorted_probs / sorted_probs.sum(dim=-1, keepdim=True).clamp_min(1e-8)
        next_idx = torch.multinomial(sorted_probs, 1)
        return sorted_idx.gather(-1, next_idx)

    @torch.no_grad()
    def _generate_text_batch_manual(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        *,
        max_new_tokens: int = 256,
        temperature: float = 0.7,
        top_p: float = 0.95,
        past_key_values: Optional[Tuple] = None,
        position_offset: Optional[torch.Tensor] = None,
        return_generated_ids: bool = False,
        return_timings: bool = False,
    ) -> Tuple:
        if input_ids.dim() != 2:
            raise ValueError("input_ids must be 2D with shape [batch, seq_len]")
        input_ids = input_ids.to(self.device)
        if attention_mask is None:
            attention_mask = torch.ones_like(input_ids, device=self.device)
        else:
            attention_mask = attention_mask.to(self.device)

        prompt_lens = attention_mask.sum(dim=1).to(self.device)
        if position_offset is None:
            position_offset = torch.zeros_like(prompt_lens)
        if isinstance(position_offset, int):
            position_offset = torch.tensor([position_offset], device=self.device)
        if position_offset.dim() == 0:
            position_offset = position_offset.view(1)
        if position_offset.numel() == 1:
            position_offset = position_offset.expand(prompt_lens.shape[0])
        position_offset = position_offset.to(self.device)

        position_ids = attention_mask.long().cumsum(-1) - 1
        position_ids = position_ids.clamp_min(0)
        position_ids = position_ids + position_offset.view(-1, 1)
        position_ids = torch.where(attention_mask.bool(), position_ids, torch.zeros_like(position_ids))

        full_mask = attention_mask
        if past_key_values is not None:
            past_len = _past_length(past_key_values)
            if past_len > 0:
                past_mask = torch.ones(
                    (attention_mask.shape[0], past_len),
                    dtype=attention_mask.dtype,
                    device=attention_mask.device,
                )
                full_mask = torch.cat([past_mask, attention_mask], dim=-1)

        timing_device = input_ids.device
        _synchronize_timing_device(timing_device)
        start_time = time.perf_counter()
        qwen_last_logit_kwargs = (
            {"logits_to_keep": 1}
            if str(getattr(self.model.config, "model_type", "")).strip().lower()
            in {"qwen3", "qwen3_moe"}
            else {}
        )
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=full_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            use_cache=True,
            return_dict=True,
            **qwen_last_logit_kwargs,
        )
        past = outputs.past_key_values
        next_logits = outputs.logits[:, -1, :]

        base_positions = prompt_lens + position_offset

        eos_id = self.tokenizer.eos_token_id
        generated_ids: List[torch.Tensor] = []
        finished = torch.zeros(input_ids.shape[0], dtype=torch.bool, device=self.device)
        ttft_sec: Optional[float] = None

        for step in range(max_new_tokens):
            if temperature <= 0:
                next_token = torch.argmax(next_logits, dim=-1, keepdim=True)
            else:
                logits = next_logits / temperature
                next_token = self._sample_top_p(logits, top_p)
            if ttft_sec is None:
                _synchronize_timing_device(timing_device)
                ttft_sec = time.perf_counter() - start_time
            if eos_id is not None:
                next_token = torch.where(
                    finished.unsqueeze(-1),
                    torch.full_like(next_token, eos_id),
                    next_token,
                )
            generated_ids.append(next_token)
            if eos_id is not None:
                finished = finished | (next_token.squeeze(-1) == eos_id)
                if bool(finished.all()):
                    break

            full_mask = torch.cat(
                [full_mask, torch.ones((full_mask.shape[0], 1), dtype=full_mask.dtype, device=full_mask.device)],
                dim=-1,
            )
            position_ids_step = (base_positions + step).view(-1, 1)
            outputs = self.model(
                input_ids=next_token,
                attention_mask=full_mask,
                position_ids=position_ids_step,
                past_key_values=past,
                use_cache=True,
                return_dict=True,
                **qwen_last_logit_kwargs,
            )
            past = outputs.past_key_values
            next_logits = outputs.logits[:, -1, :]

        if generated_ids:
            generated_ids_tensor = torch.cat(generated_ids, dim=1)
        else:
            generated_ids_tensor = torch.zeros(
                (input_ids.shape[0], 0), dtype=torch.long, device=self.device
            )

        generations: List[str] = []
        generated_ids_list: List[torch.Tensor] = []
        for row in generated_ids_tensor:
            if return_generated_ids:
                generated_ids_list.append(row.detach().clone())
            text = self.tokenizer.decode(row.tolist(), skip_special_tokens=True).strip()
            generations.append(text)

        _synchronize_timing_device(timing_device)
        timing_metrics = {
            "ttft_sec": ttft_sec,
            "generation_time_sec": time.perf_counter() - start_time,
        }
        if return_generated_ids:
            if return_timings:
                return generations, past, generated_ids_list, timing_metrics
            return generations, past, generated_ids_list
        if return_timings:
            return generations, past, timing_metrics
        return generations, past

    @torch.no_grad()
    def generate_text_batch(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        *,
        max_new_tokens: int = 256,
        temperature: float = 0.7,
        top_p: float = 0.95,
        past_key_values: Optional[Tuple] = None,
        cache_position: Optional[torch.Tensor] = None,
        return_generated_ids: bool = False,
        return_timings: bool = False,
    ) -> Tuple:
        if return_timings and cache_position is None:
            return self._generate_text_batch_manual(
                input_ids,
                attention_mask,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                top_p=top_p,
                past_key_values=past_key_values,
                position_offset=None,
                return_generated_ids=return_generated_ids,
                return_timings=return_timings,
            )

        if input_ids.dim() != 2:
            raise ValueError("input_ids must be 2D with shape [batch, seq_len]")
        if attention_mask is None:
            attention_mask = torch.ones_like(input_ids, device=self.device)
        timing_device = input_ids.device
        _synchronize_timing_device(timing_device)
        start_time = time.perf_counter()
        prompt_lengths = attention_mask.sum(dim=1).tolist()
        eos_id = self.tokenizer.eos_token_id
        if eos_id is None and hasattr(self.model, "config"):
            eos_id = getattr(self.model.config, "eos_token_id", None)
        pad_id = self.tokenizer.pad_token_id
        if pad_id is None and hasattr(self.model, "config"):
            pad_id = getattr(self.model.config, "pad_token_id", None)
        if pad_id is None:
            if isinstance(eos_id, int):
                pad_id = eos_id
            elif isinstance(eos_id, (list, tuple)) and len(eos_id) > 0:
                pad_id = int(eos_id[0])
        generation_kwargs = {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "max_new_tokens": max_new_tokens,
            "do_sample": temperature > 0,
            "eos_token_id": eos_id,
            "pad_token_id": pad_id,
            "return_dict_in_generate": True,
            "output_scores": False,
            "past_key_values": past_key_values,
            "cache_position": cache_position,
        }
        if temperature > 0:
            generation_kwargs.update(
                {
                    "temperature": temperature,
                    "top_p": top_p,
                }
            )
        outputs = self.model.generate(**generation_kwargs)
        sequences = outputs.sequences
        generations: List[str] = []
        generated_ids_list: List[torch.Tensor] = []
        for idx, length in enumerate(prompt_lengths):
            length = int(length)
            generated_ids = sequences[idx, length:]
            if return_generated_ids:
                generated_ids_list.append(generated_ids.detach())
            text = self.tokenizer.decode(generated_ids, skip_special_tokens=True).strip()
            generations.append(text)
        _synchronize_timing_device(timing_device)
        timing_metrics = {
            "ttft_sec": None,
            "generation_time_sec": time.perf_counter() - start_time,
        }
        if return_generated_ids:
            if return_timings:
                return generations, outputs.past_key_values, generated_ids_list, timing_metrics
            return generations, outputs.past_key_values, generated_ids_list
        if return_timings:
            return generations, outputs.past_key_values, timing_metrics
        return generations, outputs.past_key_values

    @torch.no_grad()
    def generate_text_batch_with_offset(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        *,
        max_new_tokens: int = 256,
        temperature: float = 0.6,
        top_p: float = 0.95,
        past_key_values: Optional[Tuple] = None,
        position_offset: Optional[torch.Tensor] = None,
        return_timings: bool = False,
    ) -> Tuple:
        return self._generate_text_batch_manual(
            input_ids,
            attention_mask,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
            past_key_values=past_key_values,
            position_offset=position_offset,
            return_generated_ids=False,
            return_timings=return_timings,
        )

    def tokenize_text(self, text: str) -> torch.Tensor:
        return self.tokenizer(
            text,
            add_special_tokens=False,
            return_tensors="pt",
        )["input_ids"].to(self.device)

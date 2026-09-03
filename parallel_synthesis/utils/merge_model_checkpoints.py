import argparse
import json
import shutil
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import torch

from parallel_synthesis.utils.checkpoints import resolve_parallel_kv_checkpoint


FLOAT_DTYPES = {
    torch.float16,
    torch.bfloat16,
    torch.float32,
    torch.float64,
}


def _prepare_output_dir(output_dir: Path, *, overwrite: bool) -> None:
    if output_dir.exists():
        if not overwrite:
            raise FileExistsError(f"Output directory already exists: {output_dir}")
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)


def _resolve_lora_dir(path: Path) -> Optional[Path]:
    if (path / "adapter_config.json").is_file():
        return path
    child = path / "judger_lora"
    if (child / "adapter_config.json").is_file():
        return child
    return None


def _load_safetensors(path: Path) -> Dict[str, torch.Tensor]:
    from safetensors.torch import load_file

    return dict(load_file(str(path)))


def _save_safetensors(state: Dict[str, torch.Tensor], path: Path) -> None:
    from safetensors.torch import save_file

    save_file(state, str(path))


def load_tensor_state(path: Path) -> Dict[str, torch.Tensor]:
    if path.suffix == ".safetensors":
        return _load_safetensors(path)
    state = torch.load(path, map_location="cpu")
    if not isinstance(state, dict):
        raise ValueError(f"Expected tensor state dict in {path}, got {type(state).__name__}")
    for key, value in state.items():
        if not torch.is_tensor(value):
            raise ValueError(f"Expected tensor for key {key!r} in {path}, got {type(value).__name__}")
    return state


def save_tensor_state(state: Dict[str, torch.Tensor], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.suffix == ".safetensors":
        _save_safetensors(state, path)
    else:
        torch.save(state, path)


def _merge_tensor(
    tensor_a: Optional[torch.Tensor],
    tensor_b: Optional[torch.Tensor],
    *,
    weight_a: float,
    weight_b: float,
    missing_policy: str,
    key: str,
) -> torch.Tensor:
    if tensor_a is None and tensor_b is None:
        raise ValueError(f"Internal error: both tensors missing for {key}")
    if tensor_a is None or tensor_b is None:
        present = tensor_a if tensor_a is not None else tensor_b
        present_weight = weight_a if tensor_a is not None else weight_b
        if missing_policy == "error":
            raise KeyError(f"Key {key!r} exists in only one source.")
        if missing_policy == "copy":
            return present.detach().cpu().clone()
        if missing_policy == "zero":
            if present.dtype not in FLOAT_DTYPES:
                return present.detach().cpu().clone()
            return (present.detach().cpu().float() * float(present_weight)).to(present.dtype)
        raise ValueError(f"Unsupported missing_policy: {missing_policy}")

    if tuple(tensor_a.shape) != tuple(tensor_b.shape):
        raise ValueError(
            f"Shape mismatch for {key!r}: {tuple(tensor_a.shape)} vs {tuple(tensor_b.shape)}"
        )
    if tensor_a.dtype not in FLOAT_DTYPES or tensor_b.dtype not in FLOAT_DTYPES:
        if torch.equal(tensor_a.cpu(), tensor_b.cpu()):
            return tensor_a.detach().cpu().clone()
        raise ValueError(f"Non-floating tensor {key!r} differs between checkpoints.")

    merged = tensor_a.detach().cpu().float() * float(weight_a)
    merged = merged + tensor_b.detach().cpu().float() * float(weight_b)
    return merged.to(tensor_a.dtype)


def merge_tensor_state_dicts(
    state_a: Dict[str, torch.Tensor],
    state_b: Dict[str, torch.Tensor],
    *,
    weight_a: float = 0.5,
    weight_b: float = 0.5,
    missing_policy: str = "error",
) -> Dict[str, torch.Tensor]:
    keys = sorted(set(state_a.keys()) | set(state_b.keys()))
    merged: Dict[str, torch.Tensor] = {}
    for key in keys:
        merged[key] = _merge_tensor(
            state_a.get(key),
            state_b.get(key),
            weight_a=weight_a,
            weight_b=weight_b,
            missing_policy=missing_policy,
            key=key,
        )
    return merged


def _load_lora_state(lora_dir: Path) -> Tuple[Dict[str, torch.Tensor], str]:
    st_path = lora_dir / "adapter_model.safetensors"
    bin_path = lora_dir / "adapter_model.bin"
    if st_path.is_file():
        return load_tensor_state(st_path), "adapter_model.safetensors"
    if bin_path.is_file():
        return load_tensor_state(bin_path), "adapter_model.bin"
    raise FileNotFoundError(
        f"Missing adapter weights in {lora_dir}; expected adapter_model.safetensors or adapter_model.bin."
    )


def _is_lora_b_key(key: str) -> bool:
    parts = str(key).split(".")
    return "lora_B" in parts


def scale_lora_delta_state(
    state: Dict[str, torch.Tensor],
    *,
    scale: float,
) -> Dict[str, torch.Tensor]:
    """Scale a LoRA adapter delta by multiplying only lora_B tensors.

    For a LoRA layer delta = scaling * B @ A. Scaling both A and B would produce
    scale^2 * delta, so only one side should be scaled.
    """

    scaled: Dict[str, torch.Tensor] = {}
    scaled_b_count = 0
    for key, tensor in state.items():
        value = tensor.detach().cpu().clone()
        if _is_lora_b_key(key):
            if value.dtype not in FLOAT_DTYPES:
                raise ValueError(f"LoRA B tensor {key!r} is not floating dtype: {value.dtype}")
            value = (value.float() * float(scale)).to(tensor.dtype)
            scaled_b_count += 1
        scaled[key] = value
    if scaled_b_count <= 0:
        raise ValueError("Could not find any lora_B tensors to scale in adapter state.")
    return scaled


def _scale_loaded_lora_b_weights(peft_model: Any, scale: float) -> int:
    scaled = 0
    with torch.no_grad():
        for module in peft_model.modules():
            lora_b = getattr(module, "lora_B", None)
            if lora_b is None:
                continue
            if isinstance(lora_b, torch.nn.ModuleDict):
                for submodule in lora_b.values():
                    weight = getattr(submodule, "weight", None)
                    if weight is not None:
                        weight.mul_(float(scale))
                        scaled += 1
                continue
            weight = getattr(lora_b, "weight", None)
            if weight is not None:
                weight.mul_(float(scale))
                scaled += 1
    if scaled <= 0:
        raise ValueError("Could not find any LoRA B weights to scale before merging.")
    return scaled


def _copy_lora_metadata(src_lora_dir: Path, dst_lora_dir: Path) -> None:
    dst_lora_dir.mkdir(parents=True, exist_ok=True)
    for filename in ("adapter_config.json", "README.md"):
        src = src_lora_dir / filename
        if src.is_file():
            shutil.copy2(src, dst_lora_dir / filename)


def _load_json(path: Path) -> Optional[Dict[str, Any]]:
    if not path.is_file():
        return None
    with path.open("r", encoding="utf-8") as fh:
        data = json.load(fh)
    if not isinstance(data, dict):
        raise ValueError(f"Expected JSON object in {path}")
    return data


def _check_lora_configs_compatible(
    config_a: Dict[str, Any],
    config_b: Dict[str, Any],
    *,
    allow_config_mismatch: bool,
) -> None:
    if allow_config_mismatch:
        return
    keys_to_check = [
        "peft_type",
        "r",
        "lora_alpha",
        "target_modules",
        "fan_in_fan_out",
        "bias",
        "task_type",
    ]
    mismatches = []
    for key in keys_to_check:
        value_a = config_a.get(key)
        value_b = config_b.get(key)
        if key == "target_modules":
            if sorted(str(item) for item in (value_a or [])) != sorted(str(item) for item in (value_b or [])):
                mismatches.append(key)
            continue
        if value_a != value_b:
            mismatches.append(key)
    if mismatches:
        raise ValueError(
            "LoRA adapter configs differ for keys "
            f"{mismatches}. Pass allow_config_mismatch=True only if this is intentional."
        )


def merge_parallel_kv_checkpoints(
    checkpoint_a: str,
    checkpoint_b: Optional[str],
    output_dir: str,
    *,
    weight_a: float = 0.5,
    weight_b: float = 0.5,
    source_b_is_base: bool = False,
    missing_policy: str = "error",
    allow_config_mismatch: bool = False,
    cache_mapper_source: str = "a",
    overwrite: bool = False,
) -> Dict[str, Any]:
    """Merge this repo's ParallelKV trainable checkpoint artifacts.

    The output is another ParallelKV checkpoint directory containing merged
    `cache_mapper.pt` and/or `judger_lora/` artifacts. If `source_b_is_base` is
    true, missing tensors in source B are treated as base-model zero deltas.
    """

    ckpt_a = resolve_parallel_kv_checkpoint(checkpoint_a)
    ckpt_b: Optional[Path] = None
    if checkpoint_b and not source_b_is_base:
        ckpt_b = resolve_parallel_kv_checkpoint(checkpoint_b)
    missing_policy_effective = missing_policy
    if cache_mapper_source not in {"a", "b", "none"}:
        raise ValueError("cache_mapper_source must be one of: a, b, none")

    out = Path(output_dir).expanduser().resolve()
    _prepare_output_dir(out, overwrite=overwrite)
    meta: Dict[str, Any] = {
        "merge_type": "parallel_kv_artifacts",
        "checkpoint_a": str(ckpt_a),
        "checkpoint_b": str(checkpoint_b or ""),
        "source_b_is_base": bool(source_b_is_base),
        "weight_a": float(weight_a),
        "weight_b": float(weight_b),
        "missing_policy": missing_policy_effective,
        "cache_mapper_source": cache_mapper_source,
        "merged_artifacts": [],
    }

    mapper_a_path = ckpt_a / "cache_mapper.pt"
    mapper_b_path = ckpt_b / "cache_mapper.pt" if ckpt_b is not None else None
    mapper_source_path: Optional[Path] = None
    if cache_mapper_source == "a" and mapper_a_path.is_file():
        mapper_source_path = mapper_a_path
    elif cache_mapper_source == "b" and mapper_b_path is not None and mapper_b_path.is_file():
        mapper_source_path = mapper_b_path
    elif cache_mapper_source == "b":
        raise FileNotFoundError("Requested cache_mapper_source=b, but checkpoint_b has no cache_mapper.pt")
    if mapper_source_path is not None:
        shutil.copy2(mapper_source_path, out / "cache_mapper.pt")
        meta["merged_artifacts"].append("cache_mapper.pt")

    lora_a_dir = _resolve_lora_dir(ckpt_a)
    lora_b_dir = _resolve_lora_dir(ckpt_b) if ckpt_b is not None else None
    if lora_a_dir is not None or lora_b_dir is not None:
        if lora_a_dir is None:
            raise FileNotFoundError("checkpoint_a does not contain judger_lora/ adapter weights.")
        state_a, weight_filename = _load_lora_state(lora_a_dir)
        state_b: Dict[str, torch.Tensor] = {}
        if lora_b_dir is not None:
            state_b, weight_filename_b = _load_lora_state(lora_b_dir)
            if weight_filename_b != weight_filename:
                weight_filename = "adapter_model.safetensors"
            config_a = _load_json(lora_a_dir / "adapter_config.json") or {}
            config_b = _load_json(lora_b_dir / "adapter_config.json") or {}
            _check_lora_configs_compatible(
                config_a,
                config_b,
                allow_config_mismatch=allow_config_mismatch,
            )
        if source_b_is_base:
            merged_lora = scale_lora_delta_state(state_a, scale=weight_a)
            meta["lora_merge_mode"] = "scale_checkpoint_a_delta_against_base"
            meta["lora_delta_scale"] = float(weight_a)
        else:
            merged_lora = merge_tensor_state_dicts(
                state_a,
                state_b,
                weight_a=weight_a,
                weight_b=weight_b,
                missing_policy=missing_policy_effective,
            )
            meta["lora_merge_mode"] = "weighted_adapter_tensor_average"
        out_lora_dir = out / "judger_lora"
        _copy_lora_metadata(lora_a_dir, out_lora_dir)
        save_tensor_state(merged_lora, out_lora_dir / weight_filename)
        meta["merged_artifacts"].append(f"judger_lora/{weight_filename}")

    if not meta["merged_artifacts"]:
        raise ValueError(f"No mergeable ParallelKV artifacts found in {ckpt_a}")

    with (out / "checkpoint_merge_meta.json").open("w", encoding="utf-8") as fh:
        json.dump(meta, fh, ensure_ascii=False, indent=2)
    return meta


def merge_lora_into_base_model(
    base_model: str,
    adapter_checkpoint: str,
    output_dir: str,
    *,
    lora_scale: float = 1.0,
    copy_parallel_kv_artifacts: bool = False,
    torch_dtype: str = "auto",
    device_map: str = "cpu",
    overwrite: bool = False,
) -> Dict[str, Any]:
    """Apply a scaled PEFT LoRA adapter to a base HF model and save a full merged model."""

    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer

    out = Path(output_dir).expanduser().resolve()
    _prepare_output_dir(out, overwrite=overwrite)
    adapter_path = Path(adapter_checkpoint).expanduser().resolve()
    lora_dir = _resolve_lora_dir(adapter_path) or adapter_path
    if not (0.0 <= float(lora_scale) <= 1.0):
        raise ValueError(f"lora_scale must be in [0, 1], got {lora_scale}.")

    dtype_arg: Any = torch_dtype
    if torch_dtype == "float16":
        dtype_arg = torch.float16
    elif torch_dtype == "bfloat16":
        dtype_arg = torch.bfloat16
    elif torch_dtype == "float32":
        dtype_arg = torch.float32

    model = AutoModelForCausalLM.from_pretrained(
        base_model,
        torch_dtype=dtype_arg,
        device_map=device_map,
    )
    tokenizer = AutoTokenizer.from_pretrained(base_model)
    from parallel_synthesis.models import align_model_vocab_to_adapter

    align_model_vocab_to_adapter(model, tokenizer, lora_dir)
    model = PeftModel.from_pretrained(model, str(lora_dir))
    scaled_lora_b_tensors = _scale_loaded_lora_b_weights(model, float(lora_scale))
    model = model.merge_and_unload()
    model.save_pretrained(str(out), safe_serialization=True)

    tokenizer.save_pretrained(str(out))

    copied_parallel_kv_artifacts = []
    if copy_parallel_kv_artifacts:
        source_ckpt = adapter_path if adapter_path.is_dir() else lora_dir.parent
        mapper_path = source_ckpt / "cache_mapper.pt"
        if mapper_path.is_file():
            shutil.copy2(mapper_path, out / "cache_mapper.pt")
            copied_parallel_kv_artifacts.append("cache_mapper.pt")
        else:
            raise FileNotFoundError(
                f"--copy_parallel_kv_artifacts was set, but cache_mapper.pt was not found in {source_ckpt}."
            )

    meta = {
        "merge_type": "lora_into_base_model",
        "base_model": base_model,
        "adapter_checkpoint": str(lora_dir),
        "output_dir": str(out),
        "lora_scale": float(lora_scale),
        "scaled_lora_b_tensors": int(scaled_lora_b_tensors),
        "copy_parallel_kv_artifacts": bool(copy_parallel_kv_artifacts),
        "copied_parallel_kv_artifacts": copied_parallel_kv_artifacts,
        "torch_dtype": torch_dtype,
    }
    with (out / "checkpoint_merge_meta.json").open("w", encoding="utf-8") as fh:
        json.dump(meta, fh, ensure_ascii=False, indent=2)
    return meta


def weighted_average_hf_models(
    model_a: str,
    model_b: str,
    output_dir: str,
    *,
    weight_a: float = 0.5,
    weight_b: float = 0.5,
    torch_dtype: str = "auto",
    device_map: str = "cpu",
    overwrite: bool = False,
) -> Dict[str, Any]:
    """Weighted-average two full Hugging Face causal-LM checkpoints."""

    from transformers import AutoModelForCausalLM, AutoTokenizer

    out = Path(output_dir).expanduser().resolve()
    _prepare_output_dir(out, overwrite=overwrite)
    dtype_arg: Any = torch_dtype
    if torch_dtype == "float16":
        dtype_arg = torch.float16
    elif torch_dtype == "bfloat16":
        dtype_arg = torch.bfloat16
    elif torch_dtype == "float32":
        dtype_arg = torch.float32

    model_a_obj = AutoModelForCausalLM.from_pretrained(
        model_a,
        torch_dtype=dtype_arg,
        device_map=device_map,
    )
    model_b_obj = AutoModelForCausalLM.from_pretrained(
        model_b,
        torch_dtype=dtype_arg,
        device_map=device_map,
    )

    state_a = model_a_obj.state_dict()
    state_b = model_b_obj.state_dict()
    merged = merge_tensor_state_dicts(
        state_a,
        state_b,
        weight_a=weight_a,
        weight_b=weight_b,
        missing_policy="error",
    )
    model_a_obj.load_state_dict(merged, strict=True)
    model_a_obj.save_pretrained(str(out), safe_serialization=True)
    tokenizer = AutoTokenizer.from_pretrained(model_a)
    tokenizer.save_pretrained(str(out))

    meta = {
        "merge_type": "weighted_average_hf_models",
        "model_a": model_a,
        "model_b": model_b,
        "weight_a": float(weight_a),
        "weight_b": float(weight_b),
        "output_dir": str(out),
        "torch_dtype": torch_dtype,
    }
    with (out / "checkpoint_merge_meta.json").open("w", encoding="utf-8") as fh:
        json.dump(meta, fh, ensure_ascii=False, indent=2)
    return meta


def _add_common_weight_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--weight_a", type=float, default=0.5)
    parser.add_argument("--weight_b", type=float, default=0.5)


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Merge model/checkpoint weights.")
    subparsers = parser.add_subparsers(dest="command")

    parallel = subparsers.add_parser(
        "parallel-kv",
        help="Merge this repo's ParallelKV artifacts: cache_mapper.pt and/or judger_lora/.",
    )
    parallel.add_argument("--checkpoint_a", required=True)
    parallel.add_argument("--checkpoint_b", default="")
    parallel.add_argument("--output_dir", required=True)
    _add_common_weight_args(parallel)
    parallel.add_argument("--source_b_is_base", action="store_true")
    parallel.add_argument(
        "--missing_policy",
        choices=["error", "copy", "zero"],
        default="error",
        help="How to handle LoRA keys present in only one checkpoint.",
    )
    parallel.add_argument(
        "--cache_mapper_source",
        choices=["a", "b", "none"],
        default="a",
        help="Copy cache_mapper.pt unchanged from checkpoint a/b, or omit it.",
    )
    parallel.add_argument("--allow_config_mismatch", action="store_true")
    parallel.add_argument("--overwrite", action="store_true")

    lora = subparsers.add_parser(
        "lora-into-base",
        help="Merge a PEFT LoRA adapter into a base HF model and save a full HF checkpoint.",
    )
    lora.add_argument("--base_model", required=True)
    lora.add_argument("--adapter_checkpoint", required=True)
    lora.add_argument("--output_dir", required=True)
    lora.add_argument(
        "--lora_scale",
        type=float,
        default=1.0,
        help="Scale lambda for the LoRA delta before merge. 0.0=base, 1.0=normal full LoRA merge.",
    )
    lora.add_argument(
        "--copy_parallel_kv_artifacts",
        action="store_true",
        help="Copy cache_mapper.pt from the adapter checkpoint directory into the output directory.",
    )
    lora.add_argument("--torch_dtype", choices=["auto", "float16", "bfloat16", "float32"], default="auto")
    lora.add_argument("--device_map", default="cpu")
    lora.add_argument("--overwrite", action="store_true")

    hf_avg = subparsers.add_parser(
        "hf-average",
        help="Weighted-average two full Hugging Face causal-LM checkpoints.",
    )
    hf_avg.add_argument("--model_a", required=True)
    hf_avg.add_argument("--model_b", required=True)
    hf_avg.add_argument("--output_dir", required=True)
    _add_common_weight_args(hf_avg)
    hf_avg.add_argument("--torch_dtype", choices=["auto", "float16", "bfloat16", "float32"], default="auto")
    hf_avg.add_argument("--device_map", default="cpu")
    hf_avg.add_argument("--overwrite", action="store_true")
    return parser


def main() -> None:
    parser = build_arg_parser()
    args = parser.parse_args()
    if not args.command:
        parser.print_help()
        raise SystemExit(2)
    if args.command == "parallel-kv":
        meta = merge_parallel_kv_checkpoints(
            args.checkpoint_a,
            args.checkpoint_b or None,
            args.output_dir,
            weight_a=args.weight_a,
            weight_b=args.weight_b,
            source_b_is_base=args.source_b_is_base,
            missing_policy=args.missing_policy,
            allow_config_mismatch=args.allow_config_mismatch,
            cache_mapper_source=args.cache_mapper_source,
            overwrite=args.overwrite,
        )
    elif args.command == "lora-into-base":
        meta = merge_lora_into_base_model(
            args.base_model,
            args.adapter_checkpoint,
            args.output_dir,
            lora_scale=args.lora_scale,
            copy_parallel_kv_artifacts=args.copy_parallel_kv_artifacts,
            torch_dtype=args.torch_dtype,
            device_map=args.device_map,
            overwrite=args.overwrite,
        )
    elif args.command == "hf-average":
        meta = weighted_average_hf_models(
            args.model_a,
            args.model_b,
            args.output_dir,
            weight_a=args.weight_a,
            weight_b=args.weight_b,
            torch_dtype=args.torch_dtype,
            device_map=args.device_map,
            overwrite=args.overwrite,
        )
    else:
        raise ValueError(f"Unsupported command: {args.command}")
    print(json.dumps(meta, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

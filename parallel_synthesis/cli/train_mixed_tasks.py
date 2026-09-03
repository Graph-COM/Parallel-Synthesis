import argparse
from contextlib import nullcontext
import json
import math
import os
import random
import shutil
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch

from parallel_synthesis.data import build_dataset_iter
from parallel_synthesis.methods.fixed_parallel_kv import FixedParallelKV
from parallel_synthesis.methods.parallel_kv import ParallelKV
from parallel_synthesis.models import ModelWrapper
from parallel_synthesis.utils.checkpoints import resolve_parallel_kv_checkpoint
from parallel_synthesis.utils.common import (
    DEFAULT_MIXED_PRETRAIN_TASKS,
    SUPPORTED_METHODS,
    canonicalize_task_name,
    clone_args_with_task,
    parse_csv,
    parse_task_caps,
    safe_exp,
    set_method_task,
)
from parallel_synthesis.utils.distributed_utils import (
    average_gradients,
    barrier,
    gather_objects,
    get_rank,
    get_world_size,
    is_distributed,
    is_main_process,
    maybe_init_distributed,
    reduce_long_sums,
    reduce_weighted_loss,
)
from parallel_synthesis.utils.utils import auto_device, set_seed


def _emit_train_metric_line(message: str, train_log_path: str = "") -> None:
    print(message)
    if not train_log_path:
        return
    with open(train_log_path, "a", encoding="utf-8") as fh:
        fh.write(message.rstrip() + "\n")

def _to_serializable(value: Any) -> Any:
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(key): _to_serializable(val) for key, val in value.items()}
    if isinstance(value, (list, tuple)):
        return [_to_serializable(item) for item in value]
    return str(value)


def to_serializable_args(args: argparse.Namespace) -> Dict[str, Any]:
    return {
        str(key): _to_serializable(value)
        for key, value in sorted(vars(args).items(), key=lambda item: item[0])
    }


def parallel_kv_component_flags(mode: str) -> Tuple[bool, bool]:
    if mode == "both":
        return True, True
    if mode == "mapper_only":
        return True, False
    if mode == "lora_only":
        return False, True
    raise ValueError(f"Unsupported component mode: {mode}")


def build_method_for_mixed_training(model: ModelWrapper, args: argparse.Namespace):
    common_kwargs = dict(
        temperature=args.temperature,
        top_p=args.top_p,
        generate_bs=args.generate_bs,
        args=args,
    )
    if args.method == "parallel_kv":
        return ParallelKV(
            model,
            judger_max_new_tokens=args.max_new_tokens,
            **common_kwargs,
        )
    if args.method == "fixed_parallel_kv":
        return FixedParallelKV(
            model,
            judger_max_new_tokens=args.max_new_tokens,
            **common_kwargs,
        )
    raise ValueError(f"Unsupported method for mixed training: {args.method}")


def load_task_dataset(
    task: str,
    split: str,
    args: argparse.Namespace,
    *,
    max_samples_for_task: int,
) -> List[Dict[str, Any]]:
    task_args = clone_args_with_task(args, task, split)
    rows = list(build_dataset_iter(task, split=split, seed=args.seed, args=task_args))
    if max_samples_for_task > 0:
        rows = rows[:max_samples_for_task]
    return rows


def _training_state_path(save_dir: str) -> str:
    return os.path.join(save_dir, "training_state.pt")


def _capture_process_rng_state() -> Dict[str, Any]:
    payload: Dict[str, Any] = {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch_cpu": torch.get_rng_state(),
    }
    if torch.cuda.is_available():
        payload["torch_cuda"] = torch.cuda.get_rng_state_all()
    return payload


def _restore_process_rng_state(payload: Dict[str, Any]) -> None:
    python_state = payload.get("python")
    if python_state is not None:
        random.setstate(python_state)
    numpy_state = payload.get("numpy")
    if numpy_state is not None:
        np.random.set_state(numpy_state)
    torch_cpu_state = payload.get("torch_cpu")
    if torch_cpu_state is not None:
        torch.set_rng_state(torch_cpu_state)
    torch_cuda_state = payload.get("torch_cuda")
    if torch_cuda_state is not None and torch.cuda.is_available():
        torch.cuda.set_rng_state_all(torch_cuda_state)


def _move_value_to_device(value: Any, device: torch.device) -> Any:
    if torch.is_tensor(value):
        return value.to(device)
    if isinstance(value, dict):
        return {key: _move_value_to_device(val, device) for key, val in value.items()}
    if isinstance(value, list):
        return [_move_value_to_device(item, device) for item in value]
    if isinstance(value, tuple):
        return tuple(_move_value_to_device(item, device) for item in value)
    return value


def _move_optimizer_state_to_device(optimizer: torch.optim.Optimizer, device: torch.device) -> None:
    for param_id, state in list(optimizer.state.items()):
        optimizer.state[param_id] = _move_value_to_device(state, device)


def _training_autocast_context(method: Any, args: argparse.Namespace):
    enabled = bool(getattr(args, "bf16_autocast", False))
    device = getattr(getattr(method, "model", None), "device", None)
    if enabled and device is not None and torch.device(device).type == "cuda":
        return torch.autocast(device_type="cuda", dtype=torch.bfloat16)
    return nullcontext()


def _reset_training_cuda_peak(method: Any, args: argparse.Namespace) -> None:
    if not bool(getattr(args, "log_cuda_memory", False)) or not torch.cuda.is_available():
        return
    device = getattr(getattr(method, "model", None), "device", None)
    if device is None or torch.device(device).type != "cuda":
        return
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats(torch.device(device))


def _log_training_cuda_memory(
    method: Any,
    args: argparse.Namespace,
    *,
    rank: int,
    step: int,
) -> None:
    if not bool(getattr(args, "log_cuda_memory", False)) or not torch.cuda.is_available():
        return
    device = getattr(getattr(method, "model", None), "device", None)
    if device is None or torch.device(device).type != "cuda":
        return
    device_obj = torch.device(device)
    gib = 1024.0**3
    print(
        f"[cuda-memory] rank={rank} step={step} "
        f"allocated_gib={torch.cuda.memory_allocated(device_obj) / gib:.3f} "
        f"reserved_gib={torch.cuda.memory_reserved(device_obj) / gib:.3f} "
        f"peak_allocated_gib={torch.cuda.max_memory_allocated(device_obj) / gib:.3f} "
        f"peak_reserved_gib={torch.cuda.max_memory_reserved(device_obj) / gib:.3f}",
        flush=True,
    )


def _load_training_state_payload(load_dir: str) -> Optional[Dict[str, Any]]:
    path = _training_state_path(load_dir)
    if not os.path.exists(path):
        return None
    return torch.load(path, map_location="cpu", weights_only=False)


def _write_checkpoint_artifacts(
    save_dir: str,
    *,
    meta_payload: Dict[str, Any],
    training_state_payload: Optional[Dict[str, Any]] = None,
) -> None:
    if training_state_payload is not None:
        torch.save(training_state_payload, _training_state_path(save_dir))
    with open(os.path.join(save_dir, "checkpoint_meta.json"), "w", encoding="utf-8") as fh:
        json.dump(meta_payload, fh, ensure_ascii=False, indent=2)


def _step_checkpoint_dir(save_root: str, step: int) -> str:
    _ = step
    return os.path.join(save_root, "step_checkpoints", "latest")


def _save_step_checkpoint(
    method: Any,
    save_root: str,
    *,
    step: int,
    epoch: int,
    metrics: Dict[str, Any],
    training_state_payload: Optional[Dict[str, Any]] = None,
) -> str:
    checkpoint_dir = _step_checkpoint_dir(save_root, step)
    shutil.rmtree(checkpoint_dir, ignore_errors=True)
    os.makedirs(checkpoint_dir, exist_ok=True)
    method.save_trainable_modules(checkpoint_dir)
    meta_payload = {
        "global_step": int(step),
        "epoch": int(epoch),
        "source_run_dir": save_root,
        "metrics": metrics,
    }
    _write_checkpoint_artifacts(
        checkpoint_dir,
        meta_payload=meta_payload,
        training_state_payload=training_state_payload,
    )
    return checkpoint_dir


def _build_epoch_batches(
    train_rows_by_task: Dict[str, List[Dict[str, Any]]],
    *,
    batch_size: int,
    rng: random.Random,
    world_size: int,
) -> Tuple[List[Tuple[str, List[Dict[str, Any]]]], Dict[str, int]]:
    if world_size <= 1:
        epoch_batches: List[Tuple[str, List[Dict[str, Any]]]] = []
        for task, rows in train_rows_by_task.items():
            order = list(range(len(rows)))
            rng.shuffle(order)
            for start in range(0, len(order), batch_size):
                idxs = order[start : start + batch_size]
                epoch_batches.append((task, [rows[i] for i in idxs]))
        rng.shuffle(epoch_batches)
        return epoch_batches, {}

    grouped_epoch_batches: List[List[Tuple[str, List[Dict[str, Any]]]]] = []
    dropped_batches_by_task: Dict[str, int] = {}
    for task, rows in train_rows_by_task.items():
        order = list(range(len(rows)))
        rng.shuffle(order)
        task_batches: List[Tuple[str, List[Dict[str, Any]]]] = []
        for start in range(0, len(order), batch_size):
            idxs = order[start : start + batch_size]
            task_batches.append((task, [rows[i] for i in idxs]))

        usable = (len(task_batches) // world_size) * world_size
        dropped = len(task_batches) - usable
        if dropped > 0:
            dropped_batches_by_task[task] = dropped
        for start in range(0, usable, world_size):
            grouped_epoch_batches.append(task_batches[start : start + world_size])

    rng.shuffle(grouped_epoch_batches)
    epoch_batches = [batch for group in grouped_epoch_batches for batch in group]
    return epoch_batches, dropped_batches_by_task


def _slice_epoch_batches_for_rank(
    epoch_batches: List[Tuple[str, List[Dict[str, Any]]]],
    *,
    world_size: int,
    rank: int,
) -> Tuple[List[Tuple[str, List[Dict[str, Any]]]], int]:
    if world_size <= 1:
        return epoch_batches, 0
    usable = (len(epoch_batches) // world_size) * world_size
    dropped = len(epoch_batches) - usable
    if usable <= 0:
        raise ValueError(
            "Number of epoch batches is smaller than world size. "
            "Reduce --nproc_per_node or increase training data."
        )
    return epoch_batches[:usable][rank::world_size], dropped


def train_mixed_single(
    method: Any,
    args: argparse.Namespace,
    train_rows_by_task: Dict[str, List[Dict[str, Any]]],
    *,
    train_log_path: str = "",
) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    params = method.trainable_parameters()
    if not params:
        raise ValueError("No trainable parameters found.")

    optimizer = torch.optim.AdamW(
        params,
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
    )
    _reset_training_cuda_peak(method, args)

    tasks = list(train_rows_by_task.keys())
    total_rows = sum(len(v) for v in train_rows_by_task.values())
    batch_size = max(1, int(args.generate_bs))
    rng = random.Random(args.seed)
    global_step = 0
    optimizer_steps = 0
    resume_target_step = int(getattr(args, "resume_from_global_step", 0) or 0)
    resume_remaining_steps = resume_target_step
    resume_logged_done = resume_target_step <= 0
    total_weighted_loss = 0.0
    total_target_tokens = 0
    last_step_loss = None
    last_step_ppl = None
    epoch_summaries: List[Dict[str, Any]] = []
    per_task_steps: Dict[str, int] = {task: 0 for task in tasks}
    per_task_seen_samples: Dict[str, int] = {task: 0 for task in tasks}
    skipped_steps = 0
    skipped_samples = 0
    stateful_resume = False
    resume_epoch_idx = 0
    resume_next_batch_idx = 0
    resume_epoch_rng_state = None
    resume_epoch_partial: Dict[str, Any] = {}

    if resume_target_step > 0 and str(getattr(args, "parallel_kv_load_dir", "")).strip():
        training_state = _load_training_state_payload(args.parallel_kv_load_dir)
        if training_state is not None:
            saved_step = int(training_state.get("global_step", -1))
            step_relabeled = saved_step != resume_target_step
            extra_batch_skip = resume_target_step - saved_step if step_relabeled else 0
            if step_relabeled:
                if extra_batch_skip < 0:
                    _emit_train_metric_line(
                        f"[resume] checkpoint global_step={saved_step} > "
                        f"--resume_from_global_step={resume_target_step}; rewinding "
                        f"data position by {-extra_batch_skip} batches within the saved epoch "
                        "(only same-epoch rewinds are supported).",
                        train_log_path,
                    )
                else:
                    _emit_train_metric_line(
                        f"[resume] checkpoint global_step={saved_step} < "
                        f"--resume_from_global_step={resume_target_step}; fast-forwarding "
                        f"{extra_batch_skip} additional batches in the saved epoch's sample "
                        f"order. Next training step will be {resume_target_step + 1}.",
                        train_log_path,
                    )
            optimizer_state = training_state.get("optimizer_state")
            if optimizer_state is None:
                raise ValueError(
                    f"training_state.pt in {args.parallel_kv_load_dir} is missing optimizer_state."
                )
            optimizer.load_state_dict(optimizer_state)
            model_device = getattr(getattr(method, "model", None), "device", torch.device(args.device))
            _move_optimizer_state_to_device(optimizer, torch.device(model_device))

            process_rng_state = training_state.get("process_rng_state")
            if isinstance(process_rng_state, dict):
                _restore_process_rng_state(process_rng_state)

            loop_state = training_state.get("single_loop_state")
            if not isinstance(loop_state, dict):
                raise ValueError(
                    f"training_state.pt in {args.parallel_kv_load_dir} is missing single_loop_state."
                )
            global_step = int(loop_state.get("global_step", global_step))
            if step_relabeled:
                global_step = resume_target_step
            optimizer_steps = int(loop_state.get("optimizer_steps", optimizer_steps))
            total_weighted_loss = float(loop_state.get("total_weighted_loss", total_weighted_loss))
            total_target_tokens = int(loop_state.get("total_target_tokens", total_target_tokens))
            last_step_loss = loop_state.get("last_step_loss")
            if last_step_loss is not None:
                last_step_loss = float(last_step_loss)
            last_step_ppl = loop_state.get("last_step_ppl")
            if last_step_ppl is not None:
                last_step_ppl = float(last_step_ppl)
            epoch_summaries = list(loop_state.get("epoch_summaries", []))
            saved_per_task_steps = loop_state.get("per_task_steps", {})
            per_task_steps = {task: int(saved_per_task_steps.get(task, 0)) for task in tasks}
            saved_per_task_seen = loop_state.get("per_task_seen_samples", {})
            per_task_seen_samples = {task: int(saved_per_task_seen.get(task, 0)) for task in tasks}
            skipped_steps = int(loop_state.get("skipped_steps", skipped_steps))
            skipped_samples = int(loop_state.get("skipped_samples", skipped_samples))
            resume_epoch_idx = int(loop_state.get("epoch_idx", 0))
            resume_next_batch_idx = int(loop_state.get("next_batch_idx_in_epoch", 0))
            if extra_batch_skip != 0:
                resume_next_batch_idx += extra_batch_skip
            resume_epoch_rng_state = loop_state.get("current_epoch_rng_state_before_build")
            resume_epoch_partial = dict(loop_state.get("current_epoch_partial", {}))
            resume_remaining_steps = 0
            resume_logged_done = True
            stateful_resume = True
            _emit_train_metric_line(
                f"[resume] restored optimizer and RNG state from {args.parallel_kv_load_dir}; "
                f"continuing from global_step={global_step} at epoch={resume_epoch_idx + 1} "
                f"batch_offset={resume_next_batch_idx}",
                train_log_path,
            )

    if resume_target_step > 0:
        if not stateful_resume:
            _emit_train_metric_line(
                f"[resume] fast-forwarding the first {resume_target_step} absolute batch steps before continuing training",
                train_log_path,
            )

    start_epoch_idx = resume_epoch_idx if stateful_resume else 0
    for epoch_idx in range(start_epoch_idx, args.num_epochs):
        if hasattr(method, "model") and hasattr(method.model, "model"):
            method.model.model.train()
        if hasattr(method, "cache_mapper") and method.cache_mapper is not None:
            method.cache_mapper.train()

        batch_index_offset = 0
        if stateful_resume and epoch_idx == resume_epoch_idx:
            if resume_epoch_rng_state is None:
                raise ValueError("Resume state is missing current_epoch_rng_state_before_build.")
            rng = random.Random()
            rng.setstate(resume_epoch_rng_state)
            epoch_rng_state_before_build = resume_epoch_rng_state
            epoch_batches, _ = _build_epoch_batches(
                train_rows_by_task,
                batch_size=batch_size,
                rng=rng,
                world_size=1,
            )
            if args.steps_per_epoch > 0:
                epoch_batches = epoch_batches[: int(args.steps_per_epoch)]
            if resume_next_batch_idx < 0 or resume_next_batch_idx > len(epoch_batches):
                raise ValueError(
                    "Resume state has an invalid batch offset for the current epoch: "
                    f"{resume_next_batch_idx} for epoch size {len(epoch_batches)}. "
                    "If you used --resume_from_global_step to override the saved step, "
                    "the requested jump crosses an epoch boundary, which is not "
                    "supported (we only have RNG state for the saved epoch). Pick a "
                    "value within the same epoch as the checkpoint."
                )
            epoch_weighted_loss = float(resume_epoch_partial.get("epoch_weighted_loss", 0.0))
            epoch_target_tokens = int(resume_epoch_partial.get("epoch_target_tokens", 0))
            epoch_attempted_steps = int(resume_epoch_partial.get("epoch_attempted_steps", 0))
            epoch_optimizer_steps = int(resume_epoch_partial.get("epoch_optimizer_steps", 0))
            epoch_skipped_steps = int(resume_epoch_partial.get("epoch_skipped_steps", 0))
            epoch_fast_forwarded_steps = int(
                resume_epoch_partial.get("epoch_fast_forwarded_steps", 0)
            )
            batch_index_offset = resume_next_batch_idx
            epoch_batches = epoch_batches[resume_next_batch_idx:]
            stateful_resume = False
        else:
            epoch_weighted_loss = 0.0
            epoch_target_tokens = 0
            epoch_attempted_steps = 0
            epoch_optimizer_steps = 0
            epoch_skipped_steps = 0
            epoch_fast_forwarded_steps = 0
            epoch_rng_state_before_build = rng.getstate()
            epoch_batches, _ = _build_epoch_batches(
                train_rows_by_task,
                batch_size=batch_size,
                rng=rng,
                world_size=1,
            )
            if args.steps_per_epoch > 0:
                epoch_batches = epoch_batches[: int(args.steps_per_epoch)]

            if resume_remaining_steps > 0:
                skip_in_epoch = min(resume_remaining_steps, len(epoch_batches))
                if skip_in_epoch > 0:
                    global_step += skip_in_epoch
                    epoch_fast_forwarded_steps = skip_in_epoch
                    resume_remaining_steps -= skip_in_epoch
                    batch_index_offset = skip_in_epoch
                    epoch_batches = epoch_batches[skip_in_epoch:]
                    if not resume_logged_done and resume_remaining_steps == 0:
                        _emit_train_metric_line(
                            f"[resume] reached global_step={global_step}; continuing training at step={global_step + 1}",
                            train_log_path,
                        )
                        resume_logged_done = True

        for batch_offset, (task, batch) in enumerate(epoch_batches):
            absolute_batch_idx = batch_index_offset + batch_offset
            current_step = global_step + 1
            set_method_task(method, task)
            epoch_attempted_steps += 1

            optimizer.zero_grad(set_to_none=True)
            with _training_autocast_context(method, args):
                stats = method.train_batch(batch)
            if bool(stats.get("skip", False)):
                skipped_steps += 1
                skipped_samples += len(batch)
                epoch_skipped_steps += 1
                global_step = current_step
                _emit_train_metric_line(
                    f"[train] skipped epoch={epoch_idx + 1} step={current_step} "
                    f"task={task} "
                    f"reason={stats.get('skip_reason', 'unspecified')} "
                    f"msg={str(stats.get('skip_message', '')).strip()}",
                    train_log_path,
                )
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                continue

            per_task_steps[task] += 1
            per_task_seen_samples[task] += len(batch)
            optimizer_steps += 1
            epoch_optimizer_steps += 1
            loss = stats["loss"]
            loss_value = float(loss.detach().item())
            target_tokens = int(stats.get("target_tokens", 0))
            step_ppl = safe_exp(loss_value)

            loss.backward()
            optimizer.step()
            _log_training_cuda_memory(
                method,
                args,
                rank=0,
                step=current_step,
            )

            epoch_weighted_loss += loss_value * max(target_tokens, 0)
            epoch_target_tokens += max(target_tokens, 0)
            total_weighted_loss += loss_value * max(target_tokens, 0)
            total_target_tokens += max(target_tokens, 0)
            last_step_loss = loss_value
            last_step_ppl = step_ppl
            global_step = current_step

            if args.log_every_steps > 0 and global_step % args.log_every_steps == 0:
                _emit_train_metric_line(
                    f"[train] epoch={epoch_idx + 1} step={global_step} task={task} "
                    f"loss={loss_value:.4f} "
                    f"ppl={'nan' if step_ppl is None else f'{step_ppl:.4f}'} "
                    f"target_tokens={target_tokens}",
                    train_log_path,
                )

            if (
                args.parallel_kv_save_every_steps > 0
                and global_step % int(args.parallel_kv_save_every_steps) == 0
            ):
                checkpoint_training_state = {
                    "checkpoint_version": 2,
                    "global_step": int(current_step),
                    "optimizer_state": optimizer.state_dict(),
                    "lr_scheduler_state": None,
                    "amp_scaler_state": None,
                    "ema_state": None,
                    "process_rng_state": _capture_process_rng_state(),
                    "single_loop_state": {
                        "epoch_idx": int(epoch_idx),
                        "next_batch_idx_in_epoch": int(absolute_batch_idx + 1),
                        "current_epoch_rng_state_before_build": epoch_rng_state_before_build,
                        "epoch_summaries": list(epoch_summaries),
                        "global_step": int(global_step),
                        "optimizer_steps": int(optimizer_steps),
                        "total_weighted_loss": float(total_weighted_loss),
                        "total_target_tokens": int(total_target_tokens),
                        "last_step_loss": None if last_step_loss is None else float(last_step_loss),
                        "last_step_ppl": None if last_step_ppl is None else float(last_step_ppl),
                        "per_task_steps": dict(per_task_steps),
                        "per_task_seen_samples": dict(per_task_seen_samples),
                        "skipped_steps": int(skipped_steps),
                        "skipped_samples": int(skipped_samples),
                        "current_epoch_partial": {
                            "epoch_weighted_loss": float(epoch_weighted_loss),
                            "epoch_target_tokens": int(epoch_target_tokens),
                            "epoch_attempted_steps": int(epoch_attempted_steps),
                            "epoch_optimizer_steps": int(epoch_optimizer_steps),
                            "epoch_skipped_steps": int(epoch_skipped_steps),
                            "epoch_fast_forwarded_steps": int(epoch_fast_forwarded_steps),
                        },
                    },
                }
                checkpoint_dir = _save_step_checkpoint(
                    method,
                    args.parallel_kv_save_dir,
                    step=current_step,
                    epoch=epoch_idx + 1,
                    metrics={
                        "task": task,
                        "loss": round(loss_value, 6),
                        "perplexity": None if step_ppl is None else round(step_ppl, 6),
                        "target_tokens": int(target_tokens),
                    },
                    training_state_payload=checkpoint_training_state,
                )
                print(f"[save] saved step checkpoint to {checkpoint_dir}")

        epoch_avg_loss = (
            epoch_weighted_loss / epoch_target_tokens if epoch_target_tokens > 0 else None
        )
        epoch_ppl = safe_exp(epoch_avg_loss)
        epoch_summaries.append(
            {
                "epoch": epoch_idx + 1,
                "steps": int(epoch_attempted_steps),
                "optimizer_steps": int(epoch_optimizer_steps),
                "skipped_steps": int(epoch_skipped_steps),
                "fast_forwarded_steps": int(epoch_fast_forwarded_steps),
                "target_tokens": epoch_target_tokens,
                "avg_loss": round(epoch_avg_loss, 6) if epoch_avg_loss is not None else None,
                "perplexity": round(epoch_ppl, 6) if epoch_ppl is not None else None,
            }
        )
        _emit_train_metric_line(
            f"[train] epoch={epoch_idx + 1} done: "
            f"avg_loss={'nan' if epoch_avg_loss is None else f'{epoch_avg_loss:.6f}'} "
            f"ppl={'nan' if epoch_ppl is None else f'{epoch_ppl:.6f}'} "
            f"target_tokens={epoch_target_tokens}",
            train_log_path,
        )

    if resume_remaining_steps > 0:
        raise ValueError(
            "Unable to resume from the requested global step: "
            f"requested absolute global_step={resume_target_step}, "
            f"but the configured run only reached global_step={global_step}. "
            "Increase --num_epochs/--steps_per_epoch or reduce --resume_from_global_step."
        )

    avg_loss = total_weighted_loss / total_target_tokens if total_target_tokens > 0 else None
    ppl = safe_exp(avg_loss)
    metrics = {
        "global_steps": global_step,
        "optimizer_steps": int(optimizer_steps),
        "steps_per_epoch": (
            int(args.steps_per_epoch)
            if args.steps_per_epoch > 0
            else max(1, math.ceil(total_rows / batch_size))
        ),
        "world_size": 1,
        "per_task_steps": per_task_steps,
        "per_task_seen_samples": per_task_seen_samples,
        "skipped_steps": int(skipped_steps),
        "skipped_samples": int(skipped_samples),
        "target_tokens": total_target_tokens,
        "avg_loss": round(avg_loss, 6) if avg_loss is not None else None,
        "perplexity": round(ppl, 6) if ppl is not None else None,
        "last_step_loss": round(last_step_loss, 6) if last_step_loss is not None else None,
        "last_step_perplexity": round(last_step_ppl, 6) if last_step_ppl is not None else None,
        "epochs": epoch_summaries,
    }
    final_training_state = {
        "checkpoint_version": 2,
        "global_step": int(global_step),
        "optimizer_state": optimizer.state_dict(),
        "lr_scheduler_state": None,
        "amp_scaler_state": None,
        "ema_state": None,
        "process_rng_state": _capture_process_rng_state(),
        "single_loop_state": {
            "epoch_idx": int(args.num_epochs),
            "next_batch_idx_in_epoch": 0,
            "current_epoch_rng_state_before_build": rng.getstate(),
            "epoch_summaries": list(epoch_summaries),
            "global_step": int(global_step),
            "optimizer_steps": int(optimizer_steps),
            "total_weighted_loss": float(total_weighted_loss),
            "total_target_tokens": int(total_target_tokens),
            "last_step_loss": None if last_step_loss is None else float(last_step_loss),
            "last_step_ppl": None if last_step_ppl is None else float(last_step_ppl),
            "per_task_steps": dict(per_task_steps),
            "per_task_seen_samples": dict(per_task_seen_samples),
            "skipped_steps": int(skipped_steps),
            "skipped_samples": int(skipped_samples),
            "current_epoch_partial": {
                "epoch_weighted_loss": 0.0,
                "epoch_target_tokens": 0,
                "epoch_attempted_steps": 0,
                "epoch_optimizer_steps": 0,
                "epoch_skipped_steps": 0,
                "epoch_fast_forwarded_steps": 0,
            },
        },
    }
    return metrics, final_training_state


def train_mixed_distributed(
    method: Any,
    args: argparse.Namespace,
    train_rows_by_task: Dict[str, List[Dict[str, Any]]],
    *,
    device: torch.device,
    train_log_path: str = "",
) -> Tuple[Dict[str, Any], Optional[Dict[str, Any]]]:
    params = method.trainable_parameters()
    if not params:
        raise ValueError("No trainable parameters found.")

    optimizer = torch.optim.AdamW(
        params,
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
    )
    _reset_training_cuda_peak(method, args)

    tasks = list(train_rows_by_task.keys())
    world_size = get_world_size()
    rank = get_rank()
    batch_size = max(1, int(args.generate_bs))
    rng = random.Random(args.seed)
    global_step = 0
    optimizer_sync_steps_local = 0
    resume_target_step = int(getattr(args, "resume_from_global_step", 0) or 0)
    resume_remaining_steps = resume_target_step
    resume_logged_done = resume_target_step <= 0
    total_weighted_loss_global = 0.0
    total_target_tokens_global = 0
    last_step_loss = None
    last_step_ppl = None
    epoch_summaries: List[Dict[str, Any]] = []
    task_to_idx = {task: idx for idx, task in enumerate(tasks)}
    per_task_steps_local = [0 for _ in tasks]
    per_task_seen_samples_local = [0 for _ in tasks]
    skipped_sync_steps_local = 0
    skipped_samples_local = 0
    stateful_resume = False
    resume_epoch_idx = 0
    resume_next_batch_idx = 0
    resume_epoch_rng_state = None
    resume_epoch_partial: Dict[str, Any] = {}

    if resume_target_step > 0 and str(getattr(args, "parallel_kv_load_dir", "")).strip():
        training_state = _load_training_state_payload(args.parallel_kv_load_dir)
        if training_state is not None:
            saved_step = int(training_state.get("global_step", -1))
            step_relabeled = saved_step != resume_target_step
            extra_batch_skip = resume_target_step - saved_step if step_relabeled else 0
            if step_relabeled and is_main_process():
                if extra_batch_skip < 0:
                    _emit_train_metric_line(
                        f"[resume] checkpoint global_step={saved_step} > "
                        f"--resume_from_global_step={resume_target_step}; rewinding "
                        f"data position by {-extra_batch_skip} batches within the saved epoch "
                        "(only same-epoch rewinds are supported).",
                        train_log_path,
                    )
                else:
                    _emit_train_metric_line(
                        f"[resume] checkpoint global_step={saved_step} < "
                        f"--resume_from_global_step={resume_target_step}; fast-forwarding "
                        f"{extra_batch_skip} additional batches in the saved epoch's sample "
                        f"order. Next training step will be {resume_target_step + 1}.",
                        train_log_path,
                    )
            optimizer_state = training_state.get("optimizer_state")
            if optimizer_state is None:
                raise ValueError(
                    f"training_state.pt in {args.parallel_kv_load_dir} is missing optimizer_state."
                )
            optimizer.load_state_dict(optimizer_state)
            model_device = getattr(getattr(method, "model", None), "device", device)
            _move_optimizer_state_to_device(optimizer, torch.device(model_device))

            loop_state = training_state.get("distributed_loop_state")
            if not isinstance(loop_state, dict):
                raise ValueError(
                    f"training_state.pt in {args.parallel_kv_load_dir} is missing distributed_loop_state."
                )
            saved_world_size = int(loop_state.get("world_size", world_size))
            if saved_world_size != world_size:
                raise ValueError(
                    "Distributed resume requires the same world size as the saved checkpoint: "
                    f"checkpoint={saved_world_size}, current={world_size}."
                )
            saved_tasks = list(loop_state.get("tasks", tasks))
            if saved_tasks != tasks:
                raise ValueError(
                    "Distributed resume requires the same task order as the saved checkpoint: "
                    f"checkpoint={saved_tasks}, current={tasks}."
                )
            global_step = int(loop_state.get("global_step", global_step))
            if step_relabeled:
                global_step = resume_target_step
            optimizer_sync_steps_local = int(
                loop_state.get("optimizer_sync_steps_local", optimizer_sync_steps_local)
            )
            total_weighted_loss_global = float(
                loop_state.get("total_weighted_loss_global", total_weighted_loss_global)
            )
            total_target_tokens_global = int(
                loop_state.get("total_target_tokens_global", total_target_tokens_global)
            )
            last_step_loss = loop_state.get("last_step_loss")
            if last_step_loss is not None:
                last_step_loss = float(last_step_loss)
            last_step_ppl = loop_state.get("last_step_ppl")
            if last_step_ppl is not None:
                last_step_ppl = float(last_step_ppl)
            epoch_summaries = list(loop_state.get("epoch_summaries", []))
            resume_epoch_idx = int(loop_state.get("epoch_idx", 0))
            resume_next_batch_idx = int(loop_state.get("next_batch_idx_in_epoch", 0))
            if extra_batch_skip != 0:
                resume_next_batch_idx += extra_batch_skip
            resume_epoch_rng_state = loop_state.get("current_epoch_rng_state_before_build")
            resume_epoch_partial = dict(loop_state.get("current_epoch_partial", {}))

            rank_state_map = training_state.get("rank_state_by_rank", {})
            rank_state = rank_state_map.get(rank)
            if rank_state is None:
                rank_state = rank_state_map.get(str(rank))
            if not isinstance(rank_state, dict):
                raise ValueError(
                    "Distributed resume checkpoint is missing per-rank state for "
                    f"rank {rank}."
                )
            process_rng_state = rank_state.get("process_rng_state")
            if isinstance(process_rng_state, dict):
                _restore_process_rng_state(process_rng_state)
            saved_local_steps = list(rank_state.get("per_task_steps_local", per_task_steps_local))
            if len(saved_local_steps) != len(tasks):
                raise ValueError("Saved per_task_steps_local length does not match current task list.")
            per_task_steps_local = [int(value) for value in saved_local_steps]
            saved_local_seen = list(
                rank_state.get("per_task_seen_samples_local", per_task_seen_samples_local)
            )
            if len(saved_local_seen) != len(tasks):
                raise ValueError(
                    "Saved per_task_seen_samples_local length does not match current task list."
                )
            per_task_seen_samples_local = [int(value) for value in saved_local_seen]
            skipped_sync_steps_local = int(
                rank_state.get("skipped_sync_steps_local", skipped_sync_steps_local)
            )
            skipped_samples_local = int(rank_state.get("skipped_samples_local", skipped_samples_local))
            resume_remaining_steps = 0
            resume_logged_done = True
            stateful_resume = True
            if is_main_process():
                _emit_train_metric_line(
                    f"[resume] restored distributed optimizer and RNG state from {args.parallel_kv_load_dir}; "
                    f"continuing from global_step={global_step} at epoch={resume_epoch_idx + 1} "
                    f"batch_offset={resume_next_batch_idx}",
                    train_log_path,
                )

    if resume_target_step > 0 and is_main_process():
        if not stateful_resume:
            _emit_train_metric_line(
                f"[resume] fast-forwarding the first {resume_target_step} absolute batch steps before continuing training",
                train_log_path,
            )

    start_epoch_idx = resume_epoch_idx if stateful_resume else 0
    for epoch_idx in range(start_epoch_idx, args.num_epochs):
        if hasattr(method, "model") and hasattr(method.model, "model"):
            method.model.model.train()
        if hasattr(method, "cache_mapper") and method.cache_mapper is not None:
            method.cache_mapper.train()

        batch_index_offset = 0
        if stateful_resume and epoch_idx == resume_epoch_idx:
            if resume_epoch_rng_state is None:
                raise ValueError("Resume state is missing current_epoch_rng_state_before_build.")
            rng = random.Random()
            rng.setstate(resume_epoch_rng_state)
            epoch_rng_state_before_build = resume_epoch_rng_state
            epoch_batches, dropped_batches_by_task = _build_epoch_batches(
                train_rows_by_task,
                batch_size=batch_size,
                rng=rng,
                world_size=world_size,
            )
            if args.steps_per_epoch > 0:
                epoch_batches = epoch_batches[: int(args.steps_per_epoch)]
            local_epoch_batches, dropped_batches = _slice_epoch_batches_for_rank(
                epoch_batches,
                world_size=world_size,
                rank=rank,
            )
            if resume_next_batch_idx < 0 or resume_next_batch_idx > len(local_epoch_batches):
                raise ValueError(
                    "Resume state has an invalid local batch offset for the current epoch: "
                    f"{resume_next_batch_idx} for local epoch size {len(local_epoch_batches)}. "
                    "If you used --resume_from_global_step to override the saved step, "
                    "the requested jump crosses an epoch boundary, which is not "
                    "supported (we only have RNG state for the saved epoch). Pick a "
                    "value within the same epoch as the checkpoint."
                )
            epoch_weighted_loss_global = float(
                resume_epoch_partial.get("epoch_weighted_loss_global", 0.0)
            )
            epoch_target_tokens_global = int(
                resume_epoch_partial.get("epoch_target_tokens_global", 0)
            )
            epoch_attempted_sync_steps_local = int(
                resume_epoch_partial.get("epoch_attempted_sync_steps_local", 0)
            )
            epoch_optimizer_sync_steps_local = int(
                resume_epoch_partial.get("epoch_optimizer_sync_steps_local", 0)
            )
            epoch_skipped_sync_steps_local = int(
                resume_epoch_partial.get("epoch_skipped_sync_steps_local", 0)
            )
            epoch_fast_forwarded_sync_steps_local = int(
                resume_epoch_partial.get("epoch_fast_forwarded_sync_steps_local", 0)
            )
            batch_index_offset = resume_next_batch_idx
            local_epoch_batches = local_epoch_batches[resume_next_batch_idx:]
            stateful_resume = False
        else:
            epoch_rng_state_before_build = rng.getstate()
            epoch_batches, dropped_batches_by_task = _build_epoch_batches(
                train_rows_by_task,
                batch_size=batch_size,
                rng=rng,
                world_size=world_size,
            )
            if args.steps_per_epoch > 0:
                epoch_batches = epoch_batches[: int(args.steps_per_epoch)]
            local_epoch_batches, dropped_batches = _slice_epoch_batches_for_rank(
                epoch_batches,
                world_size=world_size,
                rank=rank,
            )
            epoch_weighted_loss_global = 0.0
            epoch_target_tokens_global = 0
            epoch_attempted_sync_steps_local = 0
            epoch_optimizer_sync_steps_local = 0
            epoch_skipped_sync_steps_local = 0
            epoch_fast_forwarded_sync_steps_local = 0

            if resume_remaining_steps > 0:
                skip_in_epoch = min(resume_remaining_steps, len(local_epoch_batches))
                if skip_in_epoch > 0:
                    global_step += skip_in_epoch
                    epoch_fast_forwarded_sync_steps_local = skip_in_epoch
                    resume_remaining_steps -= skip_in_epoch
                    batch_index_offset = skip_in_epoch
                    local_epoch_batches = local_epoch_batches[skip_in_epoch:]
                    if not resume_logged_done and resume_remaining_steps == 0 and is_main_process():
                        _emit_train_metric_line(
                            f"[resume] reached global_step={global_step}; continuing training at step={global_step + 1}",
                            train_log_path,
                        )
                        resume_logged_done = True

        if dropped_batches_by_task and is_main_process():
            dropped_desc = ", ".join(
                f"{task}:{count}" for task, count in sorted(dropped_batches_by_task.items())
            )
            print(
                f"[dist] epoch={epoch_idx + 1} dropped_task_batches={dropped_desc} "
                f"to keep each synchronized step on a single task."
            )
        if dropped_batches > 0 and is_main_process():
            print(
                f"[dist] epoch={epoch_idx + 1} dropped_batches={dropped_batches} "
                f"to keep all ranks synchronized."
            )

        for batch_offset, (task, batch) in enumerate(local_epoch_batches):
            absolute_batch_idx = batch_index_offset + batch_offset
            current_step = global_step + 1
            set_method_task(method, task)
            epoch_attempted_sync_steps_local += 1

            optimizer.zero_grad(set_to_none=True)
            with _training_autocast_context(method, args):
                stats = method.train_batch(batch)
            local_skip = None
            if bool(stats.get("skip", False)):
                local_skip = {
                    "rank": int(rank),
                    "task": task,
                    "reason": str(stats.get("skip_reason", "unspecified")),
                    "message": str(stats.get("skip_message", "")).strip(),
                    "past_tokens": stats.get("past_tokens"),
                    "live_tokens": stats.get("live_tokens"),
                    "live_token_limit": stats.get("live_token_limit"),
                    "attention_area": stats.get("attention_area"),
                    "attention_area_limit": stats.get("attention_area_limit"),
                    "max_seq_tokens": stats.get("max_seq_tokens"),
                    "total_tokens": stats.get("total_tokens"),
                    "token_limit": stats.get("token_limit"),
                }
            skip_infos = gather_objects(local_skip)
            skip_details = [info for info in skip_infos if info is not None]
            if skip_details:
                skipped_sync_steps_local += 1
                skipped_samples_local += len(batch)
                epoch_skipped_sync_steps_local += 1
                global_step = current_step
                if is_main_process():
                    detail_text = "; ".join(
                        (
                            f"rank={int(info['rank'])} task={info['task']} reason={info['reason']}"
                            + (
                                f" live_tokens={int(info['live_tokens'])}/{int(info['live_token_limit'])}"
                                if info.get("live_tokens") is not None and info.get("live_token_limit") is not None
                                else ""
                            )
                            + (
                                f" total_tokens={int(info['total_tokens'])}/{int(info['token_limit'])}"
                                if info.get("total_tokens") is not None and info.get("token_limit") is not None
                                else ""
                            )
                            + (
                                f" attention_area={int(info['attention_area'])}/{int(info['attention_area_limit'])}"
                                if info.get("attention_area") is not None and info.get("attention_area_limit") is not None
                                else ""
                            )
                            + (f" msg={info['message']}" if info.get("message") else "")
                        )
                        for info in skip_details
                    )
                    _emit_train_metric_line(
                        f"[train-dist] skipped epoch={epoch_idx + 1} step={current_step} "
                        f"task={task} details={detail_text}",
                        train_log_path,
                    )
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                continue

            per_task_steps_local[task_to_idx[task]] += 1
            per_task_seen_samples_local[task_to_idx[task]] += len(batch)
            optimizer_sync_steps_local += 1
            epoch_optimizer_sync_steps_local += 1
            loss = stats["loss"]
            loss_value = float(loss.detach().item())
            target_tokens = int(stats.get("target_tokens", 0))

            loss.backward()
            average_gradients(params)
            optimizer.step()
            _log_training_cuda_memory(
                method,
                args,
                rank=rank,
                step=current_step,
            )

            global_loss, global_tokens = reduce_weighted_loss(loss_value, target_tokens, device)
            global_ppl = safe_exp(global_loss)
            epoch_weighted_loss_global += global_loss * max(global_tokens, 0)
            epoch_target_tokens_global += max(global_tokens, 0)
            total_weighted_loss_global += global_loss * max(global_tokens, 0)
            total_target_tokens_global += max(global_tokens, 0)
            last_step_loss = global_loss
            last_step_ppl = global_ppl
            global_step = current_step

            if args.log_every_steps > 0 and global_step % args.log_every_steps == 0:
                rank_tasks = gather_objects(task)
                if is_main_process():
                    _emit_train_metric_line(
                        f"[train-dist] epoch={epoch_idx + 1} step={global_step} "
                        f"rank_tasks={rank_tasks} "
                        f"loss={'nan' if math.isnan(global_loss) else f'{global_loss:.4f}'} "
                        f"ppl={'nan' if global_ppl is None else f'{global_ppl:.4f}'} "
                        f"target_tokens={global_tokens}",
                        train_log_path,
                    )

            if (
                args.parallel_kv_save_every_steps > 0
                and global_step % int(args.parallel_kv_save_every_steps) == 0
            ):
                local_rank_checkpoint_state = {
                    "rank": int(rank),
                    "process_rng_state": _capture_process_rng_state(),
                    "per_task_steps_local": list(per_task_steps_local),
                    "per_task_seen_samples_local": list(per_task_seen_samples_local),
                    "skipped_sync_steps_local": int(skipped_sync_steps_local),
                    "skipped_samples_local": int(skipped_samples_local),
                }
                gathered_rank_states = gather_objects(local_rank_checkpoint_state)
                if is_main_process():
                    checkpoint_training_state = {
                        "checkpoint_version": 2,
                        "global_step": int(current_step),
                        "optimizer_state": optimizer.state_dict(),
                        "lr_scheduler_state": None,
                        "amp_scaler_state": None,
                        "ema_state": None,
                        "distributed_loop_state": {
                            "tasks": list(tasks),
                            "world_size": int(world_size),
                            "epoch_idx": int(epoch_idx),
                            "next_batch_idx_in_epoch": int(absolute_batch_idx + 1),
                            "current_epoch_rng_state_before_build": epoch_rng_state_before_build,
                            "epoch_summaries": list(epoch_summaries),
                            "global_step": int(global_step),
                            "optimizer_sync_steps_local": int(optimizer_sync_steps_local),
                            "total_weighted_loss_global": float(total_weighted_loss_global),
                            "total_target_tokens_global": int(total_target_tokens_global),
                            "last_step_loss": None if last_step_loss is None else float(last_step_loss),
                            "last_step_ppl": None if last_step_ppl is None else float(last_step_ppl),
                            "current_epoch_partial": {
                                "epoch_weighted_loss_global": float(epoch_weighted_loss_global),
                                "epoch_target_tokens_global": int(epoch_target_tokens_global),
                                "epoch_attempted_sync_steps_local": int(epoch_attempted_sync_steps_local),
                                "epoch_optimizer_sync_steps_local": int(epoch_optimizer_sync_steps_local),
                                "epoch_skipped_sync_steps_local": int(epoch_skipped_sync_steps_local),
                                "epoch_fast_forwarded_sync_steps_local": int(
                                    epoch_fast_forwarded_sync_steps_local
                                ),
                            },
                        },
                        "rank_state_by_rank": {
                            int(item["rank"]): item for item in gathered_rank_states if item is not None
                        },
                    }
                    checkpoint_dir = _save_step_checkpoint(
                        method,
                        args.parallel_kv_save_dir,
                        step=current_step,
                        epoch=epoch_idx + 1,
                        metrics={
                            "task": task,
                            "loss": None if math.isnan(global_loss) else round(global_loss, 6),
                            "perplexity": None if global_ppl is None else round(global_ppl, 6),
                            "target_tokens": int(global_tokens),
                        },
                        training_state_payload=checkpoint_training_state,
                    )
                    print(f"[save] saved step checkpoint to {checkpoint_dir}")
                barrier()

        epoch_avg_loss = (
            epoch_weighted_loss_global / epoch_target_tokens_global
            if epoch_target_tokens_global > 0
            else None
        )
        epoch_ppl = safe_exp(epoch_avg_loss)
        epoch_summaries.append(
            {
                "epoch": epoch_idx + 1,
                "steps": int(epoch_attempted_sync_steps_local),
                "optimizer_steps": int(epoch_optimizer_sync_steps_local),
                "skipped_sync_steps": int(epoch_skipped_sync_steps_local),
                "fast_forwarded_steps": int(epoch_fast_forwarded_sync_steps_local),
                "dropped_batches": int(dropped_batches),
                "target_tokens": epoch_target_tokens_global,
                "avg_loss": round(epoch_avg_loss, 6) if epoch_avg_loss is not None else None,
                "perplexity": round(epoch_ppl, 6) if epoch_ppl is not None else None,
            }
        )
        if is_main_process():
            _emit_train_metric_line(
                f"[train-dist] epoch={epoch_idx + 1} done: "
                f"avg_loss={'nan' if epoch_avg_loss is None else f'{epoch_avg_loss:.6f}'} "
                f"ppl={'nan' if epoch_ppl is None else f'{epoch_ppl:.6f}'} "
                f"target_tokens={epoch_target_tokens_global}",
                train_log_path,
            )

    if resume_remaining_steps > 0:
        raise ValueError(
            "Unable to resume from the requested global step: "
            f"requested absolute global_step={resume_target_step}, "
            f"but the configured run only reached global_step={global_step}. "
            "Increase --num_epochs/--steps_per_epoch or reduce --resume_from_global_step."
        )

    per_task_steps = {
        task: value
        for task, value in zip(tasks, reduce_long_sums(per_task_steps_local, device))
    }
    per_task_seen_samples = {
        task: value
        for task, value in zip(tasks, reduce_long_sums(per_task_seen_samples_local, device))
    }
    avg_loss = (
        total_weighted_loss_global / total_target_tokens_global if total_target_tokens_global > 0 else None
    )
    ppl = safe_exp(avg_loss)
    skipped_sync_steps, skipped_samples = reduce_long_sums(
        [skipped_sync_steps_local, skipped_samples_local],
        device,
    )
    optimizer_sync_steps = reduce_long_sums([optimizer_sync_steps_local], device)[0] // max(world_size, 1)
    metrics = {
        "global_steps": global_step,
        "optimizer_steps": int(optimizer_sync_steps),
        "steps_per_epoch": epoch_summaries[-1]["steps"] if epoch_summaries else 0,
        "world_size": world_size,
        "per_task_steps": per_task_steps,
        "per_task_seen_samples": per_task_seen_samples,
        "skipped_sync_steps": int(skipped_sync_steps // max(world_size, 1)),
        "skipped_samples": int(skipped_samples),
        "target_tokens": total_target_tokens_global,
        "avg_loss": round(avg_loss, 6) if avg_loss is not None else None,
        "perplexity": round(ppl, 6) if ppl is not None else None,
        "last_step_loss": round(last_step_loss, 6) if last_step_loss is not None else None,
        "last_step_perplexity": round(last_step_ppl, 6) if last_step_ppl is not None else None,
        "epochs": epoch_summaries,
    }
    local_rank_final_state = {
        "rank": int(rank),
        "process_rng_state": _capture_process_rng_state(),
        "per_task_steps_local": list(per_task_steps_local),
        "per_task_seen_samples_local": list(per_task_seen_samples_local),
        "skipped_sync_steps_local": int(skipped_sync_steps_local),
        "skipped_samples_local": int(skipped_samples_local),
    }
    gathered_rank_states = gather_objects(local_rank_final_state)
    final_training_state: Optional[Dict[str, Any]] = None
    if is_main_process():
        final_training_state = {
            "checkpoint_version": 2,
            "global_step": int(global_step),
            "optimizer_state": optimizer.state_dict(),
            "lr_scheduler_state": None,
            "amp_scaler_state": None,
            "ema_state": None,
            "distributed_loop_state": {
                "tasks": list(tasks),
                "world_size": int(world_size),
                "epoch_idx": int(args.num_epochs),
                "next_batch_idx_in_epoch": 0,
                "current_epoch_rng_state_before_build": rng.getstate(),
                "epoch_summaries": list(epoch_summaries),
                "global_step": int(global_step),
                "optimizer_sync_steps_local": int(optimizer_sync_steps_local),
                "total_weighted_loss_global": float(total_weighted_loss_global),
                "total_target_tokens_global": int(total_target_tokens_global),
                "last_step_loss": None if last_step_loss is None else float(last_step_loss),
                "last_step_ppl": None if last_step_ppl is None else float(last_step_ppl),
                "current_epoch_partial": {
                    "epoch_weighted_loss_global": 0.0,
                    "epoch_target_tokens_global": 0,
                    "epoch_attempted_sync_steps_local": 0,
                    "epoch_optimizer_sync_steps_local": 0,
                    "epoch_skipped_sync_steps_local": 0,
                    "epoch_fast_forwarded_sync_steps_local": 0,
                },
            },
            "rank_state_by_rank": {
                int(item["rank"]): item for item in gathered_rank_states if item is not None
            },
        }
    return metrics, final_training_state


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Mixed-task ParallelKV training for parallel_kv/fixed_parallel_kv. "
            "Single-GPU and torchrun multi-GPU training are both supported from this entrypoint."
        )
    )
    parser.add_argument("--method", choices=["parallel_kv", "fixed_parallel_kv"], required=True)
    parser.add_argument("--model_name", type=str, required=True)
    parser.add_argument(
        "--tasks",
        type=str,
        default=",".join(DEFAULT_MIXED_PRETRAIN_TASKS),
        help=(
            "Comma-separated task names for mixed training. Defaults to the shared "
            "pretraining mix used across the repo."
        ),
    )
    parser.add_argument("--train_split", type=str, default="train")
    parser.add_argument(
        "--train_samples_per_task",
        type=str,
        default="",
        help="Optional CSV caps aligned with --tasks. Use -1 for full split.",
    )
    parser.add_argument(
        "--browsecomp_textmas_data_file",
        type=str,
        default="",
        help=(
            "Optional processed JSONL or JSONL.GZ override for "
            "browsecomp_textmas_toolcall. When omitted, the bundled "
            "data/browsecomp-textmas/processed/filtered_train.jsonl.gz is used."
        ),
    )

    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--think", action="store_true")
    parser.add_argument("--dist_backend", type=str, default="nccl")

    parser.add_argument("--generate_bs", type=int, default=1)
    parser.add_argument("--max_new_tokens", type=int, default=4096)
    parser.add_argument("--temperature", type=float, default=0.6)
    parser.add_argument("--top_p", type=float, default=0.95)
    parser.add_argument(
        "--attn_implementation",
        type=str,
        default="auto",
        help=(
            "Attention backend for HF model loading. "
            "Common values: auto, flash_attention_2, sdpa, eager."
        ),
    )

    parser.add_argument("--num_epochs", type=int, default=1)
    parser.add_argument("--steps_per_epoch", type=int, default=-1)
    parser.add_argument("--learning_rate", type=float, default=1e-4)
    parser.add_argument("--weight_decay", type=float, default=0.0)
    parser.add_argument("--log_every_steps", type=int, default=1)

    parser.add_argument(
        "--parallel_kv_train_components",
        choices=["both", "mapper_only", "lora_only"],
        default="both",
    )
    parser.add_argument("--parallel_kv_mapper_hidden_dim", type=int, default=32)
    parser.add_argument(
        "--parallel_kv_mapper_metadata_features",
        choices=["both", "length", "num_agents", "none"],
        default="both",
        help=(
            "Runtime metadata used to condition the affine cache mapper. Layer "
            "position is always retained; choose none for the unconditioned "
            "ablation or num_agents for the no-length ablation."
        ),
    )
    parser.add_argument(
        "--parallel_kv_num_parallel_agents",
        type=int,
        default=3,
        help="Number of worker agents to launch for parallel_kv.",
    )
    parser.add_argument("--parallel_kv_lora_r", type=int, default=16)
    parser.add_argument("--parallel_kv_lora_alpha", type=int, default=32)
    parser.add_argument("--parallel_kv_lora_dropout", type=float, default=0.0)
    parser.add_argument(
        "--parallel_kv_target_position_logits",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "For single-example batches, project vocabulary logits only at positions "
            "that predict supervised target tokens. This is mathematically equivalent "
            "to full logits with prompt labels masked by -100."
        ),
    )
    parser.add_argument(
        "--parallel_kv_target_position_logit_chunk_size",
        type=int,
        default=64,
        help=(
            "Number of supervised positions per checkpointed vocabulary-projection "
            "chunk when --parallel_kv_target_position_logits is enabled."
        ),
    )
    parser.add_argument(
        "--parallel_kv_cache_preserving_backbone_checkpoint",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "Checkpoint the complete judger backbone with a fresh differentiable "
            "copy of the mapped ParallelKV cache on each recomputation. Unlike "
            "standard Transformers layer checkpointing, this does not discard past KV."
        ),
    )
    parser.add_argument(
        "--parallel_kv_skip_train_if_total_tokens_exceed",
        type=int,
        default=20000,
        help=(
            "Skip a train batch before the judger forward when concatenated agent-cache tokens plus "
            "judger sequence tokens exceed this limit. Use -1 to disable."
        ),
    )
    parser.add_argument(
        "--parallel_kv_skip_train_if_live_tokens_exceed",
        type=int,
        default=1500,
        help=(
            "Skip a train batch before the judger forward when the live judger sequence "
            "(prompt plus supervised target) exceeds this limit. Use -1 to disable."
        ),
    )
    parser.add_argument(
        "--parallel_kv_skip_train_if_attention_area_exceed",
        type=int,
        default=20000000,
        help=(
            "Skip a train batch before the judger forward when "
            "live_tokens * (past_tokens + live_tokens) exceeds this limit. Use -1 to disable."
        ),
    )

    parser.add_argument("--parallel_kv_save_dir", type=str, default="")
    parser.add_argument(
        "--parallel_kv_load_dir",
        type=str,
        default="",
        help="Local checkpoint directory or Hugging Face Hub repository ID.",
    )
    parser.add_argument(
        "--resume_from_global_step",
        type=int,
        default=0,
        help=(
            "When >0, reload trainable modules from --parallel_kv_load_dir, deterministically skip "
            "the first N absolute batch steps from the same shuffled batch order, and continue "
            "training from batch step N+1."
        ),
    )
    parser.add_argument(
        "--parallel_kv_save_every_steps",
        type=int,
        default=5,
        help="If >0, save trainable-module checkpoints every N optimizer steps during training.",
    )
    parser.add_argument("--run_name", type=str, default="")
    parser.add_argument("--output_root", type=str, default="checkpoints")

    parser.add_argument("--fixed_parallel_kv_cache_max_tokens_per_text", type=int, default=-1)
    parser.add_argument(
        "--fixed_parallel_kv_skip_train_if_cache_total_prefill_tokens_exceed",
        type=int,
        default=25000,
        help=(
            "Skip a fixed_parallel_kv train batch before cache encoding when the total fixed-cache "
            "prefill tokens across the batch exceed this limit. Use -1 to disable."
        ),
    )
    parser.add_argument(
        "--fixed_parallel_kv_auto_cap_on_potential_oom",
        action=argparse.BooleanOptionalAction,
        default=False,
    )
    parser.add_argument("--fixed_parallel_kv_auto_cap_total_tokens_threshold", type=int, default=15000)
    parser.add_argument("--fixed_parallel_kv_auto_cap_tokens_per_text", type=int, default=1024)
    parser.add_argument("--fixed_parallel_kv_debug_print_cache_inputs", action="store_true")
    parser.add_argument(
        "--fixed_parallel_kv_debug_print_limit",
        type=int,
        default=-1,
        help="Print the first N fixed-cache samples' prefill/extract texts during training. Use -1 for no limit.",
    )
    args = parser.parse_args()
    if args.generate_bs <= 0:
        raise ValueError("--generate_bs must be > 0.")
    if args.method == "parallel_kv" and args.parallel_kv_num_parallel_agents <= 0:
        raise ValueError("--parallel_kv_num_parallel_agents must be > 0.")
    if args.parallel_kv_target_position_logit_chunk_size <= 0:
        raise ValueError("--parallel_kv_target_position_logit_chunk_size must be > 0.")
    if args.resume_from_global_step < 0:
        raise ValueError("--resume_from_global_step must be >= 0.")
    if args.resume_from_global_step > 0 and not str(args.parallel_kv_load_dir).strip():
        raise ValueError(
            "--resume_from_global_step requires --parallel_kv_load_dir so the trainable modules can be reloaded first."
        )
    return args


def main() -> None:
    args = parse_args()
    args.prompt = "hierarchical"
    args.train_parallel_kv = True
    args.use_vllm = False
    if args.parallel_kv_load_dir:
        args.parallel_kv_load_dir = str(
            resolve_parallel_kv_checkpoint(args.parallel_kv_load_dir)
        )

    tasks = [canonicalize_task_name(task) for task in parse_csv(args.tasks)]
    if not tasks:
        raise ValueError("No tasks provided. Use --tasks task1,task2,...")
    if (
        str(args.browsecomp_textmas_data_file).strip()
        and "browsecomp_textmas_toolcall" not in tasks
    ):
        raise ValueError(
            "--browsecomp_textmas_data_file requires browsecomp_textmas_toolcall "
            "to be present in --tasks."
        )

    train_affine, train_lora = parallel_kv_component_flags(args.parallel_kv_train_components)
    args.parallel_kv_enable_affine_map = train_affine
    args.parallel_kv_enable_judger_lora = train_lora

    train_sample_caps = parse_task_caps(
        tasks,
        args.train_samples_per_task,
        default_cap_fn=lambda _task: -1,
    )

    set_seed(args.seed)
    device = maybe_init_distributed(device_hint=args.device, backend=args.dist_backend)
    rank = get_rank()
    world_size = get_world_size()

    train_rows_by_task: Dict[str, List[Dict[str, Any]]] = {}
    for task in tasks:
        rows = load_task_dataset(
            task,
            split=args.train_split,
            args=args,
            max_samples_for_task=int(train_sample_caps[task]),
        )
        if not rows:
            if is_main_process():
                print(f"[warn] train split for task={task} is empty; skipping this task.")
            continue
        train_rows_by_task[task] = rows
        if is_main_process():
            print(f"[data] train task={task} rows={len(rows)} cap={int(train_sample_caps[task])}")

    if not train_rows_by_task:
        raise ValueError("No non-empty train datasets were loaded.")

    tasks = list(train_rows_by_task.keys())
    train_sample_caps = {task: train_sample_caps[task] for task in tasks}
    args.task = tasks[0]
    args.max_samples = -1

    timestamp = time.strftime("%Y%m%d_%H%M%S")
    if not args.parallel_kv_save_dir:
        safe_model = args.model_name.replace("/", "-")
        safe_tasks = "-".join(tasks)
        run_name = (
            args.run_name.strip()
            if args.run_name.strip()
            else f"mixed_train_{args.method}_{safe_tasks}_{safe_model}_{timestamp}"
        )
        args.parallel_kv_save_dir = os.path.join(args.output_root, run_name)
    if is_main_process():
        os.makedirs(args.parallel_kv_save_dir, exist_ok=True)

    args_snapshot = to_serializable_args(args)
    if is_main_process():
        with open(os.path.join(args.parallel_kv_save_dir, "run_args.json"), "w", encoding="utf-8") as fh:
            json.dump(args_snapshot, fh, ensure_ascii=False, indent=2)
    train_log_path = os.path.join(args.parallel_kv_save_dir, "train.log")
    if is_main_process():
        with open(train_log_path, "w", encoding="utf-8") as fh:
            fh.write("")

    model = ModelWrapper(
        args.model_name,
        device=device if is_distributed() else auto_device(args.device),
        use_vllm=False,
        args=args,
    )
    method = build_method_for_mixed_training(model, args)
    set_seed(args.seed + rank)

    if args.parallel_kv_load_dir:
        method.load_trainable_modules(args.parallel_kv_load_dir, trainable=True)
        if is_main_process():
            print(f"[load] loaded trainable modules from {args.parallel_kv_load_dir}")

    summary: Dict[str, Any] = {
        "mode": "mixed_task_train",
        "method": args.method,
        "model_name": args.model_name,
        "tasks": tasks,
        "train_sample_caps": train_sample_caps,
        "train_split": args.train_split,
        "save_dir": args.parallel_kv_save_dir,
        "args": args_snapshot,
        "world_size": world_size,
    }
    summary_path = os.path.join(args.parallel_kv_save_dir, "train_mixed_summary.json")

    def _write_summary_snapshot() -> None:
        if not is_main_process():
            return
        with open(summary_path, "w", encoding="utf-8") as fh:
            json.dump(summary, fh, ensure_ascii=False, indent=2)

    train_start = time.time()
    if world_size > 1:
        train_metrics, final_training_state = train_mixed_distributed(
            method,
            args,
            train_rows_by_task,
            device=device,
            train_log_path=train_log_path if is_main_process() else "",
        )
        barrier()
    else:
        train_metrics, final_training_state = train_mixed_single(
            method,
            args,
            train_rows_by_task,
            train_log_path=train_log_path,
        )

    if is_main_process():
        summary["train_metrics"] = train_metrics
        summary["train_time_sec"] = round(time.time() - train_start, 4)
        _write_summary_snapshot()
        method.save_trainable_modules(args.parallel_kv_save_dir)
        _write_checkpoint_artifacts(
            args.parallel_kv_save_dir,
            meta_payload={
                "global_step": int(train_metrics.get("global_steps", 0)),
                "epoch": int(len(train_metrics.get("epochs", []))),
                "source_run_dir": args.parallel_kv_save_dir,
                "metrics": {
                    "avg_loss": train_metrics.get("avg_loss"),
                    "perplexity": train_metrics.get("perplexity"),
                    "target_tokens": train_metrics.get("target_tokens"),
                    "optimizer_steps": train_metrics.get("optimizer_steps"),
                },
            },
            training_state_payload=final_training_state,
        )
        print(f"[save] saved trainable modules to {args.parallel_kv_save_dir}")
        summary["total_time_sec"] = round(time.time() - train_start, 4)
        _write_summary_snapshot()
        print(f"[log] wrote {summary_path}")

    if is_distributed():
        barrier()
        torch.distributed.destroy_process_group()


if __name__ == "__main__":
    main()

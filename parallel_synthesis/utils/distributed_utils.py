import os
from typing import Iterable, List, Optional

import torch
import torch.distributed as dist


def is_distributed() -> bool:
    return dist.is_available() and dist.is_initialized()


def get_rank() -> int:
    return dist.get_rank() if is_distributed() else 0


def get_world_size() -> int:
    return dist.get_world_size() if is_distributed() else 1


def is_main_process() -> bool:
    return get_rank() == 0


def maybe_init_distributed(device_hint: str = "cuda", backend: str = "nccl") -> torch.device:
    rank_env = os.environ.get("RANK")
    world_env = os.environ.get("WORLD_SIZE")
    local_rank_env = os.environ.get("LOCAL_RANK")

    if rank_env is None or world_env is None or local_rank_env is None:
        if torch.cuda.is_available() and str(device_hint).startswith("cuda"):
            return torch.device(device_hint)
        return torch.device(device_hint)

    local_rank = int(local_rank_env)
    if torch.cuda.is_available() and str(device_hint).startswith("cuda"):
        torch.cuda.set_device(local_rank)
        device = torch.device(f"cuda:{local_rank}")
    else:
        device = torch.device("cpu")
        if backend == "nccl":
            backend = "gloo"

    if not is_distributed():
        dist.init_process_group(backend=backend, init_method="env://")
    return device


def barrier() -> None:
    if is_distributed():
        dist.barrier()


def average_gradients(parameters: Iterable[torch.nn.Parameter]) -> None:
    if not is_distributed():
        return
    params = list(parameters)
    if not params:
        return

    devices = {param.device for param in params}
    if len(devices) != 1:
        raise ValueError(
            "Distributed gradient averaging expects all trainable parameters on one device; "
            f"found {sorted(str(device) for device in devices)}."
        )

    # MoE experts are conditionally active, so a parameter may have a gradient on
    # one rank and no gradient on another. Synchronize the usage mask first so all
    # ranks issue the same collectives and inactive ranks contribute zeros.
    usage = torch.tensor(
        [1 if param.grad is not None else 0 for param in params],
        dtype=torch.int32,
        device=params[0].device,
    )
    dist.all_reduce(usage, op=dist.ReduceOp.SUM)

    world_size = float(get_world_size())
    bucket_bytes = 64 * 1024 * 1024
    bucket: List[torch.nn.Parameter] = []
    bucket_size = 0

    def flush_bucket() -> None:
        nonlocal bucket, bucket_size
        if not bucket:
            return
        flat = torch.cat([param.grad.reshape(-1) for param in bucket])
        dist.all_reduce(flat, op=dist.ReduceOp.SUM)
        flat.div_(world_size)
        offset = 0
        for param in bucket:
            numel = param.numel()
            param.grad.copy_(flat[offset : offset + numel].view_as(param))
            offset += numel
        bucket = []
        bucket_size = 0

    for param, used_count in zip(params, usage.tolist()):
        if used_count <= 0:
            param.grad = None
            continue
        if param.grad is None:
            param.grad = torch.zeros_like(param)
        param_bytes = param.grad.numel() * param.grad.element_size()
        if bucket and (
            bucket[0].grad.dtype != param.grad.dtype
            or bucket_size + param_bytes > bucket_bytes
        ):
            flush_bucket()
        bucket.append(param)
        bucket_size += param_bytes
    flush_bucket()


def reduce_weighted_loss(loss_value: float, target_tokens: int, device: torch.device) -> tuple[float, int]:
    payload = torch.tensor(
        [float(loss_value) * max(int(target_tokens), 0), float(max(int(target_tokens), 0))],
        dtype=torch.float64,
        device=device,
    )
    if is_distributed():
        dist.all_reduce(payload, op=dist.ReduceOp.SUM)
    total_tokens = int(round(float(payload[1].item())))
    avg_loss = float(payload[0].item()) / total_tokens if total_tokens > 0 else float("nan")
    return avg_loss, total_tokens


def reduce_scalar_sum(value: float, device: torch.device) -> float:
    tensor = torch.tensor(float(value), dtype=torch.float64, device=device)
    if is_distributed():
        dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
    return float(tensor.item())


def reduce_long_sums(values: List[int], device: torch.device) -> List[int]:
    tensor = torch.tensor(values, dtype=torch.long, device=device)
    if is_distributed():
        dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
    return [int(x) for x in tensor.cpu().tolist()]


def gather_objects(obj) -> List[object]:
    if not is_distributed():
        return [obj]
    gathered: List[object] = [None for _ in range(get_world_size())]
    dist.all_gather_object(gathered, obj)
    return gathered

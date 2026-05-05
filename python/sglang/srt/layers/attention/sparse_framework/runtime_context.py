from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import torch

if TYPE_CHECKING:
    from sglang.srt.layers.radix_attention import RadixAttention
    from sglang.srt.model_executor.forward_batch_info import ForwardBatch
    from sglang.srt.model_executor.model_runner import ModelRunner


@dataclass
class SparseRuntimeContext:
    forward_batch: "ForwardBatch"
    layer: "RadixAttention | None"
    model_runner: "ModelRunner"
    seq_lens: torch.Tensor
    req_pool_indices: torch.Tensor
    seq_lens_cpu: list[int]
    req_pool_indices_cpu: list[int]
    out_cache_locs_cpu: list[int]
    req_to_token_pool: Any
    token_to_kv_pool: Any
    host_pool: Any | None = None
    cache_controller: Any | None = None
    memory_budget_bytes: int | None = None
    query: torch.Tensor | None = None
    key: torch.Tensor | None = None
    value: torch.Tensor | None = None
    save_kv_cache: bool = True
    kwargs: dict[str, Any] | None = None
    framework_state: dict[str, Any] | None = None

    @classmethod
    def from_batch(
        cls,
        model_runner: "ModelRunner",
        forward_batch: "ForwardBatch",
        layer: "RadixAttention | None" = None,
        host_pool: Any | None = None,
        cache_controller: Any | None = None,
        query: torch.Tensor | None = None,
        key: torch.Tensor | None = None,
        value: torch.Tensor | None = None,
        save_kv_cache: bool = True,
        kwargs: dict[str, Any] | None = None,
        framework_state: dict[str, Any] | None = None,
    ) -> "SparseRuntimeContext":
        return cls(
            forward_batch=forward_batch,
            layer=layer,
            model_runner=model_runner,
            seq_lens=forward_batch.seq_lens,
            req_pool_indices=forward_batch.req_pool_indices,
            seq_lens_cpu=_forward_batch_cpu_list(
                forward_batch,
                "seq_lens_cpu",
                fallback_name="seq_lens",
            ),
            req_pool_indices_cpu=_forward_batch_cpu_list(
                forward_batch,
                "req_pool_indices",
            ),
            out_cache_locs_cpu=_forward_batch_cpu_list(
                forward_batch,
                "out_cache_loc",
            ),
            req_to_token_pool=forward_batch.req_to_token_pool,
            token_to_kv_pool=forward_batch.token_to_kv_pool,
            host_pool=host_pool,
            cache_controller=cache_controller,
            query=query,
            key=key,
            value=value,
            save_kv_cache=save_kv_cache,
            kwargs=kwargs,
            framework_state=framework_state,
        )


def _forward_batch_cpu_list(
    forward_batch: "ForwardBatch",
    name: str,
    *,
    fallback_name: str | None = None,
) -> list[int]:
    cache = getattr(forward_batch, "_sparse_framework_cpu_lists", None)
    if cache is None:
        cache = {}
        setattr(forward_batch, "_sparse_framework_cpu_lists", cache)
    cache_key = (name, fallback_name)
    cached = cache.get(cache_key)
    if cached is not None:
        return cached

    values = _tensor_or_list_to_ints(getattr(forward_batch, name, None))
    if not values and fallback_name is not None:
        values = _tensor_or_list_to_ints(getattr(forward_batch, fallback_name, None))
    cache[cache_key] = values
    return values


def _tensor_or_list_to_ints(value: Any | None) -> list[int]:
    if value is None:
        return []
    if isinstance(value, torch.Tensor):
        return [int(x) for x in value.detach().cpu().tolist()]
    return [int(x) for x in value]

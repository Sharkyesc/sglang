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

from __future__ import annotations

import logging
from typing import Optional

import torch

from sglang.srt.layers.radix_attention import RadixAttention
from sglang.srt.mem_cache.memory_pool import MHATokenToKVPool

logger = logging.getLogger(__name__)


class SparseLayerwiseTokenToKVPool(MHATokenToKVPool):
    """One-layer GPU KV staging pool for sparse_framework.

    SGLang's regular MHA KV pool allocates one full GPU KV tensor per layer.
    For sparse_framework resident-only mode, the CPU store is the authoritative
    full KV cache and GPU KV is only a per-layer staging area used while the
    current layer runs. The same physical buffers are reused for every layer.
    """

    is_sparse_layerwise_staging_pool = True

    def __init__(
        self,
        size: int,
        page_size: int,
        dtype: torch.dtype,
        head_num: int,
        head_dim: int,
        layer_num: int,
        device: str,
        enable_memory_saver: bool,
        start_layer: Optional[int] = None,
        end_layer: Optional[int] = None,
        enable_alt_stream: bool = True,
        enable_kv_cache_copy: bool = False,
    ):
        self.logical_layer_num = int(layer_num)
        super().__init__(
            size=size,
            page_size=page_size,
            dtype=dtype,
            head_num=head_num,
            head_dim=head_dim,
            layer_num=1,
            device=device,
            enable_memory_saver=enable_memory_saver,
            start_layer=0 if start_layer is None else start_layer,
            end_layer=(0 if start_layer is None else start_layer) + 1,
            enable_alt_stream=enable_alt_stream,
            enable_kv_cache_copy=enable_kv_cache_copy,
        )
        logger.info(
            "Sparse layerwise KV staging pool enabled: logical_layers=%s "
            "physical_layers=1 size=%s page_size=%s",
            self.logical_layer_num,
            size,
            page_size,
        )

    def _get_key_buffer(self, layer_id: int):
        if self.store_dtype != self.dtype:
            return self.k_buffer[0].view(self.dtype)
        return self.k_buffer[0]

    def _get_value_buffer(self, layer_id: int):
        if self.store_dtype != self.dtype:
            return self.v_buffer[0].view(self.dtype)
        return self.v_buffer[0]

    def get_key_buffer(self, layer_id: int):
        if self.layer_transfer_counter is not None:
            self.layer_transfer_counter.wait_until(0)
        return self._get_key_buffer(layer_id)

    def get_value_buffer(self, layer_id: int):
        if self.layer_transfer_counter is not None:
            self.layer_transfer_counter.wait_until(0)
        return self._get_value_buffer(layer_id)

    def get_kv_buffer(self, layer_id: int):
        return self.get_key_buffer(layer_id), self.get_value_buffer(layer_id)

    def set_kv_buffer(
        self,
        layer: RadixAttention,
        loc: torch.Tensor,
        cache_k: torch.Tensor,
        cache_v: torch.Tensor,
        k_scale: Optional[float] = None,
        v_scale: Optional[float] = None,
        layer_id_override: Optional[int] = None,
    ):
        if cache_k.dtype != self.dtype:
            if k_scale is not None:
                cache_k.div_(k_scale)
            if v_scale is not None:
                cache_v.div_(v_scale)
            cache_k = cache_k.to(self.dtype)
            cache_v = cache_v.to(self.dtype)

        if self.store_dtype != self.dtype:
            cache_k = cache_k.view(self.store_dtype)
            cache_v = cache_v.view(self.store_dtype)

        self.k_buffer[0][loc] = cache_k
        self.v_buffer[0][loc] = cache_v

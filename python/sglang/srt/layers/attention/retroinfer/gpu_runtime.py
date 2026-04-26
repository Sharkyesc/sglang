from __future__ import annotations

from typing import Optional

import torch


class RetroInferGpuRuntime:
    """
    Placeholder runtime facade for GPU-resident metadata/buffers.

    The current integration delegates the heavy lifting to RetrievalAttention's own
    runtime object, but keeping this facade makes the backend/session/planner split
    match the intended architecture and gives us one place to move future GPU metadata
    ownership into.
    """

    def __init__(self):
        self.active_session_key: tuple[int, ...] | None = None
        self.tree_cache = None
        self.cache_controller = None
        self.host_pool = None
        self.io_backend: Optional[str] = None
        self.token_to_kv_pool_allocator = None
        self.req_to_token_pool = None
        self.device_pool = None

    def bind_session(self, session_key: tuple[int, ...]) -> None:
        self.active_session_key = session_key

    def bind_cache_manager(
        self,
        tree_cache=None,
        host_pool=None,
        io_backend: Optional[str] = None,
        token_to_kv_pool_allocator=None,
        req_to_token_pool=None,
    ) -> None:
        if tree_cache is not None:
            self.tree_cache = tree_cache
            self.cache_controller = getattr(tree_cache, "cache_controller", None)
        if host_pool is not None:
            self.host_pool = host_pool
        if io_backend is not None:
            self.io_backend = io_backend
        if token_to_kv_pool_allocator is not None:
            self.token_to_kv_pool_allocator = token_to_kv_pool_allocator
            get_kvcache = getattr(token_to_kv_pool_allocator, "get_kvcache", None)
            if callable(get_kvcache):
                self.device_pool = get_kvcache()
        if req_to_token_pool is not None:
            self.req_to_token_pool = req_to_token_pool

    def backup_device_indices_to_host(
        self,
        device_indices: torch.Tensor,
    ) -> torch.Tensor | None:
        if (
            self.host_pool is None
            or self.device_pool is None
            or device_indices.numel() == 0
        ):
            return None

        device_indices = device_indices.to(torch.int64).contiguous()
        host_indices = self.host_pool.alloc(len(device_indices))
        if host_indices is None and self.tree_cache is not None:
            evict_host = getattr(self.tree_cache, "evict_host", None)
            if callable(evict_host):
                evict_host(len(device_indices))
                host_indices = self.host_pool.alloc(len(device_indices))
        if host_indices is None:
            return None

        host_indices_for_transfer = host_indices
        if host_indices_for_transfer.device != device_indices.device:
            host_indices_for_transfer = host_indices_for_transfer.to(
                device=device_indices.device,
                dtype=torch.int64,
                non_blocking=True,
            ).contiguous()
        self.host_pool.backup_from_device_all_layer(
            self.device_pool,
            host_indices=host_indices_for_transfer,
            device_indices=device_indices,
            io_backend=self.io_backend or "kernel",
        )
        return host_indices

    def evict_device_indices(
        self,
        device_indices: torch.Tensor,
    ) -> int:
        if device_indices.numel() == 0:
            return 0
        if self.cache_controller is not None:
            return int(self.cache_controller.evict_device(device_indices))
        if self.token_to_kv_pool_allocator is not None:
            self.token_to_kv_pool_allocator.free(device_indices)
            return int(device_indices.numel())
        return 0

    def clear(self) -> None:
        self.active_session_key = None

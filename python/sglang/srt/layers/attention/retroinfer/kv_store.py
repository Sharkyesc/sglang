from __future__ import annotations

import abc
from typing import TYPE_CHECKING, Optional

import torch

from sglang.srt.layers.attention.retroinfer.types import (
    RetroInferLayerHostKVState,
    RetroInferRequestHostKVState,
)

if TYPE_CHECKING:
    from sglang.srt.mem_cache.memory_pool_host import HostKVCache


class RetroInferKVStore(abc.ABC):
    """
    Abstract CPU-side KV store for RetroInfer.

    This first layer only establishes the ownership boundary and host-residency
    bookkeeping. The existing decode path can keep using SGLang's GPU KV pool
    until later layers switch RetroInfer to host-first serving.
    """

    @abc.abstractmethod
    def is_bound(self) -> bool:
        raise NotImplementedError()

    @abc.abstractmethod
    def stage_request_from_device(
        self,
        req_pool_idx: int,
        upto_len: int,
    ) -> RetroInferRequestHostKVState:
        raise NotImplementedError()

    @abc.abstractmethod
    def get_request_host_state(
        self,
        req_pool_idx: int,
    ) -> Optional[RetroInferRequestHostKVState]:
        raise NotImplementedError()

    @abc.abstractmethod
    def get_request_layer_tensors(
        self,
        req_pool_idx: int,
        layer_id: int,
        upto_len: int,
    ) -> Optional[tuple[torch.Tensor, torch.Tensor]]:
        raise NotImplementedError()

    @abc.abstractmethod
    def get_request_layer_tensors_by_positions(
        self,
        req_pool_idx: int,
        layer_id: int,
        positions: torch.Tensor,
    ) -> Optional[tuple[torch.Tensor, torch.Tensor]]:
        raise NotImplementedError()

    @abc.abstractmethod
    def gather_batch_layer_tensors(
        self,
        req_pool_indices: list[int],
        layer_id: int,
        upto_len: int,
    ) -> Optional[tuple[torch.Tensor, torch.Tensor]]:
        raise NotImplementedError()

    @abc.abstractmethod
    def drop_request(self, req_pool_idx: int) -> None:
        raise NotImplementedError()


class RetroInferHiCacheKVStore(RetroInferKVStore):
    """
    Host-pool-backed KV bookkeeping for RetroInfer.

    The store can be created before a host pool is available. Once bound, it can
    stage a request's current KV payload from SGLang's device pool into the host
    pool and return stable residency metadata for CPU-side ownership tracking.
    """

    def __init__(
        self,
        model_runner,
        host_pool: Optional["HostKVCache"] = None,
        io_backend: Optional[str] = None,
    ):
        self.model_runner = model_runner
        self.req_to_token = model_runner.req_to_token_pool.req_to_token
        self.device_pool = model_runner.token_to_kv_pool_allocator.get_kvcache()
        self.layer_num = int(model_runner.model_config.num_hidden_layers)
        self.page_size = int(getattr(model_runner, "page_size", 1))
        self.io_backend = (
            io_backend
            or getattr(model_runner.server_args, "hicache_io_backend", "kernel")
        )
        self.host_pool = host_pool
        self.request_host_states: dict[int, RetroInferRequestHostKVState] = {}

    def bind_host_pool(
        self,
        host_pool: "HostKVCache",
        io_backend: Optional[str] = None,
    ) -> None:
        self.host_pool = host_pool
        if io_backend is not None:
            self.io_backend = io_backend

    def is_bound(self) -> bool:
        return self.host_pool is not None

    def _get_request_device_indices(
        self,
        req_pool_idx: int,
        upto_len: int,
    ) -> torch.Tensor:
        if upto_len <= 0:
            return torch.empty((0,), dtype=torch.int64, device=self.req_to_token.device)
        return self.req_to_token[req_pool_idx, :upto_len].to(torch.int64).contiguous()

    def _aligned_token_capacity(self, token_count: int) -> int:
        if token_count <= 0:
            return 0
        page_size = max(1, int(self.page_size))
        return ((token_count + page_size - 1) // page_size) * page_size

    def _page_token_counts(self, token_count: int) -> tuple[int, ...]:
        if token_count <= 0:
            return tuple()
        page_size = max(1, int(self.page_size))
        full_pages, remainder = divmod(token_count, page_size)
        counts = [page_size] * full_pages
        if remainder > 0:
            counts.append(remainder)
        return tuple(counts)

    def _page_start_indices(self, flat_indices: torch.Tensor) -> tuple[int, ...]:
        if flat_indices.numel() == 0:
            return tuple()
        page_size = max(1, int(self.page_size))
        return tuple(int(flat_indices[idx].item()) for idx in range(0, flat_indices.numel(), page_size))

    def _pad_device_indices(self, device_indices: torch.Tensor, aligned_len: int) -> torch.Tensor:
        if aligned_len <= int(device_indices.numel()):
            return device_indices
        if device_indices.numel() == 0:
            return torch.zeros(
                (aligned_len,),
                dtype=torch.int64,
                device=self.req_to_token.device,
            )
        pad_len = aligned_len - int(device_indices.numel())
        pad_value = device_indices[-1].view(1).expand(pad_len)
        return torch.cat([device_indices, pad_value], dim=0)

    def _indices_for_transfer(
        self,
        host_indices: torch.Tensor,
        device_indices: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Return index tensors on devices expected by HiCache transfer kernels."""
        device_indices = device_indices.to(torch.int64).contiguous()
        if host_indices.device != device_indices.device:
            host_indices_for_transfer = host_indices.to(
                device=device_indices.device,
                dtype=torch.int64,
                non_blocking=True,
            ).contiguous()
        else:
            host_indices_for_transfer = host_indices.to(torch.int64).contiguous()
        return host_indices_for_transfer, device_indices

    def _build_host_state(
        self,
        req_pool_idx: int,
        stored_upto: int,
        host_indices: tuple[int, ...],
        source_token_indices: tuple[int, ...],
        resident: bool,
    ) -> RetroInferRequestHostKVState:
        host_page_indices = host_indices[:: max(1, self.page_size)]
        page_token_counts = self._page_token_counts(stored_upto)
        return RetroInferRequestHostKVState(
            req_pool_idx=req_pool_idx,
            stored_upto=stored_upto,
            page_size=self.page_size,
            host_indices=host_indices,
            host_page_indices=host_page_indices,
            page_token_counts=page_token_counts,
            source_token_indices=source_token_indices,
            resident=resident,
            io_backend=self.io_backend,
            layers={
                layer_id: RetroInferLayerHostKVState(
                    layer_id=layer_id,
                    stored_upto=stored_upto,
                    page_size=self.page_size,
                    host_indices=host_indices,
                    host_page_indices=host_page_indices,
                    page_token_counts=page_token_counts,
                    source_token_indices=source_token_indices,
                    resident=resident,
                    io_backend=self.io_backend,
                )
                for layer_id in range(self.layer_num)
            },
        )

    def _host_indices_for_positions(
        self,
        request_state: RetroInferRequestHostKVState,
        positions: torch.Tensor,
    ) -> Optional[torch.Tensor]:
        if positions.numel() == 0:
            return torch.empty((0,), dtype=torch.int64, device="cpu")
        max_pos = int(positions.max().item())
        if request_state.stored_upto <= max_pos:
            return None

        page_size = max(1, int(request_state.page_size))
        host_page_indices = request_state.host_page_indices
        if not host_page_indices:
            return torch.empty((0,), dtype=torch.int64, device="cpu")

        translated = []
        for pos in positions.tolist():
            token_pos = int(pos)
            page_idx = token_pos // page_size
            page_offset = token_pos % page_size
            if page_idx >= len(host_page_indices):
                return None
            if page_idx < len(request_state.page_token_counts):
                valid_tokens = int(request_state.page_token_counts[page_idx])
                if page_offset >= valid_tokens:
                    return None
            translated.append(int(host_page_indices[page_idx]) + page_offset)

        return torch.tensor(
            translated,
            dtype=torch.int64,
            device="cpu",
        )

    def stage_request_from_device(
        self,
        req_pool_idx: int,
        upto_len: int,
    ) -> RetroInferRequestHostKVState:
        device_indices = self._get_request_device_indices(req_pool_idx, upto_len)
        source_token_indices = tuple(int(x) for x in device_indices.tolist())
        aligned_len = self._aligned_token_capacity(upto_len)
        existing_state = self.request_host_states.get(req_pool_idx)

        if not self.is_bound():
            request_state = self._build_host_state(
                req_pool_idx=req_pool_idx,
                stored_upto=upto_len,
                host_indices=tuple(),
                source_token_indices=source_token_indices,
                resident=False,
            )
            self.request_host_states[req_pool_idx] = request_state
            return request_state

        if aligned_len == 0:
            host_indices = torch.empty((0,), dtype=torch.int64)
        else:
            reusable = (
                existing_state is not None
                and existing_state.resident
                and len(existing_state.host_indices) == aligned_len
            )
            if reusable:
                host_indices = torch.tensor(
                    existing_state.host_indices,
                    dtype=torch.int64,
                )
            else:
                if (
                    existing_state is not None
                    and existing_state.resident
                    and existing_state.host_indices
                ):
                    self.host_pool.free(
                        torch.tensor(existing_state.host_indices, dtype=torch.int64)
                    )
                host_indices = self.host_pool.alloc(aligned_len)
                if host_indices is None:
                    raise RuntimeError(
                        f"RetroInfer host KV store is out of capacity for req_pool_idx={req_pool_idx}."
                    )
            padded_device_indices = self._pad_device_indices(device_indices, aligned_len)
            host_indices_for_transfer, padded_device_indices = self._indices_for_transfer(
                host_indices,
                padded_device_indices,
            )
            self.host_pool.backup_from_device_all_layer(
                self.device_pool,
                host_indices=host_indices_for_transfer,
                device_indices=padded_device_indices,
                io_backend=self.io_backend,
            )

        host_index_tuple = tuple(int(x) for x in host_indices.tolist())
        request_state = self._build_host_state(
            req_pool_idx=req_pool_idx,
            stored_upto=upto_len,
            host_indices=host_index_tuple,
            source_token_indices=source_token_indices,
            resident=True,
        )
        self.request_host_states[req_pool_idx] = request_state
        return request_state

    def append_request_chunk_from_device(
        self,
        req_pool_idx: int,
        start_pos: int,
        device_indices: torch.Tensor,
    ) -> RetroInferRequestHostKVState:
        """Append newly materialized GPU KV slots to the host KV source of truth.

        This is intentionally conservative and currently targets the host-only
        RetroInfer MVP where page_size=1. In that mode GPU slots are only a
        transient staging arena, while the host pool owns the full request KV.
        """
        if not self.is_bound():
            raise RuntimeError("RetroInfer host KV store is not bound to a host pool.")
        if self.page_size != 1:
            raise RuntimeError(
                "RetroInfer host-only incremental staging currently requires page_size=1."
            )

        device_indices = device_indices.to(torch.int64).contiguous()
        append_len = int(device_indices.numel())
        existing_state = self.request_host_states.get(req_pool_idx)
        expected_start = 0 if existing_state is None else int(existing_state.stored_upto)
        start_pos = int(start_pos)
        if start_pos < 0:
            raise RuntimeError(
                "RetroInfer host-only staging received a negative start_pos "
                f"(req_pool_idx={req_pool_idx}, start_pos={start_pos})."
            )
        if start_pos < expected_start:
            overlap = expected_start - start_pos
            if overlap >= append_len:
                return existing_state or self._build_host_state(
                    req_pool_idx=req_pool_idx,
                    stored_upto=expected_start,
                    host_indices=tuple(),
                    source_token_indices=tuple(),
                    resident=True,
                )
            device_indices = device_indices[overlap:].contiguous()
            append_len = int(device_indices.numel())
            start_pos = expected_start
        if start_pos != expected_start:
            raise RuntimeError(
                "RetroInfer host-only staging requires contiguous chunks "
                f"(req_pool_idx={req_pool_idx}, start_pos={start_pos}, "
                f"expected={expected_start})."
            )

        if append_len == 0:
            return existing_state or self._build_host_state(
                req_pool_idx=req_pool_idx,
                stored_upto=0,
                host_indices=tuple(),
                source_token_indices=tuple(),
                resident=True,
            )

        host_indices = self.host_pool.alloc(append_len)
        if host_indices is None:
            raise RuntimeError(
                f"RetroInfer host KV store is out of capacity for req_pool_idx={req_pool_idx}."
            )
        host_indices_for_transfer, device_indices = self._indices_for_transfer(
            host_indices,
            device_indices,
        )
        self.host_pool.backup_from_device_all_layer(
            self.device_pool,
            host_indices=host_indices_for_transfer,
            device_indices=device_indices,
            io_backend=self.io_backend,
        )

        old_host_indices = (
            tuple() if existing_state is None else existing_state.host_indices
        )
        old_source_indices = (
            tuple() if existing_state is None else existing_state.source_token_indices
        )
        new_host_indices = old_host_indices + tuple(
            int(x) for x in host_indices.tolist()
        )
        new_source_indices = old_source_indices + tuple(
            int(x) for x in device_indices.tolist()
        )
        request_state = self._build_host_state(
            req_pool_idx=req_pool_idx,
            stored_upto=expected_start + append_len,
            host_indices=new_host_indices,
            source_token_indices=new_source_indices,
            resident=True,
        )
        self.request_host_states[req_pool_idx] = request_state
        return request_state

    def get_request_host_state(
        self,
        req_pool_idx: int,
    ) -> Optional[RetroInferRequestHostKVState]:
        return self.request_host_states.get(req_pool_idx)

    def _gather_from_host_pool(
        self,
        layer_id: int,
        host_indices: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        assert self.host_pool is not None
        page_size = max(1, int(self.page_size))

        layout = getattr(self.host_pool, "layout", "layer_first")
        if layout == "layer_first":
            key = self.host_pool.k_buffer[layer_id][host_indices]
            value = self.host_pool.v_buffer[layer_id][host_indices]
        elif layout == "page_first":
            key = self.host_pool.k_buffer[host_indices, layer_id]
            value = self.host_pool.v_buffer[host_indices, layer_id]
        elif layout == "page_first_direct":
            page_indices = torch.div(host_indices, page_size, rounding_mode="floor")
            page_offsets = torch.remainder(host_indices, page_size)
            key = self.host_pool.k_buffer[page_indices, layer_id, page_offsets]
            value = self.host_pool.v_buffer[page_indices, layer_id, page_offsets]
        elif layout == "page_head":
            page_indices = torch.div(host_indices, page_size, rounding_mode="floor")
            page_offsets = torch.remainder(host_indices, page_size)
            key = self.host_pool.k_buffer[page_indices, :, page_offsets, layer_id, :]
            value = self.host_pool.v_buffer[page_indices, :, page_offsets, layer_id, :]
        else:
            raise NotImplementedError(
                f"RetroInferHiCacheKVStore page-aware host reads do not yet support layout '{layout}'."
            )
        return key.contiguous(), value.contiguous()

    def get_request_layer_tensors(
        self,
        req_pool_idx: int,
        layer_id: int,
        upto_len: int,
    ) -> Optional[tuple[torch.Tensor, torch.Tensor]]:
        request_state = self.get_request_host_state(req_pool_idx)
        if (
            request_state is None
            or not request_state.resident
            or not self.is_bound()
            or request_state.stored_upto < upto_len
        ):
            return None

        if upto_len <= 0:
            empty = torch.empty(
                (0,),
                dtype=self.host_pool.dtype if self.host_pool is not None else self.model_runner.dtype,
                device="cpu",
            )
            return empty, empty

        positions = torch.arange(
            upto_len,
            dtype=torch.int64,
            device="cpu",
        )
        host_indices = self._host_indices_for_positions(request_state, positions)
        if host_indices is None:
            return None
        return self._gather_from_host_pool(layer_id, host_indices)

    def get_request_layer_tensors_by_positions(
        self,
        req_pool_idx: int,
        layer_id: int,
        positions: torch.Tensor,
    ) -> Optional[tuple[torch.Tensor, torch.Tensor]]:
        request_state = self.get_request_host_state(req_pool_idx)
        if (
            request_state is None
            or not request_state.resident
            or not self.is_bound()
        ):
            return None

        if positions.numel() == 0:
            empty = torch.empty(
                (0,),
                dtype=self.host_pool.dtype if self.host_pool is not None else self.model_runner.dtype,
                device="cpu",
            )
            return empty, empty

        host_indices = self._host_indices_for_positions(request_state, positions)
        if host_indices is None:
            return None
        return self._gather_from_host_pool(layer_id, host_indices)

    def gather_batch_layer_tensors(
        self,
        req_pool_indices: list[int],
        layer_id: int,
        upto_len: int,
    ) -> Optional[tuple[torch.Tensor, torch.Tensor]]:
        if not self.is_bound():
            return None

        keys = []
        values = []
        for req_pool_idx in req_pool_indices:
            tensors = self.get_request_layer_tensors(req_pool_idx, layer_id, upto_len)
            if tensors is None:
                return None
            key, value = tensors
            keys.append(key)
            values.append(value)

        if not keys:
            empty = torch.empty((0,), dtype=self.host_pool.dtype, device="cpu")
            return empty, empty
        return torch.stack(keys, dim=0), torch.stack(values, dim=0)

    def drop_request(self, req_pool_idx: int) -> None:
        request_state = self.request_host_states.pop(req_pool_idx, None)
        if (
            request_state is None
            or not request_state.resident
            or not request_state.host_indices
            or not self.is_bound()
        ):
            return
        host_indices = torch.tensor(request_state.host_indices, dtype=torch.int64)
        self.host_pool.free(host_indices)

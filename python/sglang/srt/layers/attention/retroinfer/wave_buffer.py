from __future__ import annotations

from dataclasses import dataclass, field

import torch


@dataclass
class RetroInferLayerWorkingSetState:
    target_len: int
    slack_len: int
    req_positions: dict[int, torch.Tensor] = field(default_factory=dict)
    req_base_positions: dict[int, torch.Tensor] = field(default_factory=dict)
    req_retrieval_positions: dict[int, torch.Tensor] = field(default_factory=dict)
    req_append_positions: dict[int, torch.Tensor] = field(default_factory=dict)
    page_size: int = 1
    base_token_len: int = 0
    base_capacity_pages: int = 0
    retrieval_capacity_pages: int = 0
    retrieval_token_len: int = 0
    append_len: int = 0
    append_capacity_pages: int = 0
    base_key_pages: torch.Tensor | None = None
    base_value_pages: torch.Tensor | None = None
    retrieval_key_pages: torch.Tensor | None = None
    retrieval_value_pages: torch.Tensor | None = None
    execution_key_buffer: torch.Tensor | None = None
    execution_value_buffer: torch.Tensor | None = None
    execution_base_len: int = 0
    execution_retrieval_len: int = 0
    execution_append_len: int = 0
    retrieval_block_ids: dict[int, torch.Tensor] = field(default_factory=dict)
    retrieval_block_cache_slots: dict[int, torch.Tensor] = field(default_factory=dict)
    retrieval_block_page_table: dict[int, torch.Tensor] = field(default_factory=dict)
    retrieval_hit_block_ids: dict[int, torch.Tensor] = field(default_factory=dict)
    retrieval_hit_block_cache_slots: dict[int, torch.Tensor] = field(default_factory=dict)
    retrieval_hit_block_page_table: dict[int, torch.Tensor] = field(default_factory=dict)
    retrieval_miss_block_ids: dict[int, torch.Tensor] = field(default_factory=dict)
    retrieval_miss_block_cache_slots: dict[int, torch.Tensor] = field(default_factory=dict)
    retrieval_miss_block_page_table: dict[int, torch.Tensor] = field(default_factory=dict)
    pending_scatter_block_ids: dict[int, torch.Tensor] = field(default_factory=dict)
    pending_scatter_block_cache_slots: dict[int, torch.Tensor] = field(default_factory=dict)
    pending_scatter_block_page_table: dict[int, torch.Tensor] = field(default_factory=dict)
    cache_resident_block_ids: dict[int, torch.Tensor] = field(default_factory=dict)
    cache_resident_block_cache_slots: dict[int, torch.Tensor] = field(default_factory=dict)
    cache_resident_block_page_table: dict[int, torch.Tensor] = field(default_factory=dict)
    append_key_pages: torch.Tensor | None = None
    append_value_pages: torch.Tensor | None = None
    live_len: int = 0
    capacity_len: int = 0
    req_order: tuple[int, ...] = field(default_factory=tuple)


@dataclass
class RetroInferSessionWorkingSetState:
    session_key: tuple[int, ...]
    layers: dict[int, RetroInferLayerWorkingSetState] = field(default_factory=dict)


class RetroInferWaveBufferManager:
    """
    Minimal working-set manager for RetroInfer.

    This version only tracks per-session working-set token positions on GPU.
    Decode prefers append-only growth inside a small slack region and triggers a
    full working-set refresh only when the live positions exceed that slack.
    """

    def __init__(self):
        self.sessions: dict[tuple[int, ...], RetroInferSessionWorkingSetState] = {}

    def _empty_long(self) -> torch.Tensor:
        return torch.empty((0,), dtype=torch.long)

    def _resize_or_init_pages(
        self,
        old_pages: torch.Tensor | None,
        *,
        batch_size: int,
        capacity_pages: int,
        page_size: int,
        kv_heads: int,
        head_dim: int,
        device,
        dtype,
    ) -> torch.Tensor | None:
        if capacity_pages <= 0:
            return None
        new_pages = torch.zeros(
            (batch_size, capacity_pages, page_size, kv_heads, head_dim),
            dtype=dtype,
            device=device,
        )
        if old_pages is not None:
            copy_batch = min(int(old_pages.shape[0]), batch_size)
            copy_pages = min(int(old_pages.shape[1]), capacity_pages)
            if copy_batch > 0 and copy_pages > 0:
                new_pages[:copy_batch, :copy_pages].copy_(old_pages[:copy_batch, :copy_pages])
        return new_pages

    def _refresh_execution_buffer(
        self,
        layer_state: RetroInferLayerWorkingSetState,
    ) -> tuple[torch.Tensor, torch.Tensor, int] | None:
        parts_k = []
        parts_v = []
        layer_state.execution_base_len = 0
        layer_state.execution_retrieval_len = 0
        layer_state.execution_append_len = 0

        if layer_state.base_token_len > 0 and layer_state.base_key_pages is not None:
            base_k = layer_state.base_key_pages.reshape(
                layer_state.base_key_pages.shape[0],
                layer_state.base_capacity_pages * layer_state.page_size,
                layer_state.base_key_pages.shape[-2],
                layer_state.base_key_pages.shape[-1],
            )[:, : layer_state.base_token_len]
            base_v = layer_state.base_value_pages.reshape(
                layer_state.base_value_pages.shape[0],
                layer_state.base_capacity_pages * layer_state.page_size,
                layer_state.base_value_pages.shape[-2],
                layer_state.base_value_pages.shape[-1],
            )[:, : layer_state.base_token_len]
            parts_k.append(base_k)
            parts_v.append(base_v)
            layer_state.execution_base_len = int(base_k.shape[1])

        if layer_state.retrieval_token_len > 0 and layer_state.retrieval_key_pages is not None:
            retrieval_k = layer_state.retrieval_key_pages.reshape(
                layer_state.retrieval_key_pages.shape[0],
                layer_state.retrieval_capacity_pages * layer_state.page_size,
                layer_state.retrieval_key_pages.shape[-2],
                layer_state.retrieval_key_pages.shape[-1],
            )[:, : layer_state.retrieval_token_len]
            retrieval_v = layer_state.retrieval_value_pages.reshape(
                layer_state.retrieval_value_pages.shape[0],
                layer_state.retrieval_capacity_pages * layer_state.page_size,
                layer_state.retrieval_value_pages.shape[-2],
                layer_state.retrieval_value_pages.shape[-1],
            )[:, : layer_state.retrieval_token_len]
            parts_k.append(retrieval_k)
            parts_v.append(retrieval_v)
            layer_state.execution_retrieval_len = int(retrieval_k.shape[1])

        if layer_state.append_len > 0 and layer_state.append_key_pages is not None:
            append_k = layer_state.append_key_pages.reshape(
                layer_state.append_key_pages.shape[0],
                layer_state.append_capacity_pages * layer_state.page_size,
                layer_state.append_key_pages.shape[-2],
                layer_state.append_key_pages.shape[-1],
            )[:, : layer_state.append_len]
            append_v = layer_state.append_value_pages.reshape(
                layer_state.append_value_pages.shape[0],
                layer_state.append_capacity_pages * layer_state.page_size,
                layer_state.append_value_pages.shape[-2],
                layer_state.append_value_pages.shape[-1],
            )[:, : layer_state.append_len]
            parts_k.append(append_k)
            parts_v.append(append_v)
            layer_state.execution_append_len = int(append_k.shape[1])

        if not parts_k:
            layer_state.execution_key_buffer = None
            layer_state.execution_value_buffer = None
            layer_state.live_len = 0
            return None

        layer_state.execution_key_buffer = torch.cat(parts_k, dim=1).contiguous()
        layer_state.execution_value_buffer = torch.cat(parts_v, dim=1).contiguous()
        layer_state.live_len = int(layer_state.execution_key_buffer.shape[1])
        return (
            layer_state.execution_key_buffer,
            layer_state.execution_value_buffer,
            layer_state.live_len,
        )

    def _compose_req_positions(
        self,
        layer_state: RetroInferLayerWorkingSetState,
        req_pool_idx: int,
    ) -> torch.Tensor:
        parts = []
        base_positions = layer_state.req_base_positions.get(int(req_pool_idx))
        retrieval_positions = layer_state.req_retrieval_positions.get(int(req_pool_idx))
        append_positions = layer_state.req_append_positions.get(int(req_pool_idx))
        if base_positions is not None and base_positions.numel() > 0:
            parts.append(base_positions)
        if retrieval_positions is not None and retrieval_positions.numel() > 0:
            parts.append(retrieval_positions)
        if append_positions is not None and append_positions.numel() > 0:
            parts.append(append_positions)
        if not parts:
            return self._empty_long()
        return torch.cat(parts, dim=0).to(torch.long).contiguous()

    def _set_request_segments(
        self,
        layer_state: RetroInferLayerWorkingSetState,
        req_pool_idx: int,
        *,
        base_positions: torch.Tensor | None = None,
        retrieval_positions: torch.Tensor | None = None,
        append_positions: torch.Tensor | None = None,
    ) -> None:
        req_pool_idx = int(req_pool_idx)
        if base_positions is not None:
            layer_state.req_base_positions[req_pool_idx] = base_positions.to(torch.long).clone()
        if retrieval_positions is not None:
            layer_state.req_retrieval_positions[req_pool_idx] = retrieval_positions.to(torch.long).clone()
        if append_positions is not None:
            layer_state.req_append_positions[req_pool_idx] = append_positions.to(torch.long).clone()
        layer_state.req_positions[req_pool_idx] = self._compose_req_positions(
            layer_state,
            req_pool_idx,
        )

    def _block_ids(
        self,
        positions: torch.Tensor,
        page_size: int,
    ) -> torch.Tensor:
        if positions.numel() == 0:
            return self._empty_long()
        ordered: list[int] = []
        seen: set[int] = set()
        for pos in positions.tolist():
            block_id = int(pos) // max(1, int(page_size))
            if block_id in seen:
                continue
            seen.add(block_id)
            ordered.append(block_id)
        if not ordered:
            return self._empty_long()
        return torch.tensor(ordered, dtype=torch.long)

    def _select_block_mapping(
        self,
        source_block_ids: torch.Tensor,
        source_cache_slots: torch.Tensor,
        target_block_ids: list[int],
        page_size: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if source_block_ids.numel() == 0 or source_cache_slots.numel() == 0 or not target_block_ids:
            return self._empty_long(), self._empty_long()
        block_to_slot = {
            int(block_id): int(cache_slot)
            for block_id, cache_slot in zip(
                source_block_ids.tolist(),
                source_cache_slots.tolist(),
            )
        }
        slots = [block_to_slot[block_id] for block_id in target_block_ids if block_id in block_to_slot]
        if not slots:
            return self._empty_long(), self._empty_long()
        cache_slots = torch.tensor(slots, dtype=torch.long)
        page_table = torch.div(
            cache_slots,
            max(1, int(page_size)),
            rounding_mode="floor",
        )
        return cache_slots, page_table

    def _resolve_block_cache_mapping(
        self,
        req_pool_idx: int,
        block_ids: torch.Tensor,
        page_size: int,
        cache_slot_lookup_fn=None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if block_ids.numel() == 0 or cache_slot_lookup_fn is None:
            return self._empty_long(), self._empty_long()
        block_starts = block_ids.to(torch.long) * max(1, int(page_size))
        cache_slots = cache_slot_lookup_fn(int(req_pool_idx), block_starts)
        if cache_slots is None:
            return self._empty_long(), self._empty_long()
        cache_slots = cache_slots.to(torch.long).cpu().contiguous()
        if cache_slots.numel() != block_ids.numel():
            return self._empty_long(), self._empty_long()
        page_table = torch.div(
            cache_slots,
            max(1, int(page_size)),
            rounding_mode="floor",
        )
        return cache_slots, page_table

    def _update_retrieval_cache_state(
        self,
        layer_state: RetroInferLayerWorkingSetState,
        req_pool_idx: int,
        retrieval_positions: torch.Tensor,
        cache_slot_lookup_fn=None,
    ) -> None:
        current = self._block_ids(retrieval_positions, layer_state.page_size)
        current_cache_slots, current_page_table = self._resolve_block_cache_mapping(
            req_pool_idx=int(req_pool_idx),
            block_ids=current,
            page_size=layer_state.page_size,
            cache_slot_lookup_fn=cache_slot_lookup_fn,
        )
        resident = layer_state.cache_resident_block_ids.get(int(req_pool_idx))
        if resident is None:
            resident = self._empty_long()
        resident_set = set(int(block_id) for block_id in resident.tolist())
        current_list = [int(block_id) for block_id in current.tolist()]

        hit = [block_id for block_id in current_list if block_id in resident_set]
        miss = [block_id for block_id in current_list if block_id not in resident_set]

        layer_state.retrieval_block_ids[int(req_pool_idx)] = current
        layer_state.retrieval_block_cache_slots[int(req_pool_idx)] = current_cache_slots
        layer_state.retrieval_block_page_table[int(req_pool_idx)] = current_page_table
        layer_state.retrieval_hit_block_ids[int(req_pool_idx)] = torch.tensor(
            hit, dtype=torch.long
        )
        (
            layer_state.retrieval_hit_block_cache_slots[int(req_pool_idx)],
            layer_state.retrieval_hit_block_page_table[int(req_pool_idx)],
        ) = self._select_block_mapping(
            source_block_ids=current,
            source_cache_slots=current_cache_slots,
            target_block_ids=hit,
            page_size=layer_state.page_size,
        )
        layer_state.retrieval_miss_block_ids[int(req_pool_idx)] = torch.tensor(
            miss, dtype=torch.long
        )
        (
            layer_state.retrieval_miss_block_cache_slots[int(req_pool_idx)],
            layer_state.retrieval_miss_block_page_table[int(req_pool_idx)],
        ) = self._select_block_mapping(
            source_block_ids=current,
            source_cache_slots=current_cache_slots,
            target_block_ids=miss,
            page_size=layer_state.page_size,
        )
        layer_state.pending_scatter_block_ids[int(req_pool_idx)] = torch.tensor(
            miss, dtype=torch.long
        )
        (
            layer_state.pending_scatter_block_cache_slots[int(req_pool_idx)],
            layer_state.pending_scatter_block_page_table[int(req_pool_idx)],
        ) = self._select_block_mapping(
            source_block_ids=current,
            source_cache_slots=current_cache_slots,
            target_block_ids=miss,
            page_size=layer_state.page_size,
        )

    def mark_scatter_complete(
        self,
        session_key: tuple[int, ...],
        layer_id: int,
    ) -> None:
        session_state = self.sessions.get(session_key)
        if session_state is None:
            return
        layer_state = session_state.layers.get(layer_id)
        if layer_state is None:
            return
        for req_pool_idx, pending in list(layer_state.pending_scatter_block_ids.items()):
            if pending.numel() == 0:
                continue
            resident = layer_state.cache_resident_block_ids.get(
                int(req_pool_idx),
                self._empty_long(),
            )
            resident_cache_slots = layer_state.cache_resident_block_cache_slots.get(
                int(req_pool_idx),
                self._empty_long(),
            )
            pending_cache_slots = layer_state.pending_scatter_block_cache_slots.get(
                int(req_pool_idx),
                self._empty_long(),
            )
            merged_map = {
                int(block_id): int(cache_slot)
                for block_id, cache_slot in zip(
                    resident.tolist(),
                    resident_cache_slots.tolist(),
                )
            }
            merged_map.update(
                {
                    int(block_id): int(cache_slot)
                    for block_id, cache_slot in zip(
                        pending.tolist(),
                        pending_cache_slots.tolist(),
                    )
                }
            )
            merged_ids = sorted(merged_map.keys())
            merged_slots = [merged_map[block_id] for block_id in merged_ids]
            layer_state.cache_resident_block_ids[int(req_pool_idx)] = torch.tensor(
                merged_ids,
                dtype=torch.long,
            )
            merged_cache_slots = torch.tensor(merged_slots, dtype=torch.long)
            layer_state.cache_resident_block_cache_slots[int(req_pool_idx)] = merged_cache_slots
            layer_state.cache_resident_block_page_table[int(req_pool_idx)] = torch.div(
                merged_cache_slots,
                max(1, int(layer_state.page_size)),
                rounding_mode="floor",
            )
            layer_state.pending_scatter_block_ids[int(req_pool_idx)] = self._empty_long()
            layer_state.pending_scatter_block_cache_slots[int(req_pool_idx)] = self._empty_long()
            layer_state.pending_scatter_block_page_table[int(req_pool_idx)] = self._empty_long()

    def update_layer_retrieval_cache_state(
        self,
        session_key: tuple[int, ...],
        layer_id: int,
        req_pool_idx: int,
        retrieval_positions: torch.Tensor,
        cache_slot_lookup_fn=None,
    ) -> None:
        session_state = self.sessions.get(session_key)
        if session_state is None:
            return
        layer_state = session_state.layers.get(layer_id)
        if layer_state is None:
            return
        self._update_retrieval_cache_state(
            layer_state=layer_state,
            req_pool_idx=int(req_pool_idx),
            retrieval_positions=retrieval_positions,
            cache_slot_lookup_fn=cache_slot_lookup_fn,
        )

    def clear_layer_retrieval_cache_state(
        self,
        session_key: tuple[int, ...],
        layer_id: int,
        req_pool_idx: int,
    ) -> None:
        session_state = self.sessions.get(session_key)
        if session_state is None:
            return
        layer_state = session_state.layers.get(layer_id)
        if layer_state is None:
            return
        req_pool_idx = int(req_pool_idx)
        layer_state.retrieval_block_ids[req_pool_idx] = self._empty_long()
        layer_state.retrieval_block_cache_slots[req_pool_idx] = self._empty_long()
        layer_state.retrieval_block_page_table[req_pool_idx] = self._empty_long()
        layer_state.retrieval_hit_block_ids[req_pool_idx] = self._empty_long()
        layer_state.retrieval_hit_block_cache_slots[req_pool_idx] = self._empty_long()
        layer_state.retrieval_hit_block_page_table[req_pool_idx] = self._empty_long()
        layer_state.retrieval_miss_block_ids[req_pool_idx] = self._empty_long()
        layer_state.retrieval_miss_block_cache_slots[req_pool_idx] = self._empty_long()
        layer_state.retrieval_miss_block_page_table[req_pool_idx] = self._empty_long()
        layer_state.pending_scatter_block_ids[req_pool_idx] = self._empty_long()
        layer_state.pending_scatter_block_cache_slots[req_pool_idx] = self._empty_long()
        layer_state.pending_scatter_block_page_table[req_pool_idx] = self._empty_long()

    def bind_prepared_layout(
        self,
        session_key: tuple[int, ...],
        layer_positions: dict[int, dict[int, torch.Tensor]],
        target_len: int,
        slack_len: int,
        page_size: int = 1,
        base_len: int = 0,
    ) -> None:
        session_state = RetroInferSessionWorkingSetState(session_key=session_key)
        for layer_id, req_map in layer_positions.items():
            base_token_len = max(0, base_len)
            base_capacity_pages = (
                (base_token_len + page_size - 1) // page_size if base_token_len > 0 else 0
            )
            retrieval_token_len = max(0, target_len - base_len)
            retrieval_capacity_pages = (
                (retrieval_token_len + page_size - 1) // page_size if retrieval_token_len > 0 else 0
            )
            append_capacity_pages = (
                (slack_len + page_size - 1) // page_size if slack_len > 0 else 0
            )
            session_state.layers[layer_id] = RetroInferLayerWorkingSetState(
                target_len=target_len,
                slack_len=slack_len,
                req_positions={req: pos.clone() for req, pos in req_map.items()},
                req_base_positions={
                    int(req): pos[:base_token_len].clone() for req, pos in req_map.items()
                },
                req_retrieval_positions={
                    int(req): pos[base_token_len:target_len].clone() for req, pos in req_map.items()
                },
                req_append_positions={
                    int(req): torch.empty((0,), dtype=torch.long) for req in req_map.keys()
                },
                page_size=page_size,
                base_token_len=base_token_len,
                base_capacity_pages=base_capacity_pages,
                retrieval_capacity_pages=retrieval_capacity_pages,
                retrieval_token_len=retrieval_token_len,
                append_len=0,
                append_capacity_pages=append_capacity_pages,
                live_len=target_len,
                capacity_len=target_len + slack_len,
                req_order=tuple(req_map.keys()),
                cache_resident_block_ids={
                    int(req): torch.empty((0,), dtype=torch.long) for req in req_map.keys()
                },
                cache_resident_block_cache_slots={
                    int(req): torch.empty((0,), dtype=torch.long) for req in req_map.keys()
                },
                cache_resident_block_page_table={
                    int(req): torch.empty((0,), dtype=torch.long) for req in req_map.keys()
                },
            )
        self.sessions[session_key] = session_state

    def materialize_layer(
        self,
        session_key: tuple[int, ...],
        layer_id: int,
        req_pool_indices: list[int],
        kv_heads: int,
        qk_dim: int,
        v_dim: int,
        device: torch.device | str,
        dtype: torch.dtype,
        fetch_fn,
        cache_slot_lookup_fn=None,
    ) -> tuple[torch.Tensor, torch.Tensor, int]:
        session_state = self.sessions.get(session_key)
        if session_state is None:
            raise KeyError(f"RetroInfer wave buffer missing session {session_key}.")
        layer_state = session_state.layers.get(layer_id)
        if layer_state is None:
            raise KeyError(
                f"RetroInfer wave buffer missing layer {layer_id} for session {session_key}."
            )

        req_order = tuple(int(req) for req in req_pool_indices)
        batch_size = len(req_order)
        base_token_len = int(layer_state.base_token_len)
        base_capacity_pages = int(layer_state.base_capacity_pages)
        retrieval_token_len = int(layer_state.retrieval_token_len)
        append_capacity_pages = int(layer_state.append_capacity_pages)
        retrieval_capacity_pages = int(layer_state.retrieval_capacity_pages)
        page_size = max(1, int(layer_state.page_size))

        base_key_pages = None
        base_value_pages = None
        if base_capacity_pages > 0:
            base_key_pages = torch.zeros(
                (batch_size, base_capacity_pages, page_size, kv_heads, qk_dim),
                dtype=dtype,
                device=device,
            )
            base_value_pages = torch.zeros(
                (batch_size, base_capacity_pages, page_size, kv_heads, v_dim),
                dtype=dtype,
                device=device,
            )

        retrieval_key_pages = None
        retrieval_value_pages = None
        if retrieval_capacity_pages > 0:
            retrieval_key_pages = torch.zeros(
                (batch_size, retrieval_capacity_pages, page_size, kv_heads, qk_dim),
                dtype=dtype,
                device=device,
            )
            retrieval_value_pages = torch.zeros(
                (batch_size, retrieval_capacity_pages, page_size, kv_heads, v_dim),
                dtype=dtype,
                device=device,
            )

        append_key_pages = None
        append_value_pages = None
        if append_capacity_pages > 0:
            append_key_pages = torch.zeros(
                (batch_size, append_capacity_pages, page_size, kv_heads, qk_dim),
                dtype=dtype,
                device=device,
            )
            append_value_pages = torch.zeros(
                (batch_size, append_capacity_pages, page_size, kv_heads, v_dim),
                dtype=dtype,
                device=device,
            )

        for batch_idx, req_pool_idx in enumerate(req_order):
            base_positions = layer_state.req_base_positions.get(int(req_pool_idx), self._empty_long())
            retrieval_positions = layer_state.req_retrieval_positions.get(
                int(req_pool_idx), self._empty_long()
            )

            if base_token_len > 0:
                req_base_keys, req_base_values = fetch_fn(req_pool_idx, base_positions)
                req_base_keys = req_base_keys.to(device=device, dtype=dtype, non_blocking=True)
                req_base_values = req_base_values.to(device=device, dtype=dtype, non_blocking=True)
                padded_base = base_capacity_pages * page_size
                if req_base_keys.shape[0] < padded_base:
                    pad_k = torch.zeros(
                        (padded_base - req_base_keys.shape[0], kv_heads, qk_dim),
                        dtype=dtype,
                        device=device,
                    )
                    pad_v = torch.zeros(
                        (padded_base - req_base_values.shape[0], kv_heads, v_dim),
                        dtype=dtype,
                        device=device,
                    )
                    req_base_keys = torch.cat([req_base_keys, pad_k], dim=0)
                    req_base_values = torch.cat([req_base_values, pad_v], dim=0)
                base_key_pages[batch_idx].copy_(
                    req_base_keys.view(base_capacity_pages, page_size, kv_heads, qk_dim)
                )
                base_value_pages[batch_idx].copy_(
                    req_base_values.view(base_capacity_pages, page_size, kv_heads, v_dim)
                )

            if retrieval_token_len > 0:
                req_retrieval_keys, req_retrieval_values = fetch_fn(req_pool_idx, retrieval_positions)
                req_retrieval_keys = req_retrieval_keys.to(
                    device=device, dtype=dtype, non_blocking=True
                )
                req_retrieval_values = req_retrieval_values.to(
                    device=device, dtype=dtype, non_blocking=True
                )
                padded_retrieval = retrieval_capacity_pages * page_size
                if req_retrieval_keys.shape[0] < padded_retrieval:
                    pad_k = torch.zeros(
                        (padded_retrieval - req_retrieval_keys.shape[0], kv_heads, qk_dim),
                        dtype=dtype,
                        device=device,
                    )
                    pad_v = torch.zeros(
                        (padded_retrieval - req_retrieval_values.shape[0], kv_heads, v_dim),
                        dtype=dtype,
                        device=device,
                    )
                    req_retrieval_keys = torch.cat([req_retrieval_keys, pad_k], dim=0)
                    req_retrieval_values = torch.cat([req_retrieval_values, pad_v], dim=0)
                retrieval_key_pages[batch_idx].copy_(
                    req_retrieval_keys.view(retrieval_capacity_pages, page_size, kv_heads, qk_dim)
                )
                retrieval_value_pages[batch_idx].copy_(
                    req_retrieval_values.view(retrieval_capacity_pages, page_size, kv_heads, v_dim)
                )
                self._update_retrieval_cache_state(
                    layer_state=layer_state,
                    req_pool_idx=int(req_pool_idx),
                    retrieval_positions=retrieval_positions,
                    cache_slot_lookup_fn=cache_slot_lookup_fn,
                )
            else:
                self.clear_layer_retrieval_cache_state(
                    session_key=session_key,
                    layer_id=layer_id,
                    req_pool_idx=int(req_pool_idx),
                )

        layer_state.base_key_pages = base_key_pages
        layer_state.base_value_pages = base_value_pages
        layer_state.retrieval_key_pages = retrieval_key_pages
        layer_state.retrieval_value_pages = retrieval_value_pages
        layer_state.append_key_pages = append_key_pages
        layer_state.append_value_pages = append_value_pages
        layer_state.append_len = 0
        layer_state.capacity_len = layer_state.target_len + layer_state.slack_len
        layer_state.req_order = req_order
        return self._refresh_execution_buffer(layer_state)

    def ensure_layer_capacity(
        self,
        session_key: tuple[int, ...],
        layer_id: int,
        *,
        batch_size: int,
        target_len: int,
        base_token_len: int,
        retrieval_token_len: int,
        slack_len: int,
        kv_heads: int,
        qk_dim: int,
        v_dim: int,
        device,
        dtype,
    ) -> bool:
        session_state = self.sessions.get(session_key)
        if session_state is None:
            return False
        layer_state = session_state.layers.get(layer_id)
        if layer_state is None:
            return False

        page_size = max(1, int(layer_state.page_size))
        target_len = max(0, int(target_len))
        base_token_len = max(0, int(base_token_len))
        retrieval_token_len = max(0, int(retrieval_token_len))
        slack_len = max(int(layer_state.slack_len), int(slack_len))

        new_base_capacity_pages = (
            (base_token_len + page_size - 1) // page_size if base_token_len > 0 else 0
        )
        new_retrieval_capacity_pages = (
            (retrieval_token_len + page_size - 1) // page_size if retrieval_token_len > 0 else 0
        )
        new_append_capacity_pages = (
            (slack_len + page_size - 1) // page_size if slack_len > 0 else 0
        )

        needs_resize = (
            batch_size != len(layer_state.req_order)
            or new_base_capacity_pages > int(layer_state.base_capacity_pages)
            or new_retrieval_capacity_pages > int(layer_state.retrieval_capacity_pages)
            or new_append_capacity_pages > int(layer_state.append_capacity_pages)
            or target_len > int(layer_state.target_len)
            or base_token_len > int(layer_state.base_token_len)
            or retrieval_token_len > int(layer_state.retrieval_token_len)
        )
        if not needs_resize:
            return False

        layer_state.base_key_pages = self._resize_or_init_pages(
            layer_state.base_key_pages,
            batch_size=batch_size,
            capacity_pages=max(new_base_capacity_pages, int(layer_state.base_capacity_pages)),
            page_size=page_size,
            kv_heads=kv_heads,
            head_dim=qk_dim,
            device=device,
            dtype=dtype,
        )
        layer_state.base_value_pages = self._resize_or_init_pages(
            layer_state.base_value_pages,
            batch_size=batch_size,
            capacity_pages=max(new_base_capacity_pages, int(layer_state.base_capacity_pages)),
            page_size=page_size,
            kv_heads=kv_heads,
            head_dim=v_dim,
            device=device,
            dtype=dtype,
        )
        layer_state.retrieval_key_pages = self._resize_or_init_pages(
            layer_state.retrieval_key_pages,
            batch_size=batch_size,
            capacity_pages=max(new_retrieval_capacity_pages, int(layer_state.retrieval_capacity_pages)),
            page_size=page_size,
            kv_heads=kv_heads,
            head_dim=qk_dim,
            device=device,
            dtype=dtype,
        )
        layer_state.retrieval_value_pages = self._resize_or_init_pages(
            layer_state.retrieval_value_pages,
            batch_size=batch_size,
            capacity_pages=max(new_retrieval_capacity_pages, int(layer_state.retrieval_capacity_pages)),
            page_size=page_size,
            kv_heads=kv_heads,
            head_dim=v_dim,
            device=device,
            dtype=dtype,
        )
        layer_state.append_key_pages = self._resize_or_init_pages(
            layer_state.append_key_pages,
            batch_size=batch_size,
            capacity_pages=max(new_append_capacity_pages, int(layer_state.append_capacity_pages)),
            page_size=page_size,
            kv_heads=kv_heads,
            head_dim=qk_dim,
            device=device,
            dtype=dtype,
        )
        layer_state.append_value_pages = self._resize_or_init_pages(
            layer_state.append_value_pages,
            batch_size=batch_size,
            capacity_pages=max(new_append_capacity_pages, int(layer_state.append_capacity_pages)),
            page_size=page_size,
            kv_heads=kv_heads,
            head_dim=v_dim,
            device=device,
            dtype=dtype,
        )

        layer_state.target_len = max(int(layer_state.target_len), target_len)
        layer_state.base_token_len = max(int(layer_state.base_token_len), base_token_len)
        layer_state.retrieval_token_len = max(
            int(layer_state.retrieval_token_len), retrieval_token_len
        )
        layer_state.base_capacity_pages = max(
            int(layer_state.base_capacity_pages), new_base_capacity_pages
        )
        layer_state.retrieval_capacity_pages = max(
            int(layer_state.retrieval_capacity_pages), new_retrieval_capacity_pages
        )
        layer_state.append_capacity_pages = max(
            int(layer_state.append_capacity_pages), new_append_capacity_pages
        )
        layer_state.slack_len = slack_len
        layer_state.capacity_len = int(layer_state.target_len) + int(layer_state.slack_len)
        return True

    def get_layer_positions(
        self,
        session_key: tuple[int, ...],
        layer_id: int,
        req_pool_idx: int,
    ) -> torch.Tensor | None:
        session_state = self.sessions.get(session_key)
        if session_state is None:
            return None
        layer_state = session_state.layers.get(layer_id)
        if layer_state is None:
            return None
        return layer_state.req_positions.get(req_pool_idx)

    def get_layer_retrieval_positions(
        self,
        session_key: tuple[int, ...],
        layer_id: int,
        req_pool_idx: int,
    ) -> torch.Tensor | None:
        session_state = self.sessions.get(session_key)
        if session_state is None:
            return None
        layer_state = session_state.layers.get(layer_id)
        if layer_state is None:
            return None
        return layer_state.req_retrieval_positions.get(req_pool_idx)

    def get_layer_buffers(
        self,
        session_key: tuple[int, ...],
        layer_id: int,
    ) -> tuple[torch.Tensor, torch.Tensor, int] | None:
        session_state = self.sessions.get(session_key)
        if session_state is None:
            return None
        layer_state = session_state.layers.get(layer_id)
        if (
            layer_state is None
            or (layer_state.base_capacity_pages > 0 and layer_state.base_key_pages is None)
            or (
                layer_state.retrieval_capacity_pages > 0
                and layer_state.retrieval_key_pages is None
            )
            or (layer_state.append_capacity_pages > 0 and layer_state.append_key_pages is None)
        ):
            return None
        if (
            layer_state.execution_key_buffer is None
            or layer_state.execution_value_buffer is None
            or int(layer_state.live_len) <= 0
        ):
            return self._refresh_execution_buffer(layer_state)
        return (
            layer_state.execution_key_buffer,
            layer_state.execution_value_buffer,
            int(layer_state.live_len),
        )

    def get_layer_state(
        self,
        session_key: tuple[int, ...],
        layer_id: int,
    ) -> RetroInferLayerWorkingSetState | None:
        session_state = self.sessions.get(session_key)
        if session_state is None:
            return None
        return session_state.layers.get(layer_id)

    def replace_layer_positions(
        self,
        session_key: tuple[int, ...],
        layer_id: int,
        req_positions: dict[int, torch.Tensor],
    ) -> bool:
        session_state = self.sessions.get(session_key)
        if session_state is None:
            return False
        layer_state = session_state.layers.get(layer_id)
        if layer_state is None:
            return False
        layer_state.req_positions = {
            int(req): positions.clone() for req, positions in req_positions.items()
        }
        layer_state.req_base_positions = {
            int(req): positions[: layer_state.base_token_len].clone()
            for req, positions in req_positions.items()
        }
        layer_state.req_retrieval_positions = {
            int(req): positions[
                layer_state.base_token_len : layer_state.base_token_len + layer_state.retrieval_token_len
            ].clone()
            for req, positions in req_positions.items()
        }
        layer_state.req_append_positions = {
            int(req): positions[
                layer_state.base_token_len + layer_state.retrieval_token_len :
            ].clone()
            for req, positions in req_positions.items()
        }
        layer_state.req_order = tuple(req_positions.keys())
        return True

    def update_after_decode(
        self,
        session_key: tuple[int, ...],
        layer_id: int,
        req_pool_idx: int,
        new_position: int,
    ) -> bool:
        session_state = self.sessions.get(session_key)
        if session_state is None:
            return True
        layer_state = session_state.layers.get(layer_id)
        if layer_state is None:
            return True

        positions = layer_state.req_positions.get(req_pool_idx)
        append_positions = layer_state.req_append_positions.get(req_pool_idx)
        if positions is None or append_positions is None:
            return True

        if positions.numel() == 0:
            self._set_request_segments(
                layer_state,
                req_pool_idx,
                append_positions=torch.tensor([new_position], dtype=torch.long),
            )
            return False

        if int(positions[-1].item()) == new_position:
            return False

        if positions.numel() < layer_state.target_len + layer_state.slack_len:
            if bool((positions == int(new_position)).any().item()):
                return False
            updated_append = torch.cat(
                [append_positions, torch.tensor([new_position], dtype=torch.long)]
            )
            self._set_request_segments(
                layer_state,
                req_pool_idx,
                append_positions=updated_append,
            )
            return False

        return True

    def append_decode_tokens(
        self,
        session_key: tuple[int, ...],
        layer_id: int,
        req_pool_indices: list[int],
        new_positions: list[int],
        keys: torch.Tensor,
        values: torch.Tensor,
    ) -> bool:
        session_state = self.sessions.get(session_key)
        if session_state is None:
            return True
        layer_state = session_state.layers.get(layer_id)
        if (
            layer_state is None
            or layer_state.append_key_pages is None
            or layer_state.append_value_pages is None
        ):
            return True

        if layer_state.append_len >= layer_state.slack_len:
            return True
        if tuple(int(req) for req in req_pool_indices) != layer_state.req_order:
            return True
        if keys.shape[1] != 1 or values.shape[1] != 1:
            return True

        page_size = max(1, int(layer_state.page_size))
        page_idx = int(layer_state.append_len) // page_size
        page_offset = int(layer_state.append_len) % page_size
        if page_idx >= int(layer_state.append_capacity_pages):
            return True
        layer_state.append_key_pages[:, page_idx, page_offset : page_offset + 1].copy_(keys)
        layer_state.append_value_pages[:, page_idx, page_offset : page_offset + 1].copy_(values)
        layer_state.append_len += 1
        for req_pool_idx, new_position in zip(req_pool_indices, new_positions):
            positions = layer_state.req_positions.get(int(req_pool_idx))
            append_positions = layer_state.req_append_positions.get(int(req_pool_idx))
            if positions is None or append_positions is None:
                return True
            if bool((positions == int(new_position)).any().item()):
                continue
            updated_append = torch.cat(
                [append_positions, torch.tensor([int(new_position)], dtype=torch.long)]
            )
            self._set_request_segments(
                layer_state,
                int(req_pool_idx),
                append_positions=updated_append,
            )
        self._refresh_execution_buffer(layer_state)
        return False

    def set_layer_request_segments(
        self,
        session_key: tuple[int, ...],
        layer_id: int,
        req_pool_idx: int,
        *,
        base_positions: torch.Tensor | None = None,
        retrieval_positions: torch.Tensor | None = None,
        append_positions: torch.Tensor | None = None,
    ) -> bool:
        session_state = self.sessions.get(session_key)
        if session_state is None:
            return False
        layer_state = session_state.layers.get(layer_id)
        if layer_state is None:
            return False
        self._set_request_segments(
            layer_state,
            int(req_pool_idx),
            base_positions=base_positions,
            retrieval_positions=retrieval_positions,
            append_positions=append_positions,
        )
        return True

    def sync_execution_buffer(
        self,
        session_key: tuple[int, ...],
        layer_id: int,
    ) -> tuple[torch.Tensor, torch.Tensor, int] | None:
        session_state = self.sessions.get(session_key)
        if session_state is None:
            return None
        layer_state = session_state.layers.get(layer_id)
        if layer_state is None:
            return None
        return self._refresh_execution_buffer(layer_state)

    def drop_session(self, session_key: tuple[int, ...]) -> None:
        self.sessions.pop(session_key, None)

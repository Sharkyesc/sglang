from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass
class ChunkWorkingSetEntry:
    chunk_ids: list[int]
    key_buffers: list[torch.Tensor]
    value_buffers: list[torch.Tensor]
    valid_positions: set[int]
    active: int = 0

    @property
    def inactive(self) -> int:
        return 1 - int(self.active)


class SparseWorkingSetBuffer:
    """Reusable GPU buffers for sparse_framework subset attention."""

    def __init__(self):
        self.key_buffers: dict[tuple[int, torch.device, torch.dtype, tuple[int, ...]], torch.Tensor] = {}
        self.value_buffers: dict[tuple[int, torch.device, torch.dtype, tuple[int, ...]], torch.Tensor] = {}
        self.chunk_entries: dict[
            tuple[int, int, int, torch.device, torch.dtype, tuple[int, ...]], ChunkWorkingSetEntry
        ] = {}

    def materialize(
        self,
        *,
        layer_id: int,
        key: torch.Tensor,
        value: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        key_out = self._copy_into_buffer(self.key_buffers, layer_id, key)
        value_out = self._copy_into_buffer(self.value_buffers, layer_id, value)
        return key_out, value_out

    def _copy_into_buffer(
        self,
        buffers: dict,
        layer_id: int,
        tensor: torch.Tensor,
    ) -> torch.Tensor:
        shape_tail = tuple(tensor.shape[1:])
        cache_key = (int(layer_id), tensor.device, tensor.dtype, shape_tail)
        current = buffers.get(cache_key)
        if current is None or int(current.shape[0]) < int(tensor.shape[0]):
            current = torch.empty(
                (int(tensor.shape[0]),) + shape_tail,
                dtype=tensor.dtype,
                device=tensor.device,
            )
            buffers[cache_key] = current
        view = current[: int(tensor.shape[0])]
        view.copy_(tensor)
        return view

    def materialize_chunked_delta(
        self,
        *,
        req_pool_idx: int,
        layer_id: int,
        positions: list[int],
        key: torch.Tensor,
        value: torch.Tensor,
        chunk_size: int,
    ) -> tuple[torch.Tensor, torch.Tensor, dict]:
        if not positions or int(key.shape[0]) == 0:
            return key, value, {"enabled": False, "reason": "empty"}

        chunk_size = max(1, int(chunk_size))
        shape_tail = tuple(key.shape[1:])
        cache_key = (
            int(req_pool_idx),
            int(layer_id),
            chunk_size,
            key.device,
            key.dtype,
            shape_tail,
        )
        chunk_ids = sorted({int(pos) // chunk_size for pos in positions})
        needed_chunks = len(chunk_ids)
        entry = self.chunk_entries.get(cache_key)
        if entry is None:
            initial_capacity = max(needed_chunks, 1)
            key_buffers = [
                torch.empty(
                    (initial_capacity, chunk_size) + shape_tail,
                    dtype=key.dtype,
                    device=key.device,
                ),
                torch.empty(
                    (initial_capacity, chunk_size) + shape_tail,
                    dtype=key.dtype,
                    device=key.device,
                ),
            ]
            value_buffers = [
                torch.empty(
                    (initial_capacity, chunk_size) + tuple(value.shape[1:]),
                    dtype=value.dtype,
                    device=value.device,
                ),
                torch.empty(
                    (initial_capacity, chunk_size) + tuple(value.shape[1:]),
                    dtype=value.dtype,
                    device=value.device,
                ),
            ]
            entry = ChunkWorkingSetEntry(
                chunk_ids=[],
                key_buffers=key_buffers,
                value_buffers=value_buffers,
                valid_positions=set(),
            )
            self.chunk_entries[cache_key] = entry

        self._ensure_chunk_capacity(
            entry, key=key, value=value, needed_chunks=needed_chunks
        )

        old_chunk_to_slot = {
            int(chunk_id): slot for slot, chunk_id in enumerate(entry.chunk_ids)
        }
        new_chunk_to_slot = {int(chunk_id): slot for slot, chunk_id in enumerate(chunk_ids)}
        old_key = entry.key_buffers[entry.active]
        old_value = entry.value_buffers[entry.active]
        new_key = entry.key_buffers[entry.inactive]
        new_value = entry.value_buffers[entry.inactive]

        hit_chunks = 0
        retained_valid_positions: set[int] = set()
        for chunk_id, new_slot in new_chunk_to_slot.items():
            old_slot = old_chunk_to_slot.get(chunk_id)
            if old_slot is None:
                continue
            new_key[new_slot].copy_(old_key[old_slot])
            new_value[new_slot].copy_(old_value[old_slot])
            hit_chunks += 1
            retained_valid_positions.update(
                pos
                for pos in entry.valid_positions
                if int(pos) // chunk_size == int(chunk_id)
            )

        chunk_slot_indices = torch.tensor(
            [new_chunk_to_slot[int(pos) // chunk_size] for pos in positions],
            dtype=torch.long,
            device=key.device,
        )
        inner_offsets = torch.tensor(
            [int(pos) % chunk_size for pos in positions],
            dtype=torch.long,
            device=key.device,
        )
        linear_slots = chunk_slot_indices * chunk_size + inner_offsets
        new_key.view((-1,) + tuple(new_key.shape[2:])).index_copy_(
            0, linear_slots, key
        )
        new_value.view((-1,) + tuple(new_value.shape[2:])).index_copy_(
            0, linear_slots, value
        )

        key_out = new_key[chunk_slot_indices, inner_offsets]
        value_out = new_value[chunk_slot_indices, inner_offsets]
        entry.chunk_ids = chunk_ids
        entry.valid_positions = retained_valid_positions.union(
            {int(pos) for pos in positions}
        )
        entry.active = entry.inactive
        return key_out, value_out, {
            "enabled": True,
            "layout": "chunk",
            "chunk_size": chunk_size,
            "chunks": needed_chunks,
            "hit_chunks": hit_chunks,
            "miss_chunks": needed_chunks - hit_chunks,
            "positions": len(positions),
        }

    def plan_chunked_delta(
        self,
        *,
        req_pool_idx: int,
        layer_id: int,
        positions: list[int],
        chunk_size: int,
        device: torch.device,
        dtype: torch.dtype,
        key_shape_tail: tuple[int, ...],
        max_position: int | None = None,
        full_miss_units: bool = False,
    ) -> dict:
        chunk_size = max(1, int(chunk_size))
        chunk_ids = sorted({int(pos) // chunk_size for pos in positions})
        cache_key = (
            int(req_pool_idx),
            int(layer_id),
            chunk_size,
            device,
            dtype,
            tuple(key_shape_tail),
        )
        entry = self.chunk_entries.get(cache_key)
        resident = set(entry.chunk_ids) if entry is not None else set()
        valid_positions = entry.valid_positions if entry is not None else set()
        miss_chunks = [chunk_id for chunk_id in chunk_ids if chunk_id not in resident]
        missing_requested_positions = [
            int(pos)
            for pos in positions
            if int(pos) // chunk_size in resident and int(pos) not in valid_positions
        ]
        if full_miss_units:
            miss_positions = self._expand_units(
                miss_chunks, unit_size=chunk_size, max_position=max_position
            )
            miss_positions.extend(missing_requested_positions)
        else:
            miss_chunk_set = set(miss_chunks)
            miss_positions = [
                int(pos) for pos in positions if int(pos) // chunk_size in miss_chunk_set
            ]
            miss_positions.extend(missing_requested_positions)
        miss_positions = list(dict.fromkeys(miss_positions))
        return {
            "enabled": True,
            "layout": "unit",
            "unit_size": chunk_size,
            "chunk_size": chunk_size,
            "units": len(chunk_ids),
            "chunks": len(chunk_ids),
            "hit_units": len(chunk_ids) - len(miss_chunks),
            "miss_units": len(miss_chunks),
            "hit_chunks": len(chunk_ids) - len(miss_chunks),
            "miss_chunks": len(miss_chunks),
            "miss_positions": miss_positions,
            "positions": len(positions),
            "full_miss_units": bool(full_miss_units),
        }

    def plan_delta(self, *, unit_size: int, **kwargs) -> dict:
        return self.plan_chunked_delta(chunk_size=unit_size, **kwargs)

    def materialize_chunked_delta_from_partial(
        self,
        *,
        req_pool_idx: int,
        layer_id: int,
        positions: list[int],
        materialized_positions: list[int],
        key: torch.Tensor | None,
        value: torch.Tensor | None,
        chunk_size: int,
        device: torch.device,
        key_dtype: torch.dtype,
        value_dtype: torch.dtype,
        key_shape_tail: tuple[int, ...],
        value_shape_tail: tuple[int, ...],
    ) -> tuple[torch.Tensor, torch.Tensor, dict]:
        chunk_size = max(1, int(chunk_size))
        if not positions:
            empty_key = torch.empty((0,) + tuple(key_shape_tail), dtype=key_dtype, device=device)
            empty_value = torch.empty((0,) + tuple(value_shape_tail), dtype=value_dtype, device=device)
            return empty_key, empty_value, {"enabled": False, "reason": "empty"}

        cache_key = (
            int(req_pool_idx),
            int(layer_id),
            chunk_size,
            device,
            key_dtype,
            tuple(key_shape_tail),
        )
        chunk_ids = sorted({int(pos) // chunk_size for pos in positions})
        needed_chunks = len(chunk_ids)
        entry = self.chunk_entries.get(cache_key)
        if entry is None:
            initial_capacity = max(needed_chunks, 1)
            entry = ChunkWorkingSetEntry(
                chunk_ids=[],
                valid_positions=set(),
                key_buffers=[
                    torch.empty((initial_capacity, chunk_size) + tuple(key_shape_tail), dtype=key_dtype, device=device),
                    torch.empty((initial_capacity, chunk_size) + tuple(key_shape_tail), dtype=key_dtype, device=device),
                ],
                value_buffers=[
                    torch.empty((initial_capacity, chunk_size) + tuple(value_shape_tail), dtype=value_dtype, device=device),
                    torch.empty((initial_capacity, chunk_size) + tuple(value_shape_tail), dtype=value_dtype, device=device),
                ],
            )
            self.chunk_entries[cache_key] = entry

        key_template = entry.key_buffers[entry.active].new_empty((0,) + tuple(key_shape_tail))
        value_template = entry.value_buffers[entry.active].new_empty((0,) + tuple(value_shape_tail))
        self._ensure_chunk_capacity(
            entry,
            key=key if key is not None else key_template,
            value=value if value is not None else value_template,
            needed_chunks=needed_chunks,
        )

        old_chunk_to_slot = {
            int(chunk_id): slot for slot, chunk_id in enumerate(entry.chunk_ids)
        }
        new_chunk_to_slot = {int(chunk_id): slot for slot, chunk_id in enumerate(chunk_ids)}
        old_key = entry.key_buffers[entry.active]
        old_value = entry.value_buffers[entry.active]
        new_key = entry.key_buffers[entry.inactive]
        new_value = entry.value_buffers[entry.inactive]

        hit_chunks = 0
        retained_valid_positions: set[int] = set()
        for chunk_id, new_slot in new_chunk_to_slot.items():
            old_slot = old_chunk_to_slot.get(chunk_id)
            if old_slot is None:
                continue
            new_key[new_slot].copy_(old_key[old_slot])
            new_value[new_slot].copy_(old_value[old_slot])
            hit_chunks += 1
            retained_valid_positions.update(
                pos
                for pos in entry.valid_positions
                if int(pos) // chunk_size == int(chunk_id)
            )

        if key is not None and value is not None and materialized_positions:
            materialized_chunk_slots = torch.tensor(
                [new_chunk_to_slot[int(pos) // chunk_size] for pos in materialized_positions],
                dtype=torch.long,
                device=device,
            )
            materialized_inner_offsets = torch.tensor(
                [int(pos) % chunk_size for pos in materialized_positions],
                dtype=torch.long,
                device=device,
            )
            linear_slots = materialized_chunk_slots * chunk_size + materialized_inner_offsets
            new_key.view((-1,) + tuple(new_key.shape[2:])).index_copy_(0, linear_slots, key)
            new_value.view((-1,) + tuple(new_value.shape[2:])).index_copy_(0, linear_slots, value)

        chunk_slot_indices = torch.tensor(
            [new_chunk_to_slot[int(pos) // chunk_size] for pos in positions],
            dtype=torch.long,
            device=device,
        )
        inner_offsets = torch.tensor(
            [int(pos) % chunk_size for pos in positions],
            dtype=torch.long,
            device=device,
        )
        key_out = new_key[chunk_slot_indices, inner_offsets]
        value_out = new_value[chunk_slot_indices, inner_offsets]
        entry.chunk_ids = chunk_ids
        entry.valid_positions = retained_valid_positions.union(
            {int(pos) for pos in materialized_positions}
        )
        entry.active = entry.inactive
        return key_out, value_out, {
            "enabled": True,
            "layout": "unit",
            "unit_size": chunk_size,
            "chunk_size": chunk_size,
            "units": needed_chunks,
            "chunks": needed_chunks,
            "hit_units": hit_chunks,
            "miss_units": needed_chunks - hit_chunks,
            "hit_chunks": hit_chunks,
            "miss_chunks": needed_chunks - hit_chunks,
            "positions": len(positions),
            "materialized_positions": len(materialized_positions),
        }

    def materialize_delta_from_partial(self, *, unit_size: int, **kwargs):
        return self.materialize_chunked_delta_from_partial(
            chunk_size=unit_size, **kwargs
        )

    def drop_request(self, req_pool_idx: int) -> int:
        req_pool_idx = int(req_pool_idx)
        keys = [key for key in self.chunk_entries if int(key[0]) == req_pool_idx]
        for key in keys:
            del self.chunk_entries[key]
        return len(keys)

    def _ensure_chunk_capacity(
        self,
        entry: ChunkWorkingSetEntry,
        *,
        key: torch.Tensor,
        value: torch.Tensor,
        needed_chunks: int,
    ) -> None:
        current = int(entry.key_buffers[0].shape[0])
        if current >= needed_chunks:
            return
        new_capacity = max(needed_chunks, current * 2, 16)
        for idx in range(2):
            old_key = entry.key_buffers[idx]
            old_value = entry.value_buffers[idx]
            new_key = torch.empty(
                (new_capacity,) + tuple(old_key.shape[1:]),
                dtype=key.dtype,
                device=key.device,
            )
            new_value = torch.empty(
                (new_capacity,) + tuple(old_value.shape[1:]),
                dtype=value.dtype,
                device=value.device,
            )
            if current > 0:
                new_key[:current].copy_(old_key[:current])
                new_value[:current].copy_(old_value[:current])
            entry.key_buffers[idx] = new_key
            entry.value_buffers[idx] = new_value

    def _expand_units(
        self,
        unit_ids: list[int],
        *,
        unit_size: int,
        max_position: int | None,
    ) -> list[int]:
        positions = []
        limit = None if max_position is None else int(max_position)
        for unit_id in unit_ids:
            start = int(unit_id) * int(unit_size)
            end = start + int(unit_size)
            if limit is not None:
                end = min(end, limit)
            positions.extend(range(start, end))
        return positions


def get_working_set_buffer(framework_state: dict | None) -> SparseWorkingSetBuffer | None:
    if framework_state is None:
        return None
    buffer = framework_state.get("working_set_buffer")
    if buffer is None:
        buffer = SparseWorkingSetBuffer()
        framework_state["working_set_buffer"] = buffer
    return buffer

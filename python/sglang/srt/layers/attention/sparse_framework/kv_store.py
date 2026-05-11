from __future__ import annotations

from dataclasses import dataclass, field

import torch


@dataclass
class RequestLayerKV:
    position_to_offset: dict[int, int] = field(default_factory=dict)
    key_buffer: torch.Tensor | None = None
    value_buffer: torch.Tensor | None = None
    size: int = 0
    capacity: int = 0
    pinned: bool = False
    version: int = 0
    pending_events: dict[int, torch.cuda.Event] = field(default_factory=dict)
    chunk_size: int = 16
    chunk_id_to_offset: dict[int, int] = field(default_factory=dict)
    chunk_key_buffer: torch.Tensor | None = None
    chunk_value_buffer: torch.Tensor | None = None
    chunk_count: int = 0
    chunk_capacity: int = 0
    chunk_pinned: bool = False
    pending_chunk_events: dict[tuple[int, int], torch.cuda.Event] = field(default_factory=dict)
    chunk_index_cache_version: int = -1
    chunk_index_cache: dict[tuple[int, tuple[int, ...]], dict] = field(default_factory=dict)

    def ensure_capacity(
        self,
        needed: int,
        key_shape: tuple[int, ...],
        value_shape: tuple[int, ...],
        key_dtype: torch.dtype,
        value_dtype: torch.dtype,
    ) -> None:
        if self.capacity >= needed:
            return
        self.synchronize_pending()

        new_capacity = max(16, needed, self.capacity * 2)
        new_key_buffer, key_pinned = _empty_cpu_buffer(
            (new_capacity,) + key_shape,
            dtype=key_dtype,
        )
        new_value_buffer, value_pinned = _empty_cpu_buffer(
            (new_capacity,) + value_shape,
            dtype=value_dtype,
        )

        if self.key_buffer is not None and self.size > 0:
            new_key_buffer[: self.size].copy_(self.key_buffer[: self.size])
        if self.value_buffer is not None and self.size > 0:
            new_value_buffer[: self.size].copy_(self.value_buffer[: self.size])

        self.key_buffer = new_key_buffer
        self.value_buffer = new_value_buffer
        self.capacity = new_capacity
        self.pinned = bool(key_pinned and value_pinned)

    def put_row(
        self,
        position: int,
        key: torch.Tensor,
        value: torch.Tensor,
        *,
        copy_stream: torch.cuda.Stream | None = None,
    ) -> None:
        pos = int(position)
        offset = self.position_to_offset.get(pos)
        if offset is None:
            offset = self.size
            self.position_to_offset[pos] = offset
            self.size += 1
        else:
            self.wait_position(pos)

        assert self.key_buffer is not None
        assert self.value_buffer is not None
        if copy_stream is None:
            self.key_buffer[offset].copy_(key.detach().to("cpu", non_blocking=False))
            self.value_buffer[offset].copy_(value.detach().to("cpu", non_blocking=False))
            self.pending_events.pop(pos, None)
            return

        key = key.detach()
        value = value.detach()
        if key.device.type == "cuda":
            copy_stream.wait_stream(torch.cuda.current_stream(device=key.device))
        with torch.cuda.stream(copy_stream):
            self.key_buffer[offset].copy_(key, non_blocking=True)
            self.value_buffer[offset].copy_(value, non_blocking=True)
            if key.device.type == "cuda":
                key.record_stream(copy_stream)
                value.record_stream(copy_stream)
            event = torch.cuda.Event()
            event.record(copy_stream)
        self.pending_events[pos] = event

    def ensure_chunk_capacity(
        self,
        needed: int,
        key_shape: tuple[int, ...],
        value_shape: tuple[int, ...],
        key_dtype: torch.dtype,
        value_dtype: torch.dtype,
    ) -> None:
        if self.chunk_capacity >= needed:
            return
        self.synchronize_pending()

        new_capacity = max(16, needed, self.chunk_capacity * 2)
        new_key_buffer, key_pinned = _empty_cpu_buffer(
            (new_capacity, self.chunk_size) + key_shape,
            dtype=key_dtype,
        )
        new_value_buffer, value_pinned = _empty_cpu_buffer(
            (new_capacity, self.chunk_size) + value_shape,
            dtype=value_dtype,
        )

        if self.chunk_key_buffer is not None and self.chunk_count > 0:
            new_key_buffer[: self.chunk_count].copy_(
                self.chunk_key_buffer[: self.chunk_count]
            )
        if self.chunk_value_buffer is not None and self.chunk_count > 0:
            new_value_buffer[: self.chunk_count].copy_(
                self.chunk_value_buffer[: self.chunk_count]
            )

        self.chunk_key_buffer = new_key_buffer
        self.chunk_value_buffer = new_value_buffer
        self.chunk_capacity = new_capacity
        self.chunk_pinned = bool(key_pinned and value_pinned)

    def put_chunk_row(
        self,
        position: int,
        key: torch.Tensor,
        value: torch.Tensor,
        *,
        chunk_size: int,
        copy_stream: torch.cuda.Stream | None = None,
    ) -> None:
        if int(chunk_size) != int(self.chunk_size):
            self.reset_chunks(chunk_size=int(chunk_size))
        chunk_id = int(position) // self.chunk_size
        chunk_offset = int(position) % self.chunk_size
        offset = self.chunk_id_to_offset.get(chunk_id)
        if offset is None:
            offset = self.chunk_count
            self.chunk_id_to_offset[chunk_id] = offset
            self.chunk_count += 1
        else:
            self.wait_chunk_position(chunk_id, chunk_offset)

        assert self.chunk_key_buffer is not None
        assert self.chunk_value_buffer is not None
        if copy_stream is None:
            self.chunk_key_buffer[offset, chunk_offset].copy_(
                key.detach().to("cpu", non_blocking=False)
            )
            self.chunk_value_buffer[offset, chunk_offset].copy_(
                value.detach().to("cpu", non_blocking=False)
            )
            self.pending_chunk_events.pop((chunk_id, chunk_offset), None)
            return

        key = key.detach()
        value = value.detach()
        if key.device.type == "cuda":
            copy_stream.wait_stream(torch.cuda.current_stream(device=key.device))
        with torch.cuda.stream(copy_stream):
            self.chunk_key_buffer[offset, chunk_offset].copy_(
                key, non_blocking=True
            )
            self.chunk_value_buffer[offset, chunk_offset].copy_(
                value, non_blocking=True
            )
            if key.device.type == "cuda":
                key.record_stream(copy_stream)
                value.record_stream(copy_stream)
            event = torch.cuda.Event()
            event.record(copy_stream)
        self.pending_chunk_events[(chunk_id, chunk_offset)] = event

    def reset_chunks(self, *, chunk_size: int) -> None:
        self.synchronize_pending()
        self.chunk_size = max(1, int(chunk_size))
        self.chunk_id_to_offset.clear()
        self.chunk_key_buffer = None
        self.chunk_value_buffer = None
        self.chunk_count = 0
        self.chunk_capacity = 0
        self.chunk_pinned = False
        self.pending_chunk_events.clear()
        self.chunk_index_cache_version = -1
        self.chunk_index_cache.clear()

    def has_position_ready(self, position: int) -> bool:
        pos = int(position)
        if pos not in self.position_to_offset:
            return False
        event = self.pending_events.get(pos)
        if event is None:
            return True
        if not event.query():
            return False
        del self.pending_events[pos]
        return True

    def wait_position(self, position: int) -> None:
        event = self.pending_events.pop(int(position), None)
        if event is not None:
            event.synchronize()

    def wait_chunk_position(self, chunk_id: int, chunk_offset: int) -> None:
        event = self.pending_chunk_events.pop((int(chunk_id), int(chunk_offset)), None)
        if event is not None:
            event.synchronize()

    def synchronize_pending(self) -> None:
        for event in self.pending_events.values():
            event.synchronize()
        self.pending_events.clear()
        for event in self.pending_chunk_events.values():
            event.synchronize()
        self.pending_chunk_events.clear()

    def get_offsets(self, positions: list[int]) -> tuple[list[int], list[int], list[int]]:
        offsets = []
        found_positions = []
        missing_positions = []
        for position in positions:
            pos = int(position)
            offset = self.position_to_offset.get(pos)
            if offset is None:
                missing_positions.append(pos)
                continue
            offsets.append(offset)
            found_positions.append(pos)
        return offsets, found_positions, missing_positions

    def ready_positions(self, positions: list[int]) -> list[int]:
        ready = []
        for position in positions:
            if self.has_position_ready(int(position)):
                ready.append(int(position))
        return ready

    def get_chunk_offsets(
        self, positions: list[int]
    ) -> tuple[list[int], list[int], list[int], list[int], list[int]]:
        chunk_offsets = []
        inner_offsets = []
        chunk_slot_by_id = {}
        found_positions = []
        missing_positions = []
        for position in positions:
            pos = int(position)
            if pos not in self.position_to_offset:
                missing_positions.append(pos)
                continue
            chunk_id = pos // self.chunk_size
            chunk_offset = pos % self.chunk_size
            chunk_slot = self.chunk_id_to_offset.get(chunk_id)
            if chunk_slot is None:
                missing_positions.append(pos)
                continue
            chunk_slot_by_id.setdefault(chunk_id, len(chunk_slot_by_id))
            chunk_offsets.append(chunk_slot)
            inner_offsets.append(chunk_offset)
            found_positions.append(pos)
        unique_chunk_offsets = [
            self.chunk_id_to_offset[chunk_id]
            for chunk_id, _ in sorted(chunk_slot_by_id.items(), key=lambda item: item[1])
        ]
        remapped_chunk_indices = [
            chunk_slot_by_id[int(pos) // self.chunk_size] for pos in found_positions
        ]
        return (
            unique_chunk_offsets,
            remapped_chunk_indices,
            inner_offsets,
            found_positions,
            missing_positions,
        )


class SparseCPUKVStore:
    """CPU-side KV store owned by sparse_framework.

    This intentionally does not own or free SGLang's main KV pool. It snapshots
    KV rows into contiguous CPU buffers keyed by logical request/layer/token
    position so retrieval and working-set materialization can be developed
    independently of SGLang radix-cache ownership.
    """

    def __init__(self):
        self.layers: dict[tuple[int, int], RequestLayerKV] = {}
        self.copy_streams: dict[torch.device, torch.cuda.Stream] = {}
        self.h2d_streams: dict[torch.device, torch.cuda.Stream] = {}
        self.chunking_enabled: bool = False
        self.chunk_size: int = 16

    def configure_chunking(self, *, enabled: bool, chunk_size: int) -> None:
        self.chunking_enabled = bool(enabled)
        self.chunk_size = max(1, int(chunk_size))

    def drop_request(self, req_pool_idx: int) -> int:
        req_pool_idx = int(req_pool_idx)
        keys_to_drop = [
            key for key in self.layers.keys() if int(key[0]) == req_pool_idx
        ]
        for key in keys_to_drop:
            self.layers[key].synchronize_pending()
            del self.layers[key]
        return len(keys_to_drop)

    def put(
        self,
        *,
        req_pool_idx: int,
        layer_id: int,
        positions: list[int],
        keys: torch.Tensor,
        values: torch.Tensor,
    ) -> int:
        if not positions:
            return 0
        entry = self.layers.setdefault((int(req_pool_idx), int(layer_id)), RequestLayerKV())
        key_rows = keys.detach()
        value_rows = values.detach()
        max_rows = min(len(positions), int(key_rows.shape[0]), int(value_rows.shape[0]))
        if max_rows <= 0:
            return 0

        new_positions = [
            int(pos)
            for pos in positions[:max_rows]
            if int(pos) not in entry.position_to_offset
        ]
        entry.ensure_capacity(
            entry.size + len(new_positions),
            tuple(key_rows.shape[1:]),
            tuple(value_rows.shape[1:]),
            key_rows.dtype,
            value_rows.dtype,
        )
        written = 0
        copy_stream = self._copy_stream_for(key_rows, entry)
        for offset, position in enumerate(positions[:max_rows]):
            entry.put_row(
                int(position),
                key_rows[offset],
                value_rows[offset],
                copy_stream=copy_stream,
            )
            if self.chunking_enabled:
                if int(entry.chunk_size) != int(self.chunk_size):
                    entry.reset_chunks(chunk_size=self.chunk_size)
                entry.ensure_chunk_capacity(
                    entry.chunk_count
                    + (
                        0
                        if int(position) // self.chunk_size in entry.chunk_id_to_offset
                        else 1
                    ),
                    tuple(key_rows.shape[1:]),
                    tuple(value_rows.shape[1:]),
                    key_rows.dtype,
                    value_rows.dtype,
                )
                entry.put_chunk_row(
                    int(position),
                    key_rows[offset],
                    value_rows[offset],
                    chunk_size=self.chunk_size,
                    copy_stream=copy_stream,
                )
            written += 1
        entry.version += 1
        return written

    def get_many(
        self,
        *,
        req_pool_idx: int,
        layer_id: int,
        positions: list[int],
        device: torch.device | str,
        dtype: torch.dtype,
    ) -> tuple[torch.Tensor | None, torch.Tensor | None, list[int], list[int]]:
        entry = self.layers.get((int(req_pool_idx), int(layer_id)))
        if entry is None or entry.key_buffer is None or entry.value_buffer is None:
            return None, None, [], [int(pos) for pos in positions]

        offsets, found_positions, missing_positions = entry.get_offsets(
            [int(pos) for pos in positions]
        )
        if not offsets:
            return None, None, found_positions, missing_positions
        for position in found_positions:
            entry.wait_position(position)

        offset_tensor = torch.tensor(offsets, dtype=torch.long)
        key_rows = entry.key_buffer.index_select(0, offset_tensor)
        value_rows = entry.value_buffer.index_select(0, offset_tensor)

        return (
            key_rows.to(device=device, dtype=dtype, non_blocking=entry.pinned),
            value_rows.to(device=device, dtype=dtype, non_blocking=entry.pinned),
            found_positions,
            missing_positions,
        )

    def get_many_async(
        self,
        *,
        req_pool_idx: int,
        layer_id: int,
        positions: list[int],
        device: torch.device | str,
        dtype: torch.dtype,
    ) -> tuple[
        torch.Tensor | None,
        torch.Tensor | None,
        list[int],
        list[int],
        torch.cuda.Event | None,
    ]:
        entry = self.layers.get((int(req_pool_idx), int(layer_id)))
        if entry is None or entry.key_buffer is None or entry.value_buffer is None:
            return None, None, [], [int(pos) for pos in positions], None

        offsets, found_positions, missing_positions = entry.get_offsets(
            [int(pos) for pos in positions]
        )
        if not offsets:
            return None, None, found_positions, missing_positions, None
        for position in found_positions:
            entry.wait_position(position)

        target_device = torch.device(device)
        if target_device.type != "cuda" or not entry.pinned or not torch.cuda.is_available():
            keys, values, found_positions, missing_positions = self.get_many(
                req_pool_idx=req_pool_idx,
                layer_id=layer_id,
                positions=positions,
                device=device,
                dtype=dtype,
            )
            return keys, values, found_positions, missing_positions, None

        offset_tensor = torch.tensor(offsets, dtype=torch.long)
        key_rows = entry.key_buffer.index_select(0, offset_tensor)
        value_rows = entry.value_buffer.index_select(0, offset_tensor)
        stream = self._h2d_stream_for(target_device)
        assert stream is not None
        with torch.cuda.stream(stream):
            key_out = key_rows.to(device=target_device, dtype=dtype, non_blocking=True)
            value_out = value_rows.to(device=target_device, dtype=dtype, non_blocking=True)
            event = torch.cuda.Event()
            event.record(stream)
            key_out.record_stream(stream)
            value_out.record_stream(stream)
        return key_out, value_out, found_positions, missing_positions, event

    def get_many_chunked_async(
        self,
        *,
        req_pool_idx: int,
        layer_id: int,
        positions: list[int],
        device: torch.device | str,
        dtype: torch.dtype,
    ) -> tuple[
        torch.Tensor | None,
        torch.Tensor | None,
        list[int],
        list[int],
        torch.cuda.Event | None,
        dict,
    ]:
        entry = self.layers.get((int(req_pool_idx), int(layer_id)))
        if (
            entry is None
            or entry.chunk_key_buffer is None
            or entry.chunk_value_buffer is None
            or int(entry.chunk_size) != int(self.chunk_size)
        ):
            keys, values, found, missing, event = self.get_many_async(
                req_pool_idx=req_pool_idx,
                layer_id=layer_id,
                positions=positions,
                device=device,
                dtype=dtype,
            )
            return keys, values, found, missing, event, {"mode": "token_fallback"}

        (
            unique_chunk_offsets,
            remapped_chunk_indices,
            inner_offsets,
            found_positions,
            missing_positions,
        ) = entry.get_chunk_offsets([int(pos) for pos in positions])
        if not unique_chunk_offsets:
            return (
                None,
                None,
                found_positions,
                missing_positions,
                None,
                {"mode": "chunk", "chunks": 0},
            )
        for pos in found_positions:
            entry.wait_chunk_position(pos // entry.chunk_size, pos % entry.chunk_size)

        target_device = torch.device(device)
        chunk_offset_tensor = torch.tensor(unique_chunk_offsets, dtype=torch.long)
        chunk_keys = entry.chunk_key_buffer.index_select(0, chunk_offset_tensor)
        chunk_values = entry.chunk_value_buffer.index_select(0, chunk_offset_tensor)
        remap_cpu = torch.tensor(remapped_chunk_indices, dtype=torch.long)
        inner_cpu = torch.tensor(inner_offsets, dtype=torch.long)

        if (
            target_device.type != "cuda"
            or not entry.chunk_pinned
            or not torch.cuda.is_available()
        ):
            key_rows = chunk_keys[remap_cpu, inner_cpu].to(device=device, dtype=dtype)
            value_rows = chunk_values[remap_cpu, inner_cpu].to(device=device, dtype=dtype)
            return (
                key_rows,
                value_rows,
                found_positions,
                missing_positions,
                None,
                {
                    "mode": "chunk",
                    "chunks": len(unique_chunk_offsets),
                    "positions": len(found_positions),
                    "async": False,
                },
            )

        stream = self._h2d_stream_for(target_device)
        assert stream is not None
        with torch.cuda.stream(stream):
            chunk_keys_gpu = chunk_keys.to(
                device=target_device, dtype=dtype, non_blocking=True
            )
            chunk_values_gpu = chunk_values.to(
                device=target_device, dtype=dtype, non_blocking=True
            )
            remap_gpu = remap_cpu.to(device=target_device, non_blocking=True)
            inner_gpu = inner_cpu.to(device=target_device, non_blocking=True)
            key_out = chunk_keys_gpu[remap_gpu, inner_gpu]
            value_out = chunk_values_gpu[remap_gpu, inner_gpu]
            event = torch.cuda.Event()
            event.record(stream)
            key_out.record_stream(stream)
            value_out.record_stream(stream)
        return (
            key_out,
            value_out,
            found_positions,
            missing_positions,
            event,
            {
                "mode": "chunk",
                "chunks": len(unique_chunk_offsets),
                "positions": len(found_positions),
                "async": True,
            },
        )

    def build_chunk_index(
        self,
        *,
        req_pool_idx: int,
        layer_id: int,
        positions: list[int],
        chunk_size: int,
    ) -> dict | None:
        entry = self.layers.get((int(req_pool_idx), int(layer_id)))
        if entry is None or entry.key_buffer is None:
            return None
        chunk_size = max(1, int(chunk_size))
        if int(entry.chunk_index_cache_version) != int(entry.version):
            entry.chunk_index_cache.clear()
            entry.chunk_index_cache_version = int(entry.version)
        chunks = []
        for start in range(0, len(positions), chunk_size):
            chunk_positions = [int(pos) for pos in positions[start : start + chunk_size]]
            cache_key = (chunk_size, tuple(chunk_positions))
            cached = entry.chunk_index_cache.get(cache_key)
            if cached is not None:
                chunks.append(cached)
                continue
            item = self._build_chunk_index_item_from_chunk_buffer(
                entry,
                chunk_positions=chunk_positions,
                chunk_size=chunk_size,
            )
            if item is not None:
                entry.chunk_index_cache[cache_key] = item
                chunks.append(item)
                continue
            offsets, found_positions, _ = entry.get_offsets(chunk_positions)
            if not offsets:
                continue
            offset_tensor = torch.tensor(offsets, dtype=torch.long)
            centroid = (
                entry.key_buffer.index_select(0, offset_tensor)
                .to(torch.float32)
                .mean(dim=0)
            )
            item = {"positions": found_positions, "centroid": centroid}
            entry.chunk_index_cache[cache_key] = item
            chunks.append(item)
        return {"chunk_size": chunk_size, "chunks": chunks}

    def _build_chunk_index_item_from_chunk_buffer(
        self,
        entry: RequestLayerKV,
        *,
        chunk_positions: list[int],
        chunk_size: int,
    ) -> dict | None:
        if (
            not self.chunking_enabled
            or entry.chunk_key_buffer is None
            or int(entry.chunk_size) != int(chunk_size)
            or len(chunk_positions) != int(chunk_size)
        ):
            return None
        first = int(chunk_positions[0])
        if first % int(chunk_size) != 0:
            return None
        for offset, pos in enumerate(chunk_positions):
            if int(pos) != first + offset:
                return None
            if int(pos) not in entry.position_to_offset:
                return None
        chunk_id = first // int(chunk_size)
        chunk_slot = entry.chunk_id_to_offset.get(chunk_id)
        if chunk_slot is None:
            return None
        centroid = entry.chunk_key_buffer[int(chunk_slot)].to(torch.float32).mean(dim=0)
        return {"positions": chunk_positions, "centroid": centroid}

    def stats(self) -> dict:
        entries = 0
        capacity = 0
        pinned_layers = 0
        pending_copies = 0
        for layer_store in self.layers.values():
            entries += layer_store.size
            capacity += layer_store.capacity
            pinned_layers += int(layer_store.pinned)
            pending_copies += len(layer_store.pending_events)
            pending_copies += len(layer_store.pending_chunk_events)
        return {
            "request_layers": len(self.layers),
            "tokens": entries,
            "capacity": capacity,
            "pinned_layers": pinned_layers,
            "pending_copies": pending_copies,
            "chunking_enabled": self.chunking_enabled,
            "chunk_size": self.chunk_size,
        }

    def has_complete_ready_backup(
        self,
        *,
        req_pool_idx: int,
        position: int,
        expected_layers: int | None,
    ) -> bool:
        if expected_layers is None:
            return False
        for layer_id in range(int(expected_layers)):
            layer_store = self.layers.get((int(req_pool_idx), int(layer_id)))
            if layer_store is None or not layer_store.has_position_ready(position):
                return False
        return True

    def ready_positions(
        self,
        *,
        req_pool_idx: int,
        layer_id: int,
        positions: list[int],
    ) -> list[int]:
        entry = self.layers.get((int(req_pool_idx), int(layer_id)))
        if entry is None or entry.key_buffer is None or entry.value_buffer is None:
            return []
        return entry.ready_positions([int(pos) for pos in positions])

    def _copy_stream_for(
        self,
        key_rows: torch.Tensor,
        entry: RequestLayerKV,
    ) -> torch.cuda.Stream | None:
        if (
            not entry.pinned
            or key_rows.device.type != "cuda"
            or not torch.cuda.is_available()
        ):
            return None
        device = key_rows.device
        stream = self.copy_streams.get(device)
        if stream is None:
            stream = torch.cuda.Stream(device=device)
            self.copy_streams[device] = stream
        return stream

    def _h2d_stream_for(self, device: torch.device) -> torch.cuda.Stream | None:
        if device.type != "cuda" or not torch.cuda.is_available():
            return None
        stream = self.h2d_streams.get(device)
        if stream is None:
            stream = torch.cuda.Stream(device=device)
            self.h2d_streams[device] = stream
        return stream


def _empty_cpu_buffer(
    shape: tuple[int, ...],
    *,
    dtype: torch.dtype,
) -> tuple[torch.Tensor, bool]:
    try:
        return torch.empty(shape, dtype=dtype, device="cpu", pin_memory=True), True
    except RuntimeError:
        return torch.empty(shape, dtype=dtype, device="cpu"), False


def get_cpu_kv_store(framework_state: dict | None) -> SparseCPUKVStore | None:
    if framework_state is None:
        return None
    store = framework_state.get("cpu_kv_store")
    if store is None:
        store = SparseCPUKVStore()
        framework_state["cpu_kv_store"] = store
    return store

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

        with torch.cuda.stream(copy_stream):
            self.key_buffer[offset].copy_(key.detach(), non_blocking=True)
            self.value_buffer[offset].copy_(value.detach(), non_blocking=True)
            event = torch.cuda.Event()
            event.record(copy_stream)
        self.pending_events[pos] = event

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

    def synchronize_pending(self) -> None:
        for event in self.pending_events.values():
            event.synchronize()
        self.pending_events.clear()

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
        chunks = []
        for start in range(0, len(positions), chunk_size):
            chunk_positions = [int(pos) for pos in positions[start : start + chunk_size]]
            offsets, found_positions, _ = entry.get_offsets(chunk_positions)
            if not offsets:
                continue
            offset_tensor = torch.tensor(offsets, dtype=torch.long)
            centroid = (
                entry.key_buffer.index_select(0, offset_tensor)
                .to(torch.float32)
                .mean(dim=0)
            )
            chunks.append({"positions": found_positions, "centroid": centroid})
        return {"chunk_size": chunk_size, "chunks": chunks}

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
        return {
            "request_layers": len(self.layers),
            "tokens": entries,
            "capacity": capacity,
            "pinned_layers": pinned_layers,
            "pending_copies": pending_copies,
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

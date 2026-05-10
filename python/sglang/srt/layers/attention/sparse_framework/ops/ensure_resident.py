from __future__ import annotations

import torch

from sglang.srt.layers.attention.sparse_framework.kv_store import get_cpu_kv_store
from sglang.srt.layers.attention.sparse_framework.residency import (
    KVResidencyEntry,
    get_residency_table,
)


class EnsureFullResidentError(RuntimeError):
    pass


class EnsureFullResidentOp:
    def run(self, ctx, state: dict):
        return ensure_full_kv_resident(ctx, state)


def ensure_full_kv_resident(ctx, state: dict) -> dict:
    layer = ctx.layer
    if layer is None or ctx.framework_state is None:
        return {"needed": 0, "fetched": 0, "reason": "missing_layer_or_state"}

    table = get_residency_table(ctx.framework_state)
    store = get_cpu_kv_store(ctx.framework_state)
    layer_id = int(layer.layer_id)
    host_entries: list[KVResidencyEntry] = []
    sparse_cpu_entries: list[KVResidencyEntry] = []
    missing_entries: list[tuple[int, int, int]] = []

    req_to_token = ctx.req_to_token_pool.req_to_token

    for req_pool_idx, seq_len in zip(ctx.req_pool_indices_cpu, ctx.seq_lens_cpu):
        for position in range(seq_len):
            entry = table.get(req_pool_idx, layer_id, position)
            if entry is None:
                device_index = int(req_to_token[req_pool_idx, position].item())
                if device_index >= 0:
                    entry, _ = table.observe_gpu(
                        req_pool_idx=req_pool_idx,
                        layer_id=layer_id,
                        position=position,
                        device_index=device_index,
                        step=table.next_step(),
                    )
                else:
                    if _has_sparse_cpu_row(
                        store,
                        req_pool_idx=req_pool_idx,
                        layer_id=layer_id,
                        position=position,
                    ):
                        entry = KVResidencyEntry(
                            req_pool_idx=int(req_pool_idx),
                            layer_id=layer_id,
                            position=int(position),
                            device_index=None,
                            host_index=None,
                            state="host",
                            backup_source="sparse_cpu",
                        )
                        table.add_entry(entry)
                        sparse_cpu_entries.append(entry)
                        continue
                    missing_entries.append((req_pool_idx, layer_id, position))
                    continue
            if entry.state == "gpu" and entry.device_index is not None:
                req_to_token[req_pool_idx, position] = entry.device_index
            elif entry.host_index is not None:
                host_entries.append(entry)
            elif entry.has_sparse_cpu_backup:
                sparse_cpu_entries.append(entry)
            else:
                device_index = int(req_to_token[req_pool_idx, position].item())
                if device_index >= 0:
                    table.mark_gpu(entry, device_index=device_index)
                    req_to_token[req_pool_idx, position] = device_index
                elif _has_sparse_cpu_row(
                    store,
                    req_pool_idx=req_pool_idx,
                    layer_id=layer_id,
                    position=position,
                ):
                    table.mark_sparse_cpu_backup(entry, keep_device_index=False)
                    sparse_cpu_entries.append(entry)
                else:
                    missing_entries.append((req_pool_idx, layer_id, position))

    if missing_entries:
        preview = ", ".join(str(item) for item in missing_entries[:4])
        raise EnsureFullResidentError(
            "Cannot fallback to full attention because some KV entries are not "
            f"resident on GPU and have no host copy: {preview}"
        )

    sparse_cpu_result = _restore_sparse_cpu_entries(
        ctx,
        layer,
        sparse_cpu_entries,
    )
    prefetched = _consume_or_wait_pending_host_entries(ctx, host_entries, layer_id)
    if prefetched:
        host_entries = [
            entry
            for entry in host_entries
            if entry.state != "gpu" or entry.device_index is None
        ]

    if not host_entries:
        result = {
            "needed": len(sparse_cpu_entries) + int(prefetched),
            "fetched": int(sparse_cpu_result["restored"]) + int(prefetched),
            "prefetch_consumed": int(prefetched),
            "sparse_cpu": sparse_cpu_result,
        }
        state["ensure_full_resident"] = result
        return result

    if ctx.cache_controller is None:
        raise EnsureFullResidentError(
            "Cannot fallback to full attention because host-resident KV exists "
            "but no cache_controller is bound."
        )

    host_indices = _host_indices_tensor(
        [entry.host_index for entry in host_entries],
        ctx,
    )
    device_indices = ctx.cache_controller.load(host_indices)
    if device_indices is None:
        raise EnsureFullResidentError(
            "Cannot fallback to full attention because device allocation failed "
            "while restoring host-resident KV."
        )

    producer_id = ctx.cache_controller.start_loading()
    layer_done_counter = getattr(ctx.cache_controller, "layer_done_counter", None)
    if producer_id is not None and producer_id >= 0 and layer_done_counter is not None:
        layer_done_counter.set_consumer(producer_id)
        layer_done_counter.wait_until(layer_id)
    _consume_sparse_load_ack(ctx.cache_controller)

    device_list = device_indices.detach().cpu().tolist()
    for entry, device_index in zip(host_entries, device_list):
        table.mark_gpu(entry, device_index=int(device_index))
        req_to_token[entry.req_pool_idx, entry.position] = int(device_index)

    result = {
        "needed": len(host_entries) + len(sparse_cpu_entries),
        "fetched": len(host_entries) + int(sparse_cpu_result["restored"]) + int(prefetched),
        "producer_id": producer_id,
        "prefetch_consumed": int(prefetched),
        "host_pool": {"needed": len(host_entries), "fetched": len(host_entries)},
        "sparse_cpu": sparse_cpu_result,
    }
    state["ensure_full_resident"] = result
    return result


def _restore_sparse_cpu_entries(ctx, layer, entries: list[KVResidencyEntry]) -> dict:
    if not entries:
        return {"needed": 0, "restored": 0, "allocated_slots": 0}

    store = get_cpu_kv_store(ctx.framework_state)
    if store is None:
        raise EnsureFullResidentError(
            "Cannot fallback to full attention because sparse CPU-resident KV "
            "exists but the SparseCPUKVStore is missing."
        )

    allocator = getattr(ctx.model_runner, "token_to_kv_pool_allocator", None)
    if allocator is None:
        raise EnsureFullResidentError(
            "Cannot fallback to full attention because sparse CPU-resident KV "
            "exists but no token_to_kv_pool_allocator is bound."
        )
    if int(getattr(allocator, "page_size", 1)) != 1:
        raise EnsureFullResidentError(
            "Cannot fallback to full attention from SparseCPUKVStore with a paged "
            "KV allocator yet."
        )

    token_to_kv_pool = ctx.token_to_kv_pool
    req_to_token = ctx.req_to_token_pool.req_to_token
    layer_id = int(layer.layer_id)
    key_buffer = token_to_kv_pool.get_key_buffer(layer_id)
    device = key_buffer.device
    dtype = key_buffer.dtype

    by_request: dict[int, list[KVResidencyEntry]] = {}
    for entry in entries:
        by_request.setdefault(int(entry.req_pool_idx), []).append(entry)

    restored = 0
    allocated_slots = 0
    table = get_residency_table(ctx.framework_state)

    for req_pool_idx, request_entries in by_request.items():
        request_entries.sort(key=lambda item: int(item.position))
        positions = [int(entry.position) for entry in request_entries]

        keys, values, found_positions, missing_positions = store.get_many(
            req_pool_idx=req_pool_idx,
            layer_id=layer_id,
            positions=positions,
            device=device,
            dtype=dtype,
        )
        if missing_positions or keys is None or values is None:
            preview = ", ".join(str(pos) for pos in missing_positions[:4])
            raise EnsureFullResidentError(
                "Cannot fallback to full attention because SparseCPUKVStore is "
                f"missing KV rows for req_pool_idx={req_pool_idx}, layer={layer_id}, "
                f"positions={preview}."
            )

        slot_positions = [
            pos
            for pos in found_positions
            if int(req_to_token[req_pool_idx, pos].item()) < 0
        ]
        allocated = None
        if slot_positions:
            allocated = allocator.alloc(len(slot_positions))
            if allocated is None:
                raise EnsureFullResidentError(
                    "Cannot fallback to full attention because device allocation "
                    "failed while restoring SparseCPUKVStore KV."
                )
            allocated_slots += int(allocated.numel())
            for pos, device_index in zip(
                slot_positions,
                allocated.detach().cpu().tolist(),
            ):
                req_to_token[req_pool_idx, pos] = int(device_index)

        device_indices = [
            int(req_to_token[req_pool_idx, pos].item()) for pos in found_positions
        ]
        loc = torch.tensor(device_indices, dtype=torch.long, device=device)
        token_to_kv_pool.set_kv_buffer(layer, loc, keys, values)

        entry_by_position = {int(entry.position): entry for entry in request_entries}
        for pos, device_index in zip(found_positions, device_indices):
            entry = entry_by_position[int(pos)]
            table.mark_gpu(entry, device_index=int(device_index))
            restored += 1

    return {
        "needed": len(entries),
        "restored": restored,
        "allocated_slots": allocated_slots,
    }


def _has_sparse_cpu_row(
    store,
    *,
    req_pool_idx: int,
    layer_id: int,
    position: int,
) -> bool:
    if store is None:
        return False
    layer_store = getattr(store, "layers", {}).get((int(req_pool_idx), int(layer_id)))
    if layer_store is not None:
        position_to_offset = getattr(layer_store, "position_to_offset", {})
        if int(position) in position_to_offset:
            wait_position = getattr(layer_store, "wait_position", None)
            if callable(wait_position):
                wait_position(int(position))
            return True
    ready_positions = getattr(store, "ready_positions", None)
    if callable(ready_positions):
        return int(position) in ready_positions(
            req_pool_idx=int(req_pool_idx),
            layer_id=int(layer_id),
            positions=[int(position)],
        )
    return False


def _host_indices_tensor(indices: list[int], ctx) -> torch.Tensor:
    io_backend = getattr(ctx.cache_controller, "io_backend", None)
    device = "cpu" if io_backend in ("direct", "kernel_ascend") else ctx.seq_lens.device
    return torch.tensor(indices, dtype=torch.long, device=device)


def _consume_or_wait_pending_host_entries(ctx, entries, layer_id: int) -> int:
    if not entries or ctx.framework_state is None or ctx.cache_controller is None:
        return 0
    pending = ctx.framework_state.get("host_kv_prefetches") or {}
    if not pending:
        return 0

    layer_done_counter = getattr(ctx.cache_controller, "layer_done_counter", None)
    if layer_done_counter is None:
        return 0
    table = get_residency_table(ctx.framework_state)
    req_to_token = ctx.req_to_token_pool.req_to_token
    consumed = 0
    for entry in entries:
        key = (int(entry.req_pool_idx), int(entry.layer_id), int(entry.position))
        item = pending.get(key)
        if item is None:
            continue
        producer_id = int(item["producer_id"])
        if producer_id >= 0:
            layer_done_counter.set_consumer(producer_id)
            layer_done_counter.wait_until(layer_id)
        device_index = int(item["device_indices"][int(item["offset"])].item())
        table.mark_gpu(entry, device_index=device_index)
        req_to_token[entry.req_pool_idx, entry.position] = device_index
        del pending[key]
        consumed += 1
    if consumed:
        _consume_sparse_load_ack(ctx.cache_controller, only_finished=True)
    return consumed


def _consume_sparse_load_ack(cache_controller, *, only_finished: bool = False) -> None:
    ack_queue = getattr(cache_controller, "ack_load_queue", None)
    if not ack_queue:
        return

    cleaned_queue = []
    for ack in ack_queue:
        if only_finished:
            query = getattr(ack.finish_event, "query", None)
            if callable(query) and not query():
                cleaned_queue.append(ack)
                continue
        node_ids = list(getattr(ack, "node_ids", []))
        if -1 not in node_ids:
            cleaned_queue.append(ack)
            continue

        finish_event = ack.finish_event
        synchronize = getattr(finish_event, "synchronize", None)
        if callable(synchronize):
            synchronize()

        remaining_node_ids = [node_id for node_id in node_ids if node_id != -1]
        if remaining_node_ids:
            cleaned_queue.append(ack._replace(node_ids=remaining_node_ids))

    ack_queue[:] = cleaned_queue

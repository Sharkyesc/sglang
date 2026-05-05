from __future__ import annotations

import torch

from sglang.srt.layers.attention.sparse_framework.kv_store import get_cpu_kv_store
from sglang.srt.layers.attention.sparse_framework.ops.base import BaseSparseOp
from sglang.srt.layers.attention.sparse_framework.ops.utils import (
    rewrite_selected_kv_indices_from_entries,
)
from sglang.srt.layers.attention.sparse_framework.residency import (
    get_residency_table,
)


class FetchOp(BaseSparseOp):
    def run(self, ctx, state: dict):
        misses = state.get("cache_misses") or []
        host_misses = [entry for entry in misses if entry.host_index is not None]
        layer_id = getattr(ctx.layer, "layer_id", None)
        consumed, consumed_entries = _consume_ready_prefetches(ctx, host_misses, layer_id)
        if consumed_entries:
            rewrite_selected_kv_indices_from_entries(state, consumed_entries)
        host_misses = [
            entry
            for entry in host_misses
            if entry.state != "gpu" or entry.device_index is None
        ]
        sparse_cpu_prefetch = {
            "requested": 0,
            "scheduled": 0,
            "reason": "skipped_due_to_host_misses",
        }
        if not host_misses:
            sparse_cpu_prefetch = _schedule_sparse_cpu_prefetch(ctx, state)
            state["fetch_result"] = {
                "requested": consumed,
                "fetched": consumed,
                "prefetch_consumed": consumed,
                "sparse_cpu_prefetch": sparse_cpu_prefetch,
                "reason": "prefetch_consumed" if consumed else "no_host_resident_misses",
            }
            return None
        if ctx.cache_controller is None:
            state["fetch_result"] = {
                "requested": len(host_misses),
                "fetched": 0,
                "sparse_cpu_prefetch": sparse_cpu_prefetch,
                "reason": "missing_cache_controller",
            }
            return None

        scheduled, schedule_reason = _schedule_prefetch(ctx, host_misses)
        if scheduled > 0 or schedule_reason == "prefetch_already_pending":
            state["fetch_result"] = {
                "requested": len(host_misses) + consumed,
                "fetched": consumed,
                "prefetch_consumed": consumed,
                "prefetch_scheduled": scheduled,
                "pending": len(host_misses),
                "sparse_cpu_prefetch": sparse_cpu_prefetch,
                "reason": schedule_reason,
            }
            state["subset_unavailable_reason"] = "host_prefetch_pending"
            return None

        host_indices = _host_indices_tensor(
            [entry.host_index for entry in host_misses],
            ctx,
        )
        device_indices = ctx.cache_controller.load(host_indices)
        if device_indices is None:
            state["fetch_result"] = {
                "requested": len(host_misses),
                "fetched": 0,
                "sparse_cpu_prefetch": sparse_cpu_prefetch,
                "reason": "device_allocation_failed",
            }
            return None

        producer_id = ctx.cache_controller.start_loading()
        layer_done_counter = getattr(ctx.cache_controller, "layer_done_counter", None)
        if producer_id is not None and producer_id >= 0 and layer_id is not None:
            if layer_done_counter is not None:
                layer_done_counter.set_consumer(producer_id)
                layer_done_counter.wait_until(layer_id)
        _consume_sparse_load_ack(ctx.cache_controller)
        device_list = device_indices.detach().cpu().tolist()
        table = get_residency_table(ctx.framework_state) if ctx.framework_state is not None else None
        for entry, device_index in zip(host_misses, device_list):
            if table is not None:
                table.mark_gpu(entry, device_index=int(device_index))
            else:
                entry.device_index = int(device_index)
                entry.logical_evicted = False

        updated = rewrite_selected_kv_indices_from_entries(state, host_misses)
        state["fetch_result"] = {
            "requested": len(host_misses),
            "fetched": len(host_misses),
            "producer_id": producer_id,
            "remapped": updated,
            "sparse_cpu_prefetch": sparse_cpu_prefetch,
        }
        return None


def _host_indices_tensor(indices: list[int], ctx) -> torch.Tensor:
    io_backend = getattr(ctx.cache_controller, "io_backend", None)
    device = "cpu" if io_backend in ("direct", "kernel_ascend") else ctx.seq_lens.device
    return torch.tensor(indices, dtype=torch.long, device=device)


def _schedule_sparse_cpu_prefetch(ctx, state: dict) -> dict:
    if ctx.framework_state is None or ctx.layer is None:
        return {"requested": 0, "scheduled": 0, "reason": "missing_state_or_layer"}
    store = get_cpu_kv_store(ctx.framework_state)
    if store is None:
        return {"requested": 0, "scheduled": 0, "reason": "missing_sparse_cpu_store"}

    selected_positions = state.get("selected_positions") or []
    selected_kv_indices = state.get("selected_kv_indices") or []
    if not selected_positions or not selected_kv_indices:
        return {"requested": 0, "scheduled": 0, "reason": "missing_selection"}

    layer_id = int(ctx.layer.layer_id)
    key_buffer = ctx.token_to_kv_pool.get_key_buffer(layer_id)
    value_buffer = ctx.token_to_kv_pool.get_value_buffer(layer_id)
    device = key_buffer.device
    dtype = key_buffer.dtype
    pending = ctx.framework_state.setdefault("sparse_cpu_prefetches", {})
    requested = 0
    scheduled = 0
    already_pending = 0
    missing = 0
    missing_gpu_available = 0
    missing_gpu_unavailable = 0

    for batch_idx, positions_tensor in enumerate(selected_positions):
        if batch_idx >= len(ctx.req_pool_indices_cpu) or batch_idx >= len(selected_kv_indices):
            continue
        req_pool_idx = int(ctx.req_pool_indices_cpu[batch_idx])
        positions = [int(pos) for pos in positions_tensor.detach().cpu().tolist()]
        kv_indices_cpu = [int(x) for x in selected_kv_indices[batch_idx].detach().cpu().tolist()]
        if not positions:
            continue
        requested += len(positions)
        prefetch_key = (req_pool_idx, layer_id, tuple(positions))
        item = pending.get(prefetch_key)
        if item is not None:
            already_pending += len(positions)
            continue

        get_many_async = getattr(store, "get_many_async", None)
        if callable(get_many_async):
            keys, values, found_positions, missing_positions, h2d_event = get_many_async(
                req_pool_idx=req_pool_idx,
                layer_id=layer_id,
                positions=positions,
                device=device,
                dtype=dtype,
            )
        else:
            keys, values, found_positions, missing_positions = store.get_many(
                req_pool_idx=req_pool_idx,
                layer_id=layer_id,
                positions=positions,
                device=device,
                dtype=dtype,
            )
            h2d_event = None
        batch_missing_gpu_unavailable = 0
        if missing_positions:
            missing += len(missing_positions)
            position_to_offset = {int(pos): idx for idx, pos in enumerate(positions)}
            for pos in missing_positions:
                offset = position_to_offset.get(int(pos))
                if offset is not None and offset < len(kv_indices_cpu) and kv_indices_cpu[offset] >= 0:
                    missing_gpu_available += 1
                else:
                    missing_gpu_unavailable += 1
                    batch_missing_gpu_unavailable += 1
        if keys is None or values is None:
            if not missing_positions:
                missing += len(positions)
                missing_gpu_unavailable += len(positions)
                batch_missing_gpu_unavailable += len(positions)
            if batch_missing_gpu_unavailable > 0:
                continue
        if missing_positions and h2d_event is not None and keys is not None:
            torch.cuda.current_stream(device=keys.device).wait_event(h2d_event)
            h2d_event = None

        key_subset, value_subset, prefetch_positions, gpu_snapshot = _merge_cpu_and_gpu_rows(
            ctx,
            layer_id=layer_id,
            req_pool_idx=req_pool_idx,
            positions=positions,
            kv_indices_cpu=kv_indices_cpu,
            cpu_keys=keys,
            cpu_values=values,
            cpu_positions=found_positions,
            k_cache=key_buffer,
            v_cache=value_buffer,
            dtype=dtype,
        )
        if key_subset is None or value_subset is None:
            continue

        if gpu_snapshot["positions"]:
            store.put(
                req_pool_idx=req_pool_idx,
                layer_id=layer_id,
                positions=gpu_snapshot["positions"],
                keys=gpu_snapshot["keys"],
                values=gpu_snapshot["values"],
            )

        event = h2d_event
        if event is None and key_subset.is_cuda:
            event = torch.cuda.Event()
            event.record(torch.cuda.current_stream(device=device))
        pending[prefetch_key] = {
            "keys": key_subset,
            "values": value_subset,
            "positions": prefetch_positions,
            "event": event,
        }
        scheduled += len(prefetch_positions)

    return {
        "requested": requested,
        "scheduled": scheduled,
        "already_pending": already_pending,
        "missing": missing,
        "missing_gpu_available": missing_gpu_available,
        "missing_gpu_unavailable": missing_gpu_unavailable,
        "all_missing_but_gpu_available": bool(
            requested > 0
            and missing == requested
            and missing_gpu_available == requested
            and missing_gpu_unavailable == 0
        ),
        "reason": (
            "scheduled"
            if scheduled
            else (
                "demand_fill_gpu_available"
                if requested > 0
                and missing == requested
                and missing_gpu_available == requested
                and missing_gpu_unavailable == 0
                else "none_scheduled"
            )
        ),
    }


def _merge_cpu_and_gpu_rows(
    ctx,
    *,
    layer_id: int,
    req_pool_idx: int,
    positions: list[int],
    kv_indices_cpu: list[int],
    cpu_keys: torch.Tensor | None,
    cpu_values: torch.Tensor | None,
    cpu_positions: list[int],
    k_cache: torch.Tensor,
    v_cache: torch.Tensor,
    dtype: torch.dtype,
) -> tuple[torch.Tensor | None, torch.Tensor | None, list[int], dict]:
    cpu_lookup = {int(pos): idx for idx, pos in enumerate(cpu_positions)}
    key_parts = []
    value_parts = []
    gpu_snapshot_positions = []
    gpu_snapshot_keys = []
    gpu_snapshot_values = []

    current_key_view = None
    current_value_view = None
    current_cache_locs = getattr(ctx, "out_cache_locs_cpu", [])
    if ctx.key is not None and ctx.value is not None:
        current_key_view = ctx.key.reshape(-1, ctx.layer.tp_k_head_num, ctx.layer.qk_head_dim)
        current_value_view = ctx.value.reshape(-1, ctx.layer.tp_k_head_num, ctx.layer.v_head_dim)

    for offset, pos in enumerate(positions):
        cpu_offset = cpu_lookup.get(int(pos))
        if cpu_offset is not None and cpu_keys is not None and cpu_values is not None:
            key_parts.append(cpu_keys[cpu_offset])
            value_parts.append(cpu_values[cpu_offset])
            continue

        if offset >= len(kv_indices_cpu) or int(kv_indices_cpu[offset]) < 0:
            return None, None, [], {"positions": [], "keys": None, "values": None}

        device_index = int(kv_indices_cpu[offset])
        current_row = None
        for batch_idx, req_pool_idx_item in enumerate(ctx.req_pool_indices_cpu):
            if (
                int(req_pool_idx_item) == int(req_pool_idx)
                and batch_idx < len(current_cache_locs)
                and int(current_cache_locs[batch_idx]) == device_index
                and current_key_view is not None
                and current_value_view is not None
                and batch_idx < int(current_key_view.shape[0])
            ):
                current_row = (current_key_view[batch_idx], current_value_view[batch_idx])
                break

        if current_row is None:
            key_row = k_cache[device_index].to(dtype=dtype)
            value_row = v_cache[device_index].to(dtype=dtype)
        else:
            key_row, value_row = current_row

        key_parts.append(key_row)
        value_parts.append(value_row)
        gpu_snapshot_positions.append(int(pos))
        gpu_snapshot_keys.append(key_row)
        gpu_snapshot_values.append(value_row)

    gpu_snapshot = {"positions": gpu_snapshot_positions, "keys": None, "values": None}
    if gpu_snapshot_keys:
        gpu_snapshot["keys"] = torch.stack(gpu_snapshot_keys)
        gpu_snapshot["values"] = torch.stack(gpu_snapshot_values)
    return (
        torch.stack(key_parts),
        torch.stack(value_parts),
        [int(pos) for pos in positions],
        gpu_snapshot,
    )


def _entry_key(entry) -> tuple[int, int, int]:
    return (int(entry.req_pool_idx), int(entry.layer_id), int(entry.position))


def _pending_prefetches(ctx) -> dict:
    if ctx.framework_state is None:
        return {}
    return ctx.framework_state.setdefault("host_kv_prefetches", {})


def _producer_ready(cache_controller, producer_id: int, layer_id: int | None) -> bool:
    if producer_id is None or int(producer_id) < 0:
        return False
    layer_done_counter = getattr(cache_controller, "layer_done_counter", None)
    if layer_done_counter is None or layer_id is None:
        return False
    events = getattr(layer_done_counter, "events", None)
    if events is None or int(producer_id) >= len(events):
        return False
    load_events = getattr(events[int(producer_id)], "load_events", None)
    if load_events is None or int(layer_id) >= len(load_events):
        return False
    query = getattr(load_events[int(layer_id)], "query", None)
    return bool(query()) if callable(query) else False


def _consume_ready_prefetches(ctx, entries, layer_id: int | None) -> tuple[int, list]:
    if not entries or ctx.framework_state is None or ctx.cache_controller is None:
        return 0, []

    pending = _pending_prefetches(ctx)
    req_to_token = ctx.req_to_token_pool.req_to_token
    consumed_entries = []
    for entry in entries:
        key = _entry_key(entry)
        item = pending.get(key)
        if item is None:
            continue
        if not _producer_ready(ctx.cache_controller, item["producer_id"], layer_id):
            continue
        device_index = int(item["device_indices"][int(item["offset"])].item())
        table = get_residency_table(ctx.framework_state)
        table.mark_gpu(entry, device_index=device_index)
        req_to_token[entry.req_pool_idx, entry.position] = device_index
        consumed_entries.append(entry)
        del pending[key]
    if consumed_entries:
        _consume_sparse_load_ack(ctx.cache_controller, only_finished=True)
    return len(consumed_entries), consumed_entries


def _schedule_prefetch(ctx, entries) -> tuple[int, str]:
    if not entries or ctx.framework_state is None:
        return 0, "missing_state"
    pending = _pending_prefetches(ctx)
    to_schedule = [entry for entry in entries if _entry_key(entry) not in pending]
    if not to_schedule:
        return 0, "prefetch_already_pending"

    host_indices = _host_indices_tensor([entry.host_index for entry in to_schedule], ctx)
    device_indices = ctx.cache_controller.load(host_indices, node_id=-1)
    if device_indices is None:
        return 0, "device_allocation_failed"
    producer_id = ctx.cache_controller.start_loading()
    if producer_id is None or int(producer_id) < 0:
        return 0, "start_loading_failed"
    for offset, entry in enumerate(to_schedule):
        pending[_entry_key(entry)] = {
            "producer_id": int(producer_id),
            "device_indices": device_indices,
            "offset": int(offset),
        }
    return len(to_schedule), "prefetch_scheduled"


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

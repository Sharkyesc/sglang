from __future__ import annotations

import torch


def rebuild_packed_kv_indices(state: dict) -> None:
    selected_kv_indices = state.get("selected_kv_indices") or []
    if not selected_kv_indices:
        return
    device = selected_kv_indices[0].device
    state["kv_indices"] = (
        torch.cat(selected_kv_indices)
        if sum(int(indices.numel()) for indices in selected_kv_indices) > 0
        else torch.empty(0, dtype=torch.long, device=device)
    )


def cpu_kv_store_enabled(state: dict | None) -> bool:
    if state is None:
        return False
    plan = state.get("execution_plan")
    return bool(
        getattr(plan, "enable_host_backup_on_evict", False)
        or getattr(plan, "enable_physical_eviction", False)
    )


def configure_cpu_kv_store_from_state(store, state: dict | None) -> None:
    if store is None or state is None:
        return
    configure = getattr(store, "configure_chunking", None)
    if not callable(configure):
        return
    plan = state.get("execution_plan")
    configure(
        enabled=bool(getattr(plan, "use_chunked_cpu_store", False)),
        chunk_size=int(getattr(plan, "chunk_size", 16)),
    )


def rewrite_selected_kv_indices_from_entries(state: dict, entries) -> int:
    selected_cache_keys = state.get("selected_cache_keys") or []
    selected_kv_indices = state.get("selected_kv_indices") or []
    if not selected_cache_keys or not selected_kv_indices:
        return 0

    by_key = {
        (entry.req_pool_idx, entry.layer_id, entry.position): entry.device_index
        for entry in entries
        if entry.device_index is not None
    }
    updated = 0
    for batch_idx, batch_keys in enumerate(selected_cache_keys):
        if batch_idx >= len(selected_kv_indices):
            continue
        kv_indices = selected_kv_indices[batch_idx].clone()
        for item_idx, key in enumerate(batch_keys):
            device_index = by_key.get(key)
            if device_index is not None and item_idx < kv_indices.numel():
                if int(kv_indices[item_idx].item()) != int(device_index):
                    updated += 1
                kv_indices[item_idx] = int(device_index)
        selected_kv_indices[batch_idx] = kv_indices
    state["selected_kv_indices"] = selected_kv_indices
    rebuild_packed_kv_indices(state)
    return updated

from __future__ import annotations

import torch

from sglang.srt.layers.attention.sparse_framework.ops.base import BaseSparseOp
from sglang.srt.layers.attention.sparse_framework.ops.utils import (
    rebuild_packed_kv_indices,
)
from sglang.srt.layers.attention.sparse_framework.residency import (
    get_residency_table,
)


class RemapOp(BaseSparseOp):
    def run(self, ctx, state: dict):
        selected_kv_indices = state.get("selected_kv_indices") or []
        miss_slots = state.get("cache_miss_slots") or []
        if not miss_slots:
            state["remap_result"] = {"updated": 0, "skipped": 0, "reason": "no_misses"}
            return None
        if not selected_kv_indices or ctx.framework_state is None:
            state["remap_result"] = {"updated": 0, "reason": "missing_selection"}
            return None

        table = get_residency_table(ctx.framework_state)
        req_to_token = ctx.req_to_token_pool.req_to_token
        updated = 0
        skipped = 0
        batch_updates: dict[int, list[tuple[int, int]]] = {}
        req_updates: list[int] = []
        position_updates: list[int] = []
        device_updates: list[int] = []
        for batch_idx, item_idx, key in miss_slots:
            if batch_idx >= len(selected_kv_indices):
                skipped += 1
                continue
            entry = table.entries.get(key)
            if entry is None or entry.state != "gpu" or entry.device_index is None:
                skipped += 1
                continue
            if item_idx < selected_kv_indices[batch_idx].numel():
                batch_updates.setdefault(int(batch_idx), []).append(
                    (int(item_idx), int(entry.device_index))
                )
                updated += 1
            req_pool_idx, _, position = key
            req_updates.append(int(req_pool_idx))
            position_updates.append(int(position))
            device_updates.append(int(entry.device_index))

        for batch_idx, updates in batch_updates.items():
            kv_indices = selected_kv_indices[batch_idx].clone()
            item_indices = torch.tensor(
                [item_idx for item_idx, _ in updates],
                dtype=torch.long,
                device=kv_indices.device,
            )
            device_indices = torch.tensor(
                [device_index for _, device_index in updates],
                dtype=kv_indices.dtype,
                device=kv_indices.device,
            )
            kv_indices[item_indices] = device_indices
            selected_kv_indices[batch_idx] = kv_indices

        if req_updates:
            req_to_token[
                torch.tensor(req_updates, dtype=torch.long, device=req_to_token.device),
                torch.tensor(
                    position_updates, dtype=torch.long, device=req_to_token.device
                ),
            ] = torch.tensor(
                device_updates, dtype=req_to_token.dtype, device=req_to_token.device
            )

        if updated:
            rebuild_packed_kv_indices(state)
        state["selected_kv_indices"] = selected_kv_indices
        state["remap_result"] = {"updated": updated, "skipped": skipped}
        return None

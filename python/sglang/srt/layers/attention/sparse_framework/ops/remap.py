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
        selected_cache_keys = state.get("selected_cache_keys") or []
        selected_kv_indices = state.get("selected_kv_indices") or []
        if not selected_cache_keys or not selected_kv_indices or ctx.framework_state is None:
            state["remap_result"] = {"updated": 0, "reason": "missing_selection"}
            return None

        table = get_residency_table(ctx.framework_state)
        req_to_token = ctx.req_to_token_pool.req_to_token
        updated = 0
        skipped = 0
        for batch_idx, batch_keys in enumerate(selected_cache_keys):
            if batch_idx >= len(selected_kv_indices):
                continue
            kv_indices = selected_kv_indices[batch_idx].clone()
            for item_idx, key in enumerate(batch_keys):
                entry = table.entries.get(key)
                if entry is None or entry.state != "gpu" or entry.device_index is None:
                    skipped += 1
                    continue
                if item_idx < kv_indices.numel() and int(kv_indices[item_idx].item()) != entry.device_index:
                    kv_indices[item_idx] = entry.device_index
                    updated += 1
                req_pool_idx, _, position = key
                req_to_token[req_pool_idx, position] = entry.device_index
            selected_kv_indices[batch_idx] = kv_indices

        if selected_kv_indices:
            rebuild_packed_kv_indices(state)
        state["selected_kv_indices"] = selected_kv_indices
        state["remap_result"] = {"updated": updated, "skipped": skipped}
        return None

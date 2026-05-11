from __future__ import annotations

from sglang.srt.layers.attention.sparse_framework.eviction_tracker import (
    get_eviction_tracker,
)
from sglang.srt.layers.attention.sparse_framework.ops.base import BaseSparseOp
from sglang.srt.layers.attention.sparse_framework.residency import (
    KVResidencyEntry,
    get_residency_table,
)


class CacheOp(BaseSparseOp):
    def run(self, ctx, state: dict):
        plan = state.get("execution_plan")
        selected_positions = state.get("selected_positions") or []
        selected_kv_indices = state.get("selected_kv_indices") or []
        selection_importance = state.get("selection_importance") or []
        layer_id = getattr(ctx.layer, "layer_id", None)
        if layer_id is None or ctx.framework_state is None:
            state["cache_result"] = {"enabled": False, "reason": "missing_layer_or_state"}
            return None

        table = get_residency_table(ctx.framework_state)
        tracker = get_eviction_tracker(ctx.framework_state)
        step = table.next_step()

        hits = []
        misses = []
        miss_slots = []
        active_keys = set()
        selected_cache_keys = []
        for batch_idx, req_pool_idx in enumerate(ctx.req_pool_indices_cpu):
            batch_keys = []
            if batch_idx >= len(selected_positions) or batch_idx >= len(selected_kv_indices):
                selected_cache_keys.append(batch_keys)
                continue
            positions = selected_positions[batch_idx].detach().cpu().tolist()
            kv_indices = selected_kv_indices[batch_idx].detach().cpu().tolist()
            importance_by_position = (
                selection_importance[batch_idx]
                if batch_idx < len(selection_importance)
                else {}
            )
            for pos, device_index in zip(positions, kv_indices):
                key = (req_pool_idx, layer_id, int(pos))
                importance = float(importance_by_position.get(int(pos), 0.0))
                active_keys.add(key)
                batch_keys.append(key)
                if int(device_index) < 0:
                    entry = table.entries.get(key)
                    if entry is None:
                        entry = KVResidencyEntry(
                            req_pool_idx=req_pool_idx,
                            layer_id=layer_id,
                            position=int(pos),
                            state="evicted",
                            last_access_step=step,
                            access_count=1,
                        )
                        table.add_entry(entry)
                    else:
                        entry.last_access_step = step
                        entry.access_count += 1
                    entry.selection_priority = importance
                    entry.last_selected_step = step
                    is_miss = True
                else:
                    tracker.mark_reused(req_pool_idx, int(pos))
                    entry, is_miss = table.observe_gpu(
                        req_pool_idx=req_pool_idx,
                        layer_id=layer_id,
                        position=int(pos),
                        device_index=int(device_index),
                        step=step,
                    )
                    entry.selection_priority = importance
                    entry.last_selected_step = step
                if is_miss:
                    misses.append(entry)
                    miss_slots.append((batch_idx, len(batch_keys) - 1, key))
                else:
                    hits.append(entry)
            selected_cache_keys.append(batch_keys)

        cache_policy = getattr(plan, "cache_policy", "working_set")
        needs_eviction_stats = (
            plan is not None
            and getattr(plan, "working_set_budget_tokens", None) is not None
        )
        if needs_eviction_stats:
            live_gpu, live_gpu_tokens = table.live_gpu_stats()
        else:
            live_gpu = None
            live_gpu_tokens = None

        state["cache_hits"] = hits
        state["cache_misses"] = misses
        state["cache_miss_slots"] = miss_slots
        state["selected_cache_keys"] = selected_cache_keys
        state["active_cache_keys"] = active_keys
        state["eviction_candidates"] = []
        state["cache_result"] = {
            "enabled": True,
            "hits": len(hits),
            "misses": len(misses),
            "entries": len(table.entries),
            "live_gpu": live_gpu,
            "live_gpu_tokens": live_gpu_tokens,
            "cache_policy": cache_policy,
        }
        return None

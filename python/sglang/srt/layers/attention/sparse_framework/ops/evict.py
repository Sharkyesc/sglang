from __future__ import annotations

import torch

from sglang.srt.layers.attention.sparse_framework.eviction_tracker import (
    get_eviction_tracker,
)
from sglang.srt.layers.attention.sparse_framework.ops.base import BaseSparseOp
from sglang.srt.layers.attention.sparse_framework.residency import (
    get_residency_table,
)


class EvictOp(BaseSparseOp):
    def run(self, ctx, state: dict):
        plan = state["execution_plan"]
        budget = plan.working_set_budget_tokens
        candidates = state.get("eviction_candidates") or []
        if budget is None or ctx.framework_state is None:
            state["evict_result"] = {"evicted": 0, "reason": "no_budget"}
            return None

        table = get_residency_table(ctx.framework_state)
        live_entry_count = table.live_gpu_count()
        live_token_count = table.live_gpu_token_count()
        excess = max(0, live_token_count - int(budget))
        if excess <= 0:
            state["evict_result"] = {
                "evicted": 0,
                "live": live_token_count,
                "live_entries": live_entry_count,
                "live_tokens": live_token_count,
                "budget": int(budget),
            }
            return None

        to_evict = candidates
        physical_free_requested = bool(plan.enable_physical_eviction)

        cpu_store = None
        if ctx.framework_state is not None:
            cpu_store = ctx.framework_state.get("cpu_kv_store")
        cpu_stats = cpu_store.stats() if cpu_store is not None else None
        physical_result = self._physically_free_tokens(
            ctx,
            table,
            to_evict,
            cpu_store=cpu_store,
            enabled=physical_free_requested,
            state=state,
            excess=excess,
        )

        state["evicted_entries"] = to_evict
        state["evict_result"] = {
            "evicted": int(physical_result["freed_tokens"]),
            "logical_candidates": len(to_evict),
            "logical_evicted": int(physical_result.get("logical_evicted", 0)),
            "physical_free": bool(physical_result["freed_tokens"]),
            "physically_freed": int(physical_result["freed_tokens"]),
            "live_before": live_token_count,
            "live_entries_before": live_entry_count,
            "live_tokens_before": live_token_count,
            "budget": int(budget),
            "physical_free_requested": physical_free_requested,
            "physical_free_skipped_reason": physical_result.get("skipped_reason"),
            "backup_backend": "sparse_framework_cpu_kv_store",
            "physical_free_result": physical_result,
            "cpu_kv_store": cpu_stats,
        }
        return None

    def _physically_free_tokens(
        self,
        ctx,
        table,
        entries,
        *,
        cpu_store,
        enabled: bool,
        state: dict,
        excess: int,
    ) -> dict:
        if not enabled:
            return {"freed_tokens": 0, "skipped_reason": "disabled"}
        allocator = getattr(ctx.model_runner, "token_to_kv_pool_allocator", None)
        if allocator is None:
            return {"freed_tokens": 0, "skipped_reason": "missing_allocator"}
        if int(getattr(allocator, "page_size", 1)) != 1:
            return {"freed_tokens": 0, "skipped_reason": "paged_allocator_unsupported"}
        setattr(allocator, "_sparse_framework_physical_eviction_active", True)
        if cpu_store is None:
            return {"freed_tokens": 0, "skipped_reason": "missing_cpu_store"}

        radix_evicted = self._drain_radix_evictable(ctx)
        radix_owned = self._radix_owned_device_indices(ctx)
        active_tokens = self._active_token_positions(ctx, state=state)
        expected_layers = self._expected_num_layers(ctx)
        max_tokens_to_free = max(1, int(excess))
        tracker = get_eviction_tracker(ctx.framework_state)
        free_plan, tracker_stats = tracker.build_free_plan(
            ctx=ctx,
            entries=entries,
            cpu_store=cpu_store,
            expected_layers=expected_layers,
            active_tokens=active_tokens,
            radix_owned_device_indices=radix_owned,
            max_tokens=max_tokens_to_free,
        )

        if not free_plan:
            return {
                "freed_tokens": 0,
                "radix_evicted": radix_evicted,
                "tracker": tracker_stats,
                "skipped_reason": "no_tracker_eligible_candidates",
            }

        device = ctx.seq_lens.device
        free_slots = torch.tensor(
            sorted({device_index for _, _, device_index in free_plan}),
            dtype=torch.long,
            device=device,
        )
        allocator.free(free_slots)
        freed_slot_set = getattr(allocator, "_sparse_framework_freed_slots", None)
        if freed_slot_set is None:
            freed_slot_set = set()
            setattr(allocator, "_sparse_framework_freed_slots", freed_slot_set)
        freed_slot_set.update(int(x) for x in free_slots.detach().cpu().tolist())
        tracker.mark_freed(free_plan)

        req_to_token = ctx.req_to_token_pool.req_to_token
        logical_evicted = 0
        for req_pool_idx, position, _ in free_plan:
            if 0 <= position < int(req_to_token.shape[1]):
                req_to_token[req_pool_idx, position] = -1
            for table_entry in table.entries_for_token(req_pool_idx, position):
                if table_entry.state == "gpu":
                    table.mark_sparse_cpu_backup(
                        table_entry,
                        keep_device_index=False,
                    )
                    logical_evicted += 1

        return {
            "freed_tokens": len(free_slots),
            "freed_entries": len(free_plan),
            "logical_evicted": logical_evicted,
            "radix_evicted": radix_evicted,
            "tracker": tracker_stats,
            "skipped_reason": None,
        }

    def _expected_num_layers(self, ctx) -> int | None:
        model_config = getattr(ctx.model_runner, "model_config", None)
        num_layers = getattr(model_config, "num_hidden_layers", None)
        return int(num_layers) if num_layers is not None else None

    def _active_token_positions(self, ctx, state) -> set[tuple[int, int]]:
        selected_positions = []
        if state is not None:
            selected_positions = state.get("selected_positions") or []
        active = set()
        for batch_idx, req_pool_idx in enumerate(ctx.req_pool_indices_cpu):
            if batch_idx >= len(selected_positions):
                continue
            for pos in selected_positions[batch_idx].detach().cpu().tolist():
                active.add((int(req_pool_idx), int(pos)))
        return active

    def _drain_radix_evictable(self, ctx) -> int:
        tree_cache = None
        if ctx.framework_state is not None:
            tree_cache = ctx.framework_state.get("tree_cache")
        evictable_size = getattr(tree_cache, "evictable_size", None)
        evict = getattr(tree_cache, "evict", None)
        if not callable(evictable_size) or not callable(evict):
            return 0
        count = int(evictable_size())
        if count > 0:
            evict(count)
        return count

    def _radix_owned_device_indices(self, ctx) -> set[int]:
        tree_cache = None
        if ctx.framework_state is not None:
            tree_cache = ctx.framework_state.get("tree_cache")
        flatten = getattr(tree_cache, "all_values_flatten", None)
        if not callable(flatten):
            return set()
        try:
            values = flatten()
        except Exception:
            return set()
        if values is None or int(values.numel()) == 0:
            return set()
        return {int(x) for x in values.detach().cpu().tolist() if int(x) >= 0}

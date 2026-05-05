from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable


@dataclass
class TokenEvictionTracker:
    physically_freed: set[tuple[int, int]] = field(default_factory=set)

    def build_free_plan(
        self,
        *,
        ctx,
        entries,
        cpu_store,
        expected_layers: int | None,
        active_tokens: set[tuple[int, int]],
        radix_owned_device_indices: set[int],
        max_tokens: int,
    ) -> tuple[list[tuple[int, int, int]], dict]:
        token_entries: dict[tuple[int, int], int] = {}
        stats = {
            "candidates": 0,
            "active": 0,
            "radix_owned": 0,
            "already_freed": 0,
            "missing_backup": 0,
            "eligible": 0,
        }

        for entry in entries:
            stats["candidates"] += 1
            if entry.device_index is None or int(entry.device_index) < 0:
                continue
            device_index = int(entry.device_index)
            token_key = (int(entry.req_pool_idx), int(entry.position))
            if token_key in active_tokens:
                stats["active"] += 1
                continue
            if device_index in radix_owned_device_indices:
                stats["radix_owned"] += 1
                continue
            if token_key in self.physically_freed:
                stats["already_freed"] += 1
                continue
            token_entries.setdefault(token_key, device_index)

        free_plan = []
        for (req_pool_idx, position), device_index in token_entries.items():
            if not self._has_complete_cpu_backup(
                cpu_store,
                req_pool_idx=req_pool_idx,
                position=position,
                expected_layers=expected_layers,
            ):
                stats["missing_backup"] += 1
                continue
            free_plan.append((req_pool_idx, position, device_index))
            if len(free_plan) >= max(0, int(max_tokens)):
                break

        stats["eligible"] = len(free_plan)
        return free_plan, stats

    def mark_freed(self, free_plan: Iterable[tuple[int, int, int]]) -> None:
        for req_pool_idx, position, _ in free_plan:
            self.physically_freed.add((int(req_pool_idx), int(position)))

    def mark_reused(self, req_pool_idx: int, position: int) -> None:
        self.physically_freed.discard((int(req_pool_idx), int(position)))

    def drop_request(self, req_pool_idx: int) -> int:
        req_pool_idx = int(req_pool_idx)
        before = len(self.physically_freed)
        self.physically_freed = {
            key for key in self.physically_freed if int(key[0]) != req_pool_idx
        }
        return before - len(self.physically_freed)

    def _has_complete_cpu_backup(
        self,
        cpu_store,
        *,
        req_pool_idx: int,
        position: int,
        expected_layers: int | None,
    ) -> bool:
        if cpu_store is None or expected_layers is None:
            return False
        ready = getattr(cpu_store, "has_complete_ready_backup", None)
        if callable(ready):
            return bool(
                ready(
                    req_pool_idx=req_pool_idx,
                    position=position,
                    expected_layers=expected_layers,
                )
            )
        for layer_id in range(int(expected_layers)):
            layer_store = cpu_store.layers.get((int(req_pool_idx), int(layer_id)))
            if layer_store is None:
                return False
            if int(position) not in layer_store.position_to_offset:
                return False
        return True


def get_eviction_tracker(framework_state: dict) -> TokenEvictionTracker:
    tracker = framework_state.get("eviction_tracker")
    if tracker is None:
        tracker = TokenEvictionTracker()
        framework_state["eviction_tracker"] = tracker
    return tracker

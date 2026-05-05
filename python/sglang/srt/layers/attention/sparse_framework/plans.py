from __future__ import annotations

from dataclasses import dataclass, field

from sglang.srt.layers.attention.sparse_framework.selection_plan import SelectionPlan


@dataclass
class ExecutionPlan:
    selection_plan: SelectionPlan
    granularity: str = "token"
    placement: str = "gpu"
    cache_policy: str = "none"
    fetch_policy: str = "none"
    fallback_backend: str = "triton"
    working_set_budget_tokens: int | None = None
    enable_host_backup_on_evict: bool = False
    enable_physical_eviction: bool = False
    validate_kv_cache: bool = False
    ops: list[object] = field(default_factory=list)

    @property
    def uses_dense_fallback(self) -> bool:
        return self.selection_plan.is_full or any(
            op.__class__.__name__ == "FallbackOp" for op in self.ops
        )

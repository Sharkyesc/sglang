from __future__ import annotations

import logging

from sglang.srt.layers.attention.sparse_framework.config import SparseFrameworkConfig
from sglang.srt.layers.attention.sparse_framework.ops.attend import AttendOp
from sglang.srt.layers.attention.sparse_framework.ops.cache import CacheOp
from sglang.srt.layers.attention.sparse_framework.ops.evict import EvictOp
from sglang.srt.layers.attention.sparse_framework.ops.fallback import FallbackOp
from sglang.srt.layers.attention.sparse_framework.ops.fetch import FetchOp
from sglang.srt.layers.attention.sparse_framework.ops.remap import RemapOp
from sglang.srt.layers.attention.sparse_framework.ops.score import ScoreUpdateOp
from sglang.srt.layers.attention.sparse_framework.ops.select import SelectOp
from sglang.srt.layers.attention.sparse_framework.plans import ExecutionPlan
from sglang.srt.layers.attention.sparse_framework.selection_plan import SelectionPlan
from sglang.srt.layers.attention.sparse_framework.selection_spec import (
    FullSelectionSpec,
    HeavyHitterSelectionSpec,
    RetrievalSelectionSpec,
    SlidingWindowSelectionSpec,
    CustomSelectionSpec,
    parse_selection_spec,
)

logger = logging.getLogger(__name__)


class PlanCompiler:
    def __init__(self, config: SparseFrameworkConfig):
        self.config = config

    def compile(self, ctx) -> ExecutionPlan:
        specs = [parse_selection_spec(item) for item in self.config.selection]
        is_full = any(isinstance(spec, FullSelectionSpec) for spec in specs)
        requires_index = any(isinstance(spec, RetrievalSelectionSpec) for spec in specs)
        requires_scores = any(isinstance(spec, HeavyHitterSelectionSpec) for spec in specs)
        selection_plan = SelectionPlan(
            specs=specs,
            combine=self.config.combine,
            fallback=self.config.fallback,
            requires_scores=requires_scores,
            requires_index=requires_index,
            is_full=is_full,
        )

        execution_plan = ExecutionPlan(
            selection_plan=selection_plan,
            fallback_backend=self.config.dense_fallback_backend,
            working_set_budget_tokens=self.config.working_set_budget_tokens,
            enable_host_backup_on_evict=self.config.enable_host_backup_on_evict,
            enable_physical_eviction=self.config.enable_physical_eviction,
            validate_kv_cache=self.config.validate_kv_cache,
        )
        self._infer_strategy(execution_plan)

        if selection_plan.is_full:
            execution_plan.ops = [AttendOp(mode="dense")]
        elif self._selection_only_supported(selection_plan):
            if ctx.forward_batch.forward_mode.is_decode():
                execution_plan.ops = [
                    SelectOp(),
                    CacheOp(),
                    FetchOp(),
                    RemapOp(),
                    AttendOp(mode="subset_decode"),
                    ScoreUpdateOp(),
                    EvictOp(),
                    FallbackOp(reason="subset_decode_unavailable"),
                ]
            else:
                execution_plan.ops = [
                    SelectOp(),
                    CacheOp(),
                    EvictOp(),
                    FallbackOp(reason="phase3_prefill_dense_fallback"),
                ]
        else:
            if selection_plan.fallback != "dense":
                logger.warning(
                    "Sparse framework Phase 1 cannot execute selection without dense fallback; "
                    "forcing dense fallback for now."
                )
            execution_plan.ops = [FallbackOp(reason="unsupported_in_phase1")]
        return execution_plan

    def _selection_only_supported(self, plan: SelectionPlan) -> bool:
        return all(
            spec.type in ("fixed", "sink", "sliding_window")
            or isinstance(spec, SlidingWindowSelectionSpec)
            or isinstance(spec, HeavyHitterSelectionSpec)
            or isinstance(spec, RetrievalSelectionSpec)
            or (isinstance(spec, CustomSelectionSpec) and spec.type == "custom")
            for spec in plan.specs
        )

    def _infer_strategy(self, plan: ExecutionPlan) -> None:
        selection = plan.selection_plan
        if selection.is_full:
            plan.granularity = "token"
            plan.placement = "gpu"
            plan.cache_policy = "none"
            plan.fetch_policy = "none"
            return
        if selection.requires_index:
            plan.granularity = "cluster"
            plan.placement = "mixed"
            plan.cache_policy = "working_set"
            plan.fetch_policy = "async"
            return
        if selection.requires_scores:
            plan.granularity = "token"
            plan.placement = "gpu"
            plan.cache_policy = "priority"
            plan.fetch_policy = "none"
            return
        if any(isinstance(spec, SlidingWindowSelectionSpec) for spec in selection.specs):
            plan.granularity = "chunk"
            plan.placement = "gpu"
            plan.cache_policy = "recent"
            plan.fetch_policy = "prefetch"

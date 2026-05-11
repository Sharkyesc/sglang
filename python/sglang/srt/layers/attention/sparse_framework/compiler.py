from __future__ import annotations

import logging

from sglang.srt.layers.attention.sparse_framework.config import SparseFrameworkConfig
from sglang.srt.layers.attention.sparse_framework.ops.attend import AttendOp
from sglang.srt.layers.attention.sparse_framework.ops.cache import CacheOp
from sglang.srt.layers.attention.sparse_framework.ops.evict import EvictOp
from sglang.srt.layers.attention.sparse_framework.ops.fallback import FallbackOp
from sglang.srt.layers.attention.sparse_framework.ops.fetch import FetchOp
from sglang.srt.layers.attention.sparse_framework.ops.lookahead_prefetch import (
    LookaheadPrefetchOp,
)
from sglang.srt.layers.attention.sparse_framework.ops.remap import RemapOp
from sglang.srt.layers.attention.sparse_framework.ops.score import ScoreUpdateOp
from sglang.srt.layers.attention.sparse_framework.ops.select import SelectOp
from sglang.srt.layers.attention.sparse_framework.plans import ExecutionPlan
from sglang.srt.layers.attention.sparse_framework.selection_plan import SelectionPlan
from sglang.srt.layers.attention.sparse_framework.selection_spec import (
    FixedSelectionSpec,
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
            physical_eviction_interval=self.config.physical_eviction_interval,
            physical_eviction_slack_tokens=self.config.physical_eviction_slack_tokens,
            validate_kv_cache=self.config.validate_kv_cache,
            debug_timing=self.config.debug_timing,
            chunk_size=self.config.chunk_size,
        )
        self._infer_strategy(execution_plan)
        self._infer_working_set_layout(execution_plan)

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
                if self.config.enable_lookahead_prefetch:
                    execution_plan.ops.insert(-2, LookaheadPrefetchOp())
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
            plan.use_chunked_cpu_store = self._can_use_chunked_cpu_store(selection)
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
            plan.use_chunked_cpu_store = self._can_use_chunked_cpu_store(selection)

    def _can_use_chunked_cpu_store(self, selection: SelectionPlan) -> bool:
        mode = self.config.chunked_cpu_store
        if mode == "off":
            return False
        if mode == "on":
            return True
        if selection.combine.lower() not in ("union", "priority"):
            return False
        if not selection.specs:
            return False
        return all(self._spec_is_chunk_friendly(spec) for spec in selection.specs)

    def _spec_is_chunk_friendly(self, spec) -> bool:
        if isinstance(spec, SlidingWindowSelectionSpec):
            return True
        if isinstance(spec, RetrievalSelectionSpec):
            return True
        if isinstance(spec, FixedSelectionSpec):
            return spec.type in ("fixed", "sink")
        return False

    def _infer_working_set_layout(self, plan: ExecutionPlan) -> None:
        layout = self.config.working_set_layout
        if layout == "token":
            plan.use_chunked_working_set = False
        elif layout == "chunk":
            plan.use_chunked_working_set = True
        else:
            plan.use_chunked_working_set = bool(plan.use_chunked_cpu_store)

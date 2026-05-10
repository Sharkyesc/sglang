from __future__ import annotations

import torch

from sglang.srt.layers.attention.sparse_framework.ops.base import BaseSparseOp
from sglang.srt.layers.attention.sparse_framework.callback_registry import (
    SelectionCallbackRegistry,
)
from sglang.srt.layers.attention.sparse_framework.ops.heavy_hitter import (
    HeavyHitterStateManager,
)
from sglang.srt.layers.attention.sparse_framework.ops.retrieval import RetrievalSelector
from sglang.srt.layers.attention.sparse_framework.selection_result import (
    RequestSelectionView,
    SelectionResult,
)
from sglang.srt.layers.attention.sparse_framework.selection_spec import (
    CustomSelectionSpec,
    FixedSelectionSpec,
    HeavyHitterSelectionSpec,
    RetrievalSelectionSpec,
    SlidingWindowSelectionSpec,
)


class SelectOp(BaseSparseOp):
    def run(self, ctx, state: dict):
        execution_plan = state["execution_plan"]
        plan = execution_plan.selection_plan
        selected_positions = []
        selected_kv_indices = []
        selection_importance = []
        selection_contributions = []
        selected_chunk_ids = []
        req_to_token = ctx.req_to_token_pool.req_to_token
        use_chunked_cpu_store = bool(
            getattr(execution_plan, "use_chunked_cpu_store", False)
        )
        chunk_size = max(1, int(getattr(execution_plan, "chunk_size", 16)))

        for request_index, seq_len in enumerate(ctx.seq_lens_cpu):
            req_pool_idx = int(ctx.req_pool_indices_cpu[request_index])
            positions = self._positions_for_request(
                plan.specs,
                plan.combine,
                ctx,
                request_index=request_index,
                req_pool_idx=req_pool_idx,
                seq_len=seq_len,
            )
            selection_contributions.append(
                getattr(self, "_last_contribution_stats", {})
            )
            selection_importance.append(getattr(self, "_last_selection_importance", {}))
            position_tensor = torch.tensor(
                positions, dtype=torch.long, device=ctx.seq_lens.device
            )
            selected_positions.append(position_tensor)
            if use_chunked_cpu_store:
                selected_chunk_ids.append(
                    self._chunk_ids_for_positions(
                        positions, chunk_size=chunk_size, device=ctx.seq_lens.device
                    )
                )
            if position_tensor.numel() == 0:
                selected_kv_indices.append(
                    torch.empty(0, dtype=torch.long, device=req_to_token.device)
                )
            else:
                selected_kv_indices.append(
                    req_to_token[req_pool_idx, position_tensor].to(torch.long)
                )

        kv_indptr = torch.zeros(
            len(selected_kv_indices) + 1, dtype=torch.int32, device=ctx.seq_lens.device
        )
        if selected_kv_indices:
            lengths = torch.tensor(
                [int(indices.numel()) for indices in selected_kv_indices],
                dtype=torch.int32,
                device=ctx.seq_lens.device,
            )
            kv_indptr[1:] = torch.cumsum(lengths, dim=0)
            kv_indices = (
                torch.cat(selected_kv_indices)
                if int(kv_indptr[-1].item()) > 0
                else torch.empty(0, dtype=torch.long, device=req_to_token.device)
            )
        else:
            kv_indices = torch.empty(0, dtype=torch.long, device=req_to_token.device)

        state["selected_positions"] = selected_positions
        state["selected_kv_indices"] = selected_kv_indices
        state["kv_indptr"] = kv_indptr
        state["kv_indices"] = kv_indices
        state["selection_importance"] = selection_importance
        state["selection_contributions"] = selection_contributions
        if use_chunked_cpu_store:
            state["chunk_selection"] = {
                "enabled": True,
                "chunk_size": chunk_size,
                "selected_chunk_ids": selected_chunk_ids,
                "requested_chunks": sum(
                    int(chunk_ids.numel()) for chunk_ids in selected_chunk_ids
                ),
            }
        return selected_positions

    def _chunk_ids_for_positions(
        self,
        positions: list[int],
        *,
        chunk_size: int,
        device,
    ) -> torch.Tensor:
        if not positions:
            return torch.empty(0, dtype=torch.long, device=device)
        chunk_ids = sorted({int(pos) // int(chunk_size) for pos in positions})
        return torch.tensor(chunk_ids, dtype=torch.long, device=device)

    def _positions_for_request(
        self,
        specs,
        combine: str,
        ctx,
        *,
        request_index: int,
        req_pool_idx: int,
        seq_len: int,
    ) -> list[int]:
        per_spec_positions = []
        contribution_stats = {
            "fixed": 0,
            "window": 0,
            "prefix": 0,
            "suffix": 0,
            "retrieval": 0,
            "retrieval_total": 0,
            "retrieval_budget": 0,
            "retrieval_middle": 0,
            "retrieval_method": None,
            "final": 0,
        }
        importance_by_position = {}
        for spec in specs:
            if isinstance(spec, FixedSelectionSpec):
                positions = self._fixed_positions(spec, seq_len)
                per_spec_positions.append(positions)
                self._merge_importance(
                    importance_by_position,
                    {pos: 1.0 for pos in positions},
                )
                contribution_stats["fixed"] += len(set(positions))
            elif isinstance(spec, SlidingWindowSelectionSpec):
                window = max(0, int(spec.window_size))
                start = max(0, seq_len - window)
                positions = list(range(start, seq_len))
                per_spec_positions.append(positions)
                self._merge_importance(
                    importance_by_position,
                    {
                        pos: float(pos - start + 1)
                        for pos in positions
                    },
                )
                contribution_stats["window"] += len(set(positions))
            elif isinstance(spec, HeavyHitterSelectionSpec):
                positions, priority = self._heavy_hitter_positions(
                    spec,
                    ctx,
                    req_pool_idx=req_pool_idx,
                    seq_len=seq_len,
                )
                per_spec_positions.append(positions)
                self._merge_importance(importance_by_position, priority)
            elif isinstance(spec, RetrievalSelectionSpec):
                positions, stats, priority = self._retrieval_positions(
                    spec,
                    ctx,
                    request_index=request_index,
                    req_pool_idx=req_pool_idx,
                    seq_len=seq_len,
                )
                per_spec_positions.append(positions)
                self._merge_importance(importance_by_position, priority)
                contribution_stats["prefix"] += int(stats.get("prefix", 0))
                contribution_stats["suffix"] += int(stats.get("suffix", 0))
                contribution_stats["retrieval"] += int(stats.get("retrieval", 0))
                contribution_stats["retrieval_total"] += int(stats.get("total", 0))
                contribution_stats["retrieval_budget"] += int(stats.get("budget", 0))
                contribution_stats["retrieval_middle"] += int(stats.get("middle", 0))
                contribution_stats["retrieval_method"] = stats.get("method")
            elif isinstance(spec, CustomSelectionSpec):
                positions, priority = self._custom_positions(
                    spec,
                    ctx,
                    request_index=request_index,
                    req_pool_idx=req_pool_idx,
                    seq_len=seq_len,
                )
                per_spec_positions.append(positions)
                self._merge_importance(importance_by_position, priority)
        positions = self._combine_positions(per_spec_positions, combine, seq_len)
        contribution_stats["final"] = len(positions)
        self._last_contribution_stats = contribution_stats
        self._last_selection_importance = {
            int(pos): float(importance_by_position.get(int(pos), 0.0))
            for pos in positions
        }
        return positions

    def _merge_importance(
        self,
        target: dict[int, float],
        source: dict[int, float],
    ) -> None:
        for pos, priority in source.items():
            pos = int(pos)
            target[pos] = max(float(priority), target.get(pos, 0.0))

    def _combine_positions(
        self,
        per_spec_positions: list[list[int]],
        combine: str,
        seq_len: int,
    ) -> list[int]:
        cleaned = [
            [pos for pos in positions if 0 <= pos < seq_len]
            for positions in per_spec_positions
        ]
        if not cleaned:
            return []
        combine = combine.lower()
        if combine == "intersection":
            selected = set(cleaned[0])
            for positions in cleaned[1:]:
                selected.intersection_update(positions)
            return sorted(selected)
        if combine == "priority":
            selected = []
            seen = set()
            for positions in cleaned:
                for pos in positions:
                    if pos not in seen:
                        selected.append(pos)
                        seen.add(pos)
            return selected
        selected = set()
        for positions in cleaned:
            selected.update(positions)
        return sorted(selected)

    def _fixed_positions(self, spec: FixedSelectionSpec, seq_len: int) -> list[int]:
        positions = []
        for pos in spec.positions:
            norm = seq_len + pos if pos < 0 else pos
            positions.append(norm)
        for start, end in spec.ranges:
            norm_start = seq_len + start if start < 0 else start
            # Positive ranges use Python-style half-open [start, end). For a
            # negative end, treat -1 as the final token so [-8, -1] means the
            # last 8 tokens, matching user-facing KV selection config.
            norm_end = seq_len + end + 1 if end < 0 else end
            if norm_end < norm_start:
                norm_start, norm_end = norm_end, norm_start
            positions.extend(range(norm_start, norm_end))
        return positions

    def _custom_positions(
        self,
        spec: CustomSelectionSpec,
        ctx,
        *,
        request_index: int,
        req_pool_idx: int,
        seq_len: int,
    ) -> tuple[list[int], dict[int, float]]:
        callback = SelectionCallbackRegistry.resolve(spec)
        layer_id = getattr(ctx.layer, "layer_id", None)
        request_view = RequestSelectionView(
            request_index=request_index,
            req_pool_idx=req_pool_idx,
            seq_len=seq_len,
            layer_id=layer_id,
        )
        try:
            value = callback(
                ctx=ctx,
                request_view=request_view,
                request_index=request_index,
                req_pool_idx=req_pool_idx,
                seq_len=seq_len,
                layer=ctx.layer,
                layer_id=layer_id,
                **spec.kwargs,
            )
        except TypeError:
            value = callback(ctx, request_view, layer_id)
        result = SelectionResult.from_callback_output(value)
        positions = result.to_positions(seq_len)
        priority = {}
        if result.priority:
            priority = {
                int(pos): float(value)
                for pos, value in result.priority.items()
                if 0 <= int(pos) < seq_len
            }
        return positions, priority

    def _heavy_hitter_positions(
        self,
        spec: HeavyHitterSelectionSpec,
        ctx,
        *,
        req_pool_idx: int,
        seq_len: int,
    ) -> tuple[list[int], dict[int, float]]:
        layer_id = getattr(ctx.layer, "layer_id", None)
        framework_state = ctx.framework_state
        if layer_id is None or framework_state is None:
            return [], {}
        manager = framework_state.get("heavy_hitter_manager")
        if manager is None:
            manager = HeavyHitterStateManager()
            framework_state["heavy_hitter_manager"] = manager
        manager.drop_missing_requests(ctx.req_pool_indices_cpu)
        positions = manager.select_positions(
            spec,
            req_pool_idx=req_pool_idx,
            layer_id=layer_id,
            seq_len=seq_len,
            device=ctx.seq_lens.device,
        )
        positions_list = positions.detach().cpu().tolist()
        priority = manager.position_scores(
            req_pool_idx=req_pool_idx,
            layer_id=layer_id,
            positions=positions_list,
        )
        return positions_list, priority

    def _retrieval_positions(
        self,
        spec: RetrievalSelectionSpec,
        ctx,
        *,
        request_index: int,
        req_pool_idx: int,
        seq_len: int,
    ) -> tuple[list[int], dict, dict[int, float]]:
        framework_state = ctx.framework_state
        if framework_state is None:
            selector = RetrievalSelector()
        else:
            selector = framework_state.get("retrieval_selector")
            if selector is None:
                selector = RetrievalSelector()
                framework_state["retrieval_selector"] = selector
        positions = selector.select_positions(
            spec,
            ctx,
            request_index=request_index,
            req_pool_idx=req_pool_idx,
            seq_len=seq_len,
        )
        return (
            positions,
            dict(getattr(selector, "last_stats", {})),
            dict(getattr(selector, "last_priority", {})),
        )

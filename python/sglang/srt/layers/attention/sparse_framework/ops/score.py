from __future__ import annotations

from sglang.srt.layers.attention.sparse_framework.ops.base import BaseSparseOp
from sglang.srt.layers.attention.sparse_framework.ops.heavy_hitter import (
    HeavyHitterStateManager,
)


class ScoreUpdateOp(BaseSparseOp):
    def run(self, ctx, state: dict):
        if state.get("attention_output") is None:
            return None
        manager = self._manager(ctx)
        if manager is None:
            return None
        attention_weights = state.get("attention_weights")
        selected_positions = state.get("selected_positions")
        if not attention_weights or not selected_positions:
            return None
        layer_id = getattr(ctx.layer, "layer_id", None)
        if layer_id is None:
            return None
        req_pool_indices = [int(x) for x in ctx.req_pool_indices.tolist()]
        for batch_idx, req_pool_idx in enumerate(req_pool_indices):
            manager.update_scores(
                req_pool_idx=req_pool_idx,
                layer_id=layer_id,
                selected_positions=selected_positions[batch_idx],
                attn_weights=attention_weights[batch_idx],
            )
        state["score_update"] = "heavy_hitter"
        return None

    def _manager(self, ctx) -> HeavyHitterStateManager | None:
        framework_state = ctx.framework_state
        if framework_state is None:
            return None
        manager = framework_state.get("heavy_hitter_manager")
        if manager is None:
            manager = HeavyHitterStateManager()
            framework_state["heavy_hitter_manager"] = manager
        return manager

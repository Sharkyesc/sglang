from __future__ import annotations

from dataclasses import dataclass

from sglang.srt.layers.attention.sparse_framework.ops.base import BaseSparseOp


@dataclass
class FallbackOp(BaseSparseOp):
    reason: str = "dense"

    def run(self, ctx, state: dict):
        if state.get("attention_output") is not None:
            return None
        state["fallback_reason"] = self.reason
        return None

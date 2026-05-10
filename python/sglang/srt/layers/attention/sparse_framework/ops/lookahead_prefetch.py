from __future__ import annotations

import torch

from sglang.srt.layers.attention.sparse_framework.kv_store import get_cpu_kv_store
from sglang.srt.layers.attention.sparse_framework.ops.base import BaseSparseOp
from sglang.srt.layers.attention.sparse_framework.ops.select import SelectOp
from sglang.srt.layers.attention.sparse_framework.ops.utils import (
    configure_cpu_kv_store_from_state,
    cpu_kv_store_enabled,
)


class LookaheadPrefetchOp(BaseSparseOp):
    """Schedule CPU-backed working-set H2D for the next decode step.

    The backend cannot see the next scheduler batch while executing the current
    one, so this op performs a conservative per-request lookahead. It predicts
    the next-step selection from current metadata, prefetches only rows that are
    already present and ready in the sparse CPU store, and stores them by
    position so the next step can partially consume them.
    """

    def __init__(self):
        self.selector = SelectOp()

    def run(self, ctx, state: dict):
        if (
            ctx.framework_state is None
            or ctx.layer is None
            or ctx.forward_batch is None
            or not ctx.forward_batch.forward_mode.is_decode()
        ):
            state["lookahead_prefetch_result"] = {
                "enabled": False,
                "reason": "missing_decode_context",
            }
            return None
        if state.get("attention_output") is None:
            state["lookahead_prefetch_result"] = {
                "enabled": False,
                "reason": "subset_attention_not_ready",
            }
            return None

        if not cpu_kv_store_enabled(state):
            state["lookahead_prefetch_result"] = {
                "enabled": False,
                "reason": "cpu_kv_store_disabled",
            }
            return None

        store = get_cpu_kv_store(ctx.framework_state)
        if store is None:
            state["lookahead_prefetch_result"] = {
                "enabled": False,
                "reason": "missing_sparse_cpu_store",
            }
            return None
        configure_cpu_kv_store_from_state(store, state)

        plan = state.get("execution_plan")
        selection_plan = getattr(plan, "selection_plan", None)
        if selection_plan is None or getattr(selection_plan, "is_full", False):
            state["lookahead_prefetch_result"] = {
                "enabled": False,
                "reason": "unsupported_selection_plan",
            }
            return None

        key_buffer = ctx.token_to_kv_pool.get_key_buffer(ctx.layer.layer_id)
        device = key_buffer.device
        dtype = key_buffer.dtype
        if device.type != "cuda" or not torch.cuda.is_available():
            state["lookahead_prefetch_result"] = {
                "enabled": False,
                "reason": "non_cuda_device",
            }
            return None

        pending = ctx.framework_state.setdefault("sparse_cpu_prefetches", {})
        layer_id = int(ctx.layer.layer_id)
        requested = 0
        ready = 0
        scheduled = 0
        already_pending = 0
        missing_or_pending_backup = 0

        for request_index, seq_len in enumerate(ctx.seq_lens_cpu):
            if request_index >= len(ctx.req_pool_indices_cpu):
                continue
            req_pool_idx = int(ctx.req_pool_indices_cpu[request_index])
            next_seq_len = int(seq_len) + 1
            positions = self.selector._positions_for_request(
                selection_plan.specs,
                selection_plan.combine,
                ctx,
                request_index=request_index,
                req_pool_idx=req_pool_idx,
                seq_len=next_seq_len,
            )
            positions = [int(pos) for pos in positions if int(pos) >= 0]
            if not positions:
                continue
            req_to_token = ctx.req_to_token_pool.req_to_token
            max_len = int(req_to_token.shape[1])
            positions = [
                pos
                for pos in positions
                if pos < max_len and int(req_to_token[req_pool_idx, pos].item()) < 0
            ]
            if not positions:
                continue
            requested += len(positions)

            candidate_positions = [
                pos
                for pos in positions
                if (req_pool_idx, layer_id, pos) not in pending
            ]
            already_pending += len(positions) - len(candidate_positions)
            if not candidate_positions:
                continue
            layer_store = store.layers.get((req_pool_idx, layer_id))
            if layer_store is None or not bool(getattr(layer_store, "pinned", False)):
                missing_or_pending_backup += len(candidate_positions)
                continue

            ready_positions = store.ready_positions(
                req_pool_idx=req_pool_idx,
                layer_id=layer_id,
                positions=candidate_positions,
            )
            ready += len(ready_positions)
            missing_or_pending_backup += len(candidate_positions) - len(ready_positions)
            if not ready_positions:
                continue

            get_many_async = getattr(store, "get_many_async", None)
            if callable(get_many_async):
                keys, values, found_positions, _, h2d_event = get_many_async(
                    req_pool_idx=req_pool_idx,
                    layer_id=layer_id,
                    positions=ready_positions,
                    device=device,
                    dtype=dtype,
                )
            else:
                keys, values, found_positions, _ = store.get_many(
                    req_pool_idx=req_pool_idx,
                    layer_id=layer_id,
                    positions=ready_positions,
                    device=device,
                    dtype=dtype,
                )
                h2d_event = None
            if keys is None or values is None:
                continue

            for offset, pos in enumerate(found_positions):
                if offset >= int(keys.shape[0]) or offset >= int(values.shape[0]):
                    break
                pending[(req_pool_idx, layer_id, int(pos))] = {
                    "keys": keys[offset : offset + 1],
                    "values": values[offset : offset + 1],
                    "positions": [int(pos)],
                    "event": h2d_event,
                    "source": "lookahead",
                }
                scheduled += 1

        state["lookahead_prefetch_result"] = {
            "enabled": True,
            "requested": requested,
            "ready": ready,
            "scheduled": scheduled,
            "already_pending": already_pending,
            "missing_or_pending_backup": missing_or_pending_backup,
            "granularity": "next_step_position",
        }
        return None

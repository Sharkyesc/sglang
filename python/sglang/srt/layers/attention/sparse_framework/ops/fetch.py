from __future__ import annotations

import torch

from sglang.srt.layers.attention.sparse_framework.ops.base import BaseSparseOp
from sglang.srt.layers.attention.sparse_framework.ops.utils import (
    rewrite_selected_kv_indices_from_entries,
)
from sglang.srt.layers.attention.sparse_framework.residency import (
    get_residency_table,
)


class FetchOp(BaseSparseOp):
    def run(self, ctx, state: dict):
        misses = state.get("cache_misses") or []
        host_misses = [entry for entry in misses if entry.host_index is not None]
        if not host_misses:
            state["fetch_result"] = {
                "requested": 0,
                "fetched": 0,
                "reason": "no_host_resident_misses",
            }
            return None
        if ctx.cache_controller is None:
            state["fetch_result"] = {
                "requested": len(host_misses),
                "fetched": 0,
                "reason": "missing_cache_controller",
            }
            return None

        host_indices = _host_indices_tensor(
            [entry.host_index for entry in host_misses],
            ctx,
        )
        device_indices = ctx.cache_controller.load(host_indices)
        if device_indices is None:
            state["fetch_result"] = {
                "requested": len(host_misses),
                "fetched": 0,
                "reason": "device_allocation_failed",
            }
            return None

        producer_id = ctx.cache_controller.start_loading()
        layer_id = getattr(ctx.layer, "layer_id", None)
        layer_done_counter = getattr(ctx.cache_controller, "layer_done_counter", None)
        if producer_id is not None and producer_id >= 0 and layer_id is not None:
            if layer_done_counter is not None:
                layer_done_counter.set_consumer(producer_id)
                layer_done_counter.wait_until(layer_id)
        _consume_sparse_load_ack(ctx.cache_controller)
        device_list = device_indices.detach().cpu().tolist()
        table = get_residency_table(ctx.framework_state) if ctx.framework_state is not None else None
        for entry, device_index in zip(host_misses, device_list):
            if table is not None:
                table.mark_gpu(entry, device_index=int(device_index))
            else:
                entry.device_index = int(device_index)
                entry.logical_evicted = False

        updated = rewrite_selected_kv_indices_from_entries(state, host_misses)
        state["fetch_result"] = {
            "requested": len(host_misses),
            "fetched": len(host_misses),
            "producer_id": producer_id,
            "remapped": updated,
        }
        return None


def _host_indices_tensor(indices: list[int], ctx) -> torch.Tensor:
    io_backend = getattr(ctx.cache_controller, "io_backend", None)
    device = "cpu" if io_backend in ("direct", "kernel_ascend") else ctx.seq_lens.device
    return torch.tensor(indices, dtype=torch.long, device=device)


def _consume_sparse_load_ack(cache_controller) -> None:
    ack_queue = getattr(cache_controller, "ack_load_queue", None)
    if not ack_queue:
        return

    cleaned_queue = []
    for ack in ack_queue:
        node_ids = list(getattr(ack, "node_ids", []))
        if -1 not in node_ids:
            cleaned_queue.append(ack)
            continue

        finish_event = ack.finish_event
        synchronize = getattr(finish_event, "synchronize", None)
        if callable(synchronize):
            synchronize()

        remaining_node_ids = [node_id for node_id in node_ids if node_id != -1]
        if remaining_node_ids:
            cleaned_queue.append(ack._replace(node_ids=remaining_node_ids))

    ack_queue[:] = cleaned_queue

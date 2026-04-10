from __future__ import annotations

from sglang.srt.layers.attention.retroinfer.types import RetroInferRequestCpuState


class RetroInferCpuStore:
    """
    CPU-side bookkeeping for RetroInfer metadata derived from SGLang KV.

    First implementation keeps only lightweight metadata. The source-of-truth KV
    remains in SGLang's pool; deeper CPU-side reordered storage can be layered on top
    later without changing backend/session/planner structure.
    """

    def __init__(self):
        self.request_meta: dict[int, dict[str, int | bool]] = {}
        self.request_states: dict[int, RetroInferRequestCpuState] = {}

    def store_request_state(self, request_state: RetroInferRequestCpuState) -> None:
        self.request_states[request_state.req_pool_idx] = request_state
        self.mark_index_ready(request_state.req_pool_idx, request_state.indexed_upto)

    def mark_index_ready(self, req_pool_idx: int, indexed_upto: int) -> None:
        entry = self.request_meta.setdefault(req_pool_idx, {})
        entry["indexed_upto"] = indexed_upto
        entry["cpu_index_ready"] = True

    def get_indexed_upto(self, req_pool_idx: int) -> int:
        return int(self.request_meta.get(req_pool_idx, {}).get("indexed_upto", 0))

    def get_request_state(self, req_pool_idx: int) -> RetroInferRequestCpuState | None:
        return self.request_states.get(req_pool_idx)

    def get_layer_index(self, req_pool_idx: int, layer_id: int):
        request_state = self.get_request_state(req_pool_idx)
        if request_state is None:
            return None
        return request_state.layers.get(layer_id)

    def is_request_ready(self, req_pool_idx: int, indexed_upto: int) -> bool:
        request_state = self.get_request_state(req_pool_idx)
        if request_state is None:
            return False
        return request_state.indexed_upto >= indexed_upto

    def drop_request(self, req_pool_idx: int) -> None:
        self.request_meta.pop(req_pool_idx, None)
        self.request_states.pop(req_pool_idx, None)

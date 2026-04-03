from __future__ import annotations


class RetroInferCpuStore:
    """
    CPU-side bookkeeping for RetroInfer metadata derived from SGLang KV.

    First implementation keeps only lightweight metadata. The source-of-truth KV
    remains in SGLang's pool; deeper CPU-side reordered storage can be layered on top
    later without changing backend/session/planner structure.
    """

    def __init__(self):
        self.request_meta: dict[int, dict[str, int | bool]] = {}

    def mark_index_ready(self, req_pool_idx: int, indexed_upto: int) -> None:
        entry = self.request_meta.setdefault(req_pool_idx, {})
        entry["indexed_upto"] = indexed_upto
        entry["cpu_index_ready"] = True

    def get_indexed_upto(self, req_pool_idx: int) -> int:
        return int(self.request_meta.get(req_pool_idx, {}).get("indexed_upto", 0))

    def drop_request(self, req_pool_idx: int) -> None:
        self.request_meta.pop(req_pool_idx, None)

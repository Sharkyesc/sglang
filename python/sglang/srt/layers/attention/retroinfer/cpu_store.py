from __future__ import annotations

from sglang.srt.layers.attention.retroinfer.types import (
    RetroInferRequestCpuState,
    RetroInferRequestHostKVState,
)


class RetroInferCpuStore:
    """
    CPU-side bookkeeping for RetroInfer metadata derived from SGLang KV.

    The initial integration only stored lightweight metadata. This version keeps
    that behavior but also tracks host-resident KV placement metadata so later
    layers can switch RetroInfer to CPU-first KV ownership without changing the
    surrounding backend/session APIs.
    """

    def __init__(self, kv_store=None):
        self.kv_store = kv_store
        self.request_meta: dict[int, dict[str, int | bool | str]] = {}
        self.request_states: dict[int, RetroInferRequestCpuState] = {}

    def store_request_state(self, request_state: RetroInferRequestCpuState) -> None:
        self.request_states[request_state.req_pool_idx] = request_state
        self.mark_index_ready(request_state.req_pool_idx, request_state.indexed_upto)
        if request_state.host_kv is not None:
            self.attach_request_host_state(request_state.req_pool_idx, request_state.host_kv)

    def mark_index_ready(self, req_pool_idx: int, indexed_upto: int) -> None:
        entry = self.request_meta.setdefault(req_pool_idx, {})
        entry["indexed_upto"] = indexed_upto
        entry["index_built_upto"] = indexed_upto

    def get_last_index_source(self, req_pool_idx: int) -> str:
        return str(self.request_meta.get(req_pool_idx, {}).get("last_index_source", "unknown"))

    def mark_last_index_source(self, req_pool_idx: int, source: str) -> None:
        entry = self.request_meta.setdefault(req_pool_idx, {})
        entry["last_index_source"] = source

    def get_indexed_upto(self, req_pool_idx: int) -> int:
        return int(self.request_meta.get(req_pool_idx, {}).get("indexed_upto", 0))

    def get_request_readiness(
        self,
        req_pool_idx: int,
        upto_len: int,
    ) -> dict[str, int | bool | str]:
        host_stored_upto = self.get_host_stored_upto(req_pool_idx)
        indexed_upto = self.get_indexed_upto(req_pool_idx)
        last_kv_source = self.get_last_kv_source(req_pool_idx)
        last_index_source = self.get_last_index_source(req_pool_idx)
        return {
            "host_stored_upto": host_stored_upto,
            "indexed_upto": indexed_upto,
            "host_ready": host_stored_upto >= upto_len,
            "index_ready": indexed_upto >= upto_len,
            "last_kv_source": last_kv_source,
            "last_index_source": last_index_source,
            "gpu_fallback": (
                last_kv_source == "gpu_fallback"
                or last_index_source == "gpu_fallback"
            ),
        }

    def get_request_state(self, req_pool_idx: int) -> RetroInferRequestCpuState | None:
        return self.request_states.get(req_pool_idx)

    def get_request_host_state(
        self,
        req_pool_idx: int,
    ) -> RetroInferRequestHostKVState | None:
        request_state = self.get_request_state(req_pool_idx)
        if request_state is None:
            return None
        return request_state.host_kv

    def get_layer_index(self, req_pool_idx: int, layer_id: int):
        request_state = self.get_request_state(req_pool_idx)
        if request_state is None:
            return None
        return request_state.layers.get(layer_id)

    def get_layer_host_state(self, req_pool_idx: int, layer_id: int):
        request_state = self.get_request_state(req_pool_idx)
        if request_state is None or request_state.host_kv is None:
            return None
        return request_state.host_kv.layers.get(layer_id)

    def is_request_ready(self, req_pool_idx: int, indexed_upto: int) -> bool:
        request_state = self.get_request_state(req_pool_idx)
        if request_state is None:
            return False
        return request_state.indexed_upto >= indexed_upto

    def attach_request_host_state(
        self,
        req_pool_idx: int,
        host_state: RetroInferRequestHostKVState,
    ) -> None:
        request_state = self.get_request_state(req_pool_idx)
        if request_state is None:
            return
        request_state.host_kv = host_state
        entry = self.request_meta.setdefault(req_pool_idx, {})
        entry["host_stored_upto"] = host_state.stored_upto
        entry["host_kv_resident"] = host_state.resident
        entry["host_staged_upto"] = host_state.stored_upto

    def get_host_stored_upto(self, req_pool_idx: int) -> int:
        return int(self.request_meta.get(req_pool_idx, {}).get("host_stored_upto", 0))

    def get_last_kv_source(self, req_pool_idx: int) -> str:
        return str(self.request_meta.get(req_pool_idx, {}).get("last_kv_source", "unknown"))

    def mark_last_kv_source(self, req_pool_idx: int, source: str) -> None:
        entry = self.request_meta.setdefault(req_pool_idx, {})
        entry["last_kv_source"] = source

    def is_host_resident(self, req_pool_idx: int, stored_upto: int) -> bool:
        request_state = self.get_request_state(req_pool_idx)
        if request_state is None or request_state.host_kv is None:
            return False
        host_state = request_state.host_kv
        return host_state.resident and host_state.stored_upto >= stored_upto

    def ensure_host_resident(
        self,
        req_pool_idx: int,
        upto_len: int,
    ) -> bool:
        if self.is_host_resident(req_pool_idx, upto_len):
            self.mark_last_kv_source(req_pool_idx, "host")
            return True
        if self.kv_store is None or not self.kv_store.is_bound():
            return False
        host_state = self.stage_request_to_host(req_pool_idx, upto_len)
        if host_state is None or not host_state.resident or host_state.stored_upto < upto_len:
            return False
        self.mark_last_kv_source(req_pool_idx, "host")
        return True

    def stage_request_to_host(
        self,
        req_pool_idx: int,
        upto_len: int,
    ) -> RetroInferRequestHostKVState | None:
        if self.kv_store is None:
            return None
        host_state = self.kv_store.stage_request_from_device(req_pool_idx, upto_len)
        self.attach_request_host_state(req_pool_idx, host_state)
        if host_state is not None and host_state.resident:
            self.mark_last_kv_source(req_pool_idx, "host")
        return host_state

    def get_request_layer_tensors_from_host(
        self,
        req_pool_idx: int,
        layer_id: int,
        upto_len: int,
    ):
        if self.kv_store is None:
            return None
        tensors = self.kv_store.get_request_layer_tensors(req_pool_idx, layer_id, upto_len)
        if tensors is not None:
            self.mark_last_kv_source(req_pool_idx, "host")
        return tensors

    def get_request_layer_tensors_by_positions_from_host(
        self,
        req_pool_idx: int,
        layer_id: int,
        positions,
    ):
        if self.kv_store is None:
            return None
        tensors = self.kv_store.get_request_layer_tensors_by_positions(
            req_pool_idx=req_pool_idx,
            layer_id=layer_id,
            positions=positions,
        )
        if tensors is not None:
            self.mark_last_kv_source(req_pool_idx, "host")
        return tensors

    def gather_batch_layer_tensors_from_host(
        self,
        req_pool_indices: list[int],
        layer_id: int,
        upto_len: int,
    ):
        if self.kv_store is None:
            return None
        tensors = self.kv_store.gather_batch_layer_tensors(
            req_pool_indices=req_pool_indices,
            layer_id=layer_id,
            upto_len=upto_len,
        )
        if tensors is not None:
            for req_pool_idx in req_pool_indices:
                self.mark_last_kv_source(int(req_pool_idx), "host")
        return tensors

    def drop_request(self, req_pool_idx: int) -> None:
        if self.kv_store is not None:
            self.kv_store.drop_request(req_pool_idx)
        self.request_meta.pop(req_pool_idx, None)
        self.request_states.pop(req_pool_idx, None)

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional


@dataclass
class RetroInferDecision:
    allow: bool
    mode: str
    reason: str = ""


@dataclass
class RetroInferRequestState:
    req_pool_idx: int
    last_seq_len: int = 0
    indexed_upto: int = 0
    cpu_index_ready: bool = False
    gpu_meta_ready: bool = False
    needs_rebuild: bool = True
    last_access_step: int = 0


@dataclass
class RetroInferLayerCpuIndex:
    layer_id: int
    indexed_upto: int
    static_prefix_len: int
    static_suffix_len: int
    dense_only: bool
    key_dtype: str
    centroids: Optional[object] = None
    value_sum: Optional[object] = None
    cluster_size: Optional[object] = None
    cluster_mask: Optional[object] = None
    cluster_to_token_indices: list[object] = field(default_factory=list)


@dataclass
class RetroInferRequestCpuState:
    req_pool_idx: int
    seq_len: int
    indexed_upto: int
    dense_only: bool
    layers: dict[int, RetroInferLayerCpuIndex] = field(default_factory=dict)


@dataclass
class RetroInferSession:
    key: tuple[int, ...]
    batch_size: int
    request_states: dict[int, RetroInferRequestState]
    retro_cache: Optional[object] = None
    prepared: bool = False
    prepared_seq_len: int = 0
    decode_steps: int = 0
    warned_fallback: bool = False
    cpu_index_ready: bool = False
    gpu_meta_ready: bool = False

    def mark_prepared(self, seq_len: int) -> None:
        self.prepared = True
        self.prepared_seq_len = seq_len
        self.warned_fallback = False
        self.cpu_index_ready = True
        self.gpu_meta_ready = True
        for state in self.request_states.values():
            state.cpu_index_ready = True
            state.gpu_meta_ready = True
            state.indexed_upto = max(state.indexed_upto, seq_len)
            state.needs_rebuild = False


@dataclass
class RetroInferPlan:
    mode: str
    reason: str = ""
    session: Optional[RetroInferSession] = None
    req_pool_indices: tuple[int, ...] = field(default_factory=tuple)
    seq_len: int = 0

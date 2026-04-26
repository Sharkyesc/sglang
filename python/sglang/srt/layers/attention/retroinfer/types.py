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
    host_staged_upto: int = 0
    buffer_prepared_upto: int = 0
    cache_synced_upto: int = 0
    needs_rebuild: bool = True
    last_access_step: int = 0

    def reset_progress(self) -> None:
        self.indexed_upto = 0
        self.host_staged_upto = 0
        self.buffer_prepared_upto = 0
        self.cache_synced_upto = 0
        self.needs_rebuild = True


@dataclass
class RetroInferLayerCpuIndex:
    layer_id: int
    indexed_upto: int
    index_source: str
    static_prefix_len: int
    static_suffix_len: int
    middle_start: int
    middle_end: int
    dense_only: bool
    key_dtype: str
    static_positions: Optional[object] = None
    middle_positions: Optional[object] = None
    segment_ranges: tuple[tuple[int, int], ...] = field(default_factory=tuple)
    centroids: Optional[object] = None
    value_sum: Optional[object] = None
    cluster_value_norms: Optional[object] = None
    cluster_size: Optional[object] = None
    cluster_mask: Optional[object] = None
    cluster_representatives: Optional[object] = None
    token_cluster_ids: Optional[object] = None
    cluster_to_token_indices: list[object] = field(default_factory=list)


@dataclass
class RetroInferWorkingSetPlan:
    target_len: int
    sparse_budget: int
    retrieval_budget: int
    estimation_budget: int
    base_positions: object
    retrieval_positions: object
    estimation_positions: object
    retrieval_cluster_indices: object | None = None
    estimation_cluster_indices: object | None = None


@dataclass
class RetroInferLayerHostKVState:
    layer_id: int
    stored_upto: int
    page_size: int
    host_indices: tuple[int, ...] = field(default_factory=tuple)
    host_page_indices: tuple[int, ...] = field(default_factory=tuple)
    page_token_counts: tuple[int, ...] = field(default_factory=tuple)
    source_token_indices: tuple[int, ...] = field(default_factory=tuple)
    resident: bool = False
    io_backend: str = ""


@dataclass
class RetroInferRequestHostKVState:
    req_pool_idx: int
    stored_upto: int
    page_size: int
    host_indices: tuple[int, ...] = field(default_factory=tuple)
    host_page_indices: tuple[int, ...] = field(default_factory=tuple)
    page_token_counts: tuple[int, ...] = field(default_factory=tuple)
    source_token_indices: tuple[int, ...] = field(default_factory=tuple)
    resident: bool = False
    io_backend: str = ""
    layers: dict[int, RetroInferLayerHostKVState] = field(default_factory=dict)


@dataclass
class RetroInferRequestCpuState:
    req_pool_idx: int
    seq_len: int
    indexed_upto: int
    dense_only: bool
    layers: dict[int, RetroInferLayerCpuIndex] = field(default_factory=dict)
    host_kv: Optional[RetroInferRequestHostKVState] = None


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
    host_staged_upto: int = 0
    index_built_upto: int = 0
    buffer_prepared_upto: int = 0
    cache_synced_upto: int = 0
    prepared_working_set_len: int = 0

    def mark_prepared(self, seq_len: int, working_set_len: int) -> None:
        self.prepared = True
        self.prepared_seq_len = seq_len
        self.prepared_working_set_len = working_set_len
        self.warned_fallback = False
        self.host_staged_upto = max(self.host_staged_upto, seq_len)
        self.index_built_upto = max(self.index_built_upto, seq_len)
        self.buffer_prepared_upto = max(self.buffer_prepared_upto, seq_len)
        self.cache_synced_upto = max(self.cache_synced_upto, seq_len)
        for state in self.request_states.values():
            state.indexed_upto = max(state.indexed_upto, seq_len)
            state.host_staged_upto = max(state.host_staged_upto, seq_len)
            state.buffer_prepared_upto = max(state.buffer_prepared_upto, seq_len)
            state.cache_synced_upto = max(state.cache_synced_upto, seq_len)
            state.needs_rebuild = False

    def reset_runtime(self) -> None:
        self.retro_cache = None
        self.prepared = False
        self.prepared_seq_len = 0
        self.warned_fallback = False
        self.host_staged_upto = 0
        self.index_built_upto = 0
        self.buffer_prepared_upto = 0
        self.cache_synced_upto = 0
        self.prepared_working_set_len = 0


@dataclass
class RetroInferPlan:
    mode: str
    reason: str = ""
    session: Optional[RetroInferSession] = None
    req_pool_indices: tuple[int, ...] = field(default_factory=tuple)
    seq_len: int = 0

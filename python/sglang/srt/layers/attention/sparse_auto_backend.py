from __future__ import annotations

import json
import logging
import math
import time
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Optional

import torch

from sglang.srt.layers.attention.base_attn_backend import AttentionBackend

if TYPE_CHECKING:
    from sglang.srt.layers.radix_attention import RadixAttention
    from sglang.srt.model_executor.forward_batch_info import ForwardBatch, ForwardMode
    from sglang.srt.model_executor.model_runner import ModelRunner
    from sglang.srt.speculative.spec_info import SpecInput

logger = logging.getLogger(__name__)


@dataclass
class SparseAutoConfig:
    dense_fallback_backend: str = "triton"
    force_backend: Optional[str] = None
    selection_policy: str = "smart"
    latency_budget_ms: Optional[float] = None
    memory_budget_mb: Optional[float] = None
    target_sparsity: Optional[float] = None
    enable_online_profiling: bool = True
    profiling_warmup_steps: int = 8
    profiling_ewma_alpha: float = 0.2
    exploration_interval: int = 64
    selection_hysteresis: float = 0.08
    profiling_sync_cuda: bool = False
    lock_retroinfer_after_select: bool = True
    retroinfer_min_decode_seq_len: int = 4096
    retroinfer_max_batch_size: int = 8
    nsa_min_seq_len: int = 2048
    h2o_min_seq_len: int = 2048
    enable_h2o: bool = False

    @classmethod
    def from_server_args(cls, server_args: Any) -> "SparseAutoConfig":
        raw = getattr(server_args, "sparse_attention_config", "{}") or "{}"
        try:
            data = json.loads(raw)
        except json.JSONDecodeError as exc:
            logger.warning(
                "Invalid sparse attention config JSON, falling back to defaults (%s)",
                exc,
            )
            data = {}
        return cls(**{k: v for k, v in data.items() if k in cls.__annotations__})


@dataclass(frozen=True)
class SparseRuntimeProfile:
    forward_mode: str
    batch_size: int
    max_seq_len: int
    avg_seq_len: float
    num_tokens: int
    is_mla: bool
    speculative: bool


@dataclass(frozen=True)
class SparseSelection:
    backend_name: str
    reason: str
    score: float


@dataclass(frozen=True)
class SparseBackendEstimate:
    latency_ms: float
    memory_mb: float
    sparsity: float
    quality: float
    source: str


@dataclass
class SparseBackendStats:
    count: int = 0
    latency_ms_ewma: Optional[float] = None

    def observe(self, latency_ms: float, alpha: float):
        self.count += 1
        if self.latency_ms_ewma is None:
            self.latency_ms_ewma = latency_ms
        else:
            self.latency_ms_ewma = (
                alpha * latency_ms + (1.0 - alpha) * self.latency_ms_ewma
            )


class SparseAutoProfiler:
    def __init__(self, config: SparseAutoConfig):
        self.config = config
        self._stats: dict[tuple[str, str, int], SparseBackendStats] = {}
        self._pending_cuda_events: list[
            tuple[str, str, int, torch.cuda.Event, torch.cuda.Event]
        ] = []

    def bucket_for(self, profile: SparseRuntimeProfile) -> tuple[str, int]:
        if profile.max_seq_len <= 0:
            seq_bucket = 0
        else:
            seq_bucket = 1 << max(0, int(math.ceil(math.log2(profile.max_seq_len))))
        return profile.forward_mode, seq_bucket

    def stats_for(
        self, backend_name: str, profile: SparseRuntimeProfile
    ) -> SparseBackendStats:
        mode, seq_bucket = self.bucket_for(profile)
        return self._stats.setdefault(
            (backend_name, mode, seq_bucket), SparseBackendStats()
        )

    def latency_ms_for(
        self, backend_name: str, profile: SparseRuntimeProfile
    ) -> Optional[float]:
        stats = self.stats_for(backend_name, profile)
        if stats.count < max(0, self.config.profiling_warmup_steps):
            return None
        return stats.latency_ms_ewma

    def observe(
        self,
        backend_name: str,
        profile: SparseRuntimeProfile,
        latency_ms: float,
    ):
        self.stats_for(backend_name, profile).observe(
            latency_ms, max(0.0, min(1.0, self.config.profiling_ewma_alpha))
        )

    def harvest_cuda_events(self):
        if not self._pending_cuda_events:
            return
        remaining = []
        for (
            backend_name,
            mode,
            seq_bucket,
            start_event,
            end_event,
        ) in self._pending_cuda_events:
            if not end_event.query():
                remaining.append((backend_name, mode, seq_bucket, start_event, end_event))
                continue
            latency_ms = float(start_event.elapsed_time(end_event))
            stats = self._stats.setdefault(
                (backend_name, mode, seq_bucket), SparseBackendStats()
            )
            stats.observe(
                latency_ms, max(0.0, min(1.0, self.config.profiling_ewma_alpha))
            )
        self._pending_cuda_events = remaining

    def make_cuda_events(self, backend_name: str, profile: SparseRuntimeProfile):
        mode, seq_bucket = self.bucket_for(profile)
        start_event = torch.cuda.Event(enable_timing=True)
        end_event = torch.cuda.Event(enable_timing=True)
        self._pending_cuda_events.append(
            (backend_name, mode, seq_bucket, start_event, end_event)
        )
        return start_event, end_event


class SparseAutoSelector:
    def __init__(
        self,
        model_runner: "ModelRunner",
        config: SparseAutoConfig,
        profiler: Optional[SparseAutoProfiler] = None,
    ):
        self.model_runner = model_runner
        self.config = config
        self.profiler = profiler
        self._last_selection: Optional[SparseSelection] = None
        self._select_count = 0

    def select(
        self,
        profile: SparseRuntimeProfile,
        available: dict[str, bool],
    ) -> SparseSelection:
        self._select_count += 1
        dense_name = self.config.dense_fallback_backend
        candidates = {dense_name: True, **available}

        if self.config.force_backend:
            forced = self.config.force_backend
            if candidates.get(forced, False):
                return SparseSelection(forced, "forced by sparse_attention_config", 1e9)
            logger.warning(
                "Forced sparse backend %s is unavailable, using %s instead.",
                forced,
                dense_name,
            )

        if self.config.selection_policy == "legacy":
            return self._legacy_select(profile, available)

        scores: dict[str, tuple[float, str]] = {}
        for backend_name, is_available in candidates.items():
            if not is_available:
                continue
            if backend_name == "h2o" and not self.config.enable_h2o:
                continue
            estimate = self._estimate_backend(backend_name, profile)
            if estimate is None:
                continue
            scores[backend_name] = self._score_estimate(
                backend_name, profile, estimate
            )

        if not scores:
            return SparseSelection(dense_name, "no sparse backend is eligible", 0.0)

        exploration = self._maybe_explore(profile, scores)
        if exploration is not None:
            self._last_selection = exploration
            return exploration

        backend_name, (score, reason) = max(scores.items(), key=lambda item: item[1][0])
        selection = SparseSelection(backend_name=backend_name, reason=reason, score=score)
        selection = self._apply_hysteresis(selection, scores)
        self._last_selection = selection
        return selection

    def _legacy_select(
        self,
        profile: SparseRuntimeProfile,
        available: dict[str, bool],
    ) -> SparseSelection:
        dense_name = self.config.dense_fallback_backend
        scores: dict[str, tuple[float, str]] = {dense_name: (1.0, "dense fallback")}

        if available.get("retroinfer", False):
            scores["retroinfer"] = self._score_retroinfer(profile)
        if available.get("nsa", False):
            scores["nsa"] = self._score_nsa(profile)
        if self.config.enable_h2o and available.get("h2o", False):
            scores["h2o"] = self._score_h2o(profile)

        backend_name, (score, reason) = max(scores.items(), key=lambda item: item[1][0])
        return SparseSelection(backend_name=backend_name, reason=reason, score=score)

    def _apply_hysteresis(
        self,
        selection: SparseSelection,
        scores: dict[str, tuple[float, str]],
    ) -> SparseSelection:
        if self._last_selection is None:
            return selection
        previous = self._last_selection.backend_name
        if previous == selection.backend_name or previous not in scores:
            return selection
        previous_score = scores[previous][0]
        required_gain = max(
            0.0, abs(previous_score) * max(0.0, self.config.selection_hysteresis)
        )
        should_explore = (
            self.config.enable_online_profiling
            and self.config.exploration_interval > 0
            and self._select_count % self.config.exploration_interval == 0
        )
        if should_explore:
            return selection
        if selection.score < previous_score + required_gain:
            return SparseSelection(
                previous,
                f"kept previous backend by hysteresis; challenger={selection.backend_name}",
                previous_score,
            )
        return selection

    def _maybe_explore(
        self,
        profile: SparseRuntimeProfile,
        scores: dict[str, tuple[float, str]],
    ) -> Optional[SparseSelection]:
        if (
            not self.config.enable_online_profiling
            or self.profiler is None
            or self.config.exploration_interval <= 0
            or self._select_count % self.config.exploration_interval != 0
        ):
            return None
        warmup_steps = max(0, self.config.profiling_warmup_steps)
        if warmup_steps == 0:
            return None
        unprofiled = []
        for backend_name, (score, reason) in scores.items():
            stats = self.profiler.stats_for(backend_name, profile)
            if stats.count < warmup_steps and score > 0:
                unprofiled.append((backend_name, score, reason, stats.count))
        if not unprofiled:
            return None
        backend_name, score, reason, count = max(unprofiled, key=lambda item: item[1])
        return SparseSelection(
            backend_name,
            f"profiling exploration {count + 1}/{warmup_steps}; {reason}",
            score,
        )

    def _estimate_backend(
        self,
        backend_name: str,
        profile: SparseRuntimeProfile,
    ) -> Optional[SparseBackendEstimate]:
        if backend_name == "retroinfer":
            if profile.speculative or profile.is_mla:
                return None
            if profile.max_seq_len < self.config.retroinfer_min_decode_seq_len:
                return None
            sparsity = self._retroinfer_sparsity(profile)
            quality = 0.98
            overhead = 0.22 if profile.forward_mode == "decode" else 0.35
        elif backend_name == "nsa":
            if profile.max_seq_len < self.config.nsa_min_seq_len:
                return None
            sparsity = self._configured_sparsity(
                default=0.28 if profile.is_mla else 0.35
            )
            quality = 0.96 if profile.is_mla else 0.9
            overhead = 0.18
        elif backend_name == "h2o":
            if profile.speculative or profile.is_mla:
                return None
            if profile.max_seq_len < self.config.h2o_min_seq_len:
                return None
            sparsity = self._configured_sparsity(default=0.25)
            quality = 0.82
            overhead = 0.14
        else:
            sparsity = 1.0
            quality = 1.0
            overhead = 0.08

        analytical_latency = self._analytical_latency_ms(profile, sparsity, overhead)
        observed_latency = (
            self.profiler.latency_ms_for(backend_name, profile)
            if self.profiler is not None and self.config.enable_online_profiling
            else None
        )
        latency_ms = observed_latency if observed_latency is not None else analytical_latency
        source = "profiled" if observed_latency is not None else "estimated"
        return SparseBackendEstimate(
            latency_ms=latency_ms,
            memory_mb=self._estimate_memory_mb(profile, sparsity),
            sparsity=sparsity,
            quality=quality,
            source=source,
        )

    def _score_estimate(
        self,
        backend_name: str,
        profile: SparseRuntimeProfile,
        estimate: SparseBackendEstimate,
    ) -> tuple[float, str]:
        latency_score = self._budget_score(
            estimate.latency_ms, self.config.latency_budget_ms, lower_is_better=True
        )
        memory_score = self._budget_score(
            estimate.memory_mb, self.config.memory_budget_mb, lower_is_better=True
        )
        sparsity_score = self._sparsity_score(estimate.sparsity)
        quality_score = 20.0 * estimate.quality
        mode_bonus = 0.0
        if backend_name == "retroinfer" and profile.forward_mode == "decode":
            mode_bonus += 10.0
            if profile.batch_size <= self.config.retroinfer_max_batch_size:
                mode_bonus += 14.0
            else:
                mode_bonus -= 12.0
        if backend_name == "nsa" and (profile.is_mla or profile.batch_size >= 8):
            mode_bonus += 7.0
        if backend_name == "h2o" and self.config.memory_budget_mb is not None:
            mode_bonus += 4.0
        if backend_name == self.config.dense_fallback_backend:
            mode_bonus += 2.0

        score = (
            latency_score
            + memory_score
            + sparsity_score
            + quality_score
            + mode_bonus
        )
        reason = (
            f"{estimate.source}: latency={estimate.latency_ms:.2f}ms, "
            f"memory={estimate.memory_mb:.0f}MB, sparsity={estimate.sparsity:.2f}, "
            f"quality={estimate.quality:.2f}"
        )
        return score, reason

    def _budget_score(
        self,
        value: float,
        budget: Optional[float],
        *,
        lower_is_better: bool,
    ) -> float:
        if budget is None or budget <= 0:
            return 25.0 / max(1.0, math.log2(value + 2.0))
        ratio = value / budget if lower_is_better else budget / max(value, 1e-6)
        if ratio <= 1.0:
            return 30.0 + 15.0 * (1.0 - ratio)
        return max(-40.0, -35.0 * (ratio - 1.0))

    def _sparsity_score(self, sparsity: float) -> float:
        sparsity = max(0.0, min(1.0, sparsity))
        if self.config.target_sparsity is None:
            return 20.0 * (1.0 - sparsity)
        target = max(0.01, min(1.0, self.config.target_sparsity))
        distance = abs(sparsity - target)
        return 25.0 * max(0.0, 1.0 - distance / target)

    def _configured_sparsity(self, default: float) -> float:
        if self.config.target_sparsity is None:
            return default
        return max(0.02, min(1.0, self.config.target_sparsity))

    def _retroinfer_sparsity(self, profile: SparseRuntimeProfile) -> float:
        if profile.max_seq_len <= 0:
            return 1.0
        static_tokens = 32
        retrieval_budget = max(8, int(profile.max_seq_len * 0.02))
        recent_budget = max(16, int(profile.max_seq_len * 0.05))
        working_set = min(
            profile.max_seq_len,
            static_tokens + retrieval_budget + recent_budget,
        )
        if self.config.target_sparsity is not None:
            working_set = min(
                working_set,
                max(1, int(profile.max_seq_len * self.config.target_sparsity)),
            )
        return max(0.01, min(1.0, working_set / profile.max_seq_len))

    def _analytical_latency_ms(
        self,
        profile: SparseRuntimeProfile,
        sparsity: float,
        overhead: float,
    ) -> float:
        seq_factor = max(1.0, profile.avg_seq_len / 1024.0)
        batch_factor = max(1.0, profile.batch_size)
        token_factor = max(1.0, profile.num_tokens / max(1, profile.batch_size))
        mode_multiplier = 1.0 if profile.forward_mode == "decode" else 2.4
        return (
            overhead
            + 0.045
            * batch_factor
            * token_factor
            * seq_factor
            * sparsity
            * mode_multiplier
        )

    def _estimate_memory_mb(
        self,
        profile: SparseRuntimeProfile,
        sparsity: float,
    ) -> float:
        model_config = getattr(self.model_runner, "model_config", None)
        if model_config is None:
            return 0.0
        tp_size = int(getattr(self.model_runner, "tp_size", 1))
        try:
            num_kv_heads = int(model_config.get_num_kv_heads(tp_size))
        except Exception:
            num_kv_heads = int(getattr(model_config, "num_attention_heads", 1))
        head_dim = int(getattr(model_config, "head_dim", 128))
        dtype = getattr(self.model_runner, "dtype", torch.float16)
        bytes_per_elem = torch.empty((), dtype=dtype).element_size()
        kv_bytes = (
            profile.batch_size
            * max(1, profile.max_seq_len)
            * num_kv_heads
            * head_dim
            * 2
            * bytes_per_elem
            * max(0.01, min(1.0, sparsity))
        )
        return kv_bytes / (1024.0 * 1024.0)

    def _score_retroinfer(self, profile: SparseRuntimeProfile) -> tuple[float, str]:
        if profile.speculative:
            return (-1.0, "speculative decoding uses dense fallback")
        if profile.is_mla:
            return (-1.0, "RetroInfer path is currently tuned for non-MLA attention")
        if profile.max_seq_len < self.config.retroinfer_min_decode_seq_len:
            return (-1.0, "context is too short for RetroInfer payoff")

        score = 50.0 + min(profile.max_seq_len / 4096.0, 8.0)
        reason = "long-context decode favors RetroInfer"

        if profile.forward_mode != "decode":
            score -= 5.0
            reason = "long-context extend observes RetroInfer state for later decode"

        if profile.batch_size <= self.config.retroinfer_max_batch_size:
            score += 6.0
        else:
            score -= 8.0

        if self.config.latency_budget_ms is not None and self.config.latency_budget_ms <= 30:
            score += 4.0
            reason = "latency-sensitive long-context decode favors RetroInfer"

        if self.config.memory_budget_mb is not None and self.config.memory_budget_mb <= 4096:
            score += 4.0

        return (score, reason)

    def _score_nsa(self, profile: SparseRuntimeProfile) -> tuple[float, str]:
        if profile.max_seq_len < self.config.nsa_min_seq_len:
            return (-1.0, "context is too short for NSA payoff")

        score = 45.0 + min(profile.max_seq_len / 8192.0, 6.0)
        reason = "model-native sparse attention favors NSA"

        if profile.is_mla:
            score += 10.0
        if profile.batch_size >= 8:
            score += 5.0
            reason = "throughput-oriented decode favors NSA"
        if self.config.target_sparsity is not None:
            score += 5.0 * max(0.0, min(self.config.target_sparsity, 1.0))
        if self.config.memory_budget_mb is not None and self.config.memory_budget_mb <= 8192:
            score += 3.0

        return (score, reason)

    def _score_h2o(self, profile: SparseRuntimeProfile) -> tuple[float, str]:
        if profile.speculative:
            return (-1.0, "speculative decoding uses dense fallback")
        if profile.is_mla:
            return (-1.0, "H2O path is currently tuned for non-MLA attention")
        if profile.max_seq_len < self.config.h2o_min_seq_len:
            return (-1.0, "context is too short for H2O payoff")

        score = 40.0 + min(profile.max_seq_len / 8192.0, 4.0)
        reason = "memory-constrained long-context decode favors H2O"
        if self.config.memory_budget_mb is not None and self.config.memory_budget_mb <= 6144:
            score += 8.0
        if profile.batch_size >= 4:
            score += 3.0
        if self.config.target_sparsity is not None:
            score += 4.0 * max(0.0, min(self.config.target_sparsity, 1.0))
        return (score, reason)


class SparseAutoAttnBackend(AttentionBackend):
    """
    Unified sparse attention backend that chooses an execution backend per batch.

    Today it can dispatch between:
    - dense fallback backend (default: Triton)
    - RetroInfer
    - NSA

    H2O and finer-grained hybrid policies can be plugged in later without touching
    the ModelRunner or backend registry again.
    """

    def __init__(self, model_runner: "ModelRunner"):
        super().__init__()
        self.model_runner = model_runner
        self.config = SparseAutoConfig.from_server_args(model_runner.server_args)
        self.profiler = SparseAutoProfiler(self.config)
        self.selector = SparseAutoSelector(model_runner, self.config, self.profiler)
        self._backend_cache: dict[str, AttentionBackend] = {}
        self._availability_cache: dict[str, bool] = {}
        self._last_selection: Optional[SparseSelection] = None
        self._active_backend_name = self.config.dense_fallback_backend
        self._bound_host_pool = None
        self._bound_io_backend: Optional[str] = None
        self._retroinfer_locked = False

    def init_forward_metadata(self, forward_batch: "ForwardBatch"):
        backend = self._select_backend(forward_batch)
        backend.init_forward_metadata(forward_batch)

    def bind_kv_store_host_pool(
        self,
        host_pool,
        io_backend: Optional[str] = None,
        tree_cache=None,
    ):
        self._bound_host_pool = host_pool
        self._bound_io_backend = io_backend
        for backend in self._backend_cache.values():
            if hasattr(backend, "bind_kv_store_host_pool"):
                backend.bind_kv_store_host_pool(
                    host_pool,
                    io_backend=io_backend,
                    tree_cache=tree_cache,
                )

    def init_cuda_graph_state(self, max_bs: int, max_num_tokens: int):
        self._dense_backend().init_cuda_graph_state(max_bs, max_num_tokens)

    def init_forward_metadata_capture_cuda_graph(
        self,
        bs: int,
        num_tokens: int,
        req_pool_indices: torch.Tensor,
        seq_lens: torch.Tensor,
        encoder_lens: Optional[torch.Tensor],
        forward_mode: "ForwardMode",
        spec_info: Optional["SpecInput"],
    ):
        self._active_backend_name = self.config.dense_fallback_backend
        return self._dense_backend().init_forward_metadata_capture_cuda_graph(
            bs,
            num_tokens,
            req_pool_indices,
            seq_lens,
            encoder_lens,
            forward_mode,
            spec_info,
        )

    def init_forward_metadata_replay_cuda_graph(
        self,
        bs: int,
        req_pool_indices: torch.Tensor,
        seq_lens: torch.Tensor,
        seq_lens_sum: int,
        encoder_lens: Optional[torch.Tensor],
        forward_mode: "ForwardMode",
        spec_info: Optional["SpecInput"],
        seq_lens_cpu: Optional[torch.Tensor],
    ):
        self._active_backend_name = self.config.dense_fallback_backend
        return self._dense_backend().init_forward_metadata_replay_cuda_graph(
            bs,
            req_pool_indices,
            seq_lens,
            seq_lens_sum,
            encoder_lens,
            forward_mode,
            spec_info,
            seq_lens_cpu,
        )

    def get_cuda_graph_seq_len_fill_value(self):
        return self._dense_backend().get_cuda_graph_seq_len_fill_value()

    def get_verify_buffers_to_fill_after_draft(self):
        return self._dense_backend().get_verify_buffers_to_fill_after_draft()

    def update_verify_buffers_to_fill_after_draft(
        self, spec_info: "SpecInput", cuda_graph_bs: Optional[int]
    ):
        return self._dense_backend().update_verify_buffers_to_fill_after_draft(
            spec_info, cuda_graph_bs
        )

    def forward_decode(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        layer: "RadixAttention",
        forward_batch: "ForwardBatch",
        save_kv_cache: bool = True,
        **kwargs,
    ):
        return self._run_profiled(
            forward_batch,
            lambda backend: backend.forward_decode(
                q, k, v, layer, forward_batch, save_kv_cache=save_kv_cache, **kwargs
            ),
        )

    def forward_extend(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        layer: "RadixAttention",
        forward_batch: "ForwardBatch",
        save_kv_cache: bool = True,
        **kwargs,
    ):
        return self._run_profiled(
            forward_batch,
            lambda backend: backend.forward_extend(
                q, k, v, layer, forward_batch, save_kv_cache=save_kv_cache, **kwargs
            ),
        )

    def forward_mixed(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        layer: "RadixAttention",
        forward_batch: "ForwardBatch",
        save_kv_cache: bool = True,
        **kwargs,
    ):
        backend = self._dense_backend()
        return backend.forward_mixed(
            q, k, v, layer, forward_batch, save_kv_cache=save_kv_cache, **kwargs
        )

    def support_triton(self):
        return self._dense_backend().support_triton()

    def get_indexer_metadata(self, layer_id: int, forward_batch: "ForwardBatch"):
        backend = self._select_backend(forward_batch)
        return backend.get_indexer_metadata(layer_id, forward_batch)

    def _select_backend(self, forward_batch: "ForwardBatch") -> AttentionBackend:
        if self.config.enable_online_profiling:
            self.profiler.harvest_cuda_events()
        profile = self._build_profile(forward_batch)
        availability = {
            "retroinfer": self._is_backend_available("retroinfer"),
            "nsa": self._is_backend_available("nsa"),
            "h2o": self.config.enable_h2o and self._is_backend_available("h2o"),
        }
        if (
            self.config.lock_retroinfer_after_select
            and self._retroinfer_locked
            and availability["retroinfer"]
        ):
            selection = SparseSelection(
                "retroinfer",
                "kept RetroInfer because it owns CPU-resident full KV",
                1e9,
            )
        else:
            selection = self.selector.select(profile, availability)
        if selection.backend_name == "retroinfer":
            self._retroinfer_locked = True
        if (
            self._last_selection is None
            or self._last_selection.backend_name != selection.backend_name
            or self._last_selection.reason != selection.reason
        ):
            logger.info(
                "Sparse auto selected %s for mode=%s bs=%d max_seq=%d (%s, score=%.2f)",
                selection.backend_name,
                profile.forward_mode,
                profile.batch_size,
                profile.max_seq_len,
                selection.reason,
                selection.score,
            )
        self._last_selection = selection
        self._active_backend_name = selection.backend_name
        return self._get_backend(selection.backend_name)

    def _run_profiled(self, forward_batch: "ForwardBatch", fn):
        profile = self._build_profile(forward_batch)
        backend = self._select_backend(forward_batch)
        backend_name = self._active_backend_name

        if not self.config.enable_online_profiling:
            return fn(backend)

        use_cuda_events = (
            torch.cuda.is_available()
            and not self.config.profiling_sync_cuda
            and not self._is_cuda_graph_capturing()
        )
        if use_cuda_events:
            start_event, end_event = self.profiler.make_cuda_events(backend_name, profile)
            start_event.record()
            try:
                return fn(backend)
            finally:
                end_event.record()

        if torch.cuda.is_available() and self.config.profiling_sync_cuda:
            torch.cuda.synchronize()
        start_time = time.perf_counter()
        try:
            return fn(backend)
        finally:
            if torch.cuda.is_available() and self.config.profiling_sync_cuda:
                torch.cuda.synchronize()
            latency_ms = (time.perf_counter() - start_time) * 1000.0
            self.profiler.observe(backend_name, profile, latency_ms)

    def _is_cuda_graph_capturing(self) -> bool:
        try:
            return torch.cuda.is_current_stream_capturing()
        except Exception:
            return False

    def _build_profile(self, forward_batch: "ForwardBatch") -> SparseRuntimeProfile:
        seq_lens = forward_batch.seq_lens
        batch_size = int(seq_lens.numel())
        max_seq_len = int(seq_lens.max().item()) if batch_size else 0
        avg_seq_len = float(seq_lens.float().mean().item()) if batch_size else 0.0
        seq_lens_sum = getattr(forward_batch, "seq_lens_sum", batch_size)
        if isinstance(seq_lens_sum, torch.Tensor):
            num_tokens = int(seq_lens_sum.item())
        else:
            num_tokens = int(seq_lens_sum)
        return SparseRuntimeProfile(
            forward_mode=self._forward_mode_name(forward_batch.forward_mode),
            batch_size=batch_size,
            max_seq_len=max_seq_len,
            avg_seq_len=avg_seq_len,
            num_tokens=num_tokens,
            is_mla=self.model_runner.use_mla_backend,
            speculative=self.model_runner.server_args.speculative_algorithm is not None,
        )

    def _forward_mode_name(self, forward_mode: "ForwardMode") -> str:
        if forward_mode.is_decode_or_idle():
            return "decode"
        if forward_mode.is_mixed():
            return "mixed"
        return "extend"

    def _dense_backend(self) -> AttentionBackend:
        return self._get_backend(self.config.dense_fallback_backend)

    def _is_backend_available(self, name: str) -> bool:
        if name in self._availability_cache:
            return self._availability_cache[name]

        if name == "h2o":
            self._availability_cache[name] = False
            try:
                self._get_backend(name)
                self._availability_cache[name] = True
            except Exception as exc:
                logger.info("Sparse backend %s unavailable: %s", name, exc)
                self._availability_cache[name] = False
            return self._availability_cache[name]

        try:
            self._get_backend(name)
            self._availability_cache[name] = True
        except Exception as exc:
            logger.info("Sparse backend %s unavailable: %s", name, exc)
            self._availability_cache[name] = False
        return self._availability_cache[name]

    def _get_backend(self, name: str) -> AttentionBackend:
        if name in self._backend_cache:
            return self._backend_cache[name]

        if name == "retroinfer":
            from sglang.srt.layers.attention.retroinfer_backend import RetroInferAttnBackend

            backend = RetroInferAttnBackend(self.model_runner)
        elif name == "nsa":
            from sglang.srt.layers.attention.nsa_backend import NativeSparseAttnBackend

            backend = NativeSparseAttnBackend(self.model_runner)
        elif name == "triton":
            from sglang.srt.layers.attention.triton_backend import TritonAttnBackend

            backend = TritonAttnBackend(self.model_runner)
        elif name == "flashinfer":
            from sglang.srt.layers.attention.flashinfer_backend import FlashInferAttnBackend

            backend = FlashInferAttnBackend(
                self.model_runner,
                init_new_workspace=getattr(self.model_runner, "init_new_workspace", False),
            )
        elif name == "h2o":
            from sglang.srt.layers.attention.h2o_backend import H2OAttnBackend

            backend = H2OAttnBackend(self.model_runner)
        elif name == "fa3":
            from sglang.srt.layers.attention.flashattention_backend import FlashAttentionBackend

            backend = FlashAttentionBackend(self.model_runner)
        else:
            raise ValueError(f"Unsupported sparse auto backend: {name}")

        if self._bound_host_pool is not None and hasattr(
            backend, "bind_kv_store_host_pool"
        ):
            backend.bind_kv_store_host_pool(
                self._bound_host_pool, io_backend=self._bound_io_backend
            )
        self._backend_cache[name] = backend
        return backend

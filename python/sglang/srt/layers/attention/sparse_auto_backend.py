from __future__ import annotations

import json
import logging
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
    latency_budget_ms: Optional[float] = None
    memory_budget_mb: Optional[float] = None
    target_sparsity: Optional[float] = None
    retroinfer_min_decode_seq_len: int = 4096
    retroinfer_max_batch_size: int = 8
    nsa_min_seq_len: int = 2048
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


class SparseAutoSelector:
    def __init__(self, model_runner: "ModelRunner", config: SparseAutoConfig):
        self.model_runner = model_runner
        self.config = config

    def select(
        self,
        profile: SparseRuntimeProfile,
        available: dict[str, bool],
    ) -> SparseSelection:
        dense_name = self.config.dense_fallback_backend
        scores: dict[str, tuple[float, str]] = {
            dense_name: (1.0, "dense fallback"),
        }

        if self.config.force_backend:
            forced = self.config.force_backend
            if available.get(forced, False):
                return SparseSelection(forced, "forced by sparse_attention_config", 1e9)
            logger.warning(
                "Forced sparse backend %s is unavailable, using %s instead.",
                forced,
                dense_name,
            )

        if available.get("retroinfer", False):
            scores["retroinfer"] = self._score_retroinfer(profile)

        if available.get("nsa", False):
            scores["nsa"] = self._score_nsa(profile)

        if self.config.enable_h2o and available.get("h2o", False):
            scores["h2o"] = self._score_h2o(profile)

        backend_name, (score, reason) = max(scores.items(), key=lambda item: item[1][0])
        return SparseSelection(backend_name=backend_name, reason=reason, score=score)

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
        score = 40.0 + min(profile.max_seq_len / 8192.0, 4.0)
        return (score, "H2O provider enabled")


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
        self.selector = SparseAutoSelector(model_runner, self.config)
        self._backend_cache: dict[str, AttentionBackend] = {}
        self._availability_cache: dict[str, bool] = {}
        self._last_selection: Optional[SparseSelection] = None
        self._active_backend_name = self.config.dense_fallback_backend

    def init_forward_metadata(self, forward_batch: "ForwardBatch"):
        backend = self._select_backend(forward_batch)
        backend.init_forward_metadata(forward_batch)

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
        backend = self._select_backend(forward_batch)
        return backend.forward_decode(
            q, k, v, layer, forward_batch, save_kv_cache=save_kv_cache, **kwargs
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
        backend = self._select_backend(forward_batch)
        return backend.forward_extend(
            q, k, v, layer, forward_batch, save_kv_cache=save_kv_cache, **kwargs
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
        profile = self._build_profile(forward_batch)
        availability = {
            "retroinfer": self._is_backend_available("retroinfer"),
            "nsa": self._is_backend_available("nsa"),
            "h2o": self._is_backend_available("h2o"),
        }
        selection = self.selector.select(profile, availability)
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
            return False

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
        elif name == "fa3":
            from sglang.srt.layers.attention.flashattention_backend import FlashAttentionBackend

            backend = FlashAttentionBackend(self.model_runner)
        else:
            raise ValueError(f"Unsupported sparse auto backend: {name}")

        self._backend_cache[name] = backend
        return backend

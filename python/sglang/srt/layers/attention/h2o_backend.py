from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from typing import Optional

import torch

from sglang.srt.layers.attention.base_attn_backend import AttentionBackend
from sglang.srt.layers.attention.retroinfer.types import RetroInferDecision
from sglang.srt.layers.attention.triton_backend import TritonAttnBackend

logger = logging.getLogger(__name__)


@dataclass
class H2OConfig:
    cache_budget_ratio: float = 0.2
    recent_ratio: float = 0.5
    min_decode_seq_len: int = 2048
    max_batch_size: int = 8
    sink_token_count: int = 4

    @classmethod
    def from_server_args(cls, server_args) -> "H2OConfig":
        raw = getattr(server_args, "sparse_attention_config", "{}") or "{}"
        try:
            data = json.loads(raw)
        except json.JSONDecodeError as exc:
            logger.warning(
                "Invalid sparse attention config JSON for H2O, using defaults (%s)",
                exc,
            )
            data = {}

        mapping = {
            "h2o_cache_budget_ratio": "cache_budget_ratio",
            "h2o_recent_ratio": "recent_ratio",
            "h2o_min_seq_len": "min_decode_seq_len",
            "h2o_max_batch_size": "max_batch_size",
            "h2o_sink_token_count": "sink_token_count",
        }
        kwargs = {}
        for src_key, dst_key in mapping.items():
            if src_key in data:
                kwargs[dst_key] = data[src_key]
        return cls(**kwargs)


class H2OCapabilityChecker:
    def __init__(self, model_runner, config: H2OConfig):
        self.model_runner = model_runner
        self.config = config

    def check_extend(self, layer, forward_batch) -> RetroInferDecision:
        if not forward_batch.forward_mode.is_extend():
            return RetroInferDecision(False, "fallback", "not extend")
        if getattr(layer, "is_cross_attention", False):
            return RetroInferDecision(False, "fallback", "cross attention unsupported")
        if getattr(layer, "sliding_window_size", -1) not in (-1, None):
            return RetroInferDecision(False, "fallback", "sliding window unsupported")
        if self.model_runner.use_mla_backend:
            return RetroInferDecision(False, "fallback", "MLA backend unsupported")
        return RetroInferDecision(True, "extend_observe")

    def check_decode(self, layer, forward_batch) -> RetroInferDecision:
        if not forward_batch.forward_mode.is_decode():
            return RetroInferDecision(False, "fallback", "not decode")
        if getattr(layer, "is_cross_attention", False):
            return RetroInferDecision(False, "fallback", "cross attention unsupported")
        if getattr(layer, "sliding_window_size", -1) not in (-1, None):
            return RetroInferDecision(False, "fallback", "sliding window unsupported")
        if self.model_runner.use_mla_backend:
            return RetroInferDecision(False, "fallback", "MLA backend unsupported")

        server_args = getattr(self.model_runner, "server_args", None)
        if server_args is not None:
            if getattr(server_args, "speculative_algorithm", None):
                return RetroInferDecision(False, "fallback", "speculative decode unsupported")
            if not getattr(server_args, "disable_cuda_graph", False):
                return RetroInferDecision(False, "fallback", "cuda graph replay unsupported")

        seq_lens = forward_batch.seq_lens
        if seq_lens is None or seq_lens.numel() == 0:
            return RetroInferDecision(False, "fallback", "missing seq_lens")
        if int(seq_lens.max().item()) < self.config.min_decode_seq_len:
            return RetroInferDecision(False, "fallback", "sequence too short")
        if int(seq_lens.numel()) > self.config.max_batch_size:
            return RetroInferDecision(False, "fallback", "batch size too large")
        return RetroInferDecision(True, "decode_sparse")


@dataclass
class H2OLayerState:
    scores: torch.Tensor
    retained_positions: torch.Tensor
    last_seq_len: int = 0
    last_budget: int = 0


@dataclass
class H2OSelectionUpdate:
    positions: torch.Tensor
    update_mode: str
    prev_positions: torch.Tensor
    added_positions: torch.Tensor
    evicted_positions: torch.Tensor


@dataclass
class H2ORequestState:
    req_pool_idx: int
    last_seq_len: int = 0
    layer_states: dict[int, H2OLayerState] = field(default_factory=dict)


class H2OSessionManager:
    def __init__(self):
        self.request_states: dict[int, H2ORequestState] = {}

    def get_or_create(self, req_pool_idx: int) -> H2ORequestState:
        state = self.request_states.get(req_pool_idx)
        if state is None:
            state = H2ORequestState(req_pool_idx=req_pool_idx)
            self.request_states[req_pool_idx] = state
        return state

    def observe_extend(self, req_pool_indices: list[int], seq_lens: list[int]):
        for req_pool_idx, seq_len in zip(req_pool_indices, seq_lens):
            state = self.get_or_create(req_pool_idx)
            if seq_len < state.last_seq_len:
                state.layer_states.clear()
            state.last_seq_len = seq_len

    def drop_missing_requests(self, active_req_pool_indices: list[int]):
        active = set(active_req_pool_indices)
        stale = [idx for idx in self.request_states if idx not in active]
        for idx in stale:
            self.request_states.pop(idx, None)


class H2OAttnBackend(AttentionBackend):
    """
    H2O backend: keep a dynamic mixture of heavy hitters and recent tokens.

    This implementation follows the H2O paper's core policy:
    - keep a fixed KV budget per request
    - reserve part of the budget for the most recent tokens
    - use accumulated local attention scores to retain heavy hitters

    Integration strategy in SGLang:
    - extend/prefill still uses the dense fallback backend
    - decode uses a subset-gathered attention computation over retained tokens
    - request/layer score state is managed locally in the backend
    """

    def __init__(self, model_runner):
        super().__init__()
        self.model_runner = model_runner
        self.config = H2OConfig.from_server_args(model_runner.server_args)
        self.fallback = TritonAttnBackend(model_runner)
        self.capability_checker = H2OCapabilityChecker(model_runner, self.config)
        self.session_manager = H2OSessionManager()

    def init_forward_metadata(self, forward_batch):
        return self.fallback.init_forward_metadata(forward_batch)

    def init_cuda_graph_state(self, max_bs: int, max_num_tokens: int):
        return self.fallback.init_cuda_graph_state(max_bs, max_num_tokens)

    def init_forward_metadata_capture_cuda_graph(
        self,
        bs: int,
        num_tokens: int,
        req_pool_indices: torch.Tensor,
        seq_lens: torch.Tensor,
        encoder_lens: Optional[torch.Tensor],
        forward_mode,
        spec_info,
    ):
        return self.fallback.init_forward_metadata_capture_cuda_graph(
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
        forward_mode,
        spec_info,
        seq_lens_cpu: Optional[torch.Tensor],
    ):
        return self.fallback.init_forward_metadata_replay_cuda_graph(
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
        return self.fallback.get_cuda_graph_seq_len_fill_value()

    def get_verify_buffers_to_fill_after_draft(self):
        return self.fallback.get_verify_buffers_to_fill_after_draft()

    def update_verify_buffers_to_fill_after_draft(self, spec_info, cuda_graph_bs: Optional[int]):
        return self.fallback.update_verify_buffers_to_fill_after_draft(
            spec_info, cuda_graph_bs
        )

    def forward_extend(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        layer,
        forward_batch,
        save_kv_cache: bool = True,
        **kwargs,
    ):
        out = self.fallback.forward_extend(
            q, k, v, layer, forward_batch, save_kv_cache=save_kv_cache, **kwargs
        )
        decision = self.capability_checker.check_extend(layer, forward_batch)
        if decision.allow:
            req_pool_indices = [int(x) for x in forward_batch.req_pool_indices.tolist()]
            seq_lens = [int(x) for x in forward_batch.seq_lens.tolist()]
            self.session_manager.observe_extend(req_pool_indices, seq_lens)
            self.session_manager.drop_missing_requests(req_pool_indices)
        return out

    def forward_decode(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        layer,
        forward_batch,
        save_kv_cache: bool = True,
        **kwargs,
    ):
        decision = self.capability_checker.check_decode(layer, forward_batch)
        if not decision.allow:
            if layer.layer_id == 0:
                logger.info(
                    "H2O inactive for decode batch: reason=%s bs=%d max_seq=%d",
                    decision.reason,
                    int(forward_batch.seq_lens.numel()) if forward_batch.seq_lens is not None else 0,
                    int(forward_batch.seq_lens.max().item())
                    if forward_batch.seq_lens is not None and forward_batch.seq_lens.numel() > 0
                    else 0,
                )
            return self.fallback.forward_decode(
                q, k, v, layer, forward_batch, save_kv_cache=save_kv_cache, **kwargs
            )

        q = q.reshape(-1, layer.tp_q_head_num * layer.qk_head_dim)
        if layer.qk_head_dim != layer.v_head_dim:
            o = q.new_empty((q.shape[0], layer.tp_q_head_num * layer.v_head_dim))
        else:
            o = torch.empty_like(q)

        cache_loc = (
            forward_batch.encoder_out_cache_loc
            if layer.is_cross_attention
            else forward_batch.out_cache_loc
        )
        if save_kv_cache:
            forward_batch.token_to_kv_pool.set_kv_buffer(layer, cache_loc, k, v)

        q_ = q.view(-1, layer.tp_q_head_num, layer.qk_head_dim)
        o_ = o.view(-1, layer.tp_q_head_num, layer.v_head_dim)
        k_cache = forward_batch.token_to_kv_pool.get_key_buffer(layer.layer_id)
        v_cache = forward_batch.token_to_kv_pool.get_value_buffer(layer.layer_id)
        req_to_token = forward_batch.req_to_token_pool.req_to_token
        self.session_manager.drop_missing_requests(
            [int(x) for x in forward_batch.req_pool_indices.tolist()]
        )

        for seq_idx in range(forward_batch.seq_lens.shape[0]):
            seq_len = int(forward_batch.seq_lens[seq_idx].item())
            req_pool_idx = int(forward_batch.req_pool_indices[seq_idx].item())
            req_state = self.session_manager.get_or_create(req_pool_idx)
            layer_state = self._ensure_layer_state(req_state, layer.layer_id, seq_len, q_.device)
            selection_update = self._update_retained_positions(layer_state, seq_len)
            selected_positions = selection_update.positions
            selected_tokens = req_to_token[req_pool_idx, selected_positions]
            budget, sink_budget, recent_budget, hh_budget = self._compute_budget_split(seq_len)

            if layer.layer_id == 0:
                logger.info(
                    "H2O active: req=%d seq_len=%d budget=%d sink=%d recent=%d hh=%d selected=%d update=%s prev_selected=%s added=%s evicted=%s",
                    req_pool_idx,
                    seq_len,
                    budget,
                    sink_budget,
                    recent_budget,
                    hh_budget,
                    int(selected_positions.numel()),
                    selection_update.update_mode,
                    self._format_positions_for_log(selection_update.prev_positions),
                    self._format_positions_for_log(selection_update.added_positions),
                    self._format_positions_for_log(selection_update.evicted_positions),
                )

            per_req_q = q_[seq_idx : seq_idx + 1]
            per_req_k = k_cache[selected_tokens]
            per_req_v = v_cache[selected_tokens]
            per_req_out, attn_weights = self._compute_h2o_decode(
                per_req_q,
                per_req_k,
                per_req_v,
                scaling=layer.scaling,
                q_head_num=layer.tp_q_head_num,
                kv_head_num=layer.tp_k_head_num,
            )
            o_[seq_idx : seq_idx + 1] = per_req_out
            self._update_scores(layer_state, selected_positions, attn_weights)
            req_state.last_seq_len = seq_len

        return o

    def support_triton(self):
        return False

    def _ensure_layer_state(
        self,
        req_state: H2ORequestState,
        layer_id: int,
        seq_len: int,
        device: torch.device,
    ) -> H2OLayerState:
        layer_state = req_state.layer_states.get(layer_id)
        if layer_state is None:
            layer_state = H2OLayerState(
                scores=torch.zeros(seq_len, dtype=torch.float32, device=device),
                retained_positions=torch.empty(0, dtype=torch.long, device=device),
                last_seq_len=0,
                last_budget=0,
            )
            req_state.layer_states[layer_id] = layer_state
            return layer_state

        if seq_len < layer_state.scores.numel():
            layer_state.scores = layer_state.scores[:seq_len]
            layer_state.retained_positions = layer_state.retained_positions[
                layer_state.retained_positions < seq_len
            ]
        elif seq_len > layer_state.scores.numel():
            pad = torch.zeros(
                seq_len - layer_state.scores.numel(),
                dtype=layer_state.scores.dtype,
                device=layer_state.scores.device,
            )
            layer_state.scores = torch.cat([layer_state.scores, pad], dim=0)
        return layer_state

    def _update_retained_positions(
        self,
        layer_state: H2OLayerState,
        seq_len: int,
    ) -> H2OSelectionUpdate:
        prev_positions = layer_state.retained_positions.clone()
        budget, sink_budget, recent_budget, _ = self._compute_budget_split(seq_len)
        if seq_len <= budget:
            positions = torch.arange(
                seq_len, device=layer_state.scores.device, dtype=torch.long
            )
            layer_state.retained_positions = positions
            layer_state.last_seq_len = seq_len
            layer_state.last_budget = budget
            return self._build_selection_update(prev_positions, positions, "full")

        # Rebuild on discontinuities. The common decode path should hit the online branch below.
        if (
            layer_state.retained_positions.numel() == 0
            or seq_len != layer_state.last_seq_len + 1
            or budget != layer_state.last_budget
        ):
            positions = self._rebuild_retained_positions(layer_state, seq_len, budget)
            layer_state.retained_positions = positions
            layer_state.last_seq_len = seq_len
            layer_state.last_budget = budget
            return self._build_selection_update(prev_positions, positions, "rebuild")

        new_pos = torch.tensor([seq_len - 1], device=layer_state.scores.device, dtype=torch.long)
        protected = self._build_protected_positions(
            seq_len,
            sink_budget=sink_budget,
            recent_budget=recent_budget,
            device=layer_state.scores.device,
        )
        candidate = torch.unique(
            torch.cat([layer_state.retained_positions, new_pos, protected]), sorted=True
        )

        while candidate.numel() > budget:
            removable_mask = ~torch.isin(candidate, protected)
            removable = candidate[removable_mask]
            if removable.numel() == 0:
                break
            removable_scores = layer_state.scores[removable]
            evict_pos = removable[torch.argmin(removable_scores)]
            candidate = candidate[candidate != evict_pos]

        if candidate.numel() < budget:
            fill = self._select_fill_positions(
                layer_state,
                seq_len=seq_len,
                budget=budget - int(candidate.numel()),
                excluded=candidate,
                sink_budget=sink_budget,
                recent_budget=recent_budget,
            )
            if fill.numel() > 0:
                candidate = torch.unique(torch.cat([candidate, fill]), sorted=True)

        layer_state.retained_positions = candidate
        layer_state.last_seq_len = seq_len
        layer_state.last_budget = budget
        return self._build_selection_update(prev_positions, candidate, "online")

    def _build_selection_update(
        self,
        prev_positions: torch.Tensor,
        positions: torch.Tensor,
        update_mode: str,
    ) -> H2OSelectionUpdate:
        if prev_positions.numel() == 0:
            added = positions
            evicted = torch.empty(0, device=positions.device, dtype=torch.long)
        else:
            added = positions[~torch.isin(positions, prev_positions)]
            evicted = prev_positions[~torch.isin(prev_positions, positions)]
        return H2OSelectionUpdate(
            positions=positions,
            update_mode=update_mode,
            prev_positions=prev_positions,
            added_positions=added,
            evicted_positions=evicted,
        )

    def _rebuild_retained_positions(
        self,
        layer_state: H2OLayerState,
        seq_len: int,
        budget: int,
    ) -> torch.Tensor:
        if seq_len <= budget:
            return torch.arange(seq_len, device=layer_state.scores.device, dtype=torch.long)

        sink_budget, recent_budget = self._compute_budget_split(seq_len)[1:3]
        protected = self._build_protected_positions(
            seq_len,
            sink_budget=sink_budget,
            recent_budget=recent_budget,
            device=layer_state.scores.device,
        )
        hh_fill = self._select_fill_positions(
            layer_state,
            seq_len=seq_len,
            budget=max(0, budget - int(protected.numel())),
            excluded=protected,
            sink_budget=sink_budget,
            recent_budget=recent_budget,
        )
        positions = torch.unique(torch.cat([protected, hh_fill]), sorted=True)
        if positions.numel() < budget:
            all_positions = torch.arange(
                seq_len, device=layer_state.scores.device, dtype=torch.long
            )
            extra = all_positions[~torch.isin(all_positions, positions)][
                : budget - positions.numel()
            ]
            if extra.numel() > 0:
                positions = torch.unique(torch.cat([positions, extra]), sorted=True)
        return positions

    def _build_protected_positions(
        self,
        seq_len: int,
        sink_budget: int,
        recent_budget: int,
        device: torch.device,
    ) -> torch.Tensor:
        protected = []
        if sink_budget > 0:
            protected.append(torch.arange(sink_budget, device=device, dtype=torch.long))
        if recent_budget > 0:
            recent_start = max(0, seq_len - recent_budget)
            protected.append(
                torch.arange(recent_start, seq_len, device=device, dtype=torch.long)
            )
        if not protected:
            return torch.empty(0, device=device, dtype=torch.long)
        return torch.unique(torch.cat(protected), sorted=True)

    def _select_fill_positions(
        self,
        layer_state: H2OLayerState,
        seq_len: int,
        budget: int,
        excluded: torch.Tensor,
        sink_budget: int,
        recent_budget: int,
    ) -> torch.Tensor:
        if budget <= 0:
            return torch.empty(0, device=layer_state.scores.device, dtype=torch.long)

        recent_start = max(0, seq_len - recent_budget) if recent_budget > 0 else seq_len
        prefix_start = sink_budget
        prefix_end = max(prefix_start, recent_start)
        if prefix_end <= prefix_start:
            return torch.empty(0, device=layer_state.scores.device, dtype=torch.long)

        prefix_positions = torch.arange(
            prefix_start,
            prefix_end,
            device=layer_state.scores.device,
            dtype=torch.long,
        )
        if excluded.numel() > 0:
            prefix_positions = prefix_positions[~torch.isin(prefix_positions, excluded)]
        if prefix_positions.numel() == 0:
            return torch.empty(0, device=layer_state.scores.device, dtype=torch.long)

        budget = min(budget, int(prefix_positions.numel()))
        prefix_scores = layer_state.scores[prefix_positions]
        topk = torch.topk(prefix_scores, k=budget, largest=True).indices
        return torch.sort(prefix_positions[topk]).values

    def _compute_budget_split(self, seq_len: int) -> tuple[int, int, int, int]:
        budget = max(1, int(seq_len * self.config.cache_budget_ratio))
        budget = min(seq_len, budget)
        if budget == 1:
            sink_budget = 0
            recent_budget = 1
        else:
            sink_budget = min(self.config.sink_token_count, budget - 1)
            recent_budget = max(1, int(budget * self.config.recent_ratio))
            recent_budget = min(budget - sink_budget, recent_budget)
        hh_budget = max(0, budget - sink_budget - recent_budget)
        return budget, sink_budget, recent_budget, hh_budget

    def _format_positions_for_log(self, positions: torch.Tensor, limit: int = 8) -> str:
        if positions.numel() == 0:
            return "[]"
        values = positions.detach().cpu().tolist()
        if len(values) <= limit:
            return str(values)
        head = ", ".join(str(v) for v in values[:limit])
        return f"[{head}, ...] ({len(values)} total)"

    def _compute_h2o_decode(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        scaling: float,
        q_head_num: int,
        kv_head_num: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        # query: [1, q_heads, qk_dim]
        # key: [num_cached, kv_heads, qk_dim]
        # value: [num_cached, kv_heads, v_dim]
        q = query.movedim(0, 1)  # [q_heads, 1, qk_dim]
        k = key.movedim(0, 1)  # [kv_heads, num_cached, qk_dim]
        v = value.movedim(0, 1)  # [kv_heads, num_cached, v_dim]

        if q_head_num != kv_head_num:
            assert q_head_num % kv_head_num == 0
            repeat = q_head_num // kv_head_num
            k = k.repeat_interleave(repeat, dim=0)
            v = v.repeat_interleave(repeat, dim=0)

        attn_logits = torch.matmul(q, k.transpose(-2, -1)) * scaling
        attn_weights = torch.softmax(attn_logits, dim=-1)
        out = torch.matmul(attn_weights, v).movedim(0, 1)  # [1, q_heads, v_dim]
        return out, attn_weights[:, 0, :]

    def _update_scores(
        self,
        layer_state: H2OLayerState,
        selected_positions: torch.Tensor,
        attn_weights: torch.Tensor,
    ) -> None:
        if selected_positions.numel() == 0:
            return
        score_delta = attn_weights.sum(dim=0).to(layer_state.scores.dtype)
        layer_state.scores[selected_positions] += score_delta

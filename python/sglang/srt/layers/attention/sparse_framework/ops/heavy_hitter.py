from __future__ import annotations

from dataclasses import dataclass, field

import torch

from sglang.srt.layers.attention.sparse_framework.selection_spec import (
    HeavyHitterSelectionSpec,
)


@dataclass
class HeavyHitterLayerState:
    scores: torch.Tensor
    retained_positions: torch.Tensor
    last_seq_len: int = 0
    last_budget: int = 0


@dataclass
class HeavyHitterRequestState:
    req_pool_idx: int
    layer_states: dict[int, HeavyHitterLayerState] = field(default_factory=dict)


class HeavyHitterStateManager:
    def __init__(self):
        self.request_states: dict[int, HeavyHitterRequestState] = {}

    def drop_missing_requests(self, active_req_pool_indices: list[int]) -> None:
        active = set(active_req_pool_indices)
        for req_pool_idx in list(self.request_states):
            if req_pool_idx not in active:
                self.request_states.pop(req_pool_idx, None)

    def select_positions(
        self,
        spec: HeavyHitterSelectionSpec,
        *,
        req_pool_idx: int,
        layer_id: int,
        seq_len: int,
        device: torch.device,
    ) -> torch.Tensor:
        request_state = self.request_states.setdefault(
            req_pool_idx, HeavyHitterRequestState(req_pool_idx=req_pool_idx)
        )
        layer_state = self._ensure_layer_state(
            request_state, layer_id=layer_id, seq_len=seq_len, device=device
        )
        return self._update_retained_positions(spec, layer_state, seq_len)

    def update_scores(
        self,
        *,
        req_pool_idx: int,
        layer_id: int,
        selected_positions: torch.Tensor,
        attn_weights: torch.Tensor,
    ) -> None:
        request_state = self.request_states.get(req_pool_idx)
        if request_state is None:
            return
        layer_state = request_state.layer_states.get(layer_id)
        if layer_state is None or selected_positions.numel() == 0:
            return
        score_delta = attn_weights.sum(dim=0).to(layer_state.scores.dtype)
        layer_state.scores[selected_positions] += score_delta

    def position_scores(
        self,
        *,
        req_pool_idx: int,
        layer_id: int,
        positions: list[int],
    ) -> dict[int, float]:
        request_state = self.request_states.get(req_pool_idx)
        if request_state is None:
            return {}
        layer_state = request_state.layer_states.get(layer_id)
        if layer_state is None or not positions:
            return {}
        scores = layer_state.scores.detach()
        result = {}
        for pos in positions:
            pos = int(pos)
            if 0 <= pos < int(scores.numel()):
                result[pos] = float(scores[pos].item())
        return result

    def _ensure_layer_state(
        self,
        request_state: HeavyHitterRequestState,
        *,
        layer_id: int,
        seq_len: int,
        device: torch.device,
    ) -> HeavyHitterLayerState:
        layer_state = request_state.layer_states.get(layer_id)
        if layer_state is None:
            layer_state = HeavyHitterLayerState(
                scores=torch.zeros(seq_len, dtype=torch.float32, device=device),
                retained_positions=torch.empty(0, dtype=torch.long, device=device),
            )
            request_state.layer_states[layer_id] = layer_state
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
        spec: HeavyHitterSelectionSpec,
        layer_state: HeavyHitterLayerState,
        seq_len: int,
    ) -> torch.Tensor:
        budget, sink_budget, recent_budget = self._compute_budget_split(spec, seq_len)
        if seq_len <= budget:
            positions = torch.arange(
                seq_len, device=layer_state.scores.device, dtype=torch.long
            )
            layer_state.retained_positions = positions
            layer_state.last_seq_len = seq_len
            layer_state.last_budget = budget
            return positions

        if (
            layer_state.retained_positions.numel() == 0
            or seq_len != layer_state.last_seq_len + 1
            or budget != layer_state.last_budget
        ):
            positions = self._rebuild_positions(
                spec, layer_state, seq_len, budget, sink_budget, recent_budget
            )
            layer_state.retained_positions = positions
            layer_state.last_seq_len = seq_len
            layer_state.last_budget = budget
            return positions

        new_pos = torch.tensor(
            [seq_len - 1], device=layer_state.scores.device, dtype=torch.long
        )
        protected = self._protected_positions(
            seq_len, sink_budget, recent_budget, layer_state.scores.device
        )
        candidate = torch.unique(
            torch.cat([layer_state.retained_positions, new_pos, protected]),
            sorted=True,
        )
        while candidate.numel() > budget:
            removable = candidate[~torch.isin(candidate, protected)]
            if removable.numel() == 0:
                break
            evict_pos = removable[torch.argmin(layer_state.scores[removable])]
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
        return candidate

    def _rebuild_positions(
        self,
        spec: HeavyHitterSelectionSpec,
        layer_state: HeavyHitterLayerState,
        seq_len: int,
        budget: int,
        sink_budget: int,
        recent_budget: int,
    ) -> torch.Tensor:
        protected = self._protected_positions(
            seq_len, sink_budget, recent_budget, layer_state.scores.device
        )
        fill = self._select_fill_positions(
            layer_state,
            seq_len=seq_len,
            budget=max(0, budget - int(protected.numel())),
            excluded=protected,
            sink_budget=sink_budget,
            recent_budget=recent_budget,
        )
        positions = torch.unique(torch.cat([protected, fill]), sorted=True)
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

    def _select_fill_positions(
        self,
        layer_state: HeavyHitterLayerState,
        *,
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
        positions = torch.arange(
            prefix_start,
            prefix_end,
            device=layer_state.scores.device,
            dtype=torch.long,
        )
        if excluded.numel() > 0:
            positions = positions[~torch.isin(positions, excluded)]
        if positions.numel() == 0:
            return torch.empty(0, device=layer_state.scores.device, dtype=torch.long)
        topk = torch.topk(
            layer_state.scores[positions], k=min(budget, int(positions.numel()))
        ).indices
        return torch.sort(positions[topk]).values

    def _protected_positions(
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
            protected.append(
                torch.arange(
                    max(0, seq_len - recent_budget),
                    seq_len,
                    device=device,
                    dtype=torch.long,
                )
            )
        if not protected:
            return torch.empty(0, device=device, dtype=torch.long)
        return torch.unique(torch.cat(protected), sorted=True)

    def _compute_budget_split(
        self, spec: HeavyHitterSelectionSpec, seq_len: int
    ) -> tuple[int, int, int]:
        if spec.budget is not None:
            budget = int(spec.budget)
        else:
            budget_ratio = spec.budget_ratio if spec.budget_ratio is not None else 0.2
            budget = int(seq_len * max(0.0, min(1.0, budget_ratio)))
        budget = max(1, min(seq_len, budget))
        if budget == 1:
            return budget, 0, 1
        sink_budget = min(max(0, spec.sink_token_count), budget - 1)
        if spec.recent_window is not None:
            recent_budget = min(max(0, spec.recent_window), budget - sink_budget)
        else:
            recent_ratio = spec.recent_ratio if spec.recent_ratio is not None else 0.5
            recent_budget = max(1, int(budget * max(0.0, min(1.0, recent_ratio))))
            recent_budget = min(budget - sink_budget, recent_budget)
        return budget, sink_budget, recent_budget

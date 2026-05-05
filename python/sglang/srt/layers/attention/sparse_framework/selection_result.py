from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import torch


@dataclass(frozen=True)
class RequestSelectionView:
    request_index: int
    req_pool_idx: int
    seq_len: int
    layer_id: int | None


@dataclass
class SelectionResult:
    positions: list[int] | torch.Tensor | None = None
    ranges: list[tuple[int, int]] | None = None
    kv_indices: torch.Tensor | None = None
    page_ids: torch.Tensor | None = None
    chunk_ids: torch.Tensor | None = None
    cluster_ids: torch.Tensor | None = None
    mask: torch.Tensor | None = None
    priority: dict[int, float] | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_callback_output(cls, value: Any) -> "SelectionResult":
        if isinstance(value, SelectionResult):
            return value
        if isinstance(value, torch.Tensor):
            return cls(positions=value)
        if isinstance(value, dict):
            return cls(
                positions=value.get("positions"),
                ranges=value.get("ranges"),
                kv_indices=value.get("kv_indices"),
                page_ids=value.get("page_ids"),
                chunk_ids=value.get("chunk_ids"),
                cluster_ids=value.get("cluster_ids"),
                mask=value.get("mask"),
                priority=value.get("priority"),
                metadata=dict(value.get("metadata", {})),
            )
        if value is None:
            return cls()
        return cls(positions=list(value))

    def to_positions(self, seq_len: int) -> list[int]:
        positions: list[int] = []
        if self.positions is not None:
            if isinstance(self.positions, torch.Tensor):
                raw_positions = self.positions.detach().cpu().tolist()
            else:
                raw_positions = list(self.positions)
            for pos in raw_positions:
                norm = seq_len + int(pos) if int(pos) < 0 else int(pos)
                positions.append(norm)

        for start, end in self.ranges or ():
            norm_start = seq_len + int(start) if int(start) < 0 else int(start)
            norm_end = seq_len + int(end) + 1 if int(end) < 0 else int(end)
            if norm_end < norm_start:
                norm_start, norm_end = norm_end, norm_start
            positions.extend(range(norm_start, norm_end))

        return sorted(set(pos for pos in positions if 0 <= pos < seq_len))

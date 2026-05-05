from __future__ import annotations

from dataclasses import dataclass, field

from sglang.srt.layers.attention.sparse_framework.selection_spec import SelectionSpec


@dataclass
class SelectionPlan:
    specs: list[SelectionSpec]
    combine: str = "union"
    fallback: str = "dense"
    normalized_ranges: list[tuple[int, int]] = field(default_factory=list)
    requires_scores: bool = False
    requires_index: bool = False
    is_full: bool = False

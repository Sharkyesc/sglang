from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class SelectionSpec:
    type: str


@dataclass(frozen=True)
class FixedSelectionSpec(SelectionSpec):
    positions: tuple[int, ...] = ()
    ranges: tuple[tuple[int, int], ...] = ()


@dataclass(frozen=True)
class SlidingWindowSelectionSpec(SelectionSpec):
    window_size: int = 0


@dataclass(frozen=True)
class RetrievalSelectionSpec(SelectionSpec):
    top_k: int = 0
    method: str = "cluster"
    name: str | None = None
    import_path: str | None = None
    fn: str | None = None
    kwargs: dict[str, Any] = field(default_factory=dict)
    static_prefix: int = 16
    static_suffix: int = 16


@dataclass(frozen=True)
class HeavyHitterSelectionSpec(SelectionSpec):
    budget: int | None = None
    budget_ratio: float | None = None
    recent_window: int | None = None
    recent_ratio: float | None = None
    sink_token_count: int = 4


@dataclass(frozen=True)
class FullSelectionSpec(SelectionSpec):
    pass


@dataclass(frozen=True)
class CustomSelectionSpec(SelectionSpec):
    name: str | None = None
    import_path: str | None = None
    fn: str | None = None
    kwargs: dict[str, Any] = field(default_factory=dict)
    payload: dict[str, Any] = field(default_factory=dict)


def parse_selection_spec(data: dict[str, Any]) -> SelectionSpec:
    spec_type = str(data.get("type", "full")).lower()
    if spec_type in ("fixed", "sink"):
        positions = tuple(int(x) for x in data.get("positions", ()))
        ranges = tuple((int(start), int(end)) for start, end in data.get("ranges", ()))
        return FixedSelectionSpec(type=spec_type, positions=positions, ranges=ranges)
    if spec_type == "sliding_window":
        return SlidingWindowSelectionSpec(
            type=spec_type, window_size=int(data.get("window_size", 0))
        )
    if spec_type == "retrieval":
        kwargs = data.get("kwargs", {})
        return RetrievalSelectionSpec(
            type=spec_type,
            top_k=int(data.get("top_k", 0)),
            method=str(data.get("method", "cluster")),
            name=data.get("name"),
            import_path=data.get("import_path") or data.get("path"),
            fn=data.get("fn") or data.get("function") or data.get("callback"),
            kwargs=dict(kwargs) if isinstance(kwargs, dict) else {},
            static_prefix=int(data.get("static_prefix", data.get("prefix", 16))),
            static_suffix=int(data.get("static_suffix", data.get("suffix", 16))),
        )
    if spec_type in ("heavy_hitter", "h2o"):
        budget = data.get("budget")
        budget_ratio = data.get("budget_ratio")
        recent_window = data.get("recent_window")
        recent_ratio = data.get("recent_ratio")
        return HeavyHitterSelectionSpec(
            type="heavy_hitter",
            budget=int(budget) if budget is not None else None,
            budget_ratio=float(budget_ratio) if budget_ratio is not None else None,
            recent_window=int(recent_window) if recent_window is not None else None,
            recent_ratio=float(recent_ratio) if recent_ratio is not None else None,
            sink_token_count=int(data.get("sink_token_count", 4)),
        )
    if spec_type in ("full", "dense"):
        return FullSelectionSpec(type="full")
    if spec_type in ("custom", "callback"):
        kwargs = data.get("kwargs", {})
        return CustomSelectionSpec(
            type="custom",
            name=data.get("name"),
            import_path=data.get("import_path") or data.get("path"),
            fn=data.get("fn") or data.get("function") or data.get("callback"),
            kwargs=dict(kwargs) if isinstance(kwargs, dict) else {},
            payload=dict(data),
        )
    return CustomSelectionSpec(type=spec_type, payload=dict(data))

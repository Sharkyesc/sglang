from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from typing import Any

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class SparseFrameworkConfig:
    selection: list[dict[str, Any]]
    combine: str = "union"
    fallback: str = "dense"
    dense_fallback_backend: str = "triton"
    working_set_budget_tokens: int | None = None
    enable_host_backup_on_evict: bool = False
    enable_physical_eviction: bool = False

    @classmethod
    def from_server_args(cls, server_args: Any) -> "SparseFrameworkConfig":
        raw = getattr(server_args, "sparse_selection_config", None)
        if raw in (None, ""):
            raw = getattr(server_args, "sparse_attention_config", "{}") or "{}"
        try:
            data = json.loads(raw)
        except json.JSONDecodeError as exc:
            logger.warning(
                "Invalid sparse selection config JSON, using full dense selection (%s)",
                exc,
            )
            data = {}
        return cls.from_dict(data)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "SparseFrameworkConfig":
        selection = data.get("selection")
        if selection is None:
            selection = [{"type": "full"}]
        if isinstance(selection, dict):
            selection = [selection]
        if not isinstance(selection, list):
            logger.warning("Invalid selection config; using full dense selection.")
            selection = [{"type": "full"}]
        return cls(
            selection=selection,
            combine=str(data.get("combine", "union")),
            fallback=str(data.get("fallback", "dense")),
            dense_fallback_backend=str(data.get("dense_fallback_backend", "triton")),
            working_set_budget_tokens=(
                int(data["working_set_budget_tokens"])
                if data.get("working_set_budget_tokens") is not None
                else None
            ),
            enable_host_backup_on_evict=bool(data.get("enable_host_backup_on_evict", False)),
            enable_physical_eviction=bool(data.get("enable_physical_eviction", False)),
        )

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
    physical_eviction_interval: int = 1
    physical_eviction_slack_tokens: int = 0
    validate_kv_cache: bool = False
    debug_timing: bool = False
    debug_timing_output_file: str | None = None
    enable_lookahead_prefetch: bool = True
    profiler_config: dict[str, Any] | None = None
    chunked_cpu_store: str = "auto"
    chunk_size: int = 16
    working_set_layout: str = "auto"
    resident_only_gpu_kv: bool = False
    prefill_layerwise_offload: bool = False

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
        profiler_config = data.get("profiler")
        if isinstance(profiler_config, bool):
            profiler_config = {"enabled": profiler_config}
        elif profiler_config is not None and not isinstance(profiler_config, dict):
            logger.warning("Invalid sparse framework profiler config; disabling profiler.")
            profiler_config = None
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
            physical_eviction_interval=max(
                1, int(data.get("physical_eviction_interval", 1))
            ),
            physical_eviction_slack_tokens=max(
                0, int(data.get("physical_eviction_slack_tokens", 0))
            ),
            validate_kv_cache=bool(
                data.get("validate_kv_cache", data.get("debug_validate_kv_cache", False))
            ),
            debug_timing=bool(data.get("debug_timing", False)),
            debug_timing_output_file=(
                str(
                    data.get(
                        "debug_timing_output_file",
                        data.get("debug_timing_log_file"),
                    )
                )
                if data.get("debug_timing_output_file", data.get("debug_timing_log_file"))
                is not None
                else None
            ),
            enable_lookahead_prefetch=bool(
                data.get("enable_lookahead_prefetch", True)
            ),
            profiler_config=profiler_config,
            chunked_cpu_store=_normalize_chunked_cpu_store(
                data.get("chunked_cpu_store", data.get("enable_chunked_cpu_store", "auto"))
            ),
            chunk_size=max(1, int(data.get("chunk_size", 16))),
            working_set_layout=_normalize_working_set_layout(
                data.get("working_set_layout", "auto")
            ),
            resident_only_gpu_kv=bool(data.get("resident_only_gpu_kv", False)),
            prefill_layerwise_offload=bool(
                data.get(
                    "prefill_layerwise_offload",
                    data.get("prefill_layerwise_async_offload", False),
                )
            ),
        )


def _normalize_chunked_cpu_store(value: Any) -> str:
    if isinstance(value, bool):
        return "on" if value else "off"
    normalized = str(value).lower()
    if normalized in ("1", "true", "yes", "on", "enabled"):
        return "on"
    if normalized in ("0", "false", "no", "off", "disabled"):
        return "off"
    return "auto"


def _normalize_working_set_layout(value: Any) -> str:
    normalized = str(value).lower()
    if normalized in ("chunk", "chunked", "chunks"):
        return "chunk"
    if normalized in ("token", "tokens", "row", "rows"):
        return "token"
    return "auto"

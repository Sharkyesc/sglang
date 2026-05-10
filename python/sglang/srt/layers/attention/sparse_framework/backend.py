from __future__ import annotations

import logging
from typing import Optional

import torch

from sglang.srt.layers.attention.base_attn_backend import AttentionBackend
from sglang.srt.layers.attention.sparse_framework.compiler import PlanCompiler
from sglang.srt.layers.attention.sparse_framework.config import SparseFrameworkConfig
from sglang.srt.layers.attention.sparse_framework.eviction_tracker import (
    get_eviction_tracker,
)
from sglang.srt.layers.attention.sparse_framework.kv_store import get_cpu_kv_store
from sglang.srt.layers.attention.sparse_framework.ops.ensure_resident import (
    ensure_full_kv_resident,
)
from sglang.srt.layers.attention.sparse_framework.ops.utils import (
    configure_cpu_kv_store_from_state,
)
from sglang.srt.layers.attention.sparse_framework.profiler import (
    SparseFrameworkProfiler,
    SparseProfilerConfig,
)
from sglang.srt.layers.attention.sparse_framework.residency import get_residency_table
from sglang.srt.layers.attention.sparse_framework.runtime_context import (
    SparseRuntimeContext,
)

logger = logging.getLogger(__name__)


class SparseFrameworkAttnBackend(AttentionBackend):
    """Wrapper for KV selection based sparse attention.

    This backend parses and compiles user KV selection semantics, but delegates
    actual attention execution to a dense fallback backend in the initial phases.
    """

    def __init__(self, model_runner):
        super().__init__()
        self.model_runner = model_runner
        self.config = SparseFrameworkConfig.from_server_args(model_runner.server_args)
        self.compiler = PlanCompiler(self.config)
        self.fallback = self._create_dense_backend(self.config.dense_fallback_backend)
        self.host_pool = None
        self.tree_cache = None
        self.cache_controller = None
        self.host_io_backend: Optional[str] = None
        self.last_execution_plan = None
        self.last_selection_state: dict | None = None
        self.framework_state: dict = {}
        self.profiler = SparseFrameworkProfiler(
            SparseProfilerConfig.from_dict(self.config.profiler_config)
        )
        self._logged_init = False
        self._logged_plan_signatures: set[tuple] = set()
        self._runtime_log_count = 0
        self._runtime_log_limit_per_request = 12
        self._log_init_once()

    def init_forward_metadata(self, forward_batch):
        is_decode = getattr(forward_batch.forward_mode, "is_decode", None)
        if not callable(is_decode) or not is_decode():
            self.last_execution_plan = None
            self._runtime_log_count = 0
            self._drop_sparse_request_state(forward_batch)
            return self.fallback.init_forward_metadata(forward_batch)

        ctx = SparseRuntimeContext.from_batch(
            self.model_runner, forward_batch, host_pool=self.host_pool
        )
        self.last_execution_plan = self.compiler.compile(ctx)
        self._log_plan_once(forward_batch, self.last_execution_plan)
        return self.fallback.init_forward_metadata(forward_batch)

    def bind_kv_store_host_pool(
        self,
        host_pool,
        io_backend: Optional[str] = None,
        tree_cache=None,
    ):
        self.host_pool = host_pool
        self.tree_cache = tree_cache
        self.cache_controller = getattr(tree_cache, "cache_controller", None)
        self.host_io_backend = io_backend
        self.framework_state["tree_cache"] = tree_cache
        if hasattr(self.fallback, "bind_kv_store_host_pool"):
            self.fallback.bind_kv_store_host_pool(
                host_pool, io_backend=io_backend, tree_cache=tree_cache
            )

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

    def update_verify_buffers_to_fill_after_draft(
        self, spec_info, cuda_graph_bs: Optional[int]
    ):
        return self.fallback.update_verify_buffers_to_fill_after_draft(
            spec_info, cuda_graph_bs
        )

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
        state = self._run_plan_ops(
            layer,
            forward_batch,
            q=q,
            k=k,
            v=v,
            save_kv_cache=save_kv_cache,
            kwargs=kwargs,
        )
        if state.get("attention_output") is not None:
            self._log_runtime_path("decode", layer, forward_batch, state, "subset")
            return state["attention_output"]
        self._ensure_dense_fallback_ready(layer, forward_batch, state)
        self._log_runtime_path("decode", layer, forward_batch, state, "dense_fallback")
        return self.fallback.forward_decode(
            q, k, v, layer, forward_batch, save_kv_cache=save_kv_cache, **kwargs
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
        ctx = SparseRuntimeContext.from_batch(
            self.model_runner,
            forward_batch,
            layer=layer,
            host_pool=self.host_pool,
            cache_controller=self.cache_controller,
            query=q,
            key=k,
            value=v,
            save_kv_cache=save_kv_cache,
            kwargs=kwargs,
            framework_state=self.framework_state,
        )
        plan = self.compiler.compile(ctx)
        self._configure_cpu_store_for_plan(plan)
        output = self.fallback.forward_extend(
            q, k, v, layer, forward_batch, save_kv_cache=save_kv_cache, **kwargs
        )
        if (
            bool(getattr(plan, "enable_host_backup_on_evict", False))
            or bool(getattr(plan, "enable_physical_eviction", False))
        ):
            get_cpu_kv_store(self.framework_state)
            self._configure_cpu_store_for_plan(plan)
            state = {"execution_plan": plan}
            state["extend_store_result"] = self._store_extend_kv_to_cpu(
                layer,
                forward_batch,
                k,
                v,
                save_kv_cache=save_kv_cache,
            )
            self._log_runtime_path("extend", layer, forward_batch, state, "dense_store")
        return output

    def forward_mixed(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        layer,
        forward_batch,
        save_kv_cache: bool = True,
        **kwargs,
    ):
        return self.fallback.forward_mixed(
            q, k, v, layer, forward_batch, save_kv_cache=save_kv_cache, **kwargs
        )

    def support_triton(self):
        return self.fallback.support_triton()

    def get_indexer_metadata(self, layer_id: int, forward_batch):
        return self.fallback.get_indexer_metadata(layer_id, forward_batch)

    def _run_plan_ops(
        self,
        layer,
        forward_batch,
        *,
        q: torch.Tensor | None = None,
        k: torch.Tensor | None = None,
        v: torch.Tensor | None = None,
        save_kv_cache: bool = True,
        kwargs: dict | None = None,
    ) -> dict:
        ctx = SparseRuntimeContext.from_batch(
            self.model_runner,
            forward_batch,
            layer=layer,
            host_pool=self.host_pool,
            cache_controller=self.cache_controller,
            query=q,
            key=k,
            value=v,
            save_kv_cache=save_kv_cache,
            kwargs=kwargs,
            framework_state=self.framework_state,
        )
        plan = self.last_execution_plan or self.compiler.compile(ctx)
        self._configure_cpu_store_for_plan(plan)
        state = {"execution_plan": plan}
        for op in plan.ops:
            op_name = type(op).__name__
            with self.profiler.record(f"sparse_framework/{op_name}"):
                op.run(ctx, state)
        self.profiler.step()
        if self.profiler.enabled:
            state["profiler"] = self.profiler.state()
        self.last_selection_state = state
        return state

    def _drop_sparse_request_state(self, forward_batch) -> None:
        req_pool_indices = getattr(forward_batch, "req_pool_indices", None)
        if req_pool_indices is None:
            return
        req_pool_indices = [int(x) for x in req_pool_indices.detach().cpu().tolist()]
        if not req_pool_indices:
            return

        store = self.framework_state.get("cpu_kv_store")
        working_set = self.framework_state.get("working_set_buffer")
        table = get_residency_table(self.framework_state)
        tracker = get_eviction_tracker(self.framework_state)
        for req_pool_idx in req_pool_indices:
            if store is not None:
                store.drop_request(req_pool_idx)
            if working_set is not None:
                drop_request = getattr(working_set, "drop_request", None)
                if callable(drop_request):
                    drop_request(req_pool_idx)
            table.drop_request(req_pool_idx)
            tracker.drop_request(req_pool_idx)

    def _log_init_once(self) -> None:
        if self._logged_init:
            return
        self._logged_init = True
        logger.info(
            "Sparse framework backend initialized: fallback=%s selection=%s combine=%s "
            "working_set_budget_tokens=%s host_backup_on_evict=%s physical_eviction=%s "
            "profiler=%s",
            self.config.dense_fallback_backend,
            self.config.selection,
            self.config.combine,
            self.config.working_set_budget_tokens,
            self.config.enable_host_backup_on_evict,
            self.config.enable_physical_eviction,
            self.profiler.enabled,
        )

    def _log_plan_once(self, forward_batch, plan) -> None:
        op_names = tuple(type(op).__name__ for op in plan.ops)
        signature = (
            getattr(forward_batch.forward_mode, "name", str(forward_batch.forward_mode)),
            op_names,
            plan.granularity,
            plan.placement,
            plan.cache_policy,
            plan.fetch_policy,
        )
        if signature in self._logged_plan_signatures:
            return
        self._logged_plan_signatures.add(signature)
        logger.info(
            "Sparse framework plan: mode=%s ops=%s fallback_backend=%s granularity=%s "
            "placement=%s cache_policy=%s fetch_policy=%s chunked_working_set=%s "
            "uses_dense_fallback=%s",
            signature[0],
            "->".join(op_names),
            plan.fallback_backend,
            plan.granularity,
            plan.placement,
            plan.cache_policy,
            plan.fetch_policy,
            getattr(plan, "use_chunked_working_set", False),
            plan.uses_dense_fallback,
        )

    def _configure_cpu_store_for_plan(self, plan) -> None:
        if self.framework_state is None:
            return
        store = self.framework_state.get("cpu_kv_store")
        if store is None:
            return
        configure = getattr(store, "configure_chunking", None)
        if callable(configure):
            configure(
                enabled=bool(getattr(plan, "use_chunked_cpu_store", False)),
                chunk_size=int(getattr(plan, "chunk_size", 16)),
            )

    def _log_runtime_path(
        self,
        phase: str,
        layer,
        forward_batch,
        state: dict,
        path: str,
    ) -> None:
        layer_id = getattr(layer, "layer_id", None)
        evicted = int((state.get("evict_result") or {}).get("evicted", 0) or 0)
        if layer_id not in (None, 0) and evicted <= 0:
            return
        if phase == "extend":
            self._runtime_log_count = 0
        if self._runtime_log_count >= self._runtime_log_limit_per_request:
            return
        self._runtime_log_count += 1

        selected_kv_indices = state.get("selected_kv_indices") or []
        selected_counts = [int(indices.numel()) for indices in selected_kv_indices[:4]]
        if len(selected_kv_indices) > 4:
            selected_counts.append(-1)
        contributions = state.get("selection_contributions") or []
        contribution_preview = contributions[:2]

        logger.info(
            "Sparse framework runtime: phase=%s layer=%s path=%s attend_mode=%s "
            "fallback_reason=%s subset_unavailable_reason=%s selected_kv_counts=%s "
            "selection_contributions=%s cache_result=%s fetch_result=%s "
            "lookahead_prefetch_result=%s evict_result=%s working_set_result=%s "
            "triton_torch_output_check=%s subset_attention_kernel_replaced=%s "
            "current_decode_rewrite=%s subset_tensor_debug=%s extend_store_result=%s "
            "cpu_kv_store=%s forward_mode=%s profiler=%s",
            phase,
            layer_id,
            path,
            state.get("attend_mode"),
            state.get("fallback_reason"),
            state.get("subset_unavailable_reason"),
            selected_counts,
            contribution_preview,
            state.get("cache_result"),
            state.get("fetch_result"),
            state.get("lookahead_prefetch_result"),
            state.get("evict_result"),
            state.get("working_set_result"),
            state.get("triton_torch_output_check"),
            state.get("subset_attention_kernel_replaced"),
            state.get("current_decode_rewrite"),
            state.get("subset_tensor_debug"),
            state.get("extend_store_result"),
            self._cpu_kv_store_stats(),
            getattr(forward_batch.forward_mode, "name", str(forward_batch.forward_mode)),
            state.get("profiler"),
        )

    def _store_extend_kv_to_cpu(
        self,
        layer,
        forward_batch,
        key: torch.Tensor,
        value: torch.Tensor,
        *,
        save_kv_cache: bool,
    ) -> dict:
        if not save_kv_cache:
            return {"enabled": False, "reason": "save_kv_cache_false"}
        store = get_cpu_kv_store(self.framework_state)
        if store is None:
            return {"enabled": False, "reason": "missing_store"}

        req_pool_indices = self._tensor_or_list_to_ints(
            getattr(forward_batch, "req_pool_indices", None)
        )
        seq_lens = self._tensor_or_list_to_ints(
            getattr(forward_batch, "seq_lens_cpu", None)
        )
        if not seq_lens:
            seq_lens = self._tensor_or_list_to_ints(getattr(forward_batch, "seq_lens", None))
        extend_lens = self._tensor_or_list_to_ints(
            getattr(forward_batch, "extend_seq_lens_cpu", None)
        )
        if not extend_lens:
            extend_lens = self._tensor_or_list_to_ints(
                getattr(forward_batch, "extend_seq_lens", None)
            )

        if not req_pool_indices or not seq_lens:
            return {"enabled": False, "reason": "missing_batch_metadata"}

        key_view = key.reshape(-1, layer.tp_k_head_num, layer.qk_head_dim)
        value_view = value.reshape(-1, layer.tp_k_head_num, layer.v_head_dim)
        if not extend_lens:
            if len(req_pool_indices) == 1:
                extend_lens = [int(key_view.shape[0])]
            else:
                return {"enabled": False, "reason": "missing_extend_lens"}

        rows = int(key_view.shape[0])
        offset = 0
        written = 0
        requests = 0
        dropped_request_layers = 0
        dropped_req_pool_indices = set()
        truncated = False
        for batch_idx, req_pool_idx in enumerate(req_pool_indices):
            if batch_idx >= len(seq_lens) or batch_idx >= len(extend_lens):
                truncated = True
                break
            extend_len = max(0, int(extend_lens[batch_idx]))
            if extend_len == 0:
                continue
            next_offset = min(rows, offset + extend_len)
            actual_len = next_offset - offset
            if actual_len <= 0:
                truncated = True
                break
            seq_len = int(seq_lens[batch_idx])
            start_pos = max(0, seq_len - extend_len)
            if (
                int(layer.layer_id) == 0
                and int(req_pool_idx) not in dropped_req_pool_indices
            ):
                dropped_request_layers += store.drop_request(int(req_pool_idx))
                dropped_req_pool_indices.add(int(req_pool_idx))
            positions = list(range(start_pos, start_pos + actual_len))
            written += store.put(
                req_pool_idx=int(req_pool_idx),
                layer_id=int(layer.layer_id),
                positions=positions,
                keys=key_view[offset:next_offset],
                values=value_view[offset:next_offset],
            )
            requests += 1
            offset = next_offset
            if offset >= rows and batch_idx + 1 < len(req_pool_indices):
                truncated = True
                break

        return {
            "enabled": True,
            "requests": requests,
            "written": written,
            "rows": rows,
            "dropped_request_layers": dropped_request_layers,
            "truncated": truncated or offset != rows,
        }

    def _tensor_or_list_to_ints(self, value) -> list[int]:
        if value is None:
            return []
        if isinstance(value, torch.Tensor):
            return [int(x) for x in value.detach().cpu().tolist()]
        return [int(x) for x in value]

    def _cpu_kv_store_stats(self) -> dict | None:
        store = self.framework_state.get("cpu_kv_store")
        if store is None:
            return None
        stats = getattr(store, "stats", None)
        return stats() if callable(stats) else None

    def _ensure_dense_fallback_ready(self, layer, forward_batch, state: dict) -> None:
        ctx = SparseRuntimeContext.from_batch(
            self.model_runner,
            forward_batch,
            layer=layer,
            host_pool=self.host_pool,
            cache_controller=self.cache_controller,
            framework_state=self.framework_state,
        )
        ensure_result = ensure_full_kv_resident(ctx, state)
        if ensure_result.get("fetched", 0) > 0:
            self.fallback.init_forward_metadata(forward_batch)

    def _create_dense_backend(self, name: str) -> AttentionBackend:
        if name == "triton":
            from sglang.srt.layers.attention.triton_backend import TritonAttnBackend

            return TritonAttnBackend(self.model_runner)
        if name == "flashinfer":
            from sglang.srt.layers.attention.flashinfer_backend import FlashInferAttnBackend

            return FlashInferAttnBackend(
                self.model_runner,
                init_new_workspace=getattr(self.model_runner, "init_new_workspace", False),
            )
        if name == "fa3":
            from sglang.srt.layers.attention.flashattention_backend import (
                FlashAttentionBackend,
            )

            return FlashAttentionBackend(self.model_runner)
        raise ValueError(f"Unsupported sparse framework dense fallback backend: {name}")

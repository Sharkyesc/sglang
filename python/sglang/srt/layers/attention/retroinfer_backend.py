from __future__ import annotations

import logging
from typing import Optional

import torch

from sglang.srt.layers.attention.base_attn_backend import AttentionBackend
from sglang.srt.layers.attention.retroinfer import (
    RetroInferBatchPlanner,
    RetroInferCapabilityChecker,
    RetroInferCpuStore,
    RetroInferExecutionEngine,
    RetroInferGpuRuntime,
    RetroInferHiCacheKVStore,
    RetroInferSessionManager,
    SGLangRetroInferKVSource,
)
from sglang.srt.layers.attention.triton_backend import TritonAttnBackend

logger = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)


class RetroInferAttnBackend(AttentionBackend):
    """
    RetroInfer integration built on top of SGLang's own KV lifecycle.

    Design goals for this backend:
    - SGLang's req_to_token/token_to_kv_pool remain the source of truth.
    - Extend/prefill stays on the fallback backend and only updates RetroInfer state.
    - Decode enters RetroInfer only after a session is explicitly planned and prepared
      from SGLang KV.
    - Session/request state is managed outside the backend entrypoints, which keeps the
      backend thin and avoids hidden batch-size keyed state reuse.
    """

    def __init__(
        self,
        model_runner: ModelRunner,
    ):
        super().__init__()

        self.model_runner = model_runner
        self.fallback = TritonAttnBackend(model_runner)

        self.capability_checker = RetroInferCapabilityChecker(model_runner)
        self.kv_store = RetroInferHiCacheKVStore(model_runner)
        self.cpu_store = RetroInferCpuStore(kv_store=self.kv_store)
        self.gpu_runtime = RetroInferGpuRuntime()
        self.gpu_runtime.bind_cache_manager(
            token_to_kv_pool_allocator=model_runner.token_to_kv_pool_allocator,
            req_to_token_pool=model_runner.req_to_token_pool,
        )
        self.kv_source = SGLangRetroInferKVSource(model_runner)
        self.execution_engine = RetroInferExecutionEngine(
            model_runner,
            self.kv_source,
            self.cpu_store,
            self.gpu_runtime,
        )
        self.session_manager = RetroInferSessionManager(
            on_drop_session=self.execution_engine.drop_session,
            on_drop_request=self.cpu_store.drop_request,
        )
        self.batch_planner = RetroInferBatchPlanner(
            self.capability_checker,
            self.session_manager,
        )

    def init_forward_metadata(self, forward_batch):
        return self.fallback.init_forward_metadata(forward_batch)

    def bind_kv_store_host_pool(
        self,
        host_pool,
        io_backend: Optional[str] = None,
        tree_cache=None,
    ):
        self.kv_store.bind_host_pool(host_pool, io_backend=io_backend)
        self.gpu_runtime.bind_cache_manager(
            tree_cache=tree_cache,
            host_pool=host_pool,
            io_backend=io_backend,
            token_to_kv_pool_allocator=self.model_runner.token_to_kv_pool_allocator,
            req_to_token_pool=self.model_runner.req_to_token_pool,
        )
        logger.info(
            "RetroInfer: bound host KV pool for CPU-resident indexing and staging (io_backend=%s).",
            self.kv_store.io_backend,
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

    def update_verify_buffers_to_fill_after_draft(self, spec_info, cuda_graph_bs: Optional[int]):
        return self.fallback.update_verify_buffers_to_fill_after_draft(spec_info, cuda_graph_bs)

    def forward_extend(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        layer,
        forward_batch,
        save_kv_cache: bool = True,
        sinks=None,
    ):
        out = self.fallback.forward_extend(
            q, k, v, layer, forward_batch, save_kv_cache=save_kv_cache, sinks=sinks
        )
        plan = self.batch_planner.plan_extend(layer, forward_batch)
        if plan.mode == "fallback" and layer.layer_id == 0 and plan.reason:
            logger.debug("RetroInfer extend observe skipped: %s", plan.reason)
        return out

    def forward_decode(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        layer: RadixAttention,
        forward_batch: ForwardBatch,
        save_kv_cache: bool = True,
        sinks=None,
    ):
        plan = self.batch_planner.plan_decode(layer, forward_batch)
        if plan.mode == "fallback" or plan.session is None:
            if plan.reason and layer.layer_id == 0:
                logger.debug("RetroInfer decode fallback: %s", plan.reason)
            return self.fallback.forward_decode(
                q, k, v, layer, forward_batch, save_kv_cache=save_kv_cache, sinks=sinks
            )

        session = plan.session
        try:
            prepared = self.execution_engine.ensure_session_prepared(
                session,
                forward_batch,
                layer,
            )
            if not prepared:
                return self.fallback.forward_decode(
                    q,
                    k,
                    v,
                    layer,
                    forward_batch,
                    save_kv_cache=save_kv_cache,
                    sinks=sinks,
                )
            output = self.execution_engine.run_decode(
                session,
                q,
                k,
                v,
                layer,
                forward_batch,
                save_kv_cache=save_kv_cache,
            )
            if layer.layer_id == self.model_runner.model_config.num_hidden_layers - 1:
                self.session_manager.mark_decode_advanced(session, plan.seq_len + 1)
            return output
        except Exception as exc:
            logger.warning(
                "RetroInfer fallback to Triton: decode execution failed (%s: %s)",
                type(exc).__name__,
                exc,
            )
            return self.fallback.forward_decode(
                q, k, v, layer, forward_batch, save_kv_cache=save_kv_cache, sinks=sinks
            )

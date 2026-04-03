from __future__ import annotations

import logging
import os
import sys
from pathlib import Path

import torch


logger = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)

class RetroInferExecutionEngine:
    def __init__(
        self, 
        model_runner, 
        kv_source, 
        cpu_store, 
        gpu_runtime
    ):
        self.model_runner = model_runner
        self.kv_source = kv_source
        self.cpu_store = cpu_store
        self.gpu_runtime = gpu_runtime

        self.retrieval_budget = float(os.getenv("SGLANG_RETROINFER_RETRIEVAL_BUDGET", "0.02"))
        self.estimation_budget = float(os.getenv("SGLANG_RETROINFER_ESTIMATION_BUDGET", "0.2"))
        self.n_centroids = int(os.getenv("SGLANG_RETROINFER_N_CENTROIDS", "64"))
        self.n_segment = int(os.getenv("SGLANG_RETROINFER_N_SEGMENT", "8"))
        self.pages_per_cluster = int(os.getenv("SGLANG_RETROINFER_PAGES_PER_CLUSTER", "16"))
        self.buffer_cluster_num = int(os.getenv("SGLANG_RETROINFER_BUFFER_CLUSTER_NUM", "64"))
        self.cache_ratio = float(os.getenv("SGLANG_RETROINFER_CACHE_RATIO", "0.0"))
        self.static_pattern_start = int(os.getenv("SGLANG_RETROINFER_STATIC_START", "16"))
        self.static_pattern_end = int(os.getenv("SGLANG_RETROINFER_STATIC_END", "16"))
        self.cpu_core_num = int(os.getenv("SGLANG_RETROINFER_CPU_CORES", "8"))
        self.prefill_bsz = int(os.getenv("SGLANG_RETROINFER_PREFILL_BSZ", "4"))
        self.max_new_length_hint = int(os.getenv("SGLANG_RETROINFER_MAX_NEW_LENGTH", "4096"))
        self.debug = os.getenv("SGLANG_RETROINFER_DEBUG", "").lower() in ("1", "true", "yes")

        self._retro_available = False
        self._warned_unavailable = False
        self._retro_import_logged = False
        self._retro_build_seg_logged = False

    def ensure_import(self) -> bool:
        if self._retro_available:
            return True

        candidate_paths = []
        env_path = os.getenv("SGLANG_RETROINFER_PATH")
        if env_path:
            candidate_paths.append(env_path)
        candidate_paths.append("/home/yy/Desktop/RetrievalAttention")

        for path in candidate_paths:
            if path and Path(path).exists() and path not in sys.path:
                sys.path.append(path)
            try:
                from cache_hub.retroinfer_cache import retroinfer_cache  # noqa: F401

                self._retro_available = True
                if not self._retro_import_logged:
                    logger.info("RetroInfer: RetrievalAttention import OK (sys.path += %s)", path)
                    self._retro_import_logged = True
                return True
            except Exception:
                continue

        if not self._warned_unavailable:
            logger.warning(
                "RetroInfer dependencies not found. Set SGLANG_RETROINFER_PATH or install RetrievalAttention kernels. Falling back to Triton."
            )
            self._warned_unavailable = True
        return False

    def _retro_effective_max_length(self) -> int:
        cfg_len = int(self.model_runner.model_config.context_len)
        pool = getattr(self.model_runner, "max_total_num_tokens", None)
        env_cap = int(os.getenv("SGLANG_RETROINFER_MAX_SEQ_LEN", "0"))
        candidates = [cfg_len]
        if pool is not None and pool > 0:
            candidates.append(int(pool))
        if env_cap > 0:
            candidates.append(env_cap)
        return max(min(candidates), 128)

    def _retro_max_new_length(self, max_length: int) -> int:
        pool = getattr(self.model_runner, "max_total_num_tokens", None)
        pool = int(max_length if pool is None or pool <= 0 else pool)
        max_prefill = min(int(max_length), pool)
        sa = getattr(self.model_runner, "server_args", None)
        if sa is not None and getattr(sa, "max_prefill_tokens", None) is not None:
            max_prefill = min(max_prefill, int(sa.max_prefill_tokens))
        steady_min = int(os.getenv("SGLANG_RETROINFER_STEADY_MIN_MAX_NEW", "128"))
        steady_min = max(2, min(steady_min, max(2, int(max_length) - 1)))
        upper_by_prefill = max(2, int(max_length) - max_prefill)
        return max(min(int(self.max_new_length_hint), upper_by_prefill), steady_min)

    def build_cache(self, batch_size: int):
        if not self.ensure_import():
            return None

        sl_build = os.getenv("SGLANG_RETROINFER_BUILD_SEGMENT")
        if sl_build:
            os.environ["RETROINFER_BUILD_SEGMENT"] = sl_build.strip()
            if not self._retro_build_seg_logged:
                logger.warning(
                    "RetroInfer: RETROINFER_BUILD_SEGMENT=%s (dev/smoke only; unset for RA defaults).",
                    os.environ["RETROINFER_BUILD_SEGMENT"],
                )
                self._retro_build_seg_logged = True

        from cache_hub.retroinfer_cache import retroinfer_cache

        layer_num = self.model_runner.model_config.num_hidden_layers
        num_kv_heads = self.model_runner.model_config.get_num_kv_heads(1)
        num_heads = self.model_runner.model_config.num_attention_heads
        head_dim = self.model_runner.model_config.head_dim
        max_length = self._retro_effective_max_length()
        max_new_length = self._retro_max_new_length(max_length)

        device_str = str(self.model_runner.device)
        if device_str == "cuda":
            device_str = f"cuda:{getattr(self.model_runner, 'gpu_id', 0)}"
        layer_mapping = {str(i): device_str for i in range(layer_num)}

        sa = getattr(self.model_runner, "server_args", None)
        use_cuda_graph = sa is not None and not getattr(sa, "disable_cuda_graph", False)

        return retroinfer_cache(
            valid_start=[0 for _ in range(batch_size)],
            layer_num=layer_num,
            batch_size=batch_size,
            max_length=max_length,
            num_key_value_heads=num_kv_heads,
            num_heads=num_heads,
            head_dim=head_dim,
            dtype=self.model_runner.dtype,
            layer_mapping=layer_mapping,
            max_new_length=max_new_length,
            static_pattern_start=self.static_pattern_start,
            static_pattern_end=self.static_pattern_end,
            core=self.cpu_core_num,
            n_centroids=self.n_centroids,
            n_segment=self.n_segment,
            pages_per_cluster=self.pages_per_cluster,
            retrieval_budget=self.retrieval_budget,
            estimation_budget=self.estimation_budget,
            cache_ratio=self.cache_ratio,
            buffer_cluster_num=self.buffer_cluster_num,
            use_cuda_graph=use_cuda_graph,
            prefill_bsz=self.prefill_bsz,
            num_gpus=int(os.getenv("SGLANG_RETROINFER_NUM_GPUS", "1")),
            model_size=float(os.getenv("SGLANG_RETROINFER_MODEL_SIZE_GB", "16")),
        )

    def _min_seq_len_for_index(self, cache) -> int:
        valid_start = int(cache.valid_start_list[0])
        static_total = int(cache.static_pattern_total)
        n_segment = int(getattr(cache, "n_segment", 1))
        return valid_start + static_total + max(1, n_segment)

    def ensure_session_prepared(self, session, forward_batch, layer) -> bool:
        if session.prepared and session.retro_cache is not None:
            return True

        cache = session.retro_cache or self.build_cache(session.batch_size)
        if cache is None:
            return False

        kv_len = int(torch.min(forward_batch.seq_lens).item()) - 1
        if kv_len <= 0 or kv_len < self._min_seq_len_for_index(cache):
            return False

        q_heads = layer.tp_q_head_num
        q_dim = layer.qk_head_dim
        kv_heads = layer.tp_k_head_num
        v_dim = layer.v_head_dim
        n_layers = self.model_runner.model_config.num_hidden_layers

        torch.cuda.synchronize()
        try:
            for layer_id in range(n_layers):
                keys, values = self.kv_source.gather_batch_layer_tensors(
                    forward_batch, layer_id, kv_len
                )
                keys = keys.view(session.batch_size, kv_len, kv_heads, q_dim)
                values = values.view(session.batch_size, kv_len, kv_heads, v_dim)
                q_dummy = keys.new_zeros((session.batch_size, kv_len, q_heads, q_dim))
                cache.prefill_update_kv_cache(q_dummy, keys, values, layer_id, start_bdx=0)
                cache.sync(layer_id, start_bdx=0)
            cache.prepare_cache()
        except Exception as exc:
            logger.warning(
                "RetroInfer fallback to Triton: build from SGLang KV failed (%s: %s)",
                type(exc).__name__,
                exc,
            )
            return False

        session.retro_cache = cache
        session.mark_prepared(kv_len)
        for req_pool_idx in session.key:
            self.cpu_store.mark_index_ready(req_pool_idx, kv_len)
        self.gpu_runtime.bind_session(session.key)
        logger.info(
            "RetroInfer: prepared session %s from SGLang KV (batch_size=%s kv_len=%s).",
            session.key,
            session.batch_size,
            kv_len,
        )
        return True

    def run_decode(self, session, q, k, v, layer, forward_batch, save_kv_cache: bool = True):
        cache = session.retro_cache
        assert cache is not None, "run_decode requires a prepared RetroInfer session"

        qv = q.view(forward_batch.batch_size, 1, layer.tp_q_head_num, layer.qk_head_dim)
        kv = k.view(forward_batch.batch_size, 1, layer.tp_k_head_num, layer.qk_head_dim)
        vv = v.view(forward_batch.batch_size, 1, layer.tp_k_head_num, layer.v_head_dim)
        n_layers = self.model_runner.model_config.num_hidden_layers
        layer_id = layer.layer_id

        if save_kv_cache:
            forward_batch.token_to_kv_pool.set_kv_buffer(layer, forward_batch.out_cache_loc, k, v)

        cache.decode_update_kv_cache(kv, vv, layer_id)
        static_len = cache.static_pattern_total if layer_id == n_layers - 1 else cache.static_pattern_total + 1
        output = cache.attn_func(qv, layer_id, static_len)
        return output.view(forward_batch.batch_size, -1)

from __future__ import annotations

import logging
import os
import sys
from pathlib import Path
from typing import Optional

import torch

from sglang.srt.layers.attention.base_attn_backend import AttentionBackend
from sglang.srt.layers.attention.triton_backend import TritonAttnBackend

logger = logging.getLogger(__name__)


class RetroInferAttnBackend(AttentionBackend):
    """
    Adapter backend that integrates RetrievalAttention(RetroInfer) into SGLang.

    Current strategy:
    - Use Triton backend as default/fallback path.
    - Use RetrievalAttention ``cache.attn_func`` for decode (``dense_attention`` or
      ``sparse_attention_gpu`` per RA's own thresholds).

    Sparse smoke test (short KV pool): RA only builds a sparse index when
    ``input_length = max_length - max_new_length >= static_pattern + BUILD_SEGMENT``
    (default ``BUILD_SEGMENT`` is 16384). With ``--max-total-tokens 8192`` that
    inequality never holds unless you lower the segment, e.g.::

        export SGLANG_RETROINFER_BUILD_SEGMENT=512

    Then prefill with at least ~512+ tokens of real text and watch logs for
    ``sparse_attention_gpu`` (and non-zero ``n_centroids`` at cache creation).

    **Chunked prefill**: when ``build_index_when_prefilling`` is true, the RA mirror is
    built once on the **first decode** step by gathering the full prefill KV from
    SGLang's pool (so sparse decode matches Triton KV). Set
    ``SGLANG_RETROINFER_SYNC_PREFILL_MIRROR=1`` to restore the old per-extend mirror
    (broken under chunked prefill unless you use a single extend).

    **VRAM vs decode depth**: ``max_new_length`` has a floor (``SGLANG_RETROINFER_STEADY_MIN_MAX_NEW``,
    default ``128``). A value ``>=1026`` matches RA's full steady-zone path (thousands of
    tokens before index maintenance) but **allocates much larger GPU tensors** and often
    **CUDA OOM** on ~16GB cards with a 4B model + large KV pool. Raise it only after
    lowering ``--mem-fraction-static`` / ``--max-total-tokens`` or similar to leave headroom.
    """

    def __init__(self, model_runner):
        super().__init__()
        self.model_runner = model_runner
        self.fallback = TritonAttnBackend(model_runner)

        self._retro_cache = None
        self._retro_cache_bs = None
        self._retro_prepared = False
        self._retro_available = False
        self._warned_unavailable = False
        self._warned_fallback = False
        # Monotonic watermark for extend-mode seq_lens[0]; drop if it decreases (new req / slot reuse).
        self._retro_max_seq_len_seen: Optional[int] = None

        # Tunables. Keep conservative defaults for stable first integration.
        self.retrieval_budget = float(os.getenv("SGLANG_RETROINFER_RETRIEVAL_BUDGET", "0.02"))
        self.estimation_budget = float(os.getenv("SGLANG_RETROINFER_ESTIMATION_BUDGET", "0.2"))
        self.n_centroids = int(os.getenv("SGLANG_RETROINFER_N_CENTROIDS", "64"))
        self.n_segment = int(os.getenv("SGLANG_RETROINFER_N_SEGMENT", "8"))
        self.pages_per_cluster = int(os.getenv("SGLANG_RETROINFER_PAGES_PER_CLUSTER", "16"))
        self.buffer_cluster_num = int(os.getenv("SGLANG_RETROINFER_BUFFER_CLUSTER_NUM", "64"))
        self.static_pattern_start = int(os.getenv("SGLANG_RETROINFER_STATIC_START", "16"))
        self.static_pattern_end = int(os.getenv("SGLANG_RETROINFER_STATIC_END", "16"))
        self.cpu_core_num = int(os.getenv("SGLANG_RETROINFER_CPU_CORES", "8"))
        self.prefill_bsz = int(os.getenv("SGLANG_RETROINFER_PREFILL_BSZ", "4"))
        self.max_new_length_hint = int(os.getenv("SGLANG_RETROINFER_MAX_NEW_LENGTH", "4096"))
        self._retro_debug = os.getenv("SGLANG_RETROINFER_DEBUG", "").lower() in (
            "1",
            "true",
            "yes",
        )
        self._retro_allow_chunked_sparse = os.getenv(
            "SGLANG_RETROINFER_ALLOW_CHUNKED_SPARSE_DECODE", ""
        ).lower() in ("1", "true", "yes")
        self._retro_sparse_decode_unsafe = False
        # When True (default): do not build RA from extend chunks; build from full KV pool on first decode.
        self._retro_defer_prefill_mirror = os.getenv(
            "SGLANG_RETROINFER_SYNC_PREFILL_MIRROR", ""
        ).lower() not in ("1", "true", "yes")
        self._retro_deferred_mirror_logged = False

        sa = getattr(model_runner, "server_args", None)
        if (
            sa is not None
            and not getattr(sa, "disable_cuda_graph", False)
            and str(getattr(model_runner, "device", "")).startswith("cuda")
        ):
            logger.warning(
                "RetroInfer: CUDA graph is on. Decode graphs are captured at startup before "
                "RetrievalAttention finishes prefill, so replayed decode usually runs the "
                "Triton path captured then — not cache.attn_func (dense/sparse). Few decode logs is expected. "
                "Use --disable-cuda-graph to exercise RetroInfer decode in Python and see decode logs."
            )

    # ===== metadata/cudagraph delegates =====
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
        return self.fallback.update_verify_buffers_to_fill_after_draft(spec_info, cuda_graph_bs)

    def _ensure_retro_import(self):
        if self._retro_available:
            return True

        candidate_paths = []
        env_path = os.getenv("SGLANG_RETROINFER_PATH")
        if env_path:
            candidate_paths.append(env_path)
        candidate_paths.append("/home/yy/Desktop/RetrievalAttention")

        for p in candidate_paths:
            if p and Path(p).exists() and p not in sys.path:
                sys.path.append(p)
            try:
                from cache_hub.retroinfer_cache_gpu import retroinfer_cache_gpu

                self._retro_available = True
                if not getattr(self, "_retro_import_logged", False):
                    logger.info(
                        "RetroInfer: RetrievalAttention import OK (sys.path += %s)",
                        p,
                    )
                    self._retro_import_logged = True
                return True
            except Exception:
                continue

        if not self._warned_unavailable:
            logger.warning(
                "RetroInfer dependencies not found. "
                "Set SGLANG_RETROINFER_PATH or install RetrievalAttention kernels. "
                "Falling back to Triton."
            )
            self._warned_unavailable = True
        return False

    def _retro_effective_max_length(self) -> int:
        """
        RetrievalAttention allocates CPU pinned buffers and GPU tensors proportional to
        max_length (via input_length = max_length - max_new_length). Using the model's
        full context_len (e.g. 262144) ignores SGLang's KV pool cap and can exhaust
        host RAM. Align with the actual token pool budget.
        """
        cfg_len = int(self.model_runner.model_config.context_len)
        pool = getattr(self.model_runner, "max_total_num_tokens", None)
        env_cap = int(os.getenv("SGLANG_RETROINFER_MAX_SEQ_LEN", "0"))
        candidates = [cfg_len]
        if pool is not None and pool > 0:
            candidates.append(int(pool))
        if env_cap > 0:
            candidates.append(env_cap)
        effective = min(candidates)
        if effective < cfg_len and not getattr(self, "_retro_len_logged", False):
            logger.info(
                "RetroInfer max_length capped to %s (model context_len=%s, KV pool=%s, env=%s)",
                effective,
                cfg_len,
                pool,
                env_cap if env_cap > 0 else "unset",
            )
            self._retro_len_logged = True
        return max(effective, 128)

    def _retro_max_new_length(self, max_length: int) -> int:
        pool = getattr(self.model_runner, "max_total_num_tokens", None)
        if pool is None or pool <= 0:
            pool = int(max_length)
        else:
            pool = int(pool)
        max_prefill = min(int(max_length), pool)
        sa = getattr(self.model_runner, "server_args", None)
        if sa is not None:
            mp = getattr(sa, "max_prefill_tokens", None)
            if mp is not None:
                max_prefill = min(max_prefill, int(mp))
        steady_min = int(os.getenv("SGLANG_RETROINFER_STEADY_MIN_MAX_NEW", "128"))
        steady_min = max(2, steady_min)
        steady_min = min(steady_min, max(2, int(max_length) - 1))

        upper_by_prefill = max(2, int(max_length) - max_prefill)
        out = min(int(self.max_new_length_hint), upper_by_prefill)
        out = max(out, steady_min)
        ra_input = int(max_length) - out

        if steady_min >= 1026 and not getattr(self, "_retro_steady_full_logged", False):
            logger.warning(
                "RetroInfer: SGLANG_RETROINFER_STEADY_MIN_MAX_NEW=%s (>=1026) uses RA's largest "
                "steady/list buffers; expect high VRAM use — reduce KV pool or mem fraction if OOM.",
                steady_min,
            )
            self._retro_steady_full_logged = True
        elif steady_min < 1026 and not getattr(self, "_retro_steady_default_logged", False):
            logger.info(
                "RetroInfer: steady max_new floor=%s (set SGLANG_RETROINFER_STEADY_MIN_MAX_NEW=1026 "
                "for RA's full multi-k-token decode path if you have spare VRAM).",
                steady_min,
            )
            self._retro_steady_default_logged = True
        if ra_input < max_prefill:
            logger.info(
                "RetroInfer: max_new_length=%s (steady floor=%s) → RA input_length=%s < max_prefill_cap=%s; "
                "prefill longer than input_length will fail mirror build.",
                out,
                steady_min,
                ra_input,
                max_prefill,
            )
        return out

    def _retro_min_seq_len_for_ra_index(self, cache) -> int:
        vs = int(cache.valid_start_list[0])
        static_total = int(cache.static_pattern_total)
        n_seg = int(getattr(cache, "n_segment", 1))
        return vs + static_total + max(1, n_seg)

    def _build_retro_cache(self, batch_size: int):
        if not self._ensure_retro_import():
            return None

        sl_build = os.getenv("SGLANG_RETROINFER_BUILD_SEGMENT")
        if sl_build:
            os.environ["RETROINFER_BUILD_SEGMENT"] = sl_build.strip()
            if not getattr(self, "_retro_build_seg_logged", False):
                logger.warning(
                    "RetroInfer: RETROINFER_BUILD_SEGMENT=%s (dev/smoke only; unset for RA defaults).",
                    os.environ["RETROINFER_BUILD_SEGMENT"],
                )
                self._retro_build_seg_logged = True

        from cache_hub.retroinfer_cache_gpu import retroinfer_cache_gpu

        layer_num = self.model_runner.model_config.num_hidden_layers
        num_kv_heads = self.model_runner.model_config.get_num_kv_heads(1)
        num_heads = self.model_runner.model_config.num_attention_heads
        head_dim = self.model_runner.model_config.head_dim
        max_length = self._retro_effective_max_length()

        device_str = str(self.model_runner.device)
        if device_str == "cuda":
            gpu_id = getattr(self.model_runner, "gpu_id", 0)
            device_str = f"cuda:{gpu_id}"
        layer_mapping = {str(i): device_str for i in range(layer_num)}
        valid_start = [0 for _ in range(batch_size)]

        model_size_gb = float(os.getenv("SGLANG_RETROINFER_MODEL_SIZE_GB", "16"))
        num_gpus = int(os.getenv("SGLANG_RETROINFER_NUM_GPUS", "1"))

        max_new_length = self._retro_max_new_length(max_length)

        cache = retroinfer_cache_gpu(
            valid_start=valid_start,
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
            buffer_cluster_num=self.buffer_cluster_num,
            prefill_bsz=self.prefill_bsz,
            num_gpus=num_gpus,
            model_size=model_size_gb,
        )
        logger.info(
            "RetroInfer: retroinfer_cache_gpu created (batch_size=%s max_length=%s max_new_length=%s "
            "num_layers=%s device=%s kv_heads=%s q_heads=%s head_dim=%s dtype=%s)",
            batch_size,
            max_length,
            max_new_length,
            layer_num,
            device_str,
            num_kv_heads,
            num_heads,
            head_dim,
            self.model_runner.dtype,
        )
        return cache

    def _ensure_retro_cache(self, batch_size: int):
        if self._retro_cache is not None and self._retro_cache_bs == batch_size:
            return self._retro_cache
        self._retro_cache = self._build_retro_cache(batch_size)
        self._retro_cache_bs = batch_size if self._retro_cache is not None else None
        self._retro_prepared = False
        return self._retro_cache

    def _warn_and_fallback(self, reason: str):
        if not self._warned_fallback:
            logger.warning("RetroInfer fallback to Triton: %s", reason)
            self._warned_fallback = True

    def _drop_retro_mirror(self) -> None:
        self._retro_cache = None
        self._retro_cache_bs = None
        self._retro_prepared = False
        self._retro_max_seq_len_seen = None
        self._retro_prefill_pass = 0
        self._retro_decode_logged = False
        self._retro_prepare_logged = False
        self._warned_fallback = False
        self._retro_sparse_decode_unsafe = False
        self._retro_deferred_mirror_logged = False
        self._retro_steady_full_logged = False
        self._retro_steady_default_logged = False

    def _maybe_reset_retro_mirror_for_new_session(self, forward_batch) -> None:
        if not forward_batch.forward_mode.is_extend():
            return
        cpu = forward_batch.seq_lens_cpu
        if not cpu:
            return
        try:
            cur = int(cpu[0])
        except (IndexError, TypeError, ValueError):
            return
        prev = self._retro_max_seq_len_seen
        if prev is not None and cur < prev:
            logger.info(
                "RetroInfer: dropping RA mirror (seq_lens %s→%s: new request or recycled slot; "
                "stale mirror caused k-means/copy shape errors).",
                prev,
                cur,
            )
            self._drop_retro_mirror()
        if self._retro_cache is None:
            self._retro_max_seq_len_seen = cur
        else:
            self._retro_max_seq_len_seen = max(
                self._retro_max_seq_len_seen if self._retro_max_seq_len_seen is not None else cur,
                cur,
            )

    def _build_retro_mirror_from_kv_pool(self, cache, forward_batch, layer) -> bool:
        if not getattr(cache, "build_index_when_prefilling", False):
            return False
        bs = forward_batch.batch_size
        if bs > int(getattr(cache, "prefill_bsz", bs)):
            self._warn_and_fallback("batch_size > RetroInfer prefill_bsz for deferred mirror")
            return False
        kv_len = int(torch.min(forward_batch.seq_lens).item()) - 1
        if kv_len <= 0:
            return False
        min_tokens = self._retro_min_seq_len_for_ra_index(cache)
        if kv_len < min_tokens:
            if self._retro_debug:
                logger.debug(
                    "RetroInfer: deferred mirror skipped (kv_len=%s < %s; need middle "
                    ">= n_segment for segment_k_means)",
                    kv_len,
                    min_tokens,
                )
            else:
                logger.info(
                    "RetroInfer: skip deferred mirror (kv_len=%s < %s); short context — "
                    "RA k-means needs at least n_segment=%s middle tokens after static pattern.",
                    kv_len,
                    min_tokens,
                    int(getattr(cache, "n_segment", 1)),
                )
            return False

        req_to_token = self.fallback.req_to_token
        pool = forward_batch.token_to_kv_pool
        n_layers = self.model_runner.model_config.num_hidden_layers
        q_heads = layer.tp_q_head_num
        q_dim = layer.qk_head_dim
        kv_heads = layer.tp_k_head_num
        vdim = layer.v_head_dim

        torch.cuda.synchronize()
        try:
            for lid in range(n_layers):
                k_buf = pool.get_key_buffer(lid)
                v_buf = pool.get_value_buffer(lid)
                keys = []
                vals = []
                for b in range(bs):
                    rpi = int(forward_batch.req_pool_indices[b].item())
                    loc = req_to_token[rpi, :kv_len].long()
                    keys.append(k_buf[loc])
                    vals.append(v_buf[loc])
                k_stacked = torch.stack(keys, dim=0).view(bs, kv_len, kv_heads, q_dim)
                v_stacked = torch.stack(vals, dim=0).view(bs, kv_len, kv_heads, vdim)
                q_dummy = k_stacked.new_zeros((bs, kv_len, q_heads, q_dim))
                cache.prefill_update_kv_cache(q_dummy, k_stacked, v_stacked, lid, start_bdx=0)
                cache.sync(lid, start_bdx=0)
            cache.prepare_cache()
        except Exception as e:
            self._warn_and_fallback(
                f"deferred RA mirror from KV pool failed ({type(e).__name__}: {e})"
            )
            return False

        self._retro_prepared = True
        self._retro_sparse_decode_unsafe = False
        logger.info(
            "RetroInfer: built RA mirror from KV pool (kv_len=%s, layers=%s); decode uses RA.",
            kv_len,
            n_layers,
        )
        return True

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

        if layer.layer_id == 0:
            self._maybe_reset_retro_mirror_for_new_session(forward_batch)

        cache = self._ensure_retro_cache(forward_batch.batch_size)
        if cache is None:
            return out

        qv = q.view(forward_batch.batch_size, -1, layer.tp_q_head_num, layer.qk_head_dim)
        kv = k.view(
            forward_batch.batch_size, -1, layer.tp_k_head_num, layer.qk_head_dim
        )
        vv = v.view(
            forward_batch.batch_size, -1, layer.tp_k_head_num, layer.v_head_dim
        )
        n_layers = self.model_runner.model_config.num_hidden_layers
        lid = layer.layer_id
        chunk_tokens = int(qv.shape[1])

        if (
            self._retro_defer_prefill_mirror
            and getattr(cache, "build_index_when_prefilling", False)
        ):
            if lid == 0 and not self._retro_deferred_mirror_logged:
                logger.info(
                    "RetroInfer: deferring RA mirror to first decode (full KV gather; "
                    "compatible with chunked prefill). Set SGLANG_RETROINFER_SYNC_PREFILL_MIRROR=1 "
                    "to build mirror during extend instead."
                )
                self._retro_deferred_mirror_logged = True
            return out

        if getattr(cache, "build_index_when_prefilling", False):
            min_tokens = self._retro_min_seq_len_for_ra_index(cache)
            if chunk_tokens < min_tokens:
                if lid == 0 and not getattr(self, "_retro_short_chunk_notice", False):
                    logger.info(
                        "RetroInfer prefill: skip RA mirror for short chunk (tokens=%s < %s); "
                        "normal for startup /generate. SGLang KV still from Triton.",
                        chunk_tokens,
                        min_tokens,
                    )
                    self._retro_short_chunk_notice = True
                elif lid == 0 and self._retro_debug:
                    logger.debug(
                        "RetroInfer prefill: skip RA mirror (chunk_tokens=%s < %s)",
                        chunk_tokens,
                        min_tokens,
                    )
                return out

        try:
            if lid == 0:
                self._retro_prefill_pass = getattr(self, "_retro_prefill_pass", 0) + 1
                logger.info(
                    "RetroInfer prefill: pass %s at layer_id=0 (%s layers), batch_size=%s tokens_this_chunk=%s",
                    self._retro_prefill_pass,
                    n_layers,
                    forward_batch.batch_size,
                    chunk_tokens,
                )
            elif self._retro_debug:
                logger.debug(
                    "RetroInfer prefill: pass %s layer_id=%s/%s tokens_this_chunk=%s",
                    getattr(self, "_retro_prefill_pass", 0),
                    lid,
                    n_layers - 1,
                    chunk_tokens,
                )

            cache.prefill_update_kv_cache(qv, kv, vv, lid, start_bdx=0)
            cache.sync(lid, start_bdx=0)

            if lid == n_layers - 1:
                logger.info(
                    "RetroInfer prefill: pass %s last layer synced → prepare_cache()",
                    getattr(self, "_retro_prefill_pass", 0),
                )
                cache.prepare_cache()
                self._retro_prepared = True
                self._retro_sparse_decode_unsafe = self._retro_prefill_pass > 1
                self._warned_fallback = False
                if not getattr(self, "_retro_prepare_logged", False):
                    if self._retro_sparse_decode_unsafe and not self._retro_allow_chunked_sparse:
                        logger.info(
                            "RetroInfer: mirror prepared after %s prefill chunk(s); decode will use "
                            "Triton (chunked prefill is not compatible with RA sparse KV layout). "
                            "Set SGLANG_RETROINFER_ALLOW_CHUNKED_SPARSE_DECODE=1 to force RA decode (unsafe). "
                            "Or leave deferred mirror on (default) and omit SYNC_PREFILL_MIRROR.",
                            self._retro_prefill_pass,
                        )
                    else:
                        logger.info(
                            "RetroInfer: RetrievalAttention cache prepared; decode uses cache.attn_func "
                            "unless chunked prefill forces Triton (see log above)."
                        )
                    self._retro_prepare_logged = True
        except Exception as e:
            self._warn_and_fallback(f"prefill warmup failed ({type(e).__name__}: {e})")
        return out

    def forward_decode(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        layer,
        forward_batch,
        save_kv_cache: bool = True,
        sinks=None,
    ):
        cache = self._ensure_retro_cache(forward_batch.batch_size)
        if cache is None:
            if self._retro_debug:
                logger.debug("RetroInfer decode: no cache, using Triton")
            return self.fallback.forward_decode(
                q, k, v, layer, forward_batch, save_kv_cache=save_kv_cache, sinks=sinks
            )

        # Deferred mirror: first decode gathers full prefill KV (chunked-prefill-safe).
        if not self._retro_prepared:
            if (
                self._retro_defer_prefill_mirror
                and getattr(cache, "build_index_when_prefilling", False)
                and layer.layer_id == 0
            ):
                self._build_retro_mirror_from_kv_pool(cache, forward_batch, layer)
        if not self._retro_prepared:
            if self._retro_debug:
                logger.debug(
                    "RetroInfer decode: _retro_prepared=False (e.g. graph capture warmup), using Triton"
                )
            return self.fallback.forward_decode(
                q, k, v, layer, forward_batch, save_kv_cache=save_kv_cache, sinks=sinks
            )

        # Current RetroInfer cache path assumes same decode length within a batch.
        if torch.min(forward_batch.seq_lens).item() != torch.max(forward_batch.seq_lens).item():
            self._warn_and_fallback("non-uniform seq_lens in decode batch")
            return self.fallback.forward_decode(
                q, k, v, layer, forward_batch, save_kv_cache=save_kv_cache, sinks=sinks
            )

        if self._retro_sparse_decode_unsafe and not self._retro_allow_chunked_sparse:
            return self.fallback.forward_decode(
                q, k, v, layer, forward_batch, save_kv_cache=save_kv_cache, sinks=sinks
            )

        try:
            # q: [bs, heads*dim], k/v: [bs, kv_heads*dim] for decode
            qv = q.view(forward_batch.batch_size, 1, layer.tp_q_head_num, layer.qk_head_dim)
            kv = k.view(forward_batch.batch_size, 1, layer.tp_k_head_num, layer.qk_head_dim)
            vv = v.view(forward_batch.batch_size, 1, layer.tp_k_head_num, layer.v_head_dim)
            n_layers = self.model_runner.model_config.num_hidden_layers
            lid = layer.layer_id

            # Keep SGLang KV pool updates consistent with other backends.
            if save_kv_cache:
                forward_batch.token_to_kv_pool.set_kv_buffer(
                    layer, forward_batch.out_cache_loc, k, v
                )

            # Match RetrievalAttention/attn_hub/retroinfer_attn.py: decode_update runs before attn.
            cache.decode_update_kv_cache(kv, vv, lid)
            # static_len is steady-zone length inside RA, NOT SGLang total seq_len (would be 1000s).
            static_len = (
                cache.static_pattern_total
                if lid == n_layers - 1
                else cache.static_pattern_total + 1
            )

            if not getattr(self, "_retro_decode_logged", False) and lid == 0:
                sgl_seq = int(torch.max(forward_batch.seq_lens).item())
                logger.info(
                    "RetroInfer decode: using %s (batch_size=%s RA_static_len=%s SGLang_seq_len=%s); "
                    "SGLANG_RETROINFER_DEBUG=1 for per-layer",
                    getattr(cache.attn_func, "__name__", "attn_func"),
                    forward_batch.batch_size,
                    static_len,
                    sgl_seq,
                )
                self._retro_decode_logged = True
            elif self._retro_debug:
                logger.debug(
                    "RetroInfer decode: layer %s/%s %s RA_static_len=%s",
                    lid,
                    n_layers - 1,
                    getattr(cache.attn_func, "__name__", "?"),
                    static_len,
                )

            o = cache.attn_func(qv, lid, static_len)
            return o.view(forward_batch.batch_size, -1)
        except Exception as e:
            self._warn_and_fallback(f"decode RetrievalAttention path failed ({type(e).__name__}: {e})")
            return self.fallback.forward_decode(
                q, k, v, layer, forward_batch, save_kv_cache=save_kv_cache, sinks=sinks
            )


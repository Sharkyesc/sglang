from __future__ import annotations

import logging
import os
import sys
from pathlib import Path

import torch

from sglang.srt.layers.attention.retroinfer.index_builder import RetroInferIndexBuilder
from sglang.srt.layers.attention.merge_state import merge_state
from sglang.srt.layers.attention.retroinfer.types import (
    RetroInferLayerCpuIndex,
    RetroInferWorkingSetPlan,
)
from sglang.srt.layers.attention.retroinfer.wave_buffer import (
    RetroInferWaveBufferManager,
)

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
        self.index_builder = RetroInferIndexBuilder(model_runner, kv_source, cpu_store)
        self.wave_buffer = RetroInferWaveBufferManager()

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
        self._weighted_flash_decoding = None
        self._weighted_import_tried = False
        self._weighted_warned_unavailable = False
        self._retroinfer_kernels = None
        self._retroinfer_kernels_import_tried = False
        self._retroinfer_kernels_warned_unavailable = False
        self._partial_refresh_summary_seen: set[tuple[tuple[int, ...], int, int, int, int]] = set()

    def drop_session(self, session) -> None:
        self.wave_buffer.drop_session(session.key)
        if self.gpu_runtime.active_session_key == session.key:
            self.gpu_runtime.clear()
        cache = getattr(session, "retro_cache", None)
        if cache is not None:
            for method_name in ("clear", "close", "release"):
                method = getattr(cache, method_name, None)
                if callable(method):
                    try:
                        method()
                    except Exception:
                        pass
                    break
        session.reset_runtime()

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

    def ensure_weighted_flash_decoding(self):
        if self._weighted_import_tried:
            return self._weighted_flash_decoding

        self._weighted_import_tried = True
        candidate_paths = []
        env_path = os.getenv("SGLANG_WEIGHTED_FLASH_PATH")
        if env_path:
            candidate_paths.append(env_path)
        candidate_paths.append("/home/yy/Desktop/flash-attention")

        for path in candidate_paths:
            if path and Path(path).exists() and path not in sys.path:
                sys.path.append(path)
            try:
                from weighted_flash_decoding import weighted_flash_decoding

                self._weighted_flash_decoding = weighted_flash_decoding
                return self._weighted_flash_decoding
            except Exception:
                continue

        if not self._weighted_warned_unavailable:
            logger.warning(
                "RetroInfer weighted estimation kernel not found. Set SGLANG_WEIGHTED_FLASH_PATH or install weighted_flash_decoding. Falling back to Python estimation."
            )
            self._weighted_warned_unavailable = True
        return None

    def ensure_retroinfer_kernels(self):
        if self._retroinfer_kernels_import_tried:
            return self._retroinfer_kernels

        self._retroinfer_kernels_import_tried = True
        candidate_paths = []
        env_path = os.getenv("SGLANG_RETROINFER_KERNEL_PATH")
        if env_path:
            candidate_paths.append(env_path)
        candidate_paths.append("/home/yy/Desktop/RetrievalAttention/library/retroinfer")

        for path in candidate_paths:
            if path and Path(path).exists() and path not in sys.path:
                sys.path.append(path)
            try:
                from retroinfer_kernels import batch_gemm_softmax, gather_copy_vectors

                self._retroinfer_kernels = {
                    "batch_gemm_softmax": batch_gemm_softmax,
                    "gather_copy_vectors": gather_copy_vectors,
                }
                return self._retroinfer_kernels
            except Exception:
                continue

        if not self._retroinfer_kernels_warned_unavailable:
            logger.warning(
                "RetroInfer kernels not found. Set SGLANG_RETROINFER_KERNEL_PATH or install retroinfer_kernels. Falling back to Python/Torch planning."
            )
            self._retroinfer_kernels_warned_unavailable = True
        return None

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

    def _working_set_target_len(self, kv_len: int) -> int:
        page_size = max(1, int(getattr(self.model_runner, "page_size", 1)))
        static_total = min(kv_len, self.static_pattern_start + self.static_pattern_end)
        recent_budget = max(
            self.pages_per_cluster,
            int(kv_len * max(self.cache_ratio, 0.05)),
        )
        retrieval_budget = max(
            self.n_segment,
            int(kv_len * max(self.retrieval_budget, 0.01)),
        )
        if retrieval_budget > 0 and page_size > 1:
            retrieval_budget = ((retrieval_budget + page_size - 1) // page_size) * page_size
        target = static_total + recent_budget + retrieval_budget
        if kv_len >= self.buffer_cluster_num > 0:
            target = max(target, min(kv_len, self.buffer_cluster_num))
        return max(1, min(kv_len, target))

    def _working_set_slack_len(self, target_len: int) -> int:
        return max(1, min(self.pages_per_cluster, max(1, target_len // 8)))

    def _ordered_unique_positions(self, positions: list[int], limit: int | None = None) -> torch.Tensor:
        seen: set[int] = set()
        ordered: list[int] = []
        for pos in positions:
            token_pos = int(pos)
            if token_pos in seen:
                continue
            seen.add(token_pos)
            ordered.append(token_pos)
            if limit is not None and len(ordered) >= limit:
                break
        return torch.tensor(ordered, dtype=torch.long)

    def _static_span_lengths(
        self,
        kv_len: int,
        layer_index: RetroInferLayerCpuIndex | None = None,
    ) -> tuple[int, int]:
        if layer_index is None:
            prefix_len = min(self.static_pattern_start, kv_len)
            suffix_len = min(self.static_pattern_end, max(0, kv_len - prefix_len))
            return prefix_len, suffix_len

        prefix_len = min(max(0, int(layer_index.middle_start)), kv_len)
        suffix_start = min(max(prefix_len, int(layer_index.middle_end)), kv_len)
        suffix_len = max(0, kv_len - suffix_start)
        return prefix_len, suffix_len

    def _base_working_set_positions(
        self,
        kv_len: int,
        target_len: int,
        layer_index: RetroInferLayerCpuIndex | None = None,
    ) -> tuple[torch.Tensor, int]:
        selected: list[int] = []
        prefix_len, suffix_len = self._static_span_lengths(kv_len, layer_index)
        if layer_index is not None and layer_index.static_positions is not None:
            selected.extend(
                int(pos)
                for pos in layer_index.static_positions.tolist()
                if 0 <= int(pos) < kv_len
            )
        else:
            selected.extend(range(prefix_len))
            if suffix_len > 0:
                selected.extend(range(kv_len - suffix_len, kv_len))

        recent_budget = min(
            max(0, target_len - len(selected)),
            max(self.pages_per_cluster, int(kv_len * max(self.cache_ratio, 0.05))),
        )
        if recent_budget > 0:
            recent_start = max(prefix_len, kv_len - suffix_len - recent_budget)
            selected.extend(range(recent_start, kv_len - suffix_len))

        base_positions = torch.tensor(sorted(set(selected)), dtype=torch.long)
        retrieval_budget = max(0, target_len - int(base_positions.numel()))
        return base_positions, retrieval_budget

    def _extract_retrieval_positions(
        self,
        positions: torch.Tensor,
        base_positions: torch.Tensor,
    ) -> torch.Tensor:
        if positions.numel() == 0:
            return positions
        base_set = set(int(pos) for pos in base_positions.tolist())
        retrieval = [int(pos) for pos in positions.tolist() if int(pos) not in base_set]
        return torch.tensor(retrieval, dtype=torch.long)

    def _expand_positions_to_pages(
        self,
        positions: torch.Tensor,
        kv_len: int,
    ) -> torch.Tensor:
        if positions.numel() == 0:
            return positions
        page_size = max(1, int(getattr(self.model_runner, "page_size", 1)))
        ordered: list[int] = []
        seen: set[int] = set()
        for pos in positions.tolist():
            page_start = (int(pos) // page_size) * page_size
            page_end = min(kv_len, page_start + page_size)
            for token_pos in range(page_start, page_end):
                if token_pos in seen:
                    continue
                seen.add(token_pos)
                ordered.append(token_pos)
        return torch.tensor(ordered, dtype=torch.long)

    def _page_start_for_position(self, position: int) -> int:
        page_size = max(1, int(getattr(self.model_runner, "page_size", 1)))
        return (int(position) // page_size) * page_size

    def _page_positions(self, page_start: int, kv_len: int) -> list[int]:
        page_size = max(1, int(getattr(self.model_runner, "page_size", 1)))
        page_end = min(kv_len, int(page_start) + page_size)
        return list(range(int(page_start), page_end))

    def _retrieval_page_positions_from_candidates(
        self,
        candidates: list[int],
        kv_len: int,
        retrieval_budget: int,
        base_positions: torch.Tensor,
    ) -> torch.Tensor:
        if retrieval_budget <= 0:
            return torch.empty((0,), dtype=torch.long)

        base_set = set(int(pos) for pos in base_positions.tolist())
        ordered: list[int] = []
        seen_pages: set[int] = set()
        used_tokens = 0

        for candidate in candidates:
            token_pos = int(candidate)
            if token_pos < 0 or token_pos >= kv_len:
                continue
            page_start = self._page_start_for_position(token_pos)
            if page_start in seen_pages:
                continue
            page_tokens = [
                pos for pos in self._page_positions(page_start, kv_len) if pos not in base_set
            ]
            if not page_tokens:
                continue
            if used_tokens + len(page_tokens) > retrieval_budget and used_tokens > 0:
                continue
            seen_pages.add(page_start)
            ordered.extend(page_tokens)
            used_tokens += len(page_tokens)
            if used_tokens >= retrieval_budget:
                break

        return torch.tensor(ordered[:retrieval_budget], dtype=torch.long)

    def _rank_layer_clusters(
        self,
        layer_index: RetroInferLayerCpuIndex | None,
        query_vector: torch.Tensor | None,
        layer,
    ) -> list[int]:
        if layer_index is None:
            return []

        cluster_count = len(layer_index.cluster_to_token_indices)
        if cluster_count <= 0:
            return []

        if layer_index.centroids is None or query_vector is None:
            cluster_size = layer_index.cluster_size
            ranked = list(range(cluster_count))
            if cluster_size is not None:
                size_flat = cluster_size.reshape(-1).to(torch.float32)
                ranked.sort(
                    key=lambda idx: float(size_flat[idx].item()) if idx < int(size_flat.numel()) else 0.0,
                    reverse=True,
                )
            return ranked

        grouped_centroids, _, _ = self._grouped_cluster_tensors_from_layer_index(
            layer_index=layer_index,
            device=query_vector.device,
            dtype=query_vector.dtype,
        )
        if grouped_centroids is None or grouped_centroids.numel() == 0:
            return []

        kv_head_count = int(grouped_centroids.shape[0])
        if layer.tp_q_head_num % kv_head_count != 0:
            return []
        group_size = layer.tp_q_head_num // kv_head_count
        q_grouped = query_vector.view(kv_head_count, group_size, query_vector.shape[-1]).contiguous()

        kernels = self.ensure_retroinfer_kernels()
        if (
            kernels is not None
            and int(grouped_centroids.shape[1]) > 0
            and int(grouped_centroids.shape[1]) % 8 == 0
        ):
            try:
                n_clusters = int(grouped_centroids.shape[1])
                gemm_o = torch.zeros(
                    (1, kv_head_count, group_size, n_clusters),
                    device=q_grouped.device,
                    dtype=q_grouped.dtype,
                ).contiguous()
                softmax_o = torch.zeros(
                    (kv_head_count, group_size, n_clusters),
                    device=q_grouped.device,
                    dtype=q_grouped.dtype,
                ).contiguous()
                n_clusters_256 = (n_clusters + 255) // 256
                norm = torch.zeros(
                    (kv_head_count, group_size, n_clusters_256),
                    device=q_grouped.device,
                    dtype=torch.float32,
                ).contiguous()
                acc_sum = torch.zeros(
                    (kv_head_count, group_size, n_clusters_256),
                    device=q_grouped.device,
                    dtype=torch.float32,
                ).contiguous()
                kernels["batch_gemm_softmax"](
                    q_grouped,
                    grouped_centroids,
                    gemm_o,
                    norm,
                    acc_sum,
                    softmax_o,
                    kv_head_count,
                    group_size,
                    n_clusters,
                    int(q_grouped.shape[-1]),
                    float(getattr(layer, "scaling", 1.0)),
                    0.0,
                )
                scores = softmax_o.sum(dim=1).sum(dim=0)
                if layer_index.cluster_mask is not None:
                    cluster_mask = layer_index.cluster_mask.reshape(-1).to(
                        device=scores.device,
                        dtype=torch.bool,
                    )
                    scores = scores.masked_fill(cluster_mask[: scores.shape[0]], float("-inf"))
                ranked = torch.argsort(scores, descending=True).tolist()
                return [int(idx) for idx in ranked if torch.isfinite(scores[int(idx)]).item()]
            except Exception as exc:
                logger.warning(
                    "RetroInfer batch_gemm_softmax ranking path failed (%s: %s). Falling back to Torch ranking.",
                    type(exc).__name__,
                    exc,
                )
                self._retroinfer_kernels = None

        centroids = grouped_centroids.permute(1, 0, 2).contiguous().to(torch.float32)
        query_cpu = query_vector.detach().to(torch.float32)
        aligned_query = self._align_query_to_kv_heads(
            query_cpu,
            kv_head_count=int(centroids.shape[1]),
        )
        logits = (centroids * aligned_query.unsqueeze(0)).sum(dim=-1)
        logits = logits * float(getattr(layer, "scaling", 1.0))
        if layer_index.cluster_mask is not None:
            cluster_mask = layer_index.cluster_mask.reshape(-1).to(torch.bool)
            if int(cluster_mask.numel()) >= int(logits.shape[0]):
                logits = logits.masked_fill(cluster_mask[: logits.shape[0]].unsqueeze(-1), float("-inf"))
        finite = torch.isfinite(logits).any(dim=-1)
        if not bool(finite.any().item()):
            return []
        masked_logits = logits[finite]
        max_logits = masked_logits.max(dim=0).values
        weights = (masked_logits - max_logits.unsqueeze(0)).softmax(dim=0)
        scores = weights.sum(dim=-1)
        ranked_valid = torch.argsort(scores, descending=True).tolist()
        valid_indices = torch.nonzero(finite, as_tuple=False).reshape(-1).tolist()
        return [int(valid_indices[idx]) for idx in ranked_valid]

    def _split_sparse_cluster_budgets(
        self,
        layer_index: RetroInferLayerCpuIndex | None,
        retrieval_budget: int,
        estimation_budget: int,
    ) -> tuple[int, int]:
        if layer_index is None:
            return 0, 0
        total_clusters = len(layer_index.cluster_to_token_indices)
        if total_clusters <= 0:
            return 0, 0

        retrieval_clusters = 0
        if retrieval_budget > 0:
            retrieval_clusters = max(1, int(round(total_clusters * max(self.retrieval_budget, 0.0))))
            retrieval_clusters = min(retrieval_clusters, total_clusters)

        estimation_clusters = 0
        remaining = max(0, total_clusters - retrieval_clusters)
        if estimation_budget > 0 and remaining > 0:
            estimation_clusters = int(round(total_clusters * max(self.estimation_budget, 0.0)))
            if estimation_clusters <= 0:
                estimation_clusters = 1
            estimation_clusters = min(estimation_clusters, remaining)

        return retrieval_clusters, estimation_clusters

    def _grouped_cluster_tensors_from_layer_index(
        self,
        layer_index: RetroInferLayerCpuIndex | None,
        device: torch.device | str,
        dtype: torch.dtype,
    ) -> tuple[torch.Tensor | None, torch.Tensor | None, torch.Tensor | None]:
        if (
            layer_index is None
            or layer_index.centroids is None
            or layer_index.value_sum is None
        ):
            return None, None, None

        flat_centroids = self._flatten_cluster_tensor(
            layer_index.centroids.detach().to(torch.float32)
        )
        flat_value_sum = self._flatten_cluster_tensor(
            layer_index.value_sum.detach().to(torch.float32)
        )
        if flat_centroids is None or flat_value_sum is None:
            return None, None, None

        grouped_centroids = flat_centroids.permute(1, 0, 2).contiguous().to(device=device, dtype=dtype)
        grouped_value_sum = flat_value_sum.permute(1, 0, 2).contiguous().to(device=device, dtype=dtype)
        if layer_index.cluster_size is not None:
            flat_cluster_size = layer_index.cluster_size.reshape(-1).to(torch.float32)
        else:
            flat_cluster_size = torch.ones(
                (flat_centroids.shape[0],),
                dtype=torch.float32,
            )
        grouped_cluster_size = flat_cluster_size.unsqueeze(0).expand(grouped_centroids.shape[0], -1).contiguous()
        grouped_cluster_size = grouped_cluster_size.to(device=device)
        return grouped_centroids, grouped_value_sum, grouped_cluster_size

    def _split_sparse_budgets(
        self,
        sparse_budget: int,
        layer_index: RetroInferLayerCpuIndex | None,
    ) -> tuple[int, int]:
        if sparse_budget <= 0:
            return 0, 0
        if (
            layer_index is None
            or layer_index.dense_only
            or layer_index.value_sum is None
        ):
            return sparse_budget, 0

        estimation_ratio = max(0.0, min(float(self.estimation_budget), 1.0))
        estimation_budget = min(
            sparse_budget,
            int(round(sparse_budget * estimation_ratio)),
        )
        retrieval_budget = max(0, sparse_budget - estimation_budget)
        if retrieval_budget == 0 and sparse_budget > 0:
            retrieval_budget = 1
            estimation_budget = max(0, sparse_budget - retrieval_budget)
        return retrieval_budget, estimation_budget

    def _combine_plan_positions(
        self,
        plan: RetroInferWorkingSetPlan,
    ) -> torch.Tensor:
        positions: list[int] = []
        positions.extend(int(pos) for pos in plan.base_positions.tolist())
        positions.extend(int(pos) for pos in plan.retrieval_positions.tolist())
        return self._ordered_unique_positions(positions, limit=plan.target_len)

    def _flatten_cluster_tensor(self, tensor: torch.Tensor | None) -> torch.Tensor | None:
        if tensor is None:
            return None
        if tensor.dim() == 1:
            return tensor.reshape(-1, 1)
        if tensor.dim() == 2:
            return tensor
        return tensor.reshape(-1, tensor.shape[-2], tensor.shape[-1])

    def _align_query_to_kv_heads(
        self,
        query_heads: torch.Tensor,
        kv_head_count: int,
    ) -> torch.Tensor:
        if query_heads.dim() == 1:
            query_heads = query_heads.unsqueeze(0)
        if query_heads.shape[0] == kv_head_count:
            return query_heads
        if query_heads.shape[0] % kv_head_count == 0:
            group = query_heads.shape[0] // kv_head_count
            return query_heads.view(kv_head_count, group, query_heads.shape[-1]).mean(dim=1)
        return query_heads.mean(dim=0, keepdim=True).expand(kv_head_count, -1)

    def _expand_kv_summary_to_q_heads(
        self,
        summary: torch.Tensor,
        q_head_count: int,
    ) -> torch.Tensor:
        kv_head_count = int(summary.shape[0])
        if kv_head_count == q_head_count:
            return summary
        if q_head_count % kv_head_count == 0:
            repeat = q_head_count // kv_head_count
            return summary.repeat_interleave(repeat, dim=0)
        mapping = torch.linspace(
            0,
            max(0, kv_head_count - 1),
            steps=q_head_count,
            device=summary.device,
        ).round().to(torch.long)
        return summary[mapping]

    def _build_estimation_zone_tensors(
        self,
        req_pool_indices: list[int],
        plans: list[RetroInferWorkingSetPlan],
        layer_id: int,
        device: torch.device | str,
        dtype: torch.dtype,
    ) -> tuple[torch.Tensor | None, torch.Tensor | None, torch.Tensor | None, torch.Tensor | None]:
        batch_size = len(req_pool_indices)
        if batch_size == 0:
            return None, None, None, None

        max_clusters = 0
        kv_head_dim = None
        value_dim = None
        for req_pool_idx, plan in zip(req_pool_indices, plans):
            cluster_ids = getattr(plan, "estimation_cluster_indices", None)
            if cluster_ids is None:
                continue
            max_clusters = max(max_clusters, int(cluster_ids.numel()))
            if kv_head_dim is None and int(cluster_ids.numel()) > 0:
                layer_index = self.cpu_store.get_layer_index(int(req_pool_idx), layer_id)
                if layer_index is None or layer_index.centroids is None or layer_index.value_sum is None:
                    continue
                flat_centroids = self._flatten_cluster_tensor(
                    layer_index.centroids.detach().to(torch.float32)
                )
                flat_value_sum = self._flatten_cluster_tensor(
                    layer_index.value_sum.detach().to(torch.float32)
                )
                if flat_centroids is None or flat_value_sum is None or flat_centroids.numel() == 0:
                    continue
                kv_head_dim = int(flat_centroids.shape[1])
                value_dim = int(flat_value_sum.shape[-1])

        if max_clusters <= 0 or kv_head_dim is None or value_dim is None:
            return None, None, None, None

        es_centroids = torch.zeros(
            (batch_size, max_clusters, kv_head_dim, self.model_runner.model_config.head_dim),
            dtype=dtype,
            device=device,
        )
        es_value_sum = torch.zeros(
            (batch_size, max_clusters, kv_head_dim, value_dim),
            dtype=dtype,
            device=device,
        )
        es_cluster_size = torch.ones(
            (batch_size, max_clusters),
            dtype=torch.float32,
            device=device,
        )
        es_valid_clusters = torch.zeros(
            (batch_size,),
            dtype=torch.int32,
            device=device,
        )

        for batch_idx, (req_pool_idx, plan) in enumerate(zip(req_pool_indices, plans)):
            cluster_ids = getattr(plan, "estimation_cluster_indices", None)
            if cluster_ids is None or cluster_ids.numel() == 0:
                continue
            layer_index = self.cpu_store.get_layer_index(int(req_pool_idx), layer_id)
            if layer_index is None or layer_index.centroids is None or layer_index.value_sum is None:
                continue

            grouped_centroids, grouped_value_sum, grouped_cluster_size = (
                self._grouped_cluster_tensors_from_layer_index(
                    layer_index=layer_index,
                    device=device,
                    dtype=dtype,
                )
            )
            if (
                grouped_centroids is None
                or grouped_value_sum is None
                or grouped_cluster_size is None
            ):
                continue

            valid_cluster_ids = cluster_ids.to(torch.long)
            valid_cluster_ids = valid_cluster_ids[
                valid_cluster_ids < int(grouped_centroids.shape[1])
            ]
            if valid_cluster_ids.numel() == 0:
                continue

            cluster_count = int(valid_cluster_ids.numel())
            es_valid_clusters[batch_idx] = cluster_count
            kernels = self.ensure_retroinfer_kernels()
            if kernels is not None:
                try:
                    kv_heads = int(grouped_centroids.shape[0])
                    index_size = int(valid_cluster_ids.numel()) + int(getattr(plan.retrieval_cluster_indices, "numel", lambda: 0)())
                    if index_size > 0:
                        retrieval_cluster_indices = getattr(plan, "retrieval_cluster_indices", None)
                        if retrieval_cluster_indices is None:
                            retrieval_cluster_indices = torch.empty((0,), dtype=torch.long)
                        kernel_indices = torch.cat(
                            [retrieval_cluster_indices.to(torch.long), valid_cluster_ids],
                            dim=0,
                        )
                        kernel_indices = kernel_indices.unsqueeze(0).expand(kv_heads, -1).contiguous().to(device=device)
                        dst_key = torch.zeros(
                            (kv_heads, cluster_count, grouped_centroids.shape[-1]),
                            dtype=dtype,
                            device=device,
                        )
                        dst_value = torch.zeros(
                            (kv_heads, cluster_count, grouped_value_sum.shape[-1]),
                            dtype=dtype,
                            device=device,
                        )
                        dst_size = torch.zeros(
                            (kv_heads, cluster_count),
                            dtype=dtype,
                            device=device,
                        )
                        kernels["gather_copy_vectors"](
                            grouped_centroids,
                            dst_key,
                            grouped_value_sum,
                            dst_value,
                            grouped_cluster_size.to(dtype=dtype),
                            dst_size,
                            kernel_indices,
                            kv_heads,
                            int(grouped_centroids.shape[1]),
                            cluster_count,
                            index_size,
                            int(retrieval_cluster_indices.numel()),
                            cluster_count,
                        )
                        es_centroids[batch_idx, :cluster_count].copy_(dst_key.permute(1, 0, 2))
                        es_value_sum[batch_idx, :cluster_count].copy_(dst_value.permute(1, 0, 2))
                        es_cluster_size[batch_idx, :cluster_count].copy_(
                            dst_size[0, :cluster_count].to(torch.float32).clamp_min(1.0)
                        )
                        continue
                except Exception as exc:
                    logger.warning(
                        "RetroInfer gather_copy_vectors materialization failed (%s: %s). Falling back to direct tensor gather.",
                        type(exc).__name__,
                        exc,
                    )
                    self._retroinfer_kernels = None

            es_centroids[batch_idx, :cluster_count].copy_(
                grouped_centroids[:, valid_cluster_ids, :].permute(1, 0, 2)
            )
            es_value_sum[batch_idx, :cluster_count].copy_(
                grouped_value_sum[:, valid_cluster_ids, :].permute(1, 0, 2)
            )
            es_cluster_size[batch_idx, :cluster_count].copy_(
                grouped_cluster_size[0, valid_cluster_ids].to(torch.float32).clamp_min(1.0)
            )

        return es_centroids, es_value_sum, es_cluster_size, es_valid_clusters

    def _estimation_attention_state_from_zone_tensors(
        self,
        q: torch.Tensor,
        es_centroids: torch.Tensor | None,
        es_value_sum: torch.Tensor | None,
        es_cluster_size: torch.Tensor | None,
        es_valid_clusters: torch.Tensor | None,
        layer,
    ) -> tuple[torch.Tensor, torch.Tensor] | None:
        if (
            es_centroids is None
            or es_value_sum is None
            or es_cluster_size is None
            or es_valid_clusters is None
        ):
            return None

        weighted_flash_decoding = self.ensure_weighted_flash_decoding()
        if weighted_flash_decoding is not None:
            try:
                batch_size = int(q.shape[0])
                kv_head_count = int(es_centroids.shape[2])
                if kv_head_count > 0 and layer.tp_q_head_num % kv_head_count == 0:
                    group_size = layer.tp_q_head_num // kv_head_count
                    q_grouped = q.view(batch_size, 1, kv_head_count, group_size, q.shape[-1]).reshape(
                        batch_size * kv_head_count, 1, group_size, q.shape[-1]
                    )
                    es_centroids_grouped = (
                        es_centroids.permute(0, 2, 1, 3)
                        .reshape(batch_size * kv_head_count, es_centroids.shape[1], 1, es_centroids.shape[-1])
                        .contiguous()
                    )
                    es_value_sum_grouped = (
                        es_value_sum.permute(0, 2, 1, 3)
                        .reshape(batch_size * kv_head_count, es_value_sum.shape[1], 1, es_value_sum.shape[-1])
                        .contiguous()
                    )
                    es_cluster_size_grouped = (
                        es_cluster_size.unsqueeze(1)
                        .expand(batch_size, kv_head_count, es_cluster_size.shape[1])
                        .reshape(batch_size * kv_head_count, 1, 1, es_cluster_size.shape[1])
                        .contiguous()
                    )
                    invalid_mask = (
                        torch.arange(es_centroids.shape[1], device=q.device, dtype=torch.int32)
                        .unsqueeze(0)
                        .expand(batch_size, es_centroids.shape[1])
                        >= es_valid_clusters.unsqueeze(1)
                    )
                    if invalid_mask.any():
                        invalid_mask_grouped = (
                            invalid_mask.unsqueeze(1)
                            .expand(batch_size, kv_head_count, es_centroids.shape[1])
                            .reshape(batch_size * kv_head_count, es_centroids.shape[1], 1, 1)
                        )
                        es_centroids_grouped = es_centroids_grouped.masked_fill(invalid_mask_grouped, 0)
                        es_value_sum_grouped = es_value_sum_grouped.masked_fill(invalid_mask_grouped, 0)
                        size_mask = invalid_mask.unsqueeze(1).expand(batch_size, kv_head_count, es_cluster_size.shape[1]).reshape(
                            batch_size * kv_head_count, 1, 1, es_cluster_size.shape[1]
                        )
                        es_cluster_size_grouped = es_cluster_size_grouped.masked_fill(size_mask, 0)

                    weighted_out, weighted_lse = weighted_flash_decoding(
                        q_grouped,
                        es_centroids_grouped,
                        es_value_sum_grouped,
                        es_cluster_size_grouped,
                        previous_out=None,
                        previous_lse=None,
                        return_softmax_lse=True,
                    )
                    output = weighted_out.reshape(batch_size, kv_head_count, group_size, layer.v_head_dim).reshape(
                        batch_size, layer.tp_q_head_num, layer.v_head_dim
                    )
                    lse = weighted_lse.squeeze(-1).reshape(batch_size, kv_head_count, group_size).reshape(
                        batch_size, layer.tp_q_head_num
                    )
                    return output, lse.to(torch.float32)
            except Exception as exc:
                logger.warning(
                    "RetroInfer weighted estimation path failed (%s: %s). Falling back to Python estimation.",
                    type(exc).__name__,
                    exc,
                )
                self._weighted_flash_decoding = None

        batch_size = int(q.shape[0])
        output = torch.zeros(
            (batch_size, layer.tp_q_head_num, layer.v_head_dim),
            dtype=q.dtype,
            device=q.device,
        )
        lse = torch.full(
            (batch_size, layer.tp_q_head_num),
            float("-inf"),
            dtype=torch.float32,
            device=q.device,
        )

        for batch_idx in range(batch_size):
            cluster_count = int(es_valid_clusters[batch_idx].item())
            if cluster_count <= 0:
                continue
            query_vector = q[batch_idx, 0].detach().to(torch.float32)
            selected_centroids = es_centroids[batch_idx, :cluster_count].to(torch.float32)
            selected_value_sum = es_value_sum[batch_idx, :cluster_count].to(torch.float32)
            cluster_sizes = es_cluster_size[batch_idx, :cluster_count].to(torch.float32).clamp_min(1.0)

            aligned_query = self._align_query_to_kv_heads(
                query_vector,
                kv_head_count=int(selected_centroids.shape[1]),
            )
            logits = (selected_centroids * aligned_query.unsqueeze(0)).sum(dim=-1)
            logits = logits * float(getattr(layer, "scaling", 1.0))
            max_logits = logits.max(dim=0).values
            exp_logits = (logits - max_logits.unsqueeze(0)).exp()
            denom = (exp_logits * cluster_sizes.view(-1, 1)).sum(dim=0).clamp_min(1e-20)
            summary = (exp_logits.unsqueeze(-1) * selected_value_sum).sum(dim=0) / denom.unsqueeze(-1)
            summary_lse = max_logits + denom.log()

            output[batch_idx].copy_(
                self._expand_kv_summary_to_q_heads(
                    summary.to(dtype=q.dtype),
                    q_head_count=layer.tp_q_head_num,
                )
            )
            lse[batch_idx].copy_(
                self._expand_kv_summary_to_q_heads(
                    summary_lse.unsqueeze(-1).to(dtype=torch.float32),
                    q_head_count=layer.tp_q_head_num,
                ).squeeze(-1)
            )

        if not torch.isfinite(lse).any():
            return None
        return output, lse

    def _materialized_attention_state(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        layer,
        kv_lens: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        batch_size, _, q_head_count, q_dim = q.shape
        _, kv_len, kv_head_count, _ = k.shape
        v_dim = v.shape[-1]

        q_heads = q.squeeze(1).to(torch.float32)
        k_heads = k.permute(0, 2, 1, 3).to(torch.float32)
        v_heads = v.permute(0, 2, 1, 3).to(torch.float32)

        if kv_head_count != q_head_count:
            if q_head_count % kv_head_count == 0:
                repeat = q_head_count // kv_head_count
                k_heads = k_heads.repeat_interleave(repeat, dim=1)
                v_heads = v_heads.repeat_interleave(repeat, dim=1)
            else:
                mapping = torch.linspace(
                    0,
                    max(0, kv_head_count - 1),
                    steps=q_head_count,
                    device=q.device,
                ).round().to(torch.long)
                k_heads = k_heads.index_select(1, mapping)
                v_heads = v_heads.index_select(1, mapping)

        logits = torch.einsum("bhd,bhtd->bht", q_heads, k_heads)
        logits = logits * float(layer.scaling)

        softcap = float(getattr(layer, "logit_cap", 0.0) or 0.0)
        if softcap > 0:
            logits = softcap * torch.tanh(logits / softcap)

        token_idx = torch.arange(kv_len, device=q.device, dtype=torch.int32).view(1, 1, kv_len)
        valid_lens = kv_lens.to(device=q.device, dtype=torch.int32).view(batch_size, 1, 1)
        logits = logits.masked_fill(token_idx >= valid_lens, float("-inf"))

        softmax_lse = torch.logsumexp(logits, dim=-1)
        probs = torch.softmax(logits, dim=-1)
        output = torch.einsum("bht,bhtd->bhd", probs, v_heads)

        return (
            output.to(dtype=q.dtype).view(batch_size, 1, q_head_count, v_dim).contiguous(),
            softmax_lse.contiguous(),
        )

    def _select_layer_working_set_plan(
        self,
        layer_index: RetroInferLayerCpuIndex | None,
        layer,
        kv_len: int,
        target_len: int,
        query_vector: torch.Tensor | None = None,
    ) -> RetroInferWorkingSetPlan:
        if kv_len <= target_len:
            full = torch.arange(kv_len, dtype=torch.long)
            return RetroInferWorkingSetPlan(
                target_len=target_len,
                sparse_budget=max(0, target_len),
                retrieval_budget=max(0, target_len),
                estimation_budget=0,
                base_positions=full,
                retrieval_positions=torch.empty((0,), dtype=torch.long),
                estimation_positions=torch.empty((0,), dtype=torch.long),
            )

        base_positions, sparse_budget = self._base_working_set_positions(
            kv_len=kv_len,
            target_len=target_len,
            layer_index=layer_index,
        )
        retrieval_candidates: list[int] = []
        estimation_candidates: list[tuple[float, int, int]] = []
        retrieval_cluster_indices: list[int] = []
        estimation_cluster_indices: list[int] = []
        prefix_len, suffix_len = self._static_span_lengths(kv_len, layer_index)
        middle_start = prefix_len
        middle_end = max(middle_start, kv_len - suffix_len)
        if layer_index is not None:
            middle_start = min(max(0, int(layer_index.middle_start)), kv_len)
            middle_end = min(max(middle_start, int(layer_index.middle_end)), kv_len)
        retrieval_budget, estimation_budget = self._split_sparse_budgets(
            sparse_budget=sparse_budget,
            layer_index=layer_index,
        )
        if (
            layer_index is not None
            and retrieval_budget > 0
            and layer_index.cluster_to_token_indices
        ):
            ranked_clusters = self._rank_layer_clusters(
                layer_index=layer_index,
                query_vector=query_vector,
                layer=layer,
            )
            retrieval_cluster_budget, estimation_cluster_budget = self._split_sparse_cluster_budgets(
                layer_index=layer_index,
                retrieval_budget=retrieval_budget,
                estimation_budget=estimation_budget,
            )
            selected_retrieval_clusters = 0
            selected_estimation_clusters = 0
            for rank_pos, cluster_idx in enumerate(ranked_clusters):
                if cluster_idx >= len(layer_index.cluster_to_token_indices):
                    continue
                token_indices = layer_index.cluster_to_token_indices[cluster_idx]
                if token_indices is None or len(token_indices) == 0:
                    continue
                representative = -1
                if (
                    layer_index.cluster_representatives is not None
                    and cluster_idx < int(layer_index.cluster_representatives.numel())
                ):
                    representative = int(layer_index.cluster_representatives[cluster_idx].item())
                if selected_retrieval_clusters < retrieval_cluster_budget:
                    for token_idx in reversed(token_indices.tolist()):
                        token_idx = int(token_idx)
                        if 0 <= token_idx < kv_len:
                            retrieval_candidates.append(token_idx)
                    retrieval_cluster_indices.append(int(cluster_idx))
                    selected_retrieval_clusters += 1
                    continue

                if selected_estimation_clusters < estimation_cluster_budget and representative >= 0:
                    estimation_candidates.append((float(-rank_pos), representative, int(cluster_idx)))
                    estimation_cluster_indices.append(int(cluster_idx))
                    selected_estimation_clusters += 1

                if (
                    selected_retrieval_clusters >= retrieval_cluster_budget
                    and selected_estimation_clusters >= estimation_cluster_budget
                ):
                    break
        elif layer_index is not None and retrieval_budget > 0:
            candidates: list[tuple[int, int, int]] = []
            cluster_size = layer_index.cluster_size
            retrieval_cluster_budget, estimation_cluster_budget = self._split_sparse_cluster_budgets(
                layer_index=layer_index,
                retrieval_budget=retrieval_budget,
                estimation_budget=estimation_budget,
            )
            for idx, token_indices in enumerate(layer_index.cluster_to_token_indices):
                if token_indices is None or len(token_indices) == 0:
                    continue
                representative = -1
                if (
                    layer_index.cluster_representatives is not None
                    and idx < int(layer_index.cluster_representatives.numel())
                ):
                    representative = int(layer_index.cluster_representatives[idx].item())
                if representative < 0:
                    representative = int(token_indices[len(token_indices) // 2].item())
                score = (
                    int(cluster_size.reshape(-1)[idx].item())
                    if cluster_size is not None and idx < int(cluster_size.numel())
                    else len(token_indices)
                )
                candidates.append((score, representative, int(idx)))
            candidates.sort(key=lambda item: item[0], reverse=True)
            for rank_pos, (_, representative, cluster_idx) in enumerate(candidates):
                if rank_pos < retrieval_cluster_budget:
                    if representative < 0 or representative >= kv_len:
                        continue
                    retrieval_candidates.append(representative)
                    retrieval_cluster_indices.append(cluster_idx)
                    continue
                if rank_pos < retrieval_cluster_budget + estimation_cluster_budget:
                    if representative >= 0:
                        estimation_candidates.append((float(-rank_pos), representative, int(cluster_idx)))
                        estimation_cluster_indices.append(cluster_idx)

        retrieval_positions = self._retrieval_page_positions_from_candidates(
            candidates=retrieval_candidates,
            kv_len=kv_len,
            retrieval_budget=retrieval_budget,
            base_positions=base_positions,
        )

        if int(retrieval_positions.numel()) < retrieval_budget:
            fallback_candidates: list[int] = []
            if layer_index is not None and layer_index.segment_ranges:
                for seg_start, seg_end in layer_index.segment_ranges:
                    if seg_end <= seg_start:
                        continue
                    fallback_candidates.append(int((seg_start + seg_end - 1) // 2))
            stride = max(1, max(1, middle_end - middle_start) // max(1, target_len))
            for pos in range(middle_start, max(middle_start, middle_end), stride):
                fallback_candidates.append(pos)
            retrieval_positions = self._retrieval_page_positions_from_candidates(
                candidates=retrieval_candidates + fallback_candidates,
                kv_len=kv_len,
                retrieval_budget=retrieval_budget,
                base_positions=base_positions,
            )

        if int(retrieval_positions.numel()) < retrieval_budget:
            fallback_candidates = list(range(kv_len))
            retrieval_positions = self._retrieval_page_positions_from_candidates(
                candidates=retrieval_candidates + fallback_candidates,
                kv_len=kv_len,
                retrieval_budget=retrieval_budget,
                base_positions=base_positions,
            )

        estimation_positions = torch.empty((0,), dtype=torch.long)
        if estimation_budget > 0:
            used_sparse = set(int(pos) for pos in retrieval_positions.tolist())
            used_sparse.update(int(pos) for pos in base_positions.tolist())
            selected_estimation: list[int] = []
            for _, representative, cluster_idx in estimation_candidates:
                rep = int(representative)
                if rep < 0 or rep >= kv_len or rep in used_sparse:
                    continue
                used_sparse.add(rep)
                selected_estimation.append(rep)
                if len(selected_estimation) >= estimation_budget:
                    break
            if len(selected_estimation) < estimation_budget and layer_index is not None:
                for pos in range(middle_start, max(middle_start, middle_end)):
                    if pos in used_sparse:
                        continue
                    used_sparse.add(pos)
                    selected_estimation.append(pos)
                    if len(selected_estimation) >= estimation_budget:
                        break
            estimation_positions = torch.tensor(selected_estimation[:estimation_budget], dtype=torch.long)

        plan = RetroInferWorkingSetPlan(
            target_len=int(base_positions.numel()) + int(retrieval_positions.numel()),
            sparse_budget=sparse_budget,
            retrieval_budget=retrieval_budget,
            estimation_budget=estimation_budget,
            base_positions=base_positions,
            retrieval_positions=retrieval_positions,
            estimation_positions=estimation_positions,
            retrieval_cluster_indices=torch.tensor(
                [idx for idx in retrieval_cluster_indices if idx >= 0],
                dtype=torch.long,
            ),
            estimation_cluster_indices=torch.tensor(
                [idx for idx in estimation_cluster_indices if idx >= 0],
                dtype=torch.long,
            ),
        )
        ordered_positions = self._combine_plan_positions(plan)
        if int(ordered_positions.numel()) < plan.target_len:
            positions = [int(pos) for pos in ordered_positions.tolist()]
            for pos in range(kv_len):
                positions.append(pos)
                ordered_positions = self._ordered_unique_positions(positions)
                if int(ordered_positions.numel()) >= plan.target_len:
                    break
            sparse_positions = ordered_positions[int(plan.base_positions.numel()) : plan.target_len]
            retrieval_len = min(int(plan.retrieval_positions.numel()), int(sparse_positions.numel()))
            plan.retrieval_positions = sparse_positions[:retrieval_len]
            plan.target_len = int(plan.base_positions.numel()) + int(plan.retrieval_positions.numel())
        return plan

    def _select_layer_working_set_positions(
        self,
        layer_index: RetroInferLayerCpuIndex | None,
        layer,
        kv_len: int,
        target_len: int,
        query_vector: torch.Tensor | None = None,
    ) -> torch.Tensor:
        plan = self._select_layer_working_set_plan(
            layer_index=layer_index,
            layer=layer,
            kv_len=kv_len,
            target_len=target_len,
            query_vector=query_vector,
        )
        return self._combine_plan_positions(plan)[:target_len]

    def _query_refresh_overlap(
        self,
        current_retrieval: torch.Tensor | None,
        desired_retrieval: torch.Tensor,
    ) -> float:
        if desired_retrieval.numel() == 0:
            return 1.0
        if current_retrieval is None:
            return 0.0
        current_set = set(int(pos) for pos in current_retrieval.tolist())
        desired_set = set(int(pos) for pos in desired_retrieval.tolist())
        if not desired_set:
            return 1.0
        return len(current_set & desired_set) / max(1, len(desired_set))

    def _query_by_request_for_current_layer(
        self,
        session,
        req_pool_indices: list[int],
        layer_id: int,
        q: torch.Tensor | None,
    ) -> dict[int, dict[int, torch.Tensor]]:
        query_map: dict[int, dict[int, torch.Tensor]] = {}
        if q is None:
            return query_map

        q_tensor = q
        if q_tensor.dim() == 4 and q_tensor.shape[1] == 1:
            q_tensor = q_tensor[:, 0]
        elif q_tensor.dim() == 3:
            pass
        else:
            return query_map

        per_req = {}
        for batch_idx, req_pool_idx in enumerate(req_pool_indices):
            if batch_idx >= q_tensor.shape[0]:
                break
            per_req[int(req_pool_idx)] = q_tensor[batch_idx]
        if per_req:
            query_map[layer_id] = per_req
        return query_map

    def _materialize_request_layer_working_set(
        self,
        req_pool_idx: int,
        layer_id: int,
        positions: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        tensors, _ = self._materialize_request_layer_working_set_with_source(
            req_pool_idx=req_pool_idx,
            layer_id=layer_id,
            positions=positions,
            prefer_host=True,
        )
        return tensors

    def _ensure_request_host_materialized(
        self,
        req_pool_idx: int,
        upto_len: int,
    ) -> bool:
        if upto_len <= 0:
            return False
        try:
            return self.cpu_store.ensure_host_resident(req_pool_idx, upto_len)
        except Exception as exc:
            logger.warning(
                "RetroInfer: host staging failed for req=%s upto=%s (%s: %s)",
                req_pool_idx,
                upto_len,
                type(exc).__name__,
                exc,
            )
            return False

    def _materialize_request_layer_working_set_with_source(
        self,
        req_pool_idx: int,
        layer_id: int,
        positions: torch.Tensor,
        prefer_host: bool = True,
    ) -> tuple[tuple[torch.Tensor, torch.Tensor], str]:
        max_pos = int(positions.max().item()) + 1 if positions.numel() > 0 else 0
        if prefer_host and max_pos > 0:
            self._ensure_request_host_materialized(req_pool_idx, max_pos)
        host_tensors = self.cpu_store.get_request_layer_tensors_by_positions_from_host(
            req_pool_idx=req_pool_idx,
            layer_id=layer_id,
            positions=positions,
        )
        if host_tensors is not None:
            self.cpu_store.mark_last_kv_source(req_pool_idx, "host")
            return host_tensors, "host"
        if prefer_host and max_pos > 0:
            logger.debug(
                "RetroInfer: host KV unavailable for req=%s layer=%s upto=%s; fallback to SGLang GPU.",
                req_pool_idx,
                layer_id,
                max_pos,
            )
        keys, values = self.kv_source.gather_request_layer_tensors(
            req_pool_idx=req_pool_idx,
            layer_id=layer_id,
            upto_len=max_pos,
        )
        self.cpu_store.mark_last_kv_source(req_pool_idx, "gpu_fallback")
        return (keys[positions], values[positions]), "gpu_fallback"

    def _materialize_request_layer_positions_page_aware(
        self,
        req_pool_idx: int,
        layer_id: int,
        positions: torch.Tensor,
        kv_len: int,
    ) -> tuple[torch.Tensor, torch.Tensor, str]:
        if positions.numel() == 0:
            empty = torch.empty((0,), dtype=self.model_runner.dtype, device="cpu")
            return empty, empty, "empty"

        expanded_positions = self._expand_positions_to_pages(positions, kv_len)
        (expanded_keys, expanded_values), source = self._materialize_request_layer_working_set_with_source(
            req_pool_idx=req_pool_idx,
            layer_id=layer_id,
            positions=expanded_positions,
            prefer_host=True,
        )
        expanded_map = {
            int(pos): idx for idx, pos in enumerate(expanded_positions.tolist())
        }
        gather_indices = torch.tensor(
            [expanded_map[int(pos)] for pos in positions.tolist()],
            dtype=torch.long,
            device=expanded_keys.device,
        )
        return expanded_keys[gather_indices], expanded_values[gather_indices], source

    def _admit_pending_scatter_blocks(
        self,
        session_key: tuple[int, ...],
        layer,
        layer_id: int,
    ) -> bool:
        layer_state = self.wave_buffer.get_layer_state(session_key, layer_id)
        if (
            layer_state is None
            or layer_state.retrieval_key_pages is None
            or layer_state.retrieval_value_pages is None
        ):
            return True

        token_to_kv_pool = self.model_runner.token_to_kv_pool
        page_size = max(1, int(layer_state.page_size))
        retrieval_token_len = int(layer_state.retrieval_token_len)
        if retrieval_token_len <= 0:
            return True

        admitted_device_indices = []
        try:
            for batch_idx, req_pool_idx in enumerate(layer_state.req_order):
                pending_blocks = layer_state.pending_scatter_block_ids.get(
                    int(req_pool_idx),
                    torch.empty((0,), dtype=torch.long),
                )
                if pending_blocks.numel() == 0:
                    continue

                retrieval_positions = layer_state.req_retrieval_positions.get(int(req_pool_idx))
                if retrieval_positions is None:
                    continue
                if retrieval_positions.numel() == 0:
                    continue

                pending_set = set(int(block_id) for block_id in pending_blocks.tolist())
                selected_indices = [
                    idx
                    for idx, pos in enumerate(retrieval_positions.tolist())
                    if (int(pos) // page_size) in pending_set
                ]
                if not selected_indices:
                    continue

                flat_retrieval_k = layer_state.retrieval_key_pages[batch_idx].reshape(
                    layer_state.retrieval_capacity_pages * page_size,
                    layer_state.retrieval_key_pages.shape[-2],
                    layer_state.retrieval_key_pages.shape[-1],
                )[:retrieval_token_len]
                flat_retrieval_v = layer_state.retrieval_value_pages[batch_idx].reshape(
                    layer_state.retrieval_capacity_pages * page_size,
                    layer_state.retrieval_value_pages.shape[-2],
                    layer_state.retrieval_value_pages.shape[-1],
                )[:retrieval_token_len]

                index_tensor = torch.tensor(
                    selected_indices,
                    dtype=torch.long,
                    device=flat_retrieval_k.device,
                )
                selected_positions = retrieval_positions[index_tensor.cpu()]
                cache_loc = self.kv_source.get_req_kv_indices_by_positions(
                    int(req_pool_idx),
                    selected_positions,
                ).to(device=flat_retrieval_k.device, dtype=torch.long)
                cache_k = flat_retrieval_k.index_select(0, index_tensor)
                cache_v = flat_retrieval_v.index_select(0, index_tensor)
                original_layer_id = getattr(layer, "layer_id", layer_id)
                restore_layer_id = original_layer_id != layer_id
                if restore_layer_id:
                    layer.layer_id = layer_id
                try:
                    token_to_kv_pool.set_kv_buffer(
                        layer,
                        cache_loc,
                        cache_k,
                        cache_v,
                    )
                finally:
                    if restore_layer_id:
                        layer.layer_id = original_layer_id
                admitted_device_indices.append(cache_loc)
            if admitted_device_indices:
                all_indices = torch.cat(admitted_device_indices).unique(sorted=True)
                host_indices = self.gpu_runtime.backup_device_indices_to_host(all_indices)
                if host_indices is None and self.gpu_runtime.host_pool is not None:
                    logger.warning(
                        "RetroInfer: host backup skipped for admitted scatter blocks (session=%s layer=%s tokens=%s).",
                        session_key,
                        layer_id,
                        int(all_indices.numel()),
                    )
            return True
        except Exception as exc:
            logger.warning(
                "RetroInfer scatter-back admission failed for layer=%s session=%s (%s: %s)",
                layer_id,
                session_key,
                type(exc).__name__,
                exc,
            )
            return False

    def _build_working_set_layout(
        self,
        session,
        layer,
        kv_len: int,
        target_len: int,
        query_map: dict[int, dict[int, torch.Tensor]] | None = None,
    ) -> dict[int, dict[int, torch.Tensor]]:
        layout: dict[int, dict[int, torch.Tensor]] = {}
        n_layers = self.model_runner.model_config.num_hidden_layers
        for layer_id in range(n_layers):
            per_req = {}
            for req_pool_idx in session.key:
                layer_index = self.cpu_store.get_layer_index(req_pool_idx, layer_id)
                query_vector = None
                if query_map is not None:
                    query_vector = query_map.get(layer_id, {}).get(req_pool_idx)
                per_req[req_pool_idx] = self._select_layer_working_set_positions(
                    layer_index=layer_index,
                    layer=layer,
                    kv_len=kv_len,
                    target_len=target_len,
                    query_vector=query_vector,
                )
            layout[layer_id] = per_req
        return layout

    def _materialized_target_len_from_layout(
        self,
        layout: dict[int, dict[int, torch.Tensor]],
    ) -> int:
        max_len = 0
        for req_map in layout.values():
            for positions in req_map.values():
                max_len = max(max_len, int(positions.numel()))
        return max_len

    def _prepare_session_from_layout(
        self,
        session,
        cache,
        layer,
        kv_len: int,
        layout: dict[int, dict[int, torch.Tensor]],
        target_len: int,
    ) -> bool:
        q_heads = layer.tp_q_head_num
        q_dim = layer.qk_head_dim
        kv_heads = layer.tp_k_head_num
        v_dim = layer.v_head_dim
        n_layers = self.model_runner.model_config.num_hidden_layers
        device = self.model_runner.device
        dtype = self.model_runner.dtype

        target_len = max(1, self._materialized_target_len_from_layout(layout))
        slack_len = self._working_set_slack_len(target_len)
        first_req_idx = int(session.key[0]) if session.key else -1
        first_layer_index = (
            self.cpu_store.get_layer_index(first_req_idx, layer.layer_id)
            if first_req_idx >= 0
            else None
        )
        base_positions, _ = self._base_working_set_positions(
            kv_len,
            target_len,
            layer_index=first_layer_index,
        )
        self.wave_buffer.bind_prepared_layout(
            session_key=session.key,
            layer_positions=layout,
            target_len=target_len,
            slack_len=slack_len,
            page_size=max(1, int(getattr(self.model_runner, "page_size", 1))),
            base_len=int(base_positions.numel()),
        )

        used_host_path = False
        for layer_id in range(n_layers):
            def _fetch(req_pool_idx: int, positions: torch.Tensor):
                tensors, source = self._materialize_request_layer_working_set_with_source(
                    req_pool_idx=req_pool_idx,
                    layer_id=layer_id,
                    positions=positions,
                )
                if source == "host":
                    nonlocal used_host_path
                    used_host_path = True
                return tensors

            keys, values, live_len = self.wave_buffer.materialize_layer(
                session_key=session.key,
                layer_id=layer_id,
                req_pool_indices=list(session.key),
                kv_heads=kv_heads,
                qk_dim=q_dim,
                v_dim=v_dim,
                device=device,
                dtype=dtype,
                fetch_fn=_fetch,
                cache_slot_lookup_fn=self.kv_source.get_req_kv_indices_by_positions,
            )
            q_dummy = keys.new_zeros(
                (session.batch_size, live_len, q_heads, q_dim)
            )
            cache.prefill_update_kv_cache(q_dummy, keys, values, layer_id, start_bdx=0)
            cache.sync(layer_id, start_bdx=0)
            if self._admit_pending_scatter_blocks(session.key, layer, layer_id):
                self.wave_buffer.mark_scatter_complete(session.key, layer_id)

        cache.prepare_cache()
        session.retro_cache = cache
        session.mark_prepared(kv_len, target_len)
        for req_pool_idx in session.key:
            self.cpu_store.mark_index_ready(req_pool_idx, kv_len)
            req_state = session.request_states.get(req_pool_idx)
            if req_state is not None:
                req_state.host_staged_upto = max(req_state.host_staged_upto, kv_len)
                req_state.indexed_upto = max(req_state.indexed_upto, kv_len)
                req_state.buffer_prepared_upto = max(req_state.buffer_prepared_upto, kv_len)
                req_state.cache_synced_upto = max(req_state.cache_synced_upto, kv_len)
        self.gpu_runtime.bind_session(session.key)

        logger.info(
            "RetroInfer: prepared session %s from %s KV (batch_size=%s kv_len=%s working_set=%s slack=%s).",
            session.key,
            "HiCache host" if used_host_path else "SGLang GPU",
            session.batch_size,
            kv_len,
            target_len,
            slack_len,
        )
        return True

    def _partial_refresh_current_layer(
        self,
        session,
        layer,
        layer_id: int,
        req_pool_indices: list[int],
        req_positions: dict[int, torch.Tensor],
        kv_len: int,
    ) -> bool:
        cache = session.retro_cache
        if cache is None:
            return False

        q_heads = layer.tp_q_head_num
        q_dim = layer.qk_head_dim
        kv_heads = layer.tp_k_head_num
        v_dim = layer.v_head_dim
        current_layer_index = self.cpu_store.get_layer_index(
            int(req_pool_indices[0]),
            layer_id,
        ) if req_pool_indices else None
        base_positions, _ = self._base_working_set_positions(
            kv_len,
            max(int(session.prepared_working_set_len), self._working_set_target_len(kv_len)),
            layer_index=current_layer_index,
        )
        layer_state = self.wave_buffer.get_layer_state(session.key, layer_id)
        if (
            layer_state is None
            or layer_state.retrieval_key_pages is None
            or layer_state.retrieval_value_pages is None
        ):
            return False

        used_host_path = False
        debug_snapshots: list[dict[str, int]] = []
        target_len = max(
            int(session.prepared_working_set_len),
            max((int(pos.numel()) for pos in req_positions.values()), default=0),
        )
        current_layer_index = self.cpu_store.get_layer_index(
            int(req_pool_indices[0]),
            layer_id,
        ) if req_pool_indices else None
        base_positions, _ = self._base_working_set_positions(
            kv_len,
            target_len,
            layer_index=current_layer_index,
        )
        base_len = int(base_positions.numel())
        page_size = max(1, int(layer_state.page_size))
        retrieval_token_len = max(0, target_len - base_len)
        slack_len = self._working_set_slack_len(target_len)
        resized = self.wave_buffer.ensure_layer_capacity(
            session_key=session.key,
            layer_id=layer_id,
            batch_size=len(req_pool_indices),
            target_len=target_len,
            base_token_len=base_len,
            retrieval_token_len=retrieval_token_len,
            slack_len=slack_len,
            kv_heads=kv_heads,
            qk_dim=q_dim,
            v_dim=v_dim,
            device=layer_state.retrieval_key_pages.device,
            dtype=layer_state.retrieval_key_pages.dtype,
        )
        if resized:
            layer_state = self.wave_buffer.get_layer_state(session.key, layer_id)
            if layer_state is None:
                return False

        for batch_idx, req_pool_idx in enumerate(req_pool_indices):
            desired_positions = req_positions[int(req_pool_idx)]
            desired_retrieval = self._extract_retrieval_positions(
                desired_positions,
                base_positions,
            )
            if desired_retrieval.numel() > 0:
                retrieval_keys, retrieval_values, retrieval_source = self._materialize_request_layer_positions_page_aware(
                    req_pool_idx=int(req_pool_idx),
                    layer_id=layer_id,
                    positions=desired_retrieval,
                    kv_len=kv_len,
                )
                if retrieval_source == "host":
                    used_host_path = True
                padded_retrieval = layer_state.retrieval_capacity_pages * page_size
                padded_base = layer_state.base_capacity_pages * page_size
                base_keys, base_values = self._materialize_request_layer_working_set(
                    req_pool_idx=int(req_pool_idx),
                    layer_id=layer_id,
                    positions=base_positions,
                )
                base_keys = base_keys.to(
                    device=layer_state.base_key_pages.device,
                    dtype=layer_state.base_key_pages.dtype,
                    non_blocking=True,
                )
                base_values = base_values.to(
                    device=layer_state.base_value_pages.device,
                    dtype=layer_state.base_value_pages.dtype,
                    non_blocking=True,
                )
                if int(base_keys.shape[0]) < padded_base:
                    pad_k = torch.zeros(
                        (padded_base - int(base_keys.shape[0]), kv_heads, q_dim),
                        dtype=layer_state.base_key_pages.dtype,
                        device=layer_state.base_key_pages.device,
                    )
                    pad_v = torch.zeros(
                        (padded_base - int(base_values.shape[0]), kv_heads, v_dim),
                        dtype=layer_state.base_value_pages.dtype,
                        device=layer_state.base_value_pages.device,
                    )
                    base_keys = torch.cat([base_keys, pad_k], dim=0)
                    base_values = torch.cat([base_values, pad_v], dim=0)
                layer_state.base_key_pages[batch_idx].copy_(
                    base_keys.view(layer_state.base_capacity_pages, page_size, kv_heads, q_dim)
                )
                layer_state.base_value_pages[batch_idx].copy_(
                    base_values.view(layer_state.base_capacity_pages, page_size, kv_heads, v_dim)
                )
                retrieval_keys = retrieval_keys.to(
                    device=layer_state.retrieval_key_pages.device,
                    dtype=layer_state.retrieval_key_pages.dtype,
                    non_blocking=True,
                )
                retrieval_values = retrieval_values.to(
                    device=layer_state.retrieval_value_pages.device,
                    dtype=layer_state.retrieval_value_pages.dtype,
                    non_blocking=True,
                )
                retrieval_len = int(retrieval_keys.shape[0])
                if retrieval_len < padded_retrieval:
                    pad_k = torch.zeros(
                        (padded_retrieval - retrieval_len, kv_heads, q_dim),
                        dtype=layer_state.retrieval_key_pages.dtype,
                        device=layer_state.retrieval_key_pages.device,
                    )
                    pad_v = torch.zeros(
                        (padded_retrieval - retrieval_len, kv_heads, v_dim),
                        dtype=layer_state.retrieval_value_pages.dtype,
                        device=layer_state.retrieval_value_pages.device,
                    )
                    retrieval_keys = torch.cat([retrieval_keys, pad_k], dim=0)
                    retrieval_values = torch.cat([retrieval_values, pad_v], dim=0)
                layer_state.retrieval_key_pages[batch_idx].copy_(
                    retrieval_keys.view(layer_state.retrieval_capacity_pages, page_size, kv_heads, q_dim)
                )
                layer_state.retrieval_value_pages[batch_idx].copy_(
                    retrieval_values.view(layer_state.retrieval_capacity_pages, page_size, kv_heads, v_dim)
                )
            else:
                retrieval_len = 0

            if retrieval_len == 0:
                layer_state.retrieval_key_pages[batch_idx].zero_()
                layer_state.retrieval_value_pages[batch_idx].zero_()
                self.wave_buffer.clear_layer_retrieval_cache_state(
                    session_key=session.key,
                    layer_id=layer_id,
                    req_pool_idx=int(req_pool_idx),
                )
            else:
                self.wave_buffer.update_layer_retrieval_cache_state(
                    session_key=session.key,
                    layer_id=layer_id,
                    req_pool_idx=int(req_pool_idx),
                    retrieval_positions=desired_retrieval,
                    cache_slot_lookup_fn=self.kv_source.get_req_kv_indices_by_positions,
                )

            append_positions = layer_state.req_append_positions.get(
                int(req_pool_idx),
                torch.empty((0,), dtype=torch.long),
            )
            self.wave_buffer.set_layer_request_segments(
                session_key=session.key,
                layer_id=layer_id,
                req_pool_idx=int(req_pool_idx),
                base_positions=base_positions,
                retrieval_positions=desired_retrieval,
                append_positions=append_positions,
            )
            composed_positions = layer_state.req_positions.get(
                int(req_pool_idx),
                torch.empty((0,), dtype=torch.long),
            )
            debug_snapshots.append(
                {
                    "req_pool_idx": int(req_pool_idx),
                    "desired_total_len": int(desired_positions.numel()),
                    "base_positions_len": int(base_positions.numel()),
                    "desired_retrieval_len": int(desired_retrieval.numel()),
                    "append_positions_len": int(append_positions.numel()),
                    "composed_positions_len": int(composed_positions.numel()),
                }
            )

        layer_state.req_order = tuple(int(req) for req in req_pool_indices)
        layer_state.target_len = max(int(layer_state.target_len), target_len)
        layer_state.base_token_len = max(int(layer_state.base_token_len), base_len)
        layer_state.retrieval_token_len = max(int(layer_state.retrieval_token_len), retrieval_token_len)
        layer_state.live_len = target_len + layer_state.append_len
        session.prepared_working_set_len = max(int(session.prepared_working_set_len), target_len)

        buffer_view = self.wave_buffer.sync_execution_buffer(session.key, layer_id)
        if buffer_view is None:
            return False
        keys, values, live_len = buffer_view
        if self.debug and debug_snapshots:
            first_snapshot = debug_snapshots[0]
            logger.info(
                "RetroInfer refresh state layer=%s session=%s req=%s desired_total=%s base_positions=%s desired_retrieval=%s append_positions=%s composed_positions=%s state_base=%s state_retrieval=%s state_append=%s state_target=%s state_live=%s exec_base=%s exec_retrieval=%s exec_append=%s exec_live=%s key_shape=%s.",
                layer_id,
                session.key,
                first_snapshot["req_pool_idx"],
                first_snapshot["desired_total_len"],
                first_snapshot["base_positions_len"],
                first_snapshot["desired_retrieval_len"],
                first_snapshot["append_positions_len"],
                first_snapshot["composed_positions_len"],
                int(layer_state.base_token_len),
                int(layer_state.retrieval_token_len),
                int(layer_state.append_len),
                int(layer_state.target_len),
                int(layer_state.live_len),
                int(layer_state.execution_base_len),
                int(layer_state.execution_retrieval_len),
                int(layer_state.execution_append_len),
                int(live_len),
                tuple(int(x) for x in keys.shape),
            )
        session.cache_synced_upto = max(session.cache_synced_upto, max(0, kv_len - 1))
        session.buffer_prepared_upto = max(session.buffer_prepared_upto, kv_len)
        for req_pool_idx in req_pool_indices:
            req_state = session.request_states.get(int(req_pool_idx))
            if req_state is not None:
                req_state.buffer_prepared_upto = max(req_state.buffer_prepared_upto, kv_len)
                req_state.cache_synced_upto = max(req_state.cache_synced_upto, max(0, kv_len - 1))
        hit_blocks = 0
        miss_blocks = 0
        for req_pool_idx in req_pool_indices:
            req_idx = int(req_pool_idx)
            hit_blocks += int(
                layer_state.retrieval_hit_block_ids.get(
                    req_idx, torch.empty((0,), dtype=torch.long)
                ).numel()
            )
            miss_blocks += int(
                layer_state.retrieval_miss_block_ids.get(
                    req_idx, torch.empty((0,), dtype=torch.long)
                ).numel()
            )
        summary_key = (
            session.key,
            int(layer_state.append_len),
            int(live_len),
            int(hit_blocks),
            int(miss_blocks),
        )
        if summary_key not in self._partial_refresh_summary_seen:
            self._partial_refresh_summary_seen.add(summary_key)
            logger.info(
                "RetroInfer: partial refresh session=%s source=%s live_len=%s append=%s hit_blocks=%s miss_blocks=%s.",
                session.key,
                "HiCache host" if used_host_path else "SGLang GPU",
                live_len,
                int(layer_state.append_len),
                hit_blocks,
                miss_blocks,
            )
        return True

    def _maybe_query_refresh_current_layer(
        self,
        session,
        forward_batch,
        layer,
        layer_id: int,
        q: torch.Tensor | None,
    ) -> bool:
        if q is None:
            return False

        kv_len = int(torch.min(forward_batch.seq_lens).item())
        target_len = self._working_set_target_len(kv_len)

        query_map = self._query_by_request_for_current_layer(
            session=session,
            req_pool_indices=[int(req) for req in forward_batch.req_pool_indices.tolist()],
            layer_id=layer_id,
            q=q,
        )
        threshold = float(os.getenv("SGLANG_RETROINFER_QUERY_REFRESH_THRESHOLD", "0.5"))
        desired_req_positions: dict[int, torch.Tensor] = {}
        should_refresh = False

        for req_pool_idx in session.key:
            layer_index = self.cpu_store.get_layer_index(req_pool_idx, layer_id)
            req_base_positions, _ = self._base_working_set_positions(
                kv_len,
                target_len,
                layer_index=layer_index,
            )
            desired_positions = self._select_layer_working_set_positions(
                layer_index=layer_index,
                layer=layer,
                kv_len=kv_len,
                target_len=target_len,
                query_vector=query_map.get(layer_id, {}).get(req_pool_idx),
            )
            desired_req_positions[req_pool_idx] = desired_positions
            desired_retrieval = self._extract_retrieval_positions(
                desired_positions,
                req_base_positions,
            )
            current_retrieval = self.wave_buffer.get_layer_retrieval_positions(
                session_key=session.key,
                layer_id=layer_id,
                req_pool_idx=req_pool_idx,
            )
            if current_retrieval is None:
                should_refresh = True
                continue
            overlap = self._query_refresh_overlap(
                current_retrieval=current_retrieval,
                desired_retrieval=desired_retrieval,
            )
            if overlap < threshold:
                should_refresh = True

        if not should_refresh:
            return False

        layer_state = self.wave_buffer.get_layer_state(session.key, layer_id)
        if layer_state is None:
            return False

        retrieval_capacity_tokens = (
            int(layer_state.retrieval_capacity_pages) * max(1, int(layer_state.page_size))
        )
        max_desired_retrieval_len = 0
        for req_pool_idx in session.key:
            layer_index = self.cpu_store.get_layer_index(req_pool_idx, layer_id)
            req_base_positions, _ = self._base_working_set_positions(
                kv_len,
                target_len,
                layer_index=layer_index,
            )
            desired_positions = desired_req_positions[int(req_pool_idx)]
            desired_retrieval = self._extract_retrieval_positions(
                desired_positions,
                req_base_positions,
            )
            max_desired_retrieval_len = max(
                max_desired_retrieval_len,
                int(desired_retrieval.numel()),
            )

        if (
            int(target_len) > int(session.prepared_working_set_len)
            or int(target_len) > int(layer_state.target_len)
            or max_desired_retrieval_len > retrieval_capacity_tokens
        ):
            if self.debug:
                logger.info(
                    "RetroInfer: query refresh growing layer-local buffers for session=%s layer=%s target_len=%s prepared_target=%s retrieval_need=%s retrieval_capacity=%s.",
                    session.key,
                    layer_id,
                    int(target_len),
                    int(session.prepared_working_set_len),
                    max_desired_retrieval_len,
                    retrieval_capacity_tokens,
                )

        try:
            return self._partial_refresh_current_layer(
                session=session,
                layer=layer,
                layer_id=layer_id,
                req_pool_indices=[int(req) for req in forward_batch.req_pool_indices.tolist()],
                req_positions=desired_req_positions,
                kv_len=kv_len,
            )
        except Exception as exc:
            layer_state = self.wave_buffer.get_layer_state(session.key, layer_id)
            debug_details = ""
            if layer_state is not None:
                req_debug_parts = []
                for req_pool_idx in session.key:
                    req_debug_parts.append(
                        "req=%s/base=%s/retrieval=%s/append=%s/positions=%s"
                        % (
                            int(req_pool_idx),
                            int(
                                layer_state.req_base_positions.get(
                                    int(req_pool_idx), torch.empty((0,), dtype=torch.long)
                                ).numel()
                            ),
                            int(
                                layer_state.req_retrieval_positions.get(
                                    int(req_pool_idx), torch.empty((0,), dtype=torch.long)
                                ).numel()
                            ),
                            int(
                                layer_state.req_append_positions.get(
                                    int(req_pool_idx), torch.empty((0,), dtype=torch.long)
                                ).numel()
                            ),
                            int(
                                layer_state.req_positions.get(
                                    int(req_pool_idx), torch.empty((0,), dtype=torch.long)
                                ).numel()
                            ),
                        )
                    )
                debug_details = (
                    " state_base=%s state_retrieval=%s state_append=%s state_live=%s"
                    " state_target=%s exec_base=%s exec_retrieval=%s exec_append=%s"
                    " retrieval_capacity_pages=%s append_capacity_pages=%s req_segments=[%s]"
                ) % (
                    int(layer_state.base_token_len),
                    int(layer_state.retrieval_token_len),
                    int(layer_state.append_len),
                    int(layer_state.live_len),
                    int(layer_state.target_len),
                    int(layer_state.execution_base_len),
                    int(layer_state.execution_retrieval_len),
                    int(layer_state.execution_append_len),
                    int(layer_state.retrieval_capacity_pages),
                    int(layer_state.append_capacity_pages),
                    ", ".join(req_debug_parts),
                )
            logger.warning(
                "RetroInfer query-driven partial refresh failed (%s: %s).%s",
                type(exc).__name__,
                exc,
                debug_details,
            )
            return False

    def _refresh_session_working_set_after_decode(
        self,
        session,
        forward_batch,
        layer,
        layer_id: int,
        q: torch.Tensor | None = None,
    ) -> bool:
        kv_len = int(torch.min(forward_batch.seq_lens).item())
        target_len = self._working_set_target_len(kv_len)
        query_map = self._query_by_request_for_current_layer(
            session=session,
            req_pool_indices=[int(req) for req in forward_batch.req_pool_indices.tolist()],
            layer_id=layer_id,
            q=q,
        )
        layout = self._build_working_set_layout(
            session,
            layer,
            kv_len,
            target_len,
            query_map=query_map,
        )
        cache = self.build_cache(session.batch_size)
        if cache is None:
            return False
        try:
            self._prepare_session_from_layout(
                session=session,
                cache=cache,
                layer=layer,
                kv_len=kv_len,
                layout=layout,
                target_len=target_len,
            )
            return True
        except Exception as exc:
            logger.warning(
                "RetroInfer working-set refresh failed (%s: %s)",
                type(exc).__name__,
                exc,
            )
            return False

    def ensure_session_prepared(self, session, forward_batch, layer) -> bool:
        if session.prepared and session.retro_cache is not None:
            return True

        cache = session.retro_cache or self.build_cache(session.batch_size)
        if cache is None:
            return False

        kv_len = int(torch.min(forward_batch.seq_lens).item()) - 1
        if kv_len <= 0 or kv_len < self._min_seq_len_for_index(cache):
            return False

        min_host_staged_upto = kv_len
        min_index_built_upto = kv_len
        host_not_ready: list[int] = []
        index_not_ready: list[int] = []
        gpu_fallback_reqs: list[int] = []

        for req_pool_idx in session.key:
            self.cpu_store.ensure_host_resident(req_pool_idx, kv_len)
            req_state = session.request_states.get(req_pool_idx)
            if not self.cpu_store.is_request_ready(req_pool_idx, kv_len):
                self.index_builder.build_request(req_pool_idx, kv_len + 1)

            readiness = self.cpu_store.get_request_readiness(req_pool_idx, kv_len)
            host_stored_upto = int(readiness["host_stored_upto"])
            indexed_upto = int(readiness["indexed_upto"])
            min_host_staged_upto = min(min_host_staged_upto, host_stored_upto)
            min_index_built_upto = min(min_index_built_upto, indexed_upto)

            if not bool(readiness["host_ready"]):
                host_not_ready.append(int(req_pool_idx))
            if not bool(readiness["index_ready"]):
                index_not_ready.append(int(req_pool_idx))
            if bool(readiness["gpu_fallback"]):
                gpu_fallback_reqs.append(int(req_pool_idx))

            if req_state is not None:
                req_state.host_staged_upto = max(req_state.host_staged_upto, host_stored_upto)
                req_state.indexed_upto = max(req_state.indexed_upto, indexed_upto)
                req_state.needs_rebuild = indexed_upto < kv_len

        session.host_staged_upto = max(session.host_staged_upto, min_host_staged_upto)
        session.index_built_upto = max(session.index_built_upto, min_index_built_upto)

        if host_not_ready or index_not_ready:
            logger.warning(
                "RetroInfer readiness check failed for session=%s kv_len=%s host_not_ready=%s index_not_ready=%s host_staged_upto=%s index_built_upto=%s.",
                session.key,
                kv_len,
                host_not_ready,
                index_not_ready,
                min_host_staged_upto,
                min_index_built_upto,
            )
            return False

        if gpu_fallback_reqs:
            logger.info(
                "RetroInfer readiness check: session=%s kv_len=%s gpu_fallback_reqs=%s.",
                session.key,
                kv_len,
                gpu_fallback_reqs,
            )

        working_set_len = self._working_set_target_len(kv_len)
        layout = self._build_working_set_layout(
            session,
            layer,
            kv_len,
            working_set_len,
        )
        

        torch.cuda.synchronize()
        try:
            self._prepare_session_from_layout(
                session=session,
                cache=cache,
                layer=layer,
                kv_len=kv_len,
                layout=layout,
                target_len=working_set_len,
            )
        except Exception as exc:
            logger.warning(
                "RetroInfer fallback to Triton: build from SGLang KV failed (%s: %s)",
                type(exc).__name__,
                exc,
            )
            return False
        return True

    def run_decode(self, session, q, k, v, layer, forward_batch, save_kv_cache: bool = True):
        cache = session.retro_cache
        assert cache is not None, "run_decode requires a prepared RetroInfer session"

        qv = q.view(forward_batch.batch_size, 1, layer.tp_q_head_num, layer.qk_head_dim)
        kv = k.view(forward_batch.batch_size, 1, layer.tp_k_head_num, layer.qk_head_dim)
        vv = v.view(forward_batch.batch_size, 1, layer.tp_k_head_num, layer.v_head_dim)
        layer_id = layer.layer_id

        if save_kv_cache:
            forward_batch.token_to_kv_pool.set_kv_buffer(layer, forward_batch.out_cache_loc, k, v)

        new_positions = [int(seq_len) - 1 for seq_len in forward_batch.seq_lens.tolist()]
        req_pool_indices = [int(req_pool_idx) for req_pool_idx in forward_batch.req_pool_indices.tolist()]
        rebuild_required = self.wave_buffer.append_decode_tokens(
            session_key=session.key,
            layer_id=layer_id,
            req_pool_indices=req_pool_indices,
            new_positions=new_positions,
            keys=kv,
            values=vv,
        )

        partial_refreshed = self._maybe_query_refresh_current_layer(
            session=session,
            forward_batch=forward_batch,
            layer=layer,
            layer_id=layer_id,
            q=qv,
        )

        if rebuild_required:
            refreshed = self._refresh_session_working_set_after_decode(
                session=session,
                forward_batch=forward_batch,
                layer=layer,
                layer_id=layer_id,
                q=qv,
            )
            if not refreshed:
                cache.decode_update_kv_cache(kv, vv, layer_id)
            else:
                cache = session.retro_cache
                assert cache is not None, "working-set refresh must leave a valid RetroInfer cache"
        else:
            cache.decode_update_kv_cache(kv, vv, layer_id)
        buffer_view = self.wave_buffer.sync_execution_buffer(session.key, layer_id)
        if buffer_view is None:
            raise RuntimeError(
                f"RetroInfer working-set buffers missing for session={session.key} layer={layer_id}"
            )
        working_keys, working_values, live_len = buffer_view
        kv_lens = torch.full(
            (forward_batch.batch_size,),
            int(live_len),
            dtype=torch.int32,
            device=qv.device,
        )
        working_output, working_lse = self._materialized_attention_state(
            q=qv,
            k=working_keys,
            v=working_values,
            layer=layer,
            kv_lens=kv_lens,
        )

        estimation_plans: list[RetroInferWorkingSetPlan] = []
        for batch_idx, req_pool_idx in enumerate(req_pool_indices):
            context_kv_len = max(0, int(forward_batch.seq_lens[batch_idx].item()) - 1)
            if context_kv_len <= 0:
                estimation_plans.append(
                    RetroInferWorkingSetPlan(
                        target_len=0,
                        sparse_budget=0,
                        retrieval_budget=0,
                        estimation_budget=0,
                        base_positions=torch.empty((0,), dtype=torch.long),
                        retrieval_positions=torch.empty((0,), dtype=torch.long),
                        estimation_positions=torch.empty((0,), dtype=torch.long),
                        estimation_cluster_indices=torch.empty((0,), dtype=torch.long),
                    )
                )
                continue
            layer_index = self.cpu_store.get_layer_index(int(req_pool_idx), layer_id)
            estimation_plans.append(
                self._select_layer_working_set_plan(
                    layer_index=layer_index,
                    layer=layer,
                    kv_len=context_kv_len,
                    target_len=self._working_set_target_len(context_kv_len),
                    query_vector=qv[batch_idx, 0],
                )
            )

        es_centroids, es_value_sum, es_cluster_size, es_valid_clusters = (
            self._build_estimation_zone_tensors(
                req_pool_indices=req_pool_indices,
                plans=estimation_plans,
                layer_id=layer_id,
                device=working_output.device,
                dtype=working_output.dtype,
            )
        )
        estimation_state = self._estimation_attention_state_from_zone_tensors(
            q=qv,
            es_centroids=es_centroids,
            es_value_sum=es_value_sum,
            es_cluster_size=es_cluster_size,
            es_valid_clusters=es_valid_clusters,
            layer=layer,
        )

        if estimation_state is not None:
            estimation_output, estimation_lse = estimation_state
            output, _ = merge_state(
                estimation_output,
                estimation_lse,
                working_output.squeeze(1),
                working_lse,
            )
            output = output.unsqueeze(1)
        else:
            output = working_output
        if self._admit_pending_scatter_blocks(session.key, layer, layer_id):
            self.wave_buffer.mark_scatter_complete(session.key, layer_id)
        return output.view(forward_batch.batch_size, -1)

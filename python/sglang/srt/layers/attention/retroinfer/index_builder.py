from __future__ import annotations

import logging
import os
import sys
from pathlib import Path

import torch

from sglang.srt.layers.attention.retroinfer.types import (
    RetroInferLayerCpuIndex,
    RetroInferRequestCpuState,
)


logger = logging.getLogger(__name__)


class RetroInferIndexBuilder:
    """
    CPU-side RetroInfer index builder on top of SGLang's KV source.
    """

    def __init__(self, model_runner, kv_source, cpu_store):
        self.model_runner = model_runner
        self.kv_source = kv_source
        self.cpu_store = cpu_store

        self.static_pattern_start = int(os.getenv("SGLANG_RETROINFER_STATIC_START", "16"))
        self.static_pattern_end = int(os.getenv("SGLANG_RETROINFER_STATIC_END", "16"))
        self.n_centroids = int(os.getenv("SGLANG_RETROINFER_N_CENTROIDS", "64"))
        self.n_segment = int(os.getenv("SGLANG_RETROINFER_N_SEGMENT", "8"))
        self._segment_k_means = None
        self._segment_import_tried = False

    def _ensure_segment_kmeans(self):
        if self._segment_import_tried:
            return self._segment_k_means

        self._segment_import_tried = True
        candidate_paths = []
        env_path = os.getenv("SGLANG_RETROINFER_PATH")
        if env_path:
            candidate_paths.append(env_path)
        candidate_paths.append("/home/yy/Desktop/RetrievalAttention")

        for path in candidate_paths:
            if path and Path(path).exists() and path not in sys.path:
                sys.path.append(path)
            try:
                from cache_hub.kmeans import segment_k_means

                self._segment_k_means = segment_k_means
                return self._segment_k_means
            except Exception:
                continue
        return None

    def _get_host_layer_tensors(
        self,
        req_pool_idx: int,
        layer_id: int,
        upto_len: int,
    ):
        if upto_len <= 0:
            empty = torch.empty((0,), dtype=self.model_runner.dtype, device="cpu")
            return empty, empty

        tensors = self.cpu_store.get_request_layer_tensors_from_host(
            req_pool_idx=req_pool_idx,
            layer_id=layer_id,
            upto_len=upto_len,
        )
        if tensors is not None:
            self.cpu_store.mark_last_kv_source(req_pool_idx, "host")
            self.cpu_store.mark_last_index_source(req_pool_idx, "host")
            return tensors

        if self.cpu_store.ensure_host_resident(req_pool_idx, upto_len):
            tensors = self.cpu_store.get_request_layer_tensors_from_host(
                req_pool_idx=req_pool_idx,
                layer_id=layer_id,
                upto_len=upto_len,
            )
            if tensors is not None:
                self.cpu_store.mark_last_kv_source(req_pool_idx, "host")
                self.cpu_store.mark_last_index_source(req_pool_idx, "host")
                return tensors

        return None

    def _get_layer_tensors_host_first(
        self,
        req_pool_idx: int,
        layer_id: int,
        upto_len: int,
    ) -> tuple[torch.Tensor, torch.Tensor, str]:
        host_tensors = self._get_host_layer_tensors(req_pool_idx, layer_id, upto_len)
        if host_tensors is not None:
            return host_tensors[0], host_tensors[1], "host"

        keys, values = self.kv_source.gather_request_layer_tensors(
            req_pool_idx=req_pool_idx,
            layer_id=layer_id,
            upto_len=upto_len,
        )
        self.cpu_store.mark_last_kv_source(req_pool_idx, "gpu_fallback")
        self.cpu_store.mark_last_index_source(req_pool_idx, "gpu_fallback")
        logger.debug(
            "RetroInfer: host KV unavailable while building index for req=%s layer=%s upto=%s; fallback to SGLang GPU.",
            req_pool_idx,
            layer_id,
            upto_len,
        )
        return keys.detach().to("cpu"), values.detach().to("cpu"), "gpu_fallback"

    def _normalize_layer_tensor_layout(
        self,
        tensor: torch.Tensor,
    ) -> torch.Tensor:
        """
        Convert SGLang KV tensors from [tokens, kv_heads, head_dim] into the
        RetrievalAttention-friendly [kv_heads, tokens, head_dim] layout.
        """
        if tensor.dim() == 3:
            return tensor.transpose(0, 1).contiguous()
        if tensor.dim() == 2:
            return tensor.unsqueeze(0).contiguous()
        raise ValueError(f"Unexpected RetroInfer KV tensor rank: {tensor.dim()}")

    def _kmeans_device(self) -> torch.device:
        device = getattr(self.model_runner, "device", "cpu")
        if str(device) == "cuda":
            gpu_id = getattr(self.model_runner, "gpu_id", 0)
            return torch.device(f"cuda:{gpu_id}")
        return torch.device(device)

    def _build_static_positions(self, upto_len: int) -> torch.Tensor:
        if upto_len <= 0:
            return torch.empty((0,), dtype=torch.int64)
        prefix_len = min(self.static_pattern_start, upto_len)
        suffix_len = min(self.static_pattern_end, max(0, upto_len - prefix_len))
        positions = list(range(prefix_len))
        if suffix_len > 0:
            positions.extend(range(upto_len - suffix_len, upto_len))
        if not positions:
            return torch.empty((0,), dtype=torch.int64)
        return torch.tensor(sorted(set(positions)), dtype=torch.int64)

    def _build_middle_positions(
        self,
        middle_start: int,
        middle_end: int,
    ) -> torch.Tensor:
        if middle_end <= middle_start:
            return torch.empty((0,), dtype=torch.int64)
        return torch.arange(middle_start, middle_end, dtype=torch.int64)

    def _build_segment_ranges(
        self,
        middle_start: int,
        middle_end: int,
    ) -> tuple[tuple[int, int], ...]:
        middle_len = max(0, middle_end - middle_start)
        if middle_len <= 0:
            return tuple()
        segment_num = max(1, min(self.n_segment, middle_len))
        base = middle_len // segment_num
        remainder = middle_len % segment_num
        ranges: list[tuple[int, int]] = []
        cursor = middle_start
        for seg_idx in range(segment_num):
            seg_len = base + (1 if seg_idx < remainder else 0)
            seg_end = cursor + seg_len
            ranges.append((cursor, seg_end))
            cursor = seg_end
        return tuple(ranges)

    def _build_layer_index(
        self,
        req_pool_idx: int,
        seq_len: int,
        layer_id: int,
    ) -> RetroInferLayerCpuIndex:
        upto_len = max(0, seq_len - 1)
        keys_cpu, values_cpu, index_source = self._get_layer_tensors_host_first(
            req_pool_idx=req_pool_idx,
            layer_id=layer_id,
            upto_len=upto_len,
        )

        keys_cpu = self._normalize_layer_tensor_layout(keys_cpu)
        values_cpu = self._normalize_layer_tensor_layout(values_cpu)

        middle_start = min(self.static_pattern_start, upto_len)
        middle_end = max(middle_start, upto_len - self.static_pattern_end)
        static_positions = self._build_static_positions(upto_len)
        middle_positions = self._build_middle_positions(middle_start, middle_end)
        segment_ranges = self._build_segment_ranges(middle_start, middle_end)
        middle_keys = keys_cpu[:, middle_start:middle_end, :]
        middle_values = values_cpu[:, middle_start:middle_end, :]
        middle_len = int(middle_keys.shape[1])

        dense_only = True
        centroids = None
        value_sum = None
        cluster_value_norms = None
        cluster_size = None
        cluster_mask = None
        cluster_representatives = None
        token_cluster_ids = None
        cluster_to_token_indices: list[object] = []

        segment_k_means = self._ensure_segment_kmeans()
        if segment_k_means is not None and middle_len >= max(1, self.n_segment) and self.n_centroids > 0:
            try:
                kmeans_device = self._kmeans_device()
                middle_keys_dev = middle_keys.to(kmeans_device, non_blocking=True)
                middle_values_dev = middle_values.to(kmeans_device, non_blocking=True)
                mean_key = middle_keys_dev.mean(dim=1, keepdim=True)
                centered_keys = middle_keys_dev - mean_key
                centroids, value_sum, clusters, cluster_size = segment_k_means(
                    key=centered_keys,
                    value=middle_values_dev,
                    num_centroids=self.n_centroids,
                    num_segments=self.n_segment,
                )
                centroids = (centroids + mean_key).to("cpu").contiguous()
                value_sum = value_sum.to("cpu").contiguous()
                cluster_value_norms = value_sum.to(torch.float32).flatten(start_dim=2).norm(dim=-1)
                cluster_value_norms = cluster_value_norms.reshape(-1).contiguous()
                clusters = clusters.to("cpu").contiguous()
                cluster_size = cluster_size.to("cpu").contiguous()
                cluster_mask = (cluster_size == 0).contiguous()
                flat_cluster_count = int(cluster_size.shape[0] * cluster_size.shape[1])
                cluster_representatives = torch.full(
                    (flat_cluster_count,),
                    -1,
                    dtype=torch.int64,
                )
                token_cluster_ids = torch.full(
                    (middle_len,),
                    -1,
                    dtype=torch.int64,
                )
                flat_cluster_idx = 0
                for seg_idx in range(cluster_size.shape[0]):
                    for centroid_idx in range(cluster_size.shape[1]):
                        size = int(cluster_size[seg_idx, centroid_idx].item())
                        if size <= 0:
                            cluster_to_token_indices.append(
                                torch.empty((0,), dtype=torch.int64)
                            )
                            flat_cluster_idx += 1
                            continue
                        token_indices = (
                            clusters[seg_idx, centroid_idx, :size]
                            .reshape(-1)
                            .to(torch.int64)
                        )
                        token_indices = token_indices[token_indices >= 0]
                        if token_indices.numel() == 0:
                            cluster_to_token_indices.append(
                                torch.empty((0,), dtype=torch.int64)
                            )
                            flat_cluster_idx += 1
                            continue
                        token_cluster_ids[token_indices] = flat_cluster_idx
                        token_indices = token_indices + middle_start
                        cluster_to_token_indices.append(token_indices.contiguous())
                        cluster_representatives[flat_cluster_idx] = int(
                            token_indices[token_indices.numel() // 2].item()
                        )
                        flat_cluster_idx += 1
                dense_only = False
            except Exception as exc:
                logger.warning(
                    "RetroInfer CPU index build fell back to dense metadata for req=%s layer=%s (%s: %s)",
                    req_pool_idx,
                    layer_id,
                    type(exc).__name__,
                    exc,
                )

        return RetroInferLayerCpuIndex(
            layer_id=layer_id,
            indexed_upto=upto_len,
            index_source=index_source,
            static_prefix_len=self.static_pattern_start,
            static_suffix_len=self.static_pattern_end,
            middle_start=middle_start,
            middle_end=middle_end,
            dense_only=dense_only,
            key_dtype=str(keys_cpu.dtype),
            static_positions=static_positions,
            middle_positions=middle_positions,
            segment_ranges=segment_ranges,
            centroids=centroids,
            value_sum=value_sum,
            cluster_value_norms=cluster_value_norms,
            cluster_size=cluster_size,
            cluster_mask=cluster_mask,
            cluster_representatives=cluster_representatives,
            token_cluster_ids=token_cluster_ids,
            cluster_to_token_indices=cluster_to_token_indices,
        )

    def build_request(self, req_pool_idx: int, seq_len: int) -> RetroInferRequestCpuState:
        layer_num = self.model_runner.model_config.num_hidden_layers
        upto_len = max(0, seq_len - 1)
        layers = {}
        dense_only = True
        host_state = None
        if self.cpu_store.ensure_host_resident(req_pool_idx, upto_len):
            host_state = self.cpu_store.get_request_host_state(req_pool_idx)
        for layer_id in range(layer_num):
            layer_index = self._build_layer_index(req_pool_idx, seq_len, layer_id)
            dense_only = dense_only and layer_index.dense_only
            layers[layer_id] = layer_index
        request_state = RetroInferRequestCpuState(
            req_pool_idx=req_pool_idx,
            seq_len=seq_len,
            indexed_upto=upto_len,
            dense_only=dense_only,
            layers=layers,
            host_kv=host_state,
        )
        self.cpu_store.store_request_state(request_state)
        return request_state

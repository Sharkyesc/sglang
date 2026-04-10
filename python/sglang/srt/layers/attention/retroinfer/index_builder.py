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

    def _build_layer_index(
        self,
        req_pool_idx: int,
        seq_len: int,
        layer_id: int,
    ) -> RetroInferLayerCpuIndex:
        upto_len = max(0, seq_len - 1)
        keys, values = self.kv_source.gather_request_layer_tensors(
            req_pool_idx=req_pool_idx,
            layer_id=layer_id,
            upto_len=upto_len,
        )
        keys_cpu = keys.detach().to("cpu")
        values_cpu = values.detach().to("cpu")

        middle_start = min(self.static_pattern_start, upto_len)
        middle_end = max(middle_start, upto_len - self.static_pattern_end)
        middle_keys = keys_cpu[middle_start:middle_end]
        middle_values = values_cpu[middle_start:middle_end]
        middle_len = int(middle_keys.shape[0])

        dense_only = True
        centroids = None
        value_sum = None
        cluster_size = None
        cluster_mask = None
        cluster_to_token_indices: list[object] = []

        segment_k_means = self._ensure_segment_kmeans()
        if segment_k_means is not None and middle_len >= max(1, self.n_segment) and self.n_centroids > 0:
            try:
                mean_key = middle_keys.mean(dim=0, keepdim=True)
                centered_keys = middle_keys - mean_key
                centroids, value_sum, clusters, cluster_size = segment_k_means(
                    key=centered_keys.unsqueeze(0),
                    value=middle_values.unsqueeze(0),
                    num_centroids=self.n_centroids,
                    num_segments=self.n_segment,
                )
                centroids = (centroids[0] + mean_key).contiguous()
                value_sum = value_sum[0].contiguous()
                cluster_size = cluster_size[0].contiguous()
                cluster_mask = (cluster_size == 0).contiguous()
                clusters = clusters[0]
                for cluster_idx in range(clusters.shape[0]):
                    size = int(cluster_size[cluster_idx].item())
                    token_indices = clusters[cluster_idx, :size].to(torch.int64) + middle_start
                    cluster_to_token_indices.append(token_indices.contiguous())
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
            static_prefix_len=self.static_pattern_start,
            static_suffix_len=self.static_pattern_end,
            dense_only=dense_only,
            key_dtype=str(keys_cpu.dtype),
            centroids=centroids,
            value_sum=value_sum,
            cluster_size=cluster_size,
            cluster_mask=cluster_mask,
            cluster_to_token_indices=cluster_to_token_indices,
        )

    def build_request(self, req_pool_idx: int, seq_len: int) -> RetroInferRequestCpuState:
        layer_num = self.model_runner.model_config.num_hidden_layers
        upto_len = max(0, seq_len - 1)
        layers = {}
        dense_only = True
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
        )
        self.cpu_store.store_request_state(request_state)
        return request_state

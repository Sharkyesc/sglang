from __future__ import annotations

from sglang.srt.layers.attention.retroinfer.types import RetroInferDecision


class RetroInferCapabilityChecker:
    def __init__(self, model_runner):
        self.model_runner = model_runner

    def check_extend(self, layer, forward_batch) -> RetroInferDecision:
        if not forward_batch.forward_mode.is_extend():
            return RetroInferDecision(False, "fallback", "not extend")
        if getattr(layer, "is_cross_attention", False):
            return RetroInferDecision(False, "fallback", "cross attention unsupported")
        if getattr(layer, "sliding_window_size", -1) not in (-1, None):
            return RetroInferDecision(False, "fallback", "sliding window unsupported")
        if self.model_runner.use_mla_backend:
            return RetroInferDecision(False, "fallback", "MLA backend unsupported")
        return RetroInferDecision(True, "extend_observe")

    def check_decode(self, layer, forward_batch) -> RetroInferDecision:
        if not forward_batch.forward_mode.is_decode():
            return RetroInferDecision(False, "fallback", "not decode")
        if getattr(layer, "is_cross_attention", False):
            return RetroInferDecision(False, "fallback", "cross attention unsupported")
        if getattr(layer, "sliding_window_size", -1) not in (-1, None):
            return RetroInferDecision(False, "fallback", "sliding window unsupported")
        if self.model_runner.use_mla_backend:
            return RetroInferDecision(False, "fallback", "MLA backend unsupported")

        server_args = getattr(self.model_runner, "server_args", None)
        if server_args is not None:
            if getattr(server_args, "speculative_algorithm", None):
                return RetroInferDecision(False, "fallback", "speculative decode unsupported")
            if not getattr(server_args, "disable_cuda_graph", False):
                return RetroInferDecision(False, "fallback", "cuda graph replay unsupported")

        seq_lens = forward_batch.seq_lens
        if seq_lens is None or seq_lens.numel() == 0:
            return RetroInferDecision(False, "fallback", "missing seq_lens")
        min_len = int(seq_lens.min().item())
        max_len = int(seq_lens.max().item())
        if min_len != max_len:
            return RetroInferDecision(False, "fallback", "non-uniform decode seq_lens")
        if min_len <= 1:
            return RetroInferDecision(False, "fallback", "sequence too short")

        return RetroInferDecision(True, "decode_sparse")

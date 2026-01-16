from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Optional

import torch

from sglang.srt.layers.attention.base_attn_backend import AttentionBackend
from sglang.srt.layers.attention.cpu_kv_cache import CPUKVCache

if TYPE_CHECKING:
    from sglang.srt.layers.radix_attention import RadixAttention
    from sglang.srt.model_executor.model_runner import ModelRunner
    from sglang.srt.model_executor.forward_batch_info import ForwardBatch, ForwardMode
    from sglang.srt.speculative.spec_info import SpecInput

logger = logging.getLogger(__name__)


class CPUAttentionBackend(AttentionBackend):

    def __init__(self, model_runner: ModelRunner):
        super().__init__()
        self.model_runner = model_runner
        self.device = model_runner.device

        from sglang.srt.distributed import get_tensor_model_parallel_world_size

        tp_size = get_tensor_model_parallel_world_size()
        self.num_heads = model_runner.model_config.num_attention_heads // tp_size
        self.num_kv_heads = model_runner.model_config.get_num_kv_heads(tp_size)
        self.head_dim = model_runner.model_config.head_dim
        self.v_head_dim = getattr(model_runner.model_config, "v_head_dim", self.head_dim)
        self.num_layers = model_runner.model_config.num_hidden_layers
        self.dtype = model_runner.model_config.dtype

        max_cache_size = model_runner.model_config.context_len * 2

        self.cpu_kv_caches = [
            CPUKVCache(
                max_cache_size=max_cache_size,
                page_size=self.model_runner.page_size,
                num_kv_heads=self.num_kv_heads,
                head_dim=self.head_dim,
                v_head_dim=self.v_head_dim,
                dtype=torch.float32,
                pin_memory=True,
            )
            for _ in range(self.num_layers)
        ]

        self.transfer_stream = torch.cuda.Stream(device=self.device) if torch.cuda.is_available() else None
        
        self.pending_transfers = {}  # {layer_id: (k_gpu, v_gpu, cache_slots)}
        self.transfer_events = {}    # {layer_id: torch.cuda.Event}
        self.q_transfer_events = {}  # {layer_id: torch.cuda.Event}
        self.o_transfer_events = {}  # {layer_id: torch.cuda.Event} 用于 O 的异步传输回 GPU

        self.supports_cuda_graph = False

    def init_forward_metadata(self, forward_batch: ForwardBatch):
        """Init the metadata for a forward pass."""
        return

    def init_cuda_graph_state(self, max_bs: int, max_num_tokens: int):
        """Init the global shared states for cuda graph."""
        self.supports_cuda_graph = False
        return

    def init_forward_metadata_capture_cuda_graph(
        self,
        bs: int,
        num_tokens: int,
        req_pool_indices: torch.Tensor,
        seq_lens: torch.Tensor,
        encoder_lens: Optional[torch.Tensor],
        forward_mode: ForwardMode,
        spec_info: Optional["SpecInput"],
    ):
        """Init the metadata for a forward pass for capturing a cuda graph."""
        pass

    def init_forward_metadata_replay_cuda_graph(
        self,
        bs: int,
        req_pool_indices: torch.Tensor,
        seq_lens: torch.Tensor,
        seq_lens_sum: int,
        encoder_lens: Optional[torch.Tensor],
        forward_mode: ForwardMode,
        spec_info: Optional["SpecInput"],
        seq_lens_cpu: Optional[torch.Tensor],
    ):
        """Init the metadata for a forward pass for replaying a cuda graph."""
        pass

    def get_cuda_graph_seq_len_fill_value(self):
        """Get the fill value for padded seq lens."""
        return 0

    def _async_transfer_q_to_cpu(
        self,
        layer_id: int,
        q_gpu: torch.Tensor,
        layer: "RadixAttention",
    ) -> torch.Tensor:
        """
        异步将 Q 传输到 CPU，在独立的 CUDA stream 中执行传输。
        
        Args:
            layer_id: 层 ID
            q_gpu: [num_tokens, tp_q_head_num * qk_head_dim], GPU, bf16/fp16
            layer: RadixAttention layer
        
        Returns:
            q_cpu: [num_tokens, tp_q_head_num, qk_head_dim], CPU, float32
        """
        if torch.cuda.is_current_stream_capturing():
            return self._transfer_q_to_cpu_sync(q_gpu, layer)

        if self.transfer_stream is None:
            return self._transfer_q_to_cpu_sync(q_gpu, layer)

        if layer_id not in self.q_transfer_events:
            self.q_transfer_events[layer_id] = torch.cuda.Event()

        with torch.cuda.stream(self.transfer_stream):
            q = q_gpu.detach()
            
            if q.ndim == 2:
                T, embed = q.shape
                expected = layer.tp_q_head_num * layer.qk_head_dim
                if embed != expected:
                    raise RuntimeError(
                        f"Q embed size mismatch: got {embed}, expected {expected} "
                        f"(tp_q_head_num={layer.tp_q_head_num}, qk_head_dim={layer.qk_head_dim})"
                    )
                q = q.reshape(T, layer.tp_q_head_num, layer.qk_head_dim)

            q_fp32 = q.float()
            q_cpu = q_fp32.contiguous().cpu()
            
            self.q_transfer_events[layer_id].record(self.transfer_stream)

        return q_cpu

    def wait_for_q_transfer(self, layer_id: int):
        """
        等待 Q 的异步传输完成。
        
        Args:
            layer_id: 层 ID
        """
        if self.transfer_stream is None:
            return
        
        if layer_id in self.q_transfer_events:
            self.q_transfer_events[layer_id].wait()

    def _transfer_q_to_cpu_sync(self, q_gpu: torch.Tensor, layer: "RadixAttention") -> torch.Tensor:
        """
        同步传输 Q 到 CPU
        """
        q = q_gpu.detach()
        
        if q.ndim == 2:
            T, embed = q.shape
            expected = layer.tp_q_head_num * layer.qk_head_dim
            if embed != expected:
                raise RuntimeError(
                    f"Q embed size mismatch: got {embed}, expected {expected} "
                    f"(tp_q_head_num={layer.tp_q_head_num}, qk_head_dim={layer.qk_head_dim})"
                )
            q = q.reshape(T, layer.tp_q_head_num, layer.qk_head_dim)

        q_fp32 = q.float()
        q_cpu = q_fp32.contiguous().cpu()

        return q_cpu

    def _transfer_kv_to_cpu(self, kv_gpu: torch.Tensor, layer: "RadixAttention", is_v: bool = False) -> torch.Tensor:
        """
        Args:
            kv_gpu: [num_tokens, tp_k_head_num * head_dim], GPU, bf16/fp16
            layer: RadixAttention layer
        
        Returns:
            kv_cpu: [num_tokens, tp_k_head_num, head_dim], CPU, float32
        """
        kv_name = "V" if is_v else "K"
        
        print(f"[{kv_name}->CPU] 原始 GPU 张量: shape={kv_gpu.shape}, dtype={kv_gpu.dtype}, device={kv_gpu.device}")
        if kv_gpu.numel() > 0:
            kv_gpu_flat = kv_gpu.flatten()
            print(f"[{kv_name}->CPU] 原始 GPU 前10个元素: {kv_gpu_flat[:10].cpu().tolist()}")
            print(f"[{kv_name}->CPU] 原始 GPU 统计: min={kv_gpu_flat.min().item():.6f}, max={kv_gpu_flat.max().item():.6f}, mean={kv_gpu_flat.mean().item():.6f}")
        
        kv = kv_gpu.detach()

        # [batch, num_heads, head_dim] -> [batch, num_heads * head_dim]
        if kv.ndim == 3:
            B, H, D = kv.shape
            kv = kv.reshape(B, H * D)
        
        if kv.ndim == 2:
            T, embed = kv.shape
            if is_v:
                expected = layer.tp_k_head_num * layer.v_head_dim
                head_dim = layer.v_head_dim
            else:
                expected = layer.tp_k_head_num * layer.qk_head_dim
                head_dim = layer.qk_head_dim
            if embed != expected:
                raise RuntimeError(
                    f"KV embed size mismatch: got {embed}, expected {expected} "
                    f"(tp_k_head_num={layer.tp_k_head_num}, head_dim={head_dim})"
                )
            kv = kv.reshape(T, layer.tp_k_head_num, head_dim)
        else:
            raise RuntimeError(f"KV tensor must be 2D or 3D, got {kv.ndim}D")

        kv_fp32 = kv.float()
        kv_cpu = kv_fp32.contiguous().cpu()

        print(f"[{kv_name}->CPU] 传输后 CPU 张量: shape={kv_cpu.shape}, dtype={kv_cpu.dtype}, device={kv_cpu.device}")
        if kv_cpu.numel() > 0:
            kv_cpu_flat = kv_cpu.flatten()
            print(f"[{kv_name}->CPU] 传输后 CPU 前10个元素: {kv_cpu_flat[:10].tolist()}")
            print(f"[{kv_name}->CPU] 传输后 CPU 统计: min={kv_cpu_flat.min().item():.6f}, max={kv_cpu_flat.max().item():.6f}, mean={kv_cpu_flat.mean().item():.6f}")

        return kv_cpu

    def _transfer_kv_back_to_gpu(self, kv_cpu: torch.Tensor, original_kv: torch.Tensor, layer: "RadixAttention", is_v: bool = False) -> torch.Tensor:
        """
        Args:
            kv_cpu: [num_tokens, tp_k_head_num, head_dim], CPU, float32
            original_kv: [num_tokens, tp_k_head_num * head_dim], GPU, 原始 dtype
            layer: RadixAttention layer
            is_v: 是否为 V
        
        Returns:
            kv_gpu: [num_tokens, tp_k_head_num * head_dim], GPU, 原始 dtype
        """
        kv_name = "V" if is_v else "K"
        
        original_shape = tuple(original_kv.shape)
        original_dtype = original_kv.dtype

        kv_cpu = kv_cpu.contiguous()

        if len(original_shape) == 2:
            T, embed = original_shape
            T_cpu, H_cpu, D_cpu = kv_cpu.shape
            if is_v:
                expected_head_dim = layer.v_head_dim
            else:
                expected_head_dim = layer.qk_head_dim
            if T_cpu != T or H_cpu != layer.tp_k_head_num or D_cpu != expected_head_dim:
                raise RuntimeError(
                    f"KV shape mismatch: kv_cpu={kv_cpu.shape}, expected T={T}, "
                    f"H={layer.tp_k_head_num}, D={expected_head_dim}"
                )
            kv_flat = kv_cpu.reshape(T, embed)
            kv_flat = kv_flat.to(dtype=original_dtype)
            kv_gpu = kv_flat.to(device=self.device, non_blocking=False)

            print(f"[{kv_name}->GPU] 传输回 GPU 张量: shape={kv_gpu.shape}, dtype={kv_gpu.dtype}, device={kv_gpu.device}")
            if kv_gpu.numel() > 0:
                kv_gpu_flat = kv_gpu.flatten()
                original_kv_flat = original_kv.flatten()
                print(f"[{kv_name}->GPU] 传输回 GPU 前10个元素: {kv_gpu_flat[:10].cpu().tolist()}")
                print(f"[{kv_name}->GPU] 原始 GPU 前10个元素: {original_kv_flat[:10].cpu().tolist()}")
                
                diff = (kv_gpu_flat - original_kv_flat).abs()
                max_diff = diff.max().item()
                mean_diff = diff.mean().item()
                print(f"[{kv_name}->GPU] 差异统计: max_diff={max_diff:.8f}, mean_diff={mean_diff:.8f}")
                if max_diff > 1e-3:
                    print(f"[{kv_name}->GPU] 传输后值与原始值差异较大!")
                    mismatched = diff > 1e-3
                    if mismatched.any():
                        mismatch_indices = torch.where(mismatched)[0][:5]
                        print(f"[{kv_name}->GPU] 前5个差异较大的位置: {mismatch_indices.cpu().tolist()}")
                        for idx in mismatch_indices[:3]:
                            print(f"  位置{idx}: 原始={original_kv_flat[idx].item():.6f}, 传输后={kv_gpu_flat[idx].item():.6f}, 差异={diff[idx].item():.8f}")
                else:
                    print(f"[{kv_name}->GPU] ✓ 传输正确: 值与原始值一致")

            return kv_gpu
        elif len(original_shape) == 3:
            B, H, D = original_shape
            T_cpu, H_cpu, D_cpu = kv_cpu.shape
            if is_v:
                expected_head_dim = layer.v_head_dim
            else:
                expected_head_dim = layer.qk_head_dim
            if T_cpu != B or H_cpu != H or D_cpu != D:
                raise RuntimeError(
                    f"KV shape mismatch: kv_cpu={kv_cpu.shape}, expected B={B}, "
                    f"H={H}, D={D}"
                )
            kv_3d = kv_cpu.reshape(B, H, D)
            kv_3d = kv_3d.to(dtype=original_dtype)
            kv_gpu = kv_3d.to(device=self.device, non_blocking=False)

            print(f"[{kv_name}->GPU] 传输回 GPU 张量: shape={kv_gpu.shape}, dtype={kv_gpu.dtype}, device={kv_gpu.device}")
            if kv_gpu.numel() > 0:
                kv_gpu_flat = kv_gpu.flatten()
                original_kv_flat = original_kv.flatten()
                print(f"[{kv_name}->GPU] 传输回 GPU 前10个元素: {kv_gpu_flat[:10].cpu().tolist()}")
                print(f"[{kv_name}->GPU] 原始 GPU 前10个元素: {original_kv_flat[:10].cpu().tolist()}")
                
                diff = (kv_gpu_flat - original_kv_flat).abs()
                max_diff = diff.max().item()
                mean_diff = diff.mean().item()
                print(f"[{kv_name}->GPU] 差异统计: max_diff={max_diff:.8f}, mean_diff={mean_diff:.8f}")
                if max_diff > 1e-3:
                    print(f"[{kv_name}->GPU] 传输后值与原始值差异较大!")
                    mismatched = diff > 1e-3
                    if mismatched.any():
                        mismatch_indices = torch.where(mismatched)[0][:5]
                        print(f"[{kv_name}->GPU] 前5个差异较大的位置: {mismatch_indices.cpu().tolist()}")
                        for idx in mismatch_indices[:3]:
                            print(f"  位置{idx}: 原始={original_kv_flat[idx].item():.6f}, 传输后={kv_gpu_flat[idx].item():.6f}, 差异={diff[idx].item():.8f}")
                else:
                    print(f"[{kv_name}->GPU] ✓ 传输正确: 值与原始值一致")

            return kv_gpu
        else:
            raise RuntimeError(f"KV tensor original shape must be 2D or 3D, got {len(original_shape)}D: {original_shape}")

    def _async_transfer_kv_to_cpu_cache(
        self,
        layer_id: int,
        k_gpu: torch.Tensor,
        v_gpu: torch.Tensor,
        cache_slots: torch.Tensor,
        layer: "RadixAttention",
    ):
        """
        异步将 KV cache 传输到 CPU，在独立的 CUDA stream 中执行传输，在 FFN 计算时并行进行，不阻塞主计算流。
        
        Args:
            layer_id: 层 ID
            k_gpu: [num_tokens, tp_k_head_num * qk_head_dim] 或 [batch, num_heads, head_dim], GPU
            v_gpu: [num_tokens, tp_k_head_num * v_head_dim] 或 [batch, num_heads, head_dim], GPU
            cache_slots: [num_tokens], GPU, 缓存位置索引
            layer: RadixAttention layer
        """
        if torch.cuda.is_current_stream_capturing():
            return

        if self.transfer_stream is None:
            return

        if layer_id not in self.transfer_events:
            self.transfer_events[layer_id] = torch.cuda.Event()

        with torch.cuda.stream(self.transfer_stream):
            k = k_gpu.detach()
            if k.ndim == 3:
                B, H, D = k.shape
                k = k.reshape(B, H * D)
            if k.ndim == 2:
                T, embed = k.shape
                expected = layer.tp_k_head_num * layer.qk_head_dim
                if embed != expected:
                    raise RuntimeError(
                        f"K embed size mismatch: got {embed}, expected {expected}"
                    )
                k = k.reshape(T, layer.tp_k_head_num, layer.qk_head_dim)
            k_fp32 = k.float()
            k_cpu = k_fp32.contiguous().cpu()

            v = v_gpu.detach()
            if v.ndim == 3:
                B, H, D = v.shape
                v = v.reshape(B, H * D)
            if v.ndim == 2:
                T, embed = v.shape
                expected = layer.tp_k_head_num * layer.v_head_dim
                if embed != expected:
                    raise RuntimeError(
                        f"V embed size mismatch: got {embed}, expected {expected}"
                    )
                v = v.reshape(T, layer.tp_k_head_num, layer.v_head_dim)
            v_fp32 = v.float()
            v_cpu = v_fp32.contiguous().cpu()

            cache_slots_cpu = cache_slots.cpu()
            
            self._batch_write_kv_to_cpu_cache(layer_id, k_cpu, v_cpu, cache_slots_cpu)
            
            self.transfer_events[layer_id].record(self.transfer_stream)

    def _async_transfer_o_to_gpu(
        self,
        layer_id: int,
        o_cpu: torch.Tensor,
        original_dtype: torch.dtype,
    ) -> torch.Tensor:
        """
        异步将 O 从 CPU 传输回 GPU，在独立的 CUDA stream 中执行传输。
        这样可以让下一层的计算与当前层 O 的传输并行。
        
        Args:
            layer_id: 层 ID
            o_cpu: [num_tokens, num_heads * head_dim], CPU, float32
            original_dtype: 原始 dtype (bf16/fp16)
        
        Returns:
            o_gpu: [num_tokens, num_heads * head_dim], GPU, original_dtype
        """
        if torch.cuda.is_current_stream_capturing():
            # 如果正在捕获 CUDA graph，回退到同步传输
            o_flat = o_cpu.reshape(o_cpu.shape[0], -1)
            o_flat = o_flat.to(dtype=original_dtype)
            return o_flat.to(device=self.device, non_blocking=False)

        if self.transfer_stream is None:
            o_flat = o_cpu.reshape(o_cpu.shape[0], -1)
            o_flat = o_flat.to(dtype=original_dtype)
            return o_flat.to(device=self.device, non_blocking=False)

        if layer_id not in self.o_transfer_events:
            self.o_transfer_events[layer_id] = torch.cuda.Event()

        with torch.cuda.stream(self.transfer_stream):
            o_flat = o_cpu.reshape(o_cpu.shape[0], -1)
            o_flat = o_flat.to(dtype=original_dtype)
            o_gpu = o_flat.to(device=self.device, non_blocking=True)
            
            self.o_transfer_events[layer_id].record(self.transfer_stream)

        return o_gpu

    def wait_for_o_transfer(self, layer_id: int):
        """
        等待 O 的异步传输完成。
        
        Args:
            layer_id: 层 ID
        """
        if self.transfer_stream is None:
            return
        
        if layer_id in self.o_transfer_events:
            self.o_transfer_events[layer_id].wait()

    def wait_for_pending_transfers(self, layer_id: Optional[int] = None):
        """
        等待待处理的传输任务完成。
        
        Args:
            layer_id: 如果指定，只等待该层的传输；否则等待所有层的传输
        """
        if self.transfer_stream is None:
            return
        
        if layer_id is not None:
            if layer_id in self.pending_transfers:
                self.transfer_stream.synchronize()
                del self.pending_transfers[layer_id]
        else:
            if self.pending_transfers:
                self.transfer_stream.synchronize()
                self.pending_transfers.clear()

    def _batch_write_kv_to_cpu_cache(
        self,
        layer_id: int,
        k_cpu: torch.Tensor,
        v_cpu: torch.Tensor,
        cache_slots: torch.Tensor
    ):
        """
        Args:
            layer_id: 层 ID
            k_cpu: [B, N_kv, D], CPU, float32
            v_cpu: [B, N_kv, D], CPU, float32
            cache_slots: [B], CPU 或 GPU, 缓存位置索引
        """
        if cache_slots.is_cuda:
            slots_cpu = cache_slots.cpu().long()
        else:
            slots_cpu = cache_slots.long()
        page_size = self.cpu_kv_caches[layer_id].page_size
        page_ids = slots_cpu // page_size
        offsets = slots_cpu % page_size

        cpu_cache = self.cpu_kv_caches[layer_id]
        unique_pages = torch.unique(page_ids)

        for page_id in unique_pages:
            page_id_int = int(page_id.item())
            mask = page_ids == page_id
            page_offsets = offsets[mask]  # [N]
            k_page = k_cpu[mask]  # [N, N_kv, D]
            v_page = v_cpu[mask]  # [N, N_kv, D]

            for i, offset in enumerate(page_offsets):
                offset_int = int(offset.item())
                cpu_cache.k_buffer[page_id_int, offset_int] = k_page[i]
                cpu_cache.v_buffer[page_id_int, offset_int] = v_page[i]

    def _gather_kv_from_pages(self, layer_id: int, token_slots: torch.Tensor):
        """        
        Args:
            layer_id: 层 ID
            token_slots: [total_tokens], GPU, token 位置索引
        
        Returns:
            k_tokens: [total_tokens, N_kv, D], CPU
            v_tokens: [total_tokens, N_kv, D], CPU
        """
        token_slots_cpu = token_slots.cpu().long()
        page_size = self.cpu_kv_caches[layer_id].page_size
        page_ids_all = token_slots_cpu // page_size
        unique_pages = torch.unique(page_ids_all)

        k_pages, v_pages = self.cpu_kv_caches[layer_id].read_pages(unique_pages)
        k_flat = k_pages.reshape(-1, k_pages.shape[2], k_pages.shape[3])
        v_flat = v_pages.reshape(-1, v_pages.shape[2], v_pages.shape[3])
        page_ids_all_long = page_ids_all.long()
        page_indices_in_unique = torch.searchsorted(unique_pages, page_ids_all_long, right=False)
        valid_mask = (page_indices_in_unique < len(unique_pages)) & (unique_pages[page_indices_in_unique] == page_ids_all_long)
        if not torch.all(valid_mask):
            invalid_indices = torch.where(~valid_mask)[0]
            missing_pages = page_ids_all_long[invalid_indices].unique()
            raise RuntimeError(f"Requested slot pages {missing_pages.tolist()} not present in fetched pages")
        
        offsets = token_slots_cpu % page_size  # [T]
        page_starts = page_indices_in_unique * page_size  # [T]
        local_indices = page_starts + offsets  # [T]

        k_tokens = k_flat[local_indices]  # [total_tokens, N_kv, D]
        v_tokens = v_flat[local_indices]  # [total_tokens, N_kv, D]

        k_tokens = k_tokens.float().contiguous()
        v_tokens = v_tokens.float().contiguous()

        return k_tokens, v_tokens

    def _get_all_token_indices(self, forward_batch: ForwardBatch) -> torch.Tensor:
        """
        Returns:
            token_slots: [total_tokens], GPU
        """
        if hasattr(forward_batch, "req_to_token_pool") and forward_batch.req_to_token_pool is not None:
            req_pool_indices = forward_batch.req_pool_indices
            token_indices_list = []
            for req_idx in req_pool_indices:
                req_to_token = forward_batch.req_to_token_pool.req_to_token[req_idx]
                allocated_len = len(req_to_token)
                token_indices_list.extend(req_to_token[:allocated_len].cpu().tolist())
            return torch.tensor(token_indices_list, dtype=torch.long, device=self.device)
        else:
            num_tokens = int(forward_batch.seq_lens.sum().item())
            return torch.arange(num_tokens, dtype=torch.long, device=self.device)

    def forward_decode(
        self,
        q: torch.Tensor,
        k: Optional[torch.Tensor],
        v: Optional[torch.Tensor],
        layer: RadixAttention,
        forward_batch: ForwardBatch,
        save_kv_cache: bool = True,
    ) -> torch.Tensor:
    
        q = q.reshape(-1, layer.tp_q_head_num * layer.qk_head_dim)

        q_cpu = self._async_transfer_q_to_cpu(layer.layer_id, q, layer)
        
        if k is not None and v is not None and save_kv_cache:
            if layer.is_cross_attention:
                cache_loc = forward_batch.encoder_out_cache_loc
            else:
                cache_loc = forward_batch.out_cache_loc
                
            self._async_transfer_kv_to_cpu_cache(
                layer.layer_id, k, v, cache_loc, layer
            )
            self.pending_transfers[layer.layer_id] = (k, v, cache_loc)
            k_cpu = None
            v_cpu = None
        else:
            k_cpu = None
            v_cpu = None

        if layer.qk_head_dim != layer.v_head_dim:
            o_cpu = torch.empty((q_cpu.shape[0], layer.tp_q_head_num, layer.v_head_dim), dtype=torch.float32, device="cpu")
        else:
            o_cpu = torch.empty_like(q_cpu)

        self.wait_for_q_transfer(layer.layer_id)
        if save_kv_cache and k is not None and v is not None:
            self.wait_for_pending_transfers(layer.layer_id)

        use_gqa = layer.tp_q_head_num != layer.tp_k_head_num

        self._run_cpu_sdpa_forward_decode(
            q_cpu,
            o_cpu,
            layer.layer_id,
            forward_batch.req_to_token_pool.req_to_token,
            forward_batch.req_pool_indices,
            forward_batch.seq_lens,
            scaling=layer.scaling,
            enable_gqa=use_gqa,
            causal=False,
        )

        o = self._async_transfer_o_to_gpu(layer.layer_id, o_cpu, q.dtype)

        if save_kv_cache and k is not None and v is not None:
            self.wait_for_pending_transfers(layer.layer_id)
        
        return o

    def _run_cpu_sdpa_forward_decode(
        self,
        query: torch.Tensor,  # [num_tokens, num_heads, head_size]
        output: torch.Tensor,  # [num_tokens, num_heads, head_size]
        layer_id: int,
        req_to_token: torch.Tensor,
        req_pool_indices: torch.Tensor,
        seq_lens: torch.Tensor,
        scaling=None,
        enable_gqa=False,
        causal=False,
    ):
        """
        使用 torch.nn.functional.scaled_dot_product_attention 在 CPU 上进行计算
        """
        from torch.nn.functional import scaled_dot_product_attention

        req_to_token_cpu = req_to_token.cpu()
        req_pool_indices_cpu = req_pool_indices.cpu()
        seq_lens_cpu = seq_lens.cpu()

        num_tokens, num_heads, head_size = query.shape
        query_transposed = query.movedim(0, 1)  # [num_heads, num_tokens, head_size]

        start_q = 0
        for seq_idx in range(seq_lens_cpu.shape[0]):
            seq_len_q = 1
            seq_len_kv = int(seq_lens_cpu[seq_idx].item())
            end_q = start_q + seq_len_q

            per_req_query = query_transposed[:, start_q:end_q, :]  # [num_heads, 1, head_size]

            req_pool_idx = int(req_pool_indices_cpu[seq_idx].item())
            per_req_tokens = req_to_token_cpu[req_pool_idx, :seq_len_kv].long()
            
            token_slots_gpu = per_req_tokens.to(device=self.device)
            k_from_cache, v_from_cache = self._gather_kv_from_pages(layer_id, token_slots_gpu)
            
            if enable_gqa:
                per_req_key = k_from_cache.movedim(0, 1)  # [num_kv_heads, seq_len_kv, head_size]
                per_req_value = v_from_cache.movedim(0, 1)  # [num_kv_heads, seq_len_kv, head_size]
            else:
                per_req_key = k_from_cache.movedim(0, 1)  # [num_heads, seq_len_kv, head_size]
                per_req_value = v_from_cache.movedim(0, 1)  # [num_heads, seq_len_kv, head_size]

            per_req_query_batch = per_req_query.unsqueeze(0)  # [1, num_heads, 1, head_size]
            per_req_key_batch = per_req_key.unsqueeze(0)  # [1, num_kv_heads, seq_len_kv, head_size]
            per_req_value_batch = per_req_value.unsqueeze(0)  # [1, num_kv_heads, seq_len_kv, head_size]

            per_req_out = scaled_dot_product_attention(
                per_req_query_batch,
                per_req_key_batch,
                per_req_value_batch,
                scale=scaling,
                is_causal=causal,
            )
            
            per_req_out = per_req_out.squeeze(0)  # [num_heads, 1, head_size]
            per_req_out = per_req_out.movedim(1, 0)  # [1, num_heads, head_size] -> [num_tokens, num_heads, head_size]
            
            output[start_q:end_q, :, :] = per_req_out
            start_q = end_q

        return output

    def _run_cpu_sdpa_forward_extend(
        self,
        query: torch.Tensor,  # [num_tokens, num_heads, head_size]
        output: torch.Tensor,  # [num_tokens, num_heads, head_size]
        layer_id: int,
        req_to_token: torch.Tensor,
        req_pool_indices: torch.Tensor,
        seq_lens: torch.Tensor,
        extend_prefix_lens: torch.Tensor,
        extend_seq_lens: torch.Tensor,
        scaling=None,
        enable_gqa=False,
        causal=False,
    ):
        """
        使用 torch.nn.functional.scaled_dot_product_attention 在 CPU 上进行计算
        """
        from torch.nn.functional import scaled_dot_product_attention

        req_to_token_cpu = req_to_token.cpu()
        req_pool_indices_cpu = req_pool_indices.cpu()
        seq_lens_cpu = seq_lens.cpu()
        extend_prefix_lens_cpu = extend_prefix_lens.cpu()
        extend_seq_lens_cpu = extend_seq_lens.cpu()

        query_transposed = query.movedim(0, 1)  # [num_heads, num_tokens, head_size]

        start_q = 0
        for seq_idx in range(seq_lens_cpu.shape[0]):
            extend_seq_len_q = int(extend_seq_lens_cpu[seq_idx].item())
            prefill_seq_len_q = int(extend_prefix_lens_cpu[seq_idx].item())
            seq_len_kv = int(seq_lens_cpu[seq_idx].item())
            end_q = start_q + extend_seq_len_q

            per_req_query = query_transposed[:, start_q:end_q, :]  # [num_heads, extend_seq_len_q, head_size]

            num_heads, extend_len, head_size = per_req_query.shape
            per_req_query_full = torch.zeros(
                (num_heads, seq_len_kv, head_size),
                dtype=per_req_query.dtype,
                device=per_req_query.device,
            )
            per_req_query_full[:, prefill_seq_len_q:, :] = per_req_query

            req_pool_idx = int(req_pool_indices_cpu[seq_idx].item())
            per_req_tokens = req_to_token_cpu[req_pool_idx, :seq_len_kv].long()
            
            token_slots_gpu = per_req_tokens.to(device=self.device)
            k_from_cache, v_from_cache = self._gather_kv_from_pages(layer_id, token_slots_gpu)
            
            if enable_gqa:
                per_req_key = k_from_cache.movedim(0, 1)  # [num_kv_heads, seq_len_kv, head_size]
                per_req_value = v_from_cache.movedim(0, 1)  # [num_kv_heads, seq_len_kv, head_size]
            else:
                per_req_key = k_from_cache.movedim(0, 1)  # [num_heads, seq_len_kv, head_size]
                per_req_value = v_from_cache.movedim(0, 1)  # [num_heads, seq_len_kv, head_size]

            per_req_query_batch = per_req_query_full.unsqueeze(0)  # [1, num_heads, seq_len_kv, head_size]
            per_req_key_batch = per_req_key.unsqueeze(0)  # [1, num_kv_heads, seq_len_kv, head_size]
            per_req_value_batch = per_req_value.unsqueeze(0)  # [1, num_kv_heads, seq_len_kv, head_size]

            per_req_out_full = scaled_dot_product_attention(
                per_req_query_batch,
                per_req_key_batch,
                per_req_value_batch,
                scale=scaling,
                is_causal=causal,
            )
            
            per_req_out_full = per_req_out_full.squeeze(0)
            
            per_req_out = per_req_out_full[:, prefill_seq_len_q:, :]  # [num_heads, extend_seq_len_q, head_size]
            
            per_req_out = per_req_out.movedim(0, 1)  # [extend_seq_len_q, num_heads, head_size]
            
            output[start_q:end_q, :, :] = per_req_out
            start_q = end_q

        return output

    def forward_extend(
        self,
        q: torch.Tensor,
        k: Optional[torch.Tensor],
        v: Optional[torch.Tensor],
        layer: RadixAttention,
        forward_batch: ForwardBatch,
        save_kv_cache: bool = True,
    ) -> torch.Tensor:
        """
        Run a forward for extend (prefill phase).
        """
        
        q_cpu = self._async_transfer_q_to_cpu(layer.layer_id, q, layer)
        
        if k is not None and v is not None and save_kv_cache:
            cache_slots = forward_batch.out_cache_loc
            self._async_transfer_kv_to_cpu_cache(
                layer.layer_id, k, v, cache_slots, layer
            )
            self.pending_transfers[layer.layer_id] = (k, v, cache_slots)
            k_cpu = None
            v_cpu = None
        else:
            k_cpu = None
            v_cpu = None

        if layer.qk_head_dim != layer.v_head_dim:
            o_cpu = torch.empty((q_cpu.shape[0], layer.tp_q_head_num, layer.v_head_dim), dtype=torch.float32, device="cpu")
        else:
            o_cpu = torch.empty_like(q_cpu)

        self.wait_for_q_transfer(layer.layer_id)
        if save_kv_cache and k is not None and v is not None:
            self.wait_for_pending_transfers(layer.layer_id)

        use_gqa = layer.tp_q_head_num != layer.tp_k_head_num

        from sglang.srt.layers.radix_attention import AttentionType
        causal = True
        if layer.is_cross_attention or layer.attn_type == AttentionType.ENCODER_ONLY:
            causal = False

        self._run_cpu_sdpa_forward_extend(
            q_cpu,
            o_cpu,
            layer.layer_id,
            forward_batch.req_to_token_pool.req_to_token,
            forward_batch.req_pool_indices,
            forward_batch.seq_lens,
            forward_batch.extend_prefix_lens,
            forward_batch.extend_seq_lens,
            scaling=layer.scaling,
            enable_gqa=use_gqa,
            causal=causal,
        )

        o = self._async_transfer_o_to_gpu(layer.layer_id, o_cpu, q.dtype)

        if save_kv_cache and k is not None and v is not None:
            self.wait_for_pending_transfers(layer.layer_id)
        
        return o

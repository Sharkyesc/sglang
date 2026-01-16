from __future__ import annotations

import torch


class CPUKVCache:
    """
    CPU KV Cache with page_first_direct layout.
    
    Layout: [num_pages, page_size, num_kv_heads, head_dim]
    """
    
    def __init__(
        self,
        max_cache_size: int,
        page_size: int,
        num_kv_heads: int,
        head_dim: int,
        v_head_dim: int,
        dtype=torch.float32,
        pin_memory=True,
    ):
        assert max_cache_size % page_size == 0, f"max_cache_size ({max_cache_size}) must be divisible by page_size ({page_size})"
        
        self.page_size = page_size
        self.num_pages = max_cache_size // page_size
        self.num_kv_heads = num_kv_heads
        self.head_dim = head_dim
        self.v_head_dim = v_head_dim
        self.dtype = dtype

        # [num_pages, page_size, num_kv_heads, head_dim]
        self.k_buffer = torch.zeros(
            (self.num_pages, page_size, num_kv_heads, head_dim),
            dtype=dtype,
            device="cpu",
        )
        self.v_buffer = torch.zeros(
            (self.num_pages, page_size, num_kv_heads, v_head_dim),
            dtype=dtype,
            device="cpu",
        )

        if pin_memory:
            try:
                self.k_buffer = self.k_buffer.pin_memory()
                self.v_buffer = self.v_buffer.pin_memory()
            except Exception:
                pass

        self.next_free_slot = 0

    def write_kv(self, k_gpu: torch.Tensor, v_gpu: torch.Tensor, slots: torch.Tensor):
        """
        Args:
            k_gpu: [B, N_kv * D] or [B, N_kv, D], GPU
            v_gpu: [B, N_kv * D] or [B, N_kv, D], GPU
            slots: [B], GPU, 缓存位置索引
        """
        slots_cpu = slots.cpu().long()
        T = slots_cpu.shape[0]

        k_cpu = k_gpu.detach().cpu().to(self.dtype)
        v_cpu = v_gpu.detach().cpu().to(self.dtype)

        # Reshape 为 [B, N_kv, D]
        if k_cpu.ndim == 2:
            k_cpu = k_cpu.view(T, self.num_kv_heads, self.head_dim)
        if v_cpu.ndim == 2:
            v_cpu = v_cpu.view(T, self.num_kv_heads, self.v_head_dim)

        for i in range(T):
            slot = int(slots_cpu[i])
            page_id = slot // self.page_size
            offset = slot % self.page_size
            self.k_buffer[page_id, offset] = k_cpu[i]
            self.v_buffer[page_id, offset] = v_cpu[i]

    def read_pages(self, page_ids: torch.Tensor):
        """
        Args:
            page_ids: [P], CPU, 页 ID 列表
        
        Returns:
            k_pages: [P, page_size, num_kv_heads, head_dim]
            v_pages: [P, page_size, num_kv_heads, v_head_dim]
        """
        page_ids = page_ids.cpu().long()
        return (
            self.k_buffer[page_ids],  # [P, S, H, D]
            self.v_buffer[page_ids],  # [P, S, H, Dv]
        )

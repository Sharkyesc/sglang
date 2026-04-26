from __future__ import annotations

import torch


class SGLangRetroInferKVSource:
    def __init__(self, model_runner):
        self.model_runner = model_runner
        self.req_to_token = model_runner.req_to_token_pool.req_to_token

    def get_page_size(self) -> int:
        return int(getattr(self.model_runner.token_to_kv_pool, "page_size", 1))

    def get_seq_len(self, req_pool_idx: int) -> int:
        indices = self.req_to_token[req_pool_idx]
        nonzero = torch.nonzero(indices, as_tuple=False)
        if nonzero.numel() == 0:
            return 0
        return int(nonzero[-1].item()) + 1

    def get_req_kv_indices(self, req_pool_idx: int, upto_len: int) -> torch.Tensor:
        return self.req_to_token[req_pool_idx, :upto_len].long()

    def get_req_kv_indices_by_positions(
        self,
        req_pool_idx: int,
        positions: torch.Tensor,
    ) -> torch.Tensor:
        if positions.numel() == 0:
            return torch.empty((0,), dtype=torch.long, device=self.req_to_token.device)
        return self.req_to_token[req_pool_idx, positions.to(torch.long)].long()

    def gather_batch_layer_tensors(
        self,
        forward_batch,
        layer_id: int,
        upto_len: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        pool = forward_batch.token_to_kv_pool
        k_buf = pool.get_key_buffer(layer_id)
        v_buf = pool.get_value_buffer(layer_id)
        keys = []
        values = []
        for req_pool_idx in forward_batch.req_pool_indices.tolist():
            loc = self.get_req_kv_indices(int(req_pool_idx), upto_len)
            keys.append(k_buf[loc])
            values.append(v_buf[loc])
        return torch.stack(keys, dim=0), torch.stack(values, dim=0)

    def gather_request_layer_tensors(
        self,
        req_pool_idx: int,
        layer_id: int,
        upto_len: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        k_buf = self.model_runner.token_to_kv_pool.get_key_buffer(layer_id)
        v_buf = self.model_runner.token_to_kv_pool.get_value_buffer(layer_id)
        loc = self.get_req_kv_indices(req_pool_idx, upto_len)
        return k_buf[loc], v_buf[loc]

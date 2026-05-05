from __future__ import annotations

from dataclasses import dataclass

import torch

from sglang.srt.layers.attention.sparse_framework.ops.batched_decode_attention import (
    batched_sparse_decode_attention,
    can_use_batched_sparse_decode,
)
from sglang.srt.layers.attention.sparse_framework.kv_store import get_cpu_kv_store
from sglang.srt.layers.attention.sparse_framework.ops.base import BaseSparseOp
from sglang.srt.layers.attention.sparse_framework.working_set import (
    get_working_set_buffer,
)


@dataclass
class AttendOp(BaseSparseOp):
    mode: str = "dense"

    def run(self, ctx, state: dict):
        state["attend_mode"] = self.mode
        if self.mode == "subset_decode":
            return self._run_subset_decode(ctx, state)
        return None

    def _run_subset_decode(self, ctx, state: dict):
        forward_batch = ctx.forward_batch
        layer = ctx.layer
        q = ctx.query
        k = ctx.key
        v = ctx.value
        if (
            layer is None
            or q is None
            or k is None
            or v is None
            or not forward_batch.forward_mode.is_decode()
        ):
            state["subset_unavailable_reason"] = "subset_decode_requires_decode_qkv"
            return None

        selected_kv_indices = state.get("selected_kv_indices")
        if not selected_kv_indices:
            state["subset_unavailable_reason"] = "missing_selected_kv_indices"
            return None
        if state.get("subset_unavailable_reason") == "host_prefetch_pending":
            return None
        if any(indices.numel() == 0 for indices in selected_kv_indices):
            state["subset_unavailable_reason"] = "empty_selection"
            return None

        if ctx.save_kv_cache:
            forward_batch.token_to_kv_pool.set_kv_buffer(
                layer, forward_batch.out_cache_loc, k, v
            )
            self._store_current_decode_kv(ctx, layer, k, v)

        q = q.reshape(-1, layer.tp_q_head_num * layer.qk_head_dim)
        if layer.qk_head_dim != layer.v_head_dim:
            output = q.new_empty((q.shape[0], layer.tp_q_head_num * layer.v_head_dim))
        else:
            output = torch.empty_like(q)

        q_view = q.view(-1, layer.tp_q_head_num, layer.qk_head_dim)
        output_view = output.view(-1, layer.tp_q_head_num, layer.v_head_dim)
        k_cache = forward_batch.token_to_kv_pool.get_key_buffer(layer.layer_id)
        v_cache = forward_batch.token_to_kv_pool.get_value_buffer(layer.layer_id)
        attention_weights = []
        working_set_stats = []
        key_subsets = []
        value_subsets = []

        selected_positions = state.get("selected_positions") or []
        for batch_idx, token_indices in enumerate(selected_kv_indices):
            token_positions = (
                selected_positions[batch_idx].detach().cpu().tolist()
                if batch_idx < len(selected_positions)
                else []
            )
            key_subset, value_subset, ws_stats = self._materialize_working_set(
                ctx,
                layer,
                batch_idx=batch_idx,
                token_positions=token_positions,
                token_indices=token_indices,
                k_cache=k_cache,
                v_cache=v_cache,
                state=state,
            )
            key_subsets.append(key_subset)
            value_subsets.append(value_subset)
            working_set_stats.append(ws_stats)

        plan = state.get("execution_plan")
        requires_scores = bool(
            getattr(getattr(plan, "selection_plan", None), "requires_scores", False)
        )
        if (
            not requires_scores
            and key_subsets
            and all(item is not None for item in key_subsets)
            and can_use_batched_sparse_decode(
                query=q_view,
                key=key_subsets[0],
                value=value_subsets[0],
                q_head_num=layer.tp_q_head_num,
                kv_head_num=layer.tp_k_head_num,
            )
        ):
            packed_keys = torch.cat(key_subsets, dim=0)
            packed_values = torch.cat(value_subsets, dim=0)
            lengths = torch.tensor(
                [int(item.shape[0]) for item in key_subsets],
                dtype=torch.int32,
                device=q_view.device,
            )
            packed_indptr = torch.zeros(
                len(key_subsets) + 1,
                dtype=torch.int32,
                device=q_view.device,
            )
            packed_indptr[1:] = torch.cumsum(lengths, dim=0)
            output_view.copy_(
                batched_sparse_decode_attention(
                    query=q_view,
                    key=packed_keys,
                    value=packed_values,
                    kv_indptr=packed_indptr,
                    scaling=layer.scaling,
                    q_head_num=layer.tp_q_head_num,
                    kv_head_num=layer.tp_k_head_num,
                )
            )
            state["subset_attention_kernel"] = "triton_batched_sparse_decode"
        else:
            for batch_idx, (key_subset, value_subset) in enumerate(
                zip(key_subsets, value_subsets)
            ):
                per_req_out, per_req_weights = self._compute_subset_attention(
                    q_view[batch_idx : batch_idx + 1],
                    key_subset,
                    value_subset,
                    scaling=layer.scaling,
                    q_head_num=layer.tp_q_head_num,
                    kv_head_num=layer.tp_k_head_num,
                )
                output_view[batch_idx : batch_idx + 1] = per_req_out
                attention_weights.append(per_req_weights)
            state["subset_attention_kernel"] = "torch_per_request"

        state["attention_output"] = output
        state["attention_weights"] = attention_weights
        state["working_set_result"] = working_set_stats
        return output

    def _store_current_decode_kv(self, ctx, layer, key: torch.Tensor, value: torch.Tensor) -> None:
        store = get_cpu_kv_store(ctx.framework_state)
        if store is None:
            return
        out_cache_loc = getattr(ctx.forward_batch, "out_cache_loc", None)
        if out_cache_loc is None:
            return
        key_view = key.reshape(-1, layer.tp_k_head_num, layer.qk_head_dim)
        value_view = value.reshape(-1, layer.tp_k_head_num, layer.v_head_dim)
        for batch_idx, req_pool_idx in enumerate(ctx.req_pool_indices_cpu):
            if batch_idx >= key_view.shape[0]:
                break
            cache_loc = (
                ctx.out_cache_locs_cpu[batch_idx]
                if batch_idx < len(ctx.out_cache_locs_cpu)
                else -1
            )
            position = self._decode_position_from_cache_loc(
                ctx,
                req_pool_idx=req_pool_idx,
                seq_len=int(ctx.seq_lens_cpu[batch_idx]),
                cache_loc=cache_loc,
            )
            store.put(
                req_pool_idx=req_pool_idx,
                layer_id=int(layer.layer_id),
                positions=[position],
                keys=key_view[batch_idx : batch_idx + 1],
                values=value_view[batch_idx : batch_idx + 1],
            )

    def _decode_position_from_cache_loc(
        self,
        ctx,
        *,
        req_pool_idx: int,
        seq_len: int,
        cache_loc: int,
    ) -> int:
        if cache_loc < 0:
            return max(0, int(seq_len) - 1)
        req_to_token = ctx.req_to_token_pool.req_to_token
        candidates = [int(seq_len), int(seq_len) - 1, int(seq_len) + 1]
        max_len = int(req_to_token.shape[1])
        for position in candidates:
            if 0 <= position < max_len:
                value = int(req_to_token[req_pool_idx, position].item())
                if value == int(cache_loc):
                    return int(position)
        window_start = max(0, int(seq_len) - 4)
        window_end = min(max_len, int(seq_len) + 5)
        for position in range(window_start, window_end):
            value = int(req_to_token[req_pool_idx, position].item())
            if value == int(cache_loc):
                return int(position)
        return max(0, int(seq_len) - 1)

    def _materialize_working_set(
        self,
        ctx,
        layer,
        *,
        batch_idx: int,
        token_positions: list[int],
        token_indices: torch.Tensor,
        k_cache: torch.Tensor,
        v_cache: torch.Tensor,
        state: dict,
    ) -> tuple[torch.Tensor, torch.Tensor, dict]:
        store = get_cpu_kv_store(ctx.framework_state)
        buffer = get_working_set_buffer(ctx.framework_state)
        req_pool_idx = int(ctx.req_pool_indices_cpu[batch_idx])

        cpu_keys = None
        cpu_values = None
        cpu_positions = []
        missing_positions = list(token_positions)
        cpu_missing_gpu_available = 0
        cpu_missing_gpu_unavailable = 0
        prefetch = self._consume_sparse_cpu_prefetch(
            ctx,
            req_pool_idx=req_pool_idx,
            layer_id=int(layer.layer_id),
            positions=token_positions,
        )
        if prefetch is not None:
            cpu_keys, cpu_values, cpu_positions = prefetch
            cpu_position_set = {int(pos) for pos in cpu_positions}
            missing_positions = [
                int(pos) for pos in token_positions if int(pos) not in cpu_position_set
            ]
        elif store is not None and token_positions:
            cpu_keys, cpu_values, cpu_positions, missing_positions = store.get_many(
                req_pool_idx=req_pool_idx,
                layer_id=int(layer.layer_id),
                positions=token_positions,
                device=k_cache.device,
                dtype=k_cache.dtype,
            )

        if not missing_positions:
            key_subset = cpu_keys
            value_subset = cpu_values
            gpu_fallback_count = 0
        elif cpu_keys is None or cpu_values is None:
            key_subset = k_cache[token_indices]
            value_subset = v_cache[token_indices]
            gpu_fallback_count = int(token_indices.numel())
            cpu_missing_gpu_available = int((token_indices >= 0).sum().item())
            cpu_missing_gpu_unavailable = int((token_indices < 0).sum().item())
            self._snapshot_missing_to_cpu(
                store,
                req_pool_idx=req_pool_idx,
                layer_id=int(layer.layer_id),
                positions=token_positions,
                keys=key_subset,
                values=value_subset,
            )
        else:
            position_to_index = {int(pos): i for i, pos in enumerate(token_positions)}
            missing_offsets = [
                position_to_index[pos]
                for pos in missing_positions
                if pos in position_to_index
            ]
            missing_index_tensor = torch.tensor(
                missing_offsets,
                dtype=torch.long,
                device=token_indices.device,
            )
            missing_token_indices = token_indices[missing_index_tensor]
            cpu_missing_gpu_available = int((missing_token_indices >= 0).sum().item())
            cpu_missing_gpu_unavailable = int((missing_token_indices < 0).sum().item())
            gpu_keys = k_cache[missing_token_indices]
            gpu_values = v_cache[missing_token_indices]
            key_parts = []
            value_parts = []
            cpu_lookup = {pos: i for i, pos in enumerate(cpu_positions)}
            gpu_lookup = {pos: i for i, pos in enumerate(missing_positions)}
            for pos in token_positions:
                if pos in cpu_lookup:
                    key_parts.append(cpu_keys[cpu_lookup[pos]])
                    value_parts.append(cpu_values[cpu_lookup[pos]])
                elif pos in gpu_lookup:
                    key_parts.append(gpu_keys[gpu_lookup[pos]])
                    value_parts.append(gpu_values[gpu_lookup[pos]])
            key_subset = torch.stack(key_parts)
            value_subset = torch.stack(value_parts)
            gpu_fallback_count = int(gpu_keys.shape[0])
            self._snapshot_missing_to_cpu(
                store,
                req_pool_idx=req_pool_idx,
                layer_id=int(layer.layer_id),
                positions=missing_positions,
                keys=gpu_keys,
                values=gpu_values,
            )

        validation = None
        plan = state.get("execution_plan")
        if (
            key_subset is not None
            and value_subset is not None
            and bool(getattr(plan, "validate_kv_cache", False))
        ):
            validation = self._validate_against_gpu_cache(
                token_indices=token_indices,
                key_subset=key_subset,
                value_subset=value_subset,
                k_cache=k_cache,
                v_cache=v_cache,
            )
            if self._kv_cache_check_failed(validation):
                dropped_request_layers = (
                    store.drop_request(req_pool_idx)
                    if store is not None and int(layer.layer_id) == 0
                    else 0
                )
                key_subset = k_cache[token_indices]
                value_subset = v_cache[token_indices]
                gpu_fallback_count = int(token_indices.numel())
                cpu_positions = []
                self._snapshot_missing_to_cpu(
                    store,
                    req_pool_idx=req_pool_idx,
                    layer_id=int(layer.layer_id),
                    positions=token_positions,
                    keys=key_subset,
                    values=value_subset,
                )
                validation["fallback_reason"] = "cpu_store_mismatch"
                validation["dropped_request_layers"] = dropped_request_layers

        if buffer is not None and key_subset is not None and value_subset is not None:
            key_subset, value_subset = buffer.materialize(
                layer_id=int(layer.layer_id),
                key=key_subset,
                value=value_subset,
            )

        return key_subset, value_subset, {
            "requested": len(token_positions),
            "cpu": len(cpu_positions),
            "gpu_fallback": gpu_fallback_count,
            "cpu_missing": len(missing_positions),
            "cpu_missing_gpu_available": cpu_missing_gpu_available,
            "cpu_missing_gpu_unavailable": cpu_missing_gpu_unavailable,
            "buffered": int(key_subset.shape[0]) if key_subset is not None else 0,
            "kv_cache_check": validation,
            "prefetched": prefetch is not None,
        }

    def _consume_sparse_cpu_prefetch(
        self,
        ctx,
        *,
        req_pool_idx: int,
        layer_id: int,
        positions: list[int],
    ) -> tuple[torch.Tensor, torch.Tensor, list[int]] | None:
        if ctx.framework_state is None or not positions:
            return None
        pending = ctx.framework_state.get("sparse_cpu_prefetches") or {}
        key = (int(req_pool_idx), int(layer_id), tuple(int(pos) for pos in positions))
        item = pending.pop(key, None)
        if item is None:
            return None
        event = item.get("event")
        if event is not None:
            torch.cuda.current_stream(device=item["keys"].device).wait_event(event)
        return item["keys"], item["values"], item["positions"]

    def _snapshot_missing_to_cpu(
        self,
        store,
        *,
        req_pool_idx: int,
        layer_id: int,
        positions: list[int],
        keys: torch.Tensor,
        values: torch.Tensor,
    ) -> None:
        if store is None or not positions:
            return
        store.put(
            req_pool_idx=req_pool_idx,
            layer_id=layer_id,
            positions=positions,
            keys=keys,
            values=values,
        )

    def _compute_subset_attention(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        *,
        scaling: float,
        q_head_num: int,
        kv_head_num: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        q = query.movedim(0, 1)
        k = key.movedim(0, 1)
        v = value.movedim(0, 1)

        if q_head_num != kv_head_num:
            assert q_head_num % kv_head_num == 0
            repeat = q_head_num // kv_head_num
            k = k.repeat_interleave(repeat, dim=0)
            v = v.repeat_interleave(repeat, dim=0)

        q_float = q.float()
        k_float = k.float()
        v_float = v.float()
        attn_logits = torch.matmul(q_float, k_float.transpose(-2, -1)) * scaling
        attn_weights = torch.softmax(attn_logits, dim=-1)
        output = torch.matmul(attn_weights, v_float).to(dtype=query.dtype)
        return output.movedim(0, 1), attn_weights[:, 0, :]

    def _validate_against_gpu_cache(
        self,
        *,
        token_indices: torch.Tensor,
        key_subset: torch.Tensor,
        value_subset: torch.Tensor,
        k_cache: torch.Tensor,
        v_cache: torch.Tensor,
    ) -> dict | None:
        if int(token_indices.numel()) == 0:
            return None
        if bool((token_indices < 0).any().item()):
            return {"skipped": "offloaded_slots"}
        limit = min(8, int(token_indices.numel()), int(key_subset.shape[0]))
        sample_indices = token_indices[:limit]
        gpu_keys = k_cache[sample_indices].to(dtype=key_subset.dtype)
        gpu_values = v_cache[sample_indices].to(dtype=value_subset.dtype)
        key_diff = (key_subset[:limit] - gpu_keys).abs().max()
        value_diff = (value_subset[:limit] - gpu_values).abs().max()
        return {
            "sample": limit,
            "key_max_abs_diff": float(key_diff.item()),
            "value_max_abs_diff": float(value_diff.item()),
        }

    def _kv_cache_check_failed(self, validation: dict | None) -> bool:
        if validation is None:
            return False
        return (
            float(validation.get("key_max_abs_diff", 0.0)) > 1e-3
            or float(validation.get("value_max_abs_diff", 0.0)) > 1e-3
        )

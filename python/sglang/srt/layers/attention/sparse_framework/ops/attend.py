from __future__ import annotations

from dataclasses import dataclass

import torch

from sglang.srt.layers.attention.sparse_framework.ops.batched_decode_attention import (
    batched_sparse_decode_attention,
    batched_sparse_decode_attention_with_weights,
    can_use_batched_sparse_decode,
)
from sglang.srt.layers.attention.sparse_framework.kv_store import get_cpu_kv_store
from sglang.srt.layers.attention.sparse_framework.ops.base import BaseSparseOp
from sglang.srt.layers.attention.sparse_framework.ops.utils import (
    configure_cpu_kv_store_from_state,
    cpu_kv_store_enabled,
    rebuild_packed_kv_indices,
)
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
            self._rewrite_current_decode_indices(ctx, state)
            self._store_current_decode_kv(ctx, layer, k, v, state)

        q = q.reshape(-1, layer.tp_q_head_num * layer.qk_head_dim)
        if layer.qk_head_dim != layer.v_head_dim:
            output = q.new_empty((q.shape[0], layer.tp_q_head_num * layer.v_head_dim))
        else:
            output = torch.empty_like(q)

        q_view = q.view(-1, layer.tp_q_head_num, layer.qk_head_dim)
        output_view = output.view(-1, layer.tp_q_head_num, layer.v_head_dim)
        k_cache = forward_batch.token_to_kv_pool.get_key_buffer(layer.layer_id)
        v_cache = forward_batch.token_to_kv_pool.get_value_buffer(layer.layer_id)
        attention_weights = [None] * len(selected_kv_indices)
        working_set_stats = []
        records = []

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
            h2d_event = ws_stats.pop("_h2d_event", None)
            if key_subset is None or value_subset is None:
                working_set_stats.append(ws_stats)
                state["working_set_result"] = working_set_stats
                state.setdefault("subset_unavailable_reason", "missing_working_set")
                return None
            records.append(
                {
                    "batch_idx": batch_idx,
                    "key": key_subset,
                    "value": value_subset,
                    "event": h2d_event,
                }
            )
            working_set_stats.append(ws_stats)

        plan = state.get("execution_plan")
        requires_scores = bool(
            getattr(getattr(plan, "selection_plan", None), "requires_scores", False)
        )
        ready_records = []
        pending_records = []
        for record in records:
            event = record["event"]
            if event is None:
                ready_records.append(record)
                continue
            if self._event_ready(event):
                ready_records.append(record)
            else:
                pending_records.append(record)

        kernel_names = []
        if ready_records:
            kernel_names.append(
                self._compute_record_group(
                    ready_records,
                    q_view=q_view,
                    output_view=output_view,
                    attention_weights=attention_weights,
                    layer=layer,
                    requires_scores=requires_scores,
                    state=state,
                )
            )
        for record in pending_records:
            event = record["event"]
            if event is not None:
                self._wait_event(event, device=q_view.device)
        if pending_records:
            kernel_names.append(
                self._compute_record_group(
                    pending_records,
                    q_view=q_view,
                    output_view=output_view,
                    attention_weights=attention_weights,
                    layer=layer,
                    requires_scores=requires_scores,
                    state=state,
                )
            )

        state["subset_attention_kernel"] = "+".join(kernel_names) if kernel_names else "none"
        state["inter_request_overlap"] = {
            "ready": len(ready_records),
            "pending": len(pending_records),
            "granularity": "request_working_set_h2d",
        }

        state["attention_output"] = output
        state["attention_weights"] = [item for item in attention_weights if item is not None]
        state["working_set_result"] = working_set_stats
        return output

    def _compute_record_group(
        self,
        records: list[dict],
        *,
        q_view: torch.Tensor,
        output_view: torch.Tensor,
        attention_weights: list,
        layer,
        requires_scores: bool,
        state: dict,
    ) -> str:
        if not records:
            return "none"
        key_subsets = [record["key"] for record in records]
        value_subsets = [record["value"] for record in records]
        state["subset_tensor_debug"] = {
            "records": len(records),
            "key_shapes": [tuple(item.shape) for item in key_subsets if item is not None],
            "value_shapes": [tuple(item.shape) for item in value_subsets if item is not None],
            "q_shape": tuple(q_view.shape),
        }
        if all(
            key is not None and value is not None
            for key, value in zip(key_subsets, value_subsets)
        ) and all(
            can_use_batched_sparse_decode(
                query=q_view,
                key=key,
                value=value,
                q_head_num=layer.tp_q_head_num,
                kv_head_num=layer.tp_k_head_num,
            )
            for key, value in zip(key_subsets, value_subsets)
        ):
            batch_indices = [int(record["batch_idx"]) for record in records]
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
            group_q = q_view[batch_indices]
            if requires_scores:
                group_output, packed_weights = batched_sparse_decode_attention_with_weights(
                    query=group_q,
                    key=packed_keys,
                    value=packed_values,
                    kv_indptr=packed_indptr,
                    scaling=layer.scaling,
                    q_head_num=layer.tp_q_head_num,
                    kv_head_num=layer.tp_k_head_num,
                )
                for local_idx, batch_idx in enumerate(batch_indices):
                    start = int(packed_indptr[local_idx].item())
                    end = int(packed_indptr[local_idx + 1].item())
                    attention_weights[batch_idx] = packed_weights[start:end].transpose(
                        0, 1
                    )
            else:
                group_output = batched_sparse_decode_attention(
                    query=group_q,
                    key=packed_keys,
                    value=packed_values,
                    kv_indptr=packed_indptr,
                    scaling=layer.scaling,
                    q_head_num=layer.tp_q_head_num,
                    kv_head_num=layer.tp_k_head_num,
                )
            if self._should_validate_subset_kernel(state):
                torch_output, torch_weights = self._compute_record_group_torch(
                    records,
                    q_view=q_view,
                    layer=layer,
                )
                max_abs_diff = (group_output - torch_output).abs().max()
                check = {
                    "max_abs_diff": float(max_abs_diff.item()),
                    "records": len(records),
                }
                state["triton_torch_output_check"] = check
                if float(check["max_abs_diff"]) > 1e-2:
                    group_output = torch_output
                    for local_idx, batch_idx in enumerate(batch_indices):
                        attention_weights[batch_idx] = torch_weights[local_idx]
                    state["subset_attention_kernel_replaced"] = (
                        "triton_to_torch_per_request"
                    )
            output_view[batch_indices] = group_output
            return (
                "triton_batched_sparse_decode_with_weights"
                if requires_scores
                else "triton_batched_sparse_decode"
            )

        for record in records:
            batch_idx = int(record["batch_idx"])
            per_req_out, per_req_weights = self._compute_subset_attention(
                q_view[batch_idx : batch_idx + 1],
                record["key"],
                record["value"],
                scaling=layer.scaling,
                q_head_num=layer.tp_q_head_num,
                kv_head_num=layer.tp_k_head_num,
            )
            output_view[batch_idx : batch_idx + 1] = per_req_out
            attention_weights[batch_idx] = per_req_weights
        return "torch_per_request"

    def _should_validate_subset_kernel(self, state: dict) -> bool:
        plan = state.get("execution_plan")
        return bool(getattr(plan, "validate_kv_cache", False))

    def _compute_record_group_torch(
        self,
        records: list[dict],
        *,
        q_view: torch.Tensor,
        layer,
    ) -> tuple[torch.Tensor, list[torch.Tensor]]:
        outputs = []
        weights = []
        for record in records:
            batch_idx = int(record["batch_idx"])
            per_req_out, per_req_weights = self._compute_subset_attention(
                q_view[batch_idx : batch_idx + 1],
                record["key"],
                record["value"],
                scaling=layer.scaling,
                q_head_num=layer.tp_q_head_num,
                kv_head_num=layer.tp_k_head_num,
            )
            outputs.append(per_req_out[0])
            weights.append(per_req_weights)
        return torch.stack(outputs), weights

    def _rewrite_current_decode_indices(self, ctx, state: dict) -> None:
        selected_positions = state.get("selected_positions") or []
        selected_kv_indices = state.get("selected_kv_indices") or []
        out_cache_locs = getattr(ctx, "out_cache_locs_cpu", [])
        if not selected_positions or not selected_kv_indices or not out_cache_locs:
            return
        req_to_token = ctx.req_to_token_pool.req_to_token
        updated = 0
        for batch_idx, req_pool_idx in enumerate(ctx.req_pool_indices_cpu):
            if (
                batch_idx >= len(selected_positions)
                or batch_idx >= len(selected_kv_indices)
                or batch_idx >= len(out_cache_locs)
                or batch_idx >= len(ctx.seq_lens_cpu)
            ):
                continue
            current_position = int(ctx.seq_lens_cpu[batch_idx]) - 1
            cache_loc = int(out_cache_locs[batch_idx])
            if current_position < 0 or cache_loc < 0:
                continue
            if current_position < int(req_to_token.shape[1]):
                req_to_token[int(req_pool_idx), current_position] = cache_loc
            positions = selected_positions[batch_idx]
            if int(positions.numel()) == 0:
                continue
            matches = positions == current_position
            if not bool(matches.any().item()):
                continue
            kv_indices = selected_kv_indices[batch_idx].clone()
            kv_indices[matches.to(device=kv_indices.device)] = cache_loc
            selected_kv_indices[batch_idx] = kv_indices
            updated += int(matches.sum().item())
        if updated:
            state["selected_kv_indices"] = selected_kv_indices
            rebuild_packed_kv_indices(state)
        state["current_decode_rewrite"] = {"updated": updated}

    def _store_current_decode_kv(
        self, ctx, layer, key: torch.Tensor, value: torch.Tensor, state: dict
    ) -> None:
        if not cpu_kv_store_enabled(state):
            return
        store = get_cpu_kv_store(ctx.framework_state)
        if store is None:
            return
        configure_cpu_kv_store_from_state(store, state)
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
        store = (
            get_cpu_kv_store(ctx.framework_state)
            if cpu_kv_store_enabled(state)
            else None
        )
        configure_cpu_kv_store_from_state(store, state)
        buffer = get_working_set_buffer(ctx.framework_state)
        req_pool_idx = int(ctx.req_pool_indices_cpu[batch_idx])
        plan = state.get("execution_plan")
        chunk_selection = state.get("chunk_selection") or {}
        unit_size = (
            int(chunk_selection.get("chunk_size", 16))
            if (
                bool(chunk_selection.get("enabled"))
                and bool(getattr(plan, "use_chunked_working_set", False))
            )
            else 1
        )
        use_delta = buffer is not None and bool(token_positions)
        delta_plan = None
        fetch_positions = list(token_positions)
        fetch_token_indices = token_indices
        if use_delta:
            delta_plan = buffer.plan_delta(
                req_pool_idx=req_pool_idx,
                layer_id=int(layer.layer_id),
                positions=token_positions,
                unit_size=unit_size,
                device=k_cache.device,
                dtype=k_cache.dtype,
                key_shape_tail=tuple(k_cache.shape[1:]),
                max_position=int(ctx.seq_lens_cpu[batch_idx]),
                full_miss_units=unit_size > 1,
            )
            fetch_positions = list(delta_plan["miss_positions"])
            if fetch_positions:
                fetch_position_tensor = torch.tensor(
                    fetch_positions, dtype=torch.long, device=token_indices.device
                )
                fetch_token_indices = ctx.req_to_token_pool.req_to_token[
                    req_pool_idx, fetch_position_tensor
                ].to(torch.long)
            else:
                fetch_token_indices = token_indices[:0]

        cpu_keys = None
        cpu_values = None
        cpu_positions = []
        missing_positions = list(fetch_positions)
        cpu_missing_gpu_available = 0
        cpu_missing_gpu_unavailable = 0
        h2d_event = None
        chunk_fetch_stats = None
        resident_only_gpu_kv = bool(
            getattr(ctx.token_to_kv_pool, "is_sparse_layerwise_staging_pool", False)
        )
        gpu_resident_fetch = bool(
            not resident_only_gpu_kv
            and
            fetch_positions
            and int(fetch_token_indices.numel()) == len(fetch_positions)
            and bool((fetch_token_indices >= 0).all().item())
        )
        if gpu_resident_fetch:
            key_subset = k_cache[fetch_token_indices]
            value_subset = v_cache[fetch_token_indices]
            cpu_keys = None
            cpu_values = None
            cpu_positions = []
            missing_positions = []
            gpu_fallback_count = int(fetch_token_indices.numel())
            prefetch = None
        else:
            key_subset = None
            value_subset = None
            prefetch = self._consume_sparse_cpu_prefetch(
                ctx,
                req_pool_idx=req_pool_idx,
                layer_id=int(layer.layer_id),
                positions=fetch_positions,
            )
        if gpu_resident_fetch:
            pass
        elif prefetch is not None:
            cpu_keys, cpu_values, cpu_positions, h2d_event = prefetch
            cpu_position_set = {int(pos) for pos in cpu_positions}
            missing_positions = [
                int(pos) for pos in fetch_positions if int(pos) not in cpu_position_set
            ]
            if missing_positions and h2d_event is not None:
                self._wait_event(h2d_event, device=k_cache.device)
                h2d_event = None
        elif store is not None and fetch_positions:
            get_many_chunked_async = getattr(store, "get_many_chunked_async", None)
            if (
                unit_size > 1
                and bool(chunk_selection.get("enabled"))
                and callable(get_many_chunked_async)
            ):
                (
                    cpu_keys,
                    cpu_values,
                    cpu_positions,
                    missing_positions,
                    h2d_event,
                    chunk_fetch_stats,
                ) = get_many_chunked_async(
                    req_pool_idx=req_pool_idx,
                    layer_id=int(layer.layer_id),
                    positions=fetch_positions,
                    device=k_cache.device,
                    dtype=k_cache.dtype,
                )
            else:
                get_many_async = getattr(store, "get_many_async", None)
                if callable(get_many_async):
                    (
                        cpu_keys,
                        cpu_values,
                        cpu_positions,
                        missing_positions,
                        h2d_event,
                    ) = get_many_async(
                        req_pool_idx=req_pool_idx,
                        layer_id=int(layer.layer_id),
                        positions=fetch_positions,
                        device=k_cache.device,
                        dtype=k_cache.dtype,
                    )
                else:
                    cpu_keys, cpu_values, cpu_positions, missing_positions = store.get_many(
                        req_pool_idx=req_pool_idx,
                        layer_id=int(layer.layer_id),
                        positions=fetch_positions,
                        device=k_cache.device,
                        dtype=k_cache.dtype,
                    )
        elif use_delta and not fetch_positions:
            missing_positions = []
        if gpu_resident_fetch:
            pass
        elif not missing_positions:
            key_subset = cpu_keys
            value_subset = cpu_values
            gpu_fallback_count = 0
        elif cpu_keys is None or cpu_values is None:
            cpu_missing_gpu_available = int((fetch_token_indices >= 0).sum().item())
            cpu_missing_gpu_unavailable = int((fetch_token_indices < 0).sum().item())
            if resident_only_gpu_kv or cpu_missing_gpu_unavailable:
                state["subset_unavailable_reason"] = "missing_kv_not_resident"
                return None, None, {
                    "requested": len(token_positions),
                    "cpu": 0,
                    "gpu_fallback": 0,
                    "cpu_missing": len(missing_positions),
                    "cpu_missing_gpu_available": cpu_missing_gpu_available,
                    "cpu_missing_gpu_unavailable": cpu_missing_gpu_unavailable,
                    "buffered": 0,
                    "kv_cache_check": None,
                    "prefetched": prefetch is not None,
                    "h2d_async": False,
                    "chunk_fetch": chunk_fetch_stats,
                    "working_set_delta": {
                        "enabled": bool(use_delta),
                        "reason": "missing_kv_not_resident",
                    },
                    "_h2d_event": None,
                }
            key_subset = k_cache[fetch_token_indices]
            value_subset = v_cache[fetch_token_indices]
            h2d_event = None
            gpu_fallback_count = int(fetch_token_indices.numel())
            self._snapshot_missing_to_cpu(
                store,
                req_pool_idx=req_pool_idx,
                layer_id=int(layer.layer_id),
                positions=fetch_positions,
                keys=key_subset,
                values=value_subset,
            )
        else:
            position_to_index = {int(pos): i for i, pos in enumerate(fetch_positions)}
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
            missing_token_indices = fetch_token_indices[missing_index_tensor]
            cpu_missing_gpu_available = int((missing_token_indices >= 0).sum().item())
            cpu_missing_gpu_unavailable = int((missing_token_indices < 0).sum().item())
            if resident_only_gpu_kv or cpu_missing_gpu_unavailable:
                state["subset_unavailable_reason"] = "missing_kv_not_resident"
                return None, None, {
                    "requested": len(token_positions),
                    "cpu": len(cpu_positions),
                    "gpu_fallback": 0,
                    "cpu_missing": len(missing_positions),
                    "cpu_missing_gpu_available": cpu_missing_gpu_available,
                    "cpu_missing_gpu_unavailable": cpu_missing_gpu_unavailable,
                    "buffered": 0,
                    "kv_cache_check": None,
                    "prefetched": prefetch is not None,
                    "h2d_async": False,
                    "chunk_fetch": chunk_fetch_stats,
                    "working_set_delta": {
                        "enabled": bool(use_delta),
                        "reason": "missing_kv_not_resident",
                    },
                    "_h2d_event": None,
                }
            gpu_keys = k_cache[missing_token_indices]
            gpu_values = v_cache[missing_token_indices]
            key_parts = []
            value_parts = []
            cpu_lookup = {pos: i for i, pos in enumerate(cpu_positions)}
            gpu_lookup = {pos: i for i, pos in enumerate(missing_positions)}
            for pos in fetch_positions:
                if pos in cpu_lookup:
                    key_parts.append(cpu_keys[cpu_lookup[pos]])
                    value_parts.append(cpu_values[cpu_lookup[pos]])
                elif pos in gpu_lookup:
                    key_parts.append(gpu_keys[gpu_lookup[pos]])
                    value_parts.append(gpu_values[gpu_lookup[pos]])
            key_subset = torch.stack(key_parts)
            value_subset = torch.stack(value_parts)
            h2d_event = None
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
            and not use_delta
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
                h2d_event = None
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

        if buffer is not None and h2d_event is None:
            if use_delta:
                materialized_positions = (
                    fetch_positions
                    if key_subset is not None and value_subset is not None
                    else []
                )
                key_subset, value_subset, delta_stats = (
                    buffer.materialize_delta_from_partial(
                        req_pool_idx=req_pool_idx,
                        layer_id=int(layer.layer_id),
                        positions=token_positions,
                        materialized_positions=materialized_positions,
                        key=key_subset,
                        value=value_subset,
                        unit_size=unit_size,
                        device=k_cache.device,
                        key_dtype=k_cache.dtype,
                        value_dtype=v_cache.dtype,
                        key_shape_tail=tuple(k_cache.shape[1:]),
                        value_shape_tail=tuple(v_cache.shape[1:]),
                    )
                )
            elif key_subset is not None and value_subset is not None:
                key_subset, value_subset = buffer.materialize(
                    layer_id=int(layer.layer_id),
                    key=key_subset,
                    value=value_subset,
                )
                delta_stats = {"enabled": False, "layout": "token"}
            else:
                delta_stats = {"enabled": False, "reason": "missing_tensor"}
        else:
            delta_stats = {"enabled": False, "reason": "pending_or_missing_tensor"}

        if (
            use_delta
            and key_subset is not None
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
                key_subset = k_cache[token_indices]
                value_subset = v_cache[token_indices]
                h2d_event = None
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
                key_subset, value_subset = buffer.materialize(
                    layer_id=int(layer.layer_id),
                    key=key_subset,
                    value=value_subset,
                )
                validation["fallback_reason"] = "chunked_working_set_mismatch"
                delta_stats = {"enabled": False, "layout": "token_fallback"}

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
            "gpu_resident_preferred": gpu_resident_fetch,
            "h2d_async": h2d_event is not None,
            "chunk_fetch": chunk_fetch_stats,
            "working_set_delta": delta_stats,
            "_h2d_event": h2d_event,
        }

    def _consume_sparse_cpu_prefetch(
        self,
        ctx,
        *,
        req_pool_idx: int,
        layer_id: int,
        positions: list[int],
    ) -> tuple[torch.Tensor, torch.Tensor, list[int], torch.cuda.Event | None] | None:
        if ctx.framework_state is None or not positions:
            return None
        pending = ctx.framework_state.get("sparse_cpu_prefetches") or {}
        key = (int(req_pool_idx), int(layer_id), tuple(int(pos) for pos in positions))
        item = pending.pop(key, None)
        if item is None:
            return self._consume_sparse_cpu_position_prefetch(
                pending,
                req_pool_idx=req_pool_idx,
                layer_id=layer_id,
                positions=positions,
            )
        return item["keys"], item["values"], item["positions"], item.get("event")

    def _consume_sparse_cpu_position_prefetch(
        self,
        pending: dict,
        *,
        req_pool_idx: int,
        layer_id: int,
        positions: list[int],
    ) -> tuple[torch.Tensor, torch.Tensor, list[int], object] | None:
        key_parts = []
        value_parts = []
        consumed_positions = []
        events = []
        for pos in positions:
            item = pending.pop((int(req_pool_idx), int(layer_id), int(pos)), None)
            if item is None:
                continue
            keys = item["keys"]
            values = item["values"]
            if keys is None or values is None or int(keys.shape[0]) == 0:
                continue
            key_parts.append(keys[0])
            value_parts.append(values[0])
            consumed_positions.append(int(pos))
            event = item.get("event")
            if event is not None:
                events.append(event)
        if not key_parts:
            return None
        event = None
        if events:
            event = events[0] if all(item is events[0] for item in events) else events
        return (
            torch.stack(key_parts),
            torch.stack(value_parts),
            consumed_positions,
            event,
        )

    def _event_ready(self, event) -> bool:
        if event is None:
            return True
        if isinstance(event, (list, tuple)):
            return all(self._event_ready(item) for item in event)
        query_event = getattr(event, "query", None)
        return bool(query_event()) if callable(query_event) else False

    def _wait_event(self, event, *, device) -> None:
        if event is None:
            return
        stream = torch.cuda.current_stream(device=device)
        if isinstance(event, (list, tuple)):
            for item in event:
                self._wait_event(item, device=device)
            return
        stream.wait_event(event)

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

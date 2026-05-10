from __future__ import annotations

import torch
from sglang.srt.layers.attention.sparse_framework.callback_registry import (
    SelectionCallbackRegistry,
)
from sglang.srt.layers.attention.sparse_framework.selection_result import (
    RequestSelectionView,
    SelectionResult,
)
from sglang.srt.layers.attention.sparse_framework.selection_spec import (
    RetrievalSelectionSpec,
)


class RetrievalSelector:
    def __init__(self):
        self.last_stats: dict = {}
        self.last_priority: dict[int, float] = {}

    def select_positions(
        self,
        spec: RetrievalSelectionSpec,
        ctx,
        *,
        request_index: int,
        req_pool_idx: int,
        seq_len: int,
    ) -> list[int]:
        self.last_stats = {}
        self.last_priority = {}
        if spec.name or spec.import_path or spec.fn:
            positions = self._callback_positions(
                spec,
                ctx,
                request_index=request_index,
                req_pool_idx=req_pool_idx,
                seq_len=seq_len,
            )
            self._set_stats(
                prefix_len=0,
                suffix_len=0,
                retrieval_len=len(positions),
                total_len=len(set(positions)),
                method="callback",
                budget=spec.top_k,
                middle_len=seq_len,
            )
            return positions
        if spec.method == "similarity":
            positions = self._similarity_positions(
                spec,
                ctx,
                request_index=request_index,
                req_pool_idx=req_pool_idx,
                seq_len=seq_len,
            )
            if positions is not None:
                return positions
        return self._fallback_positions(spec, seq_len)

    def _callback_positions(
        self,
        spec: RetrievalSelectionSpec,
        ctx,
        *,
        request_index: int,
        req_pool_idx: int,
        seq_len: int,
    ) -> list[int]:
        callback = SelectionCallbackRegistry.resolve_parts(
            name=spec.name,
            import_path=spec.import_path,
            fn=spec.fn,
        )
        layer_id = getattr(ctx.layer, "layer_id", None)
        request_view = RequestSelectionView(
            request_index=request_index,
            req_pool_idx=req_pool_idx,
            seq_len=seq_len,
            layer_id=layer_id,
        )
        try:
            value = callback(
                ctx=ctx,
                request_view=request_view,
                request_index=request_index,
                req_pool_idx=req_pool_idx,
                seq_len=seq_len,
                layer=ctx.layer,
                layer_id=layer_id,
                query=ctx.query,
                top_k=spec.top_k,
                method=spec.method,
                **spec.kwargs,
            )
        except TypeError:
            value = callback(ctx, request_view, layer_id)
        result = SelectionResult.from_callback_output(value)
        positions = result.to_positions(seq_len)
        if result.priority:
            self.last_priority = {
                int(pos): float(priority)
                for pos, priority in result.priority.items()
                if 0 <= int(pos) < seq_len
            }
        else:
            self.last_priority = self._rank_priority(positions)
        return positions

    def _similarity_positions(
        self,
        spec: RetrievalSelectionSpec,
        ctx,
        *,
        request_index: int,
        req_pool_idx: int,
        seq_len: int,
    ) -> list[int] | None:
        layer = ctx.layer
        query = ctx.query
        if layer is None or query is None or seq_len <= 0:
            return None
        if ctx.req_to_token_pool is None or ctx.token_to_kv_pool is None:
            return None

        prefix_len, suffix_len = self._static_span_lengths(spec, seq_len)
        selected = self._static_positions(prefix_len, suffix_len, seq_len)

        middle_start = prefix_len
        middle_end = max(middle_start, seq_len - suffix_len)
        if middle_end <= middle_start:
            positions = sorted(selected)
            self.last_priority = self._rank_priority(positions)
            self._set_stats(
                prefix_len=prefix_len,
                suffix_len=suffix_len,
                retrieval_len=0,
                total_len=len(positions),
                method="similarity",
                budget=0,
                middle_len=0,
            )
            return positions

        candidate_positions = torch.arange(
            middle_start,
            middle_end,
            dtype=torch.long,
            device=ctx.seq_lens.device,
        )
        if candidate_positions.numel() == 0:
            positions = sorted(selected)
            self.last_priority = self._rank_priority(positions)
            self._set_stats(
                prefix_len=prefix_len,
                suffix_len=suffix_len,
                retrieval_len=0,
                total_len=len(positions),
                method="similarity",
                budget=0,
                middle_len=0,
            )
            return positions

        cpu_positions = self._cpu_index_positions(
            spec,
            ctx,
            req_pool_idx=req_pool_idx,
            layer_id=int(layer.layer_id),
            query=query.reshape(-1, layer.tp_q_head_num, layer.qk_head_dim)[request_index],
            candidate_positions=candidate_positions.detach().cpu().tolist(),
            selected=selected,
            seq_len=seq_len,
            prefix_len=prefix_len,
            suffix_len=suffix_len,
        )
        if cpu_positions is not None:
            return cpu_positions

        req_to_token = ctx.req_to_token_pool.req_to_token
        candidate_kv_indices = req_to_token[req_pool_idx, candidate_positions].to(
            torch.long
        )
        key_cache = ctx.token_to_kv_pool.get_key_buffer(layer.layer_id)
        candidate_keys = key_cache[candidate_kv_indices]
        if candidate_keys.numel() == 0:
            positions = sorted(selected)
            self.last_priority = self._rank_priority(positions)
            self._set_stats(
                prefix_len=prefix_len,
                suffix_len=suffix_len,
                retrieval_len=0,
                total_len=len(positions),
                method="similarity",
                budget=0,
                middle_len=int(candidate_positions.numel()),
            )
            return positions

        q = query.reshape(-1, layer.tp_q_head_num, layer.qk_head_dim)[request_index]
        scores = self._score_candidates(
            q,
            candidate_keys,
            q_head_num=layer.tp_q_head_num,
            kv_head_num=layer.tp_k_head_num,
        )
        retrieval_budget = self._retrieval_budget(spec, int(scores.numel()))
        k = min(retrieval_budget, int(scores.numel()))
        if k <= 0:
            positions = sorted(selected)
            self.last_priority = self._rank_priority(positions)
            self._set_stats(
                prefix_len=prefix_len,
                suffix_len=suffix_len,
                retrieval_len=0,
                total_len=len(positions),
                method="similarity",
                budget=retrieval_budget,
                middle_len=int(candidate_positions.numel()),
            )
            return positions
        top_indices = torch.topk(scores, k=k, largest=True).indices
        retrieval_positions = [
            int(pos) for pos in candidate_positions[top_indices].tolist()
        ]
        retrieval_positions = self._expand_positions_to_pages(
            retrieval_positions,
            seq_len=seq_len,
            limit=retrieval_budget,
            exclude=selected,
            page_size=max(1, int(getattr(ctx.model_runner, "page_size", 1))),
        )
        selected.update(retrieval_positions)
        positions = sorted(selected)
        self.last_priority = self._rank_priority(retrieval_positions)
        self._set_stats(
            prefix_len=prefix_len,
            suffix_len=suffix_len,
            retrieval_len=len(retrieval_positions),
            total_len=len(positions),
            method="similarity",
            budget=retrieval_budget,
            middle_len=int(candidate_positions.numel()),
        )
        return positions

    def _cpu_index_positions(
        self,
        spec: RetrievalSelectionSpec,
        ctx,
        *,
        req_pool_idx: int,
        layer_id: int,
        query: torch.Tensor,
        candidate_positions: list[int],
        selected: set[int],
        seq_len: int,
        prefix_len: int,
        suffix_len: int,
    ) -> list[int] | None:
        store = (
            ctx.framework_state.get("cpu_kv_store")
            if ctx.framework_state is not None
            else None
        )
        if store is None or not candidate_positions:
            return None

        retrieval_budget = self._retrieval_budget(spec, len(candidate_positions))
        chunk_size = int(spec.kwargs.get("chunk_size", 16)) if spec.kwargs else 16
        index = store.build_chunk_index(
            req_pool_idx=req_pool_idx,
            layer_id=layer_id,
            positions=candidate_positions,
            chunk_size=chunk_size,
        )
        if not index or not index.get("chunks"):
            return None

        q = query.detach().to("cpu", dtype=torch.float32)
        chunk_scores = []
        for chunk in index["chunks"]:
            centroid = chunk["centroid"]
            if q_head_num := int(q.shape[0]):
                if centroid.shape[0] != q_head_num and q_head_num % centroid.shape[0] == 0:
                    centroid = centroid.repeat_interleave(q_head_num // centroid.shape[0], dim=0)
            score = torch.einsum("hd,hd->h", q, centroid.to(torch.float32)).mean()
            chunk_scores.append((float(score.item()), chunk["positions"]))
        if not chunk_scores:
            return None

        retrieval_positions = []
        seen = set(selected)
        for _, chunk_positions in sorted(chunk_scores, key=lambda item: item[0], reverse=True):
            for pos in chunk_positions:
                pos = int(pos)
                if pos in seen:
                    continue
                seen.add(pos)
                retrieval_positions.append(pos)
                if len(retrieval_positions) >= retrieval_budget:
                    break
            if len(retrieval_positions) >= retrieval_budget:
                break

        selected.update(retrieval_positions)
        positions = sorted(selected)
        self.last_priority = self._rank_priority(retrieval_positions)
        self._set_stats(
            prefix_len=prefix_len,
            suffix_len=suffix_len,
            retrieval_len=len(retrieval_positions),
            total_len=len(positions),
            method="cpu_chunk_similarity",
            budget=retrieval_budget,
            middle_len=len(candidate_positions),
        )
        return positions

    def _score_candidates(
        self,
        query: torch.Tensor,
        candidate_keys: torch.Tensor,
        *,
        q_head_num: int,
        kv_head_num: int,
    ) -> torch.Tensor:
        keys = candidate_keys
        if q_head_num != kv_head_num:
            assert q_head_num % kv_head_num == 0
            repeat = q_head_num // kv_head_num
            keys = keys.repeat_interleave(repeat, dim=1)
        scores = torch.einsum("hd,thd->ht", query.float(), keys.float())
        return scores.mean(dim=0)

    def _fallback_positions(
        self,
        spec: RetrievalSelectionSpec,
        seq_len: int,
    ) -> list[int]:
        prefix_len, suffix_len = self._static_span_lengths(spec, seq_len)
        selected = self._static_positions(prefix_len, suffix_len, seq_len)

        middle_start = prefix_len
        middle_end = max(middle_start, seq_len - suffix_len)
        middle_len = max(0, middle_end - middle_start)
        if middle_len == 0:
            positions = sorted(selected)
            self.last_priority = self._rank_priority(positions)
            self._set_stats(
                prefix_len=prefix_len,
                suffix_len=suffix_len,
                retrieval_len=0,
                total_len=len(positions),
                method="fallback",
                budget=0,
                middle_len=0,
            )
            return positions

        budget = self._retrieval_budget(spec, middle_len)
        stride = max(1, middle_len // budget)
        pos = middle_start + stride // 2
        retrieval_positions = []
        while len(retrieval_positions) < budget and pos < middle_end:
            if pos not in selected:
                retrieval_positions.append(pos)
                selected.add(pos)
            pos += stride
        pos = middle_start
        while len(retrieval_positions) < budget and pos < middle_end:
            if pos not in selected:
                retrieval_positions.append(pos)
                selected.add(pos)
            pos += 1
        positions = sorted(selected)
        self.last_priority = self._rank_priority(retrieval_positions)
        self._set_stats(
            prefix_len=prefix_len,
            suffix_len=suffix_len,
            retrieval_len=len(retrieval_positions),
            total_len=len(positions),
            method="fallback",
            budget=budget,
            middle_len=middle_len,
        )
        return positions

    def _static_span_lengths(
        self,
        spec: RetrievalSelectionSpec,
        seq_len: int,
    ) -> tuple[int, int]:
        prefix_len = min(max(0, spec.static_prefix), seq_len)
        suffix_len = min(max(0, spec.static_suffix), max(0, seq_len - prefix_len))
        return prefix_len, suffix_len

    def _static_positions(
        self,
        prefix_len: int,
        suffix_len: int,
        seq_len: int,
    ) -> set[int]:
        selected = set(range(prefix_len))
        if suffix_len > 0:
            selected.update(range(max(prefix_len, seq_len - suffix_len), seq_len))
        return selected

    def _retrieval_budget(self, spec: RetrievalSelectionSpec, middle_len: int) -> int:
        if middle_len <= 0:
            return 0
        top_k = spec.top_k if spec.top_k > 0 else middle_len
        return max(1, min(middle_len, int(top_k)))

    def _expand_positions_to_pages(
        self,
        positions: list[int],
        *,
        seq_len: int,
        limit: int,
        exclude: set[int],
        page_size: int,
    ) -> list[int]:
        if limit <= 0:
            return []
        ordered = []
        seen = set(exclude)
        seen_pages = set()
        for pos in positions:
            page_start = (int(pos) // page_size) * page_size
            if page_start in seen_pages:
                continue
            seen_pages.add(page_start)
            page_end = min(seq_len, page_start + page_size)
            for token_pos in range(page_start, page_end):
                if token_pos in seen:
                    continue
                seen.add(token_pos)
                ordered.append(token_pos)
                if len(ordered) >= limit:
                    return ordered
        return ordered

    def _rank_priority(self, positions: list[int]) -> dict[int, float]:
        count = len(positions)
        if count <= 0:
            return {}
        return {int(pos): float(count - rank) for rank, pos in enumerate(positions)}

    def _set_stats(
        self,
        *,
        prefix_len: int,
        suffix_len: int,
        retrieval_len: int,
        total_len: int,
        method: str,
        budget: int,
        middle_len: int,
    ) -> None:
        self.last_stats = {
            "prefix": int(prefix_len),
            "suffix": int(suffix_len),
            "retrieval": int(retrieval_len),
            "total": int(total_len),
            "method": method,
            "budget": int(budget),
            "middle": int(middle_len),
        }

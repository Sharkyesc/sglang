from __future__ import annotations

from sglang.srt.layers.attention.retroinfer.types import RetroInferPlan


class RetroInferBatchPlanner:
    def __init__(self, capability_checker, session_manager):
        self.capability_checker = capability_checker
        self.session_manager = session_manager

    def plan_extend(self, layer, forward_batch) -> RetroInferPlan:
        decision = self.capability_checker.check_extend(layer, forward_batch)
        if not decision.allow:
            return RetroInferPlan(mode="fallback", reason=decision.reason)
        req_pool_indices = [int(x) for x in forward_batch.req_pool_indices.tolist()]
        seq_lens = [int(x) for x in forward_batch.seq_lens.tolist()]
        self.session_manager.observe_extend(req_pool_indices, seq_lens)
        self.session_manager.drop_missing_requests(req_pool_indices)
        return RetroInferPlan(mode="extend_observe", req_pool_indices=tuple(req_pool_indices))

    def plan_decode(self, layer, forward_batch) -> RetroInferPlan:
        decision = self.capability_checker.check_decode(layer, forward_batch)
        if not decision.allow:
            return RetroInferPlan(mode="fallback", reason=decision.reason)

        req_pool_indices = [int(x) for x in forward_batch.req_pool_indices.tolist()]
        seq_lens = [int(x) for x in forward_batch.seq_lens.tolist()]
        session = self.session_manager.get_or_create_session(req_pool_indices, seq_lens)
        return RetroInferPlan(
            mode=decision.mode,
            session=session,
            req_pool_indices=tuple(req_pool_indices),
            seq_len=seq_lens[0],
        )

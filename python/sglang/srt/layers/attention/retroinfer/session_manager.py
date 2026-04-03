from __future__ import annotations

from sglang.srt.layers.attention.retroinfer.types import (
    RetroInferRequestState,
    RetroInferSession,
)


class RetroInferSessionManager:
    def __init__(self):
        self.request_states: dict[int, RetroInferRequestState] = {}
        self.sessions: dict[tuple[int, ...], RetroInferSession] = {}
        self.step = 0

    def _get_request_state(self, req_pool_idx: int) -> RetroInferRequestState:
        state = self.request_states.get(req_pool_idx)
        if state is None:
            state = RetroInferRequestState(req_pool_idx=req_pool_idx)
            self.request_states[req_pool_idx] = state
        return state

    def observe_extend(self, req_pool_indices: list[int], seq_lens: list[int]) -> None:
        self.step += 1
        changed = set()
        for req_pool_idx, seq_len in zip(req_pool_indices, seq_lens):
            state = self._get_request_state(req_pool_idx)
            if seq_len < state.last_seq_len:
                changed.add(req_pool_idx)
            elif seq_len > state.last_seq_len:
                state.needs_rebuild = True
                changed.add(req_pool_idx)
            state.last_seq_len = seq_len
            state.last_access_step = self.step
        if changed:
            self.invalidate_requests(changed)

    def get_or_create_session(
        self,
        req_pool_indices: list[int],
        seq_lens: list[int],
    ) -> RetroInferSession:
        self.step += 1
        key = tuple(req_pool_indices)
        for req_pool_idx, seq_len in zip(req_pool_indices, seq_lens):
            state = self._get_request_state(req_pool_idx)
            if seq_len < state.last_seq_len:
                self.invalidate_requests({req_pool_idx})
                state = self._get_request_state(req_pool_idx)
            state.last_seq_len = seq_len
            state.last_access_step = self.step

        self._invalidate_overlapping_sessions(key)

        session = self.sessions.get(key)
        if session is None:
            session = RetroInferSession(
                key=key,
                batch_size=len(req_pool_indices),
                request_states={idx: self._get_request_state(idx) for idx in req_pool_indices},
            )
            self.sessions[key] = session
        return session

    def mark_decode_advanced(self, session: RetroInferSession, new_seq_len: int) -> None:
        for state in session.request_states.values():
            state.last_seq_len = new_seq_len
            state.last_access_step = self.step
            state.cpu_index_ready = True
            state.gpu_meta_ready = True
        session.prepared_seq_len = new_seq_len
        session.decode_steps += 1

    def invalidate_requests(self, req_pool_indices: set[int]) -> None:
        drop_keys = []
        for key, session in self.sessions.items():
            if any(req_pool_idx in session.request_states for req_pool_idx in req_pool_indices):
                drop_keys.append(key)
        for key in drop_keys:
            self.sessions.pop(key, None)
        for req_pool_idx in req_pool_indices:
            state = self.request_states.get(req_pool_idx)
            if state is not None:
                state.cpu_index_ready = False
                state.gpu_meta_ready = False
                state.needs_rebuild = True

    def _invalidate_overlapping_sessions(self, active_key: tuple[int, ...]) -> None:
        active = set(active_key)
        drop_keys = []
        for key in self.sessions:
            if key == active_key:
                continue
            if active.intersection(key):
                drop_keys.append(key)
        for key in drop_keys:
            self.sessions.pop(key, None)

    def drop_missing_requests(self, active_req_pool_indices: list[int]) -> None:
        active = set(active_req_pool_indices)
        to_drop = [req_pool_idx for req_pool_idx in self.request_states if req_pool_idx not in active]
        if to_drop:
            self.invalidate_requests(set(to_drop))
            for req_pool_idx in to_drop:
                self.request_states.pop(req_pool_idx, None)

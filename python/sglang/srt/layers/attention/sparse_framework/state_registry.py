from __future__ import annotations

import weakref


_REGISTERED_STATES: list[weakref.ReferenceType[dict]] = []


class _StateBox:
    __slots__ = ("state", "__weakref__")

    def __init__(self, state: dict):
        self.state = state


_BOXES: list[_StateBox] = []


def register_framework_state(state: dict) -> None:
    for box in _BOXES:
        if box.state is state:
            return
    box = _StateBox(state)
    _BOXES.append(box)
    _REGISTERED_STATES.append(weakref.ref(box))


def drop_request_state(req_pool_idx: int) -> dict:
    req_pool_idx = int(req_pool_idx)
    dropped = {
        "states": 0,
        "cpu_layers": 0,
        "working_set": 0,
        "pending_prefetches": 0,
        "residency": 0,
        "tracker": 0,
    }
    alive_refs = []
    alive_boxes = []
    for ref in _REGISTERED_STATES:
        box = ref()
        if box is None:
            continue
        alive_refs.append(ref)
        alive_boxes.append(box)
        state = box.state
        dropped["states"] += 1

        store = state.get("cpu_kv_store")
        if store is not None:
            drop_request = getattr(store, "drop_request", None)
            if callable(drop_request):
                dropped["cpu_layers"] += int(drop_request(req_pool_idx) or 0)

        working_set = state.get("working_set_buffer")
        if working_set is not None:
            drop_request = getattr(working_set, "drop_request", None)
            if callable(drop_request):
                dropped["working_set"] += int(drop_request(req_pool_idx) or 0)

        pending = state.get("sparse_cpu_prefetches")
        if pending:
            keys = [
                key
                for key in pending
                if isinstance(key, tuple) and key and int(key[0]) == req_pool_idx
            ]
            for key in keys:
                del pending[key]
            dropped["pending_prefetches"] += len(keys)

        table = state.get("residency_table")
        if table is not None:
            drop_request = getattr(table, "drop_request", None)
            if callable(drop_request):
                dropped["residency"] += int(drop_request(req_pool_idx) or 0)

        tracker = state.get("eviction_tracker")
        if tracker is not None:
            drop_request = getattr(tracker, "drop_request", None)
            if callable(drop_request):
                dropped["tracker"] += int(drop_request(req_pool_idx) or 0)

    _REGISTERED_STATES[:] = alive_refs
    _BOXES[:] = alive_boxes
    return dropped

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal


KVResidencyState = Literal["gpu", "host", "loading", "evicted"]
KVBackupSource = Literal["none", "host_pool", "sparse_cpu"]


@dataclass
class KVResidencyEntry:
    req_pool_idx: int
    layer_id: int
    position: int
    device_index: int | None = None
    host_index: int | None = None
    state: KVResidencyState = "gpu"
    backup_source: KVBackupSource = "none"
    owner: str = "request"
    version: int = 0
    last_access_step: int = 0
    access_count: int = 0
    selection_priority: float = 0.0
    last_selected_step: int = 0

    @property
    def logical_evicted(self) -> bool:
        return self.state in ("host", "evicted")

    @logical_evicted.setter
    def logical_evicted(self, value: bool) -> None:
        if value and self.state == "gpu":
            self.state = "evicted"
        elif not value and self.device_index is not None:
            self.state = "gpu"

    @property
    def has_sparse_cpu_backup(self) -> bool:
        return self.backup_source == "sparse_cpu"


class KVResidencyTable:
    def __init__(self):
        self.entries: dict[tuple[int, int, int], KVResidencyEntry] = {}
        self.entries_by_token: dict[tuple[int, int], set[tuple[int, int, int]]] = {}
        self.step = 0

    def next_step(self) -> int:
        self.step += 1
        return self.step

    def key(self, req_pool_idx: int, layer_id: int, position: int) -> tuple[int, int, int]:
        return (int(req_pool_idx), int(layer_id), int(position))

    def get(
        self, req_pool_idx: int, layer_id: int, position: int
    ) -> KVResidencyEntry | None:
        return self.entries.get(self.key(req_pool_idx, layer_id, position))

    def add_entry(self, entry: KVResidencyEntry) -> None:
        key = self.key(entry.req_pool_idx, entry.layer_id, entry.position)
        self.entries[key] = entry
        token_key = (int(entry.req_pool_idx), int(entry.position))
        self.entries_by_token.setdefault(token_key, set()).add(key)

    def entries_for_token(
        self, req_pool_idx: int, position: int
    ) -> list[KVResidencyEntry]:
        token_key = (int(req_pool_idx), int(position))
        keys = self.entries_by_token.get(token_key)
        if keys is None:
            keys = {
                key
                for key in self.entries
                if int(key[0]) == int(req_pool_idx) and int(key[2]) == int(position)
            }
            if keys:
                self.entries_by_token[token_key] = keys
        return [self.entries[key] for key in keys or () if key in self.entries]

    def observe_gpu(
        self,
        *,
        req_pool_idx: int,
        layer_id: int,
        position: int,
        device_index: int,
        step: int,
    ) -> tuple[KVResidencyEntry, bool]:
        key = self.key(req_pool_idx, layer_id, position)
        entry = self.entries.get(key)
        is_miss = entry is None or entry.state != "gpu"
        if entry is None:
            entry = KVResidencyEntry(
                req_pool_idx=int(req_pool_idx),
                layer_id=int(layer_id),
                position=int(position),
                device_index=int(device_index),
                state="gpu",
                last_access_step=step,
                access_count=1,
            )
            self.add_entry(entry)
            return entry, True
        if entry.state == "gpu":
            entry.device_index = int(device_index)
        elif entry.host_index is None or device_index >= 0:
            entry.device_index = int(device_index)
            entry.state = "gpu"
        entry.backup_source = "none"
        entry.last_access_step = step
        entry.access_count += 1
        return entry, is_miss

    def mark_host(
        self,
        entry: KVResidencyEntry,
        *,
        host_index: int | None,
        keep_device_index: bool,
    ) -> None:
        if host_index is not None:
            entry.host_index = int(host_index)
            entry.backup_source = "host_pool"
        else:
            entry.backup_source = "none"
        if not keep_device_index:
            entry.device_index = None
        entry.state = "host" if entry.host_index is not None else "evicted"
        entry.version += 1

    def mark_sparse_cpu_backup(
        self,
        entry: KVResidencyEntry,
        *,
        keep_device_index: bool,
    ) -> None:
        if not keep_device_index:
            entry.device_index = None
        entry.host_index = None
        entry.backup_source = "sparse_cpu"
        entry.state = "host"
        entry.version += 1

    def mark_gpu(
        self,
        entry: KVResidencyEntry,
        *,
        device_index: int,
    ) -> None:
        entry.device_index = int(device_index)
        entry.state = "gpu"
        entry.backup_source = "none"
        entry.version += 1

    def live_gpu_count(self) -> int:
        return sum(1 for entry in self.entries.values() if entry.state == "gpu")

    def live_gpu_token_count(self) -> int:
        return len(
            {
                int(entry.device_index)
                for entry in self.entries.values()
                if (
                    entry.state == "gpu"
                    and entry.device_index is not None
                    and int(entry.device_index) >= 0
                )
            }
        )

    def drop_request(self, req_pool_idx: int) -> int:
        req_pool_idx = int(req_pool_idx)
        keys_to_drop = [
            key for key in self.entries.keys() if int(key[0]) == req_pool_idx
        ]
        for key in keys_to_drop:
            del self.entries[key]
            token_key = (int(key[0]), int(key[2]))
            token_entries = self.entries_by_token.get(token_key)
            if token_entries is not None:
                token_entries.discard(key)
                if not token_entries:
                    del self.entries_by_token[token_key]
        return len(keys_to_drop)

    def eviction_candidates(
        self,
        active_keys: set[tuple[int, int, int]],
        cache_policy: str = "working_set",
    ) -> list[KVResidencyEntry]:
        candidates = [
            entry
            for key, entry in self.entries.items()
            if key not in active_keys and entry.state == "gpu" and entry.device_index is not None
        ]
        cache_policy = (cache_policy or "working_set").lower()
        if cache_policy == "recent":
            candidates.sort(
                key=lambda entry: (
                    entry.position,
                    entry.last_access_step,
                    entry.access_count,
                )
            )
        elif cache_policy == "priority":
            candidates.sort(
                key=lambda entry: (
                    entry.selection_priority,
                    entry.last_access_step,
                    entry.access_count,
                )
            )
        else:
            candidates.sort(
                key=lambda entry: (
                    entry.last_selected_step,
                    entry.selection_priority,
                    entry.last_access_step,
                    entry.access_count,
                )
            )
        return candidates


def get_residency_table(framework_state: dict) -> KVResidencyTable:
    table = framework_state.get("kv_residency")
    if table is None:
        table = KVResidencyTable()
        framework_state["kv_residency"] = table
    return table

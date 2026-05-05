from __future__ import annotations

import torch


class SparseWorkingSetBuffer:
    """Reusable GPU buffers for sparse_framework subset attention."""

    def __init__(self):
        self.key_buffers: dict[tuple[int, torch.device, torch.dtype, tuple[int, ...]], torch.Tensor] = {}
        self.value_buffers: dict[tuple[int, torch.device, torch.dtype, tuple[int, ...]], torch.Tensor] = {}

    def materialize(
        self,
        *,
        layer_id: int,
        key: torch.Tensor,
        value: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        key_out = self._copy_into_buffer(self.key_buffers, layer_id, key)
        value_out = self._copy_into_buffer(self.value_buffers, layer_id, value)
        return key_out, value_out

    def _copy_into_buffer(
        self,
        buffers: dict,
        layer_id: int,
        tensor: torch.Tensor,
    ) -> torch.Tensor:
        shape_tail = tuple(tensor.shape[1:])
        cache_key = (int(layer_id), tensor.device, tensor.dtype, shape_tail)
        current = buffers.get(cache_key)
        if current is None or int(current.shape[0]) < int(tensor.shape[0]):
            current = torch.empty(
                (int(tensor.shape[0]),) + shape_tail,
                dtype=tensor.dtype,
                device=tensor.device,
            )
            buffers[cache_key] = current
        view = current[: int(tensor.shape[0])]
        view.copy_(tensor)
        return view


def get_working_set_buffer(framework_state: dict | None) -> SparseWorkingSetBuffer | None:
    if framework_state is None:
        return None
    buffer = framework_state.get("working_set_buffer")
    if buffer is None:
        buffer = SparseWorkingSetBuffer()
        framework_state["working_set_buffer"] = buffer
    return buffer

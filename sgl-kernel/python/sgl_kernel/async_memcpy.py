import torch


def async_memcpy_d2h(src: torch.Tensor, dst: torch.Tensor) -> None:
  """Asynchronously copy tensor data from device (CUDA) to host (CPU).

  Both tensors must have the same shape and dtype. `src` must be CUDA,
  `dst` must be CPU (ideally pinned for best performance).
  """
  torch.ops.sgl_kernel.async_memcpy_d2h(src, dst)


def async_memcpy_h2d(src: torch.Tensor, dst: torch.Tensor) -> None:
  """Asynchronously copy tensor data from host (CPU) to device (CUDA)."""
  torch.ops.sgl_kernel.async_memcpy_h2d(src, dst)


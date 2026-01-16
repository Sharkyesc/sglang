// Asynchronous host <-> device memcpy helpers.
// These are thin wrappers around cudaMemcpyAsync that use PyTorch's current
// CUDA stream so they compose naturally with torch.cuda.Stream contexts.

#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAException.h>

#include <cuda_runtime.h>

#include <torch/torch.h>

namespace {

void check_tensors_for_copy(const at::Tensor& src, const at::Tensor& dst) {
  TORCH_CHECK(src.defined(), "async_memcpy: src tensor must be defined");
  TORCH_CHECK(dst.defined(), "async_memcpy: dst tensor must be defined");
  TORCH_CHECK(
      src.numel() == dst.numel(),
      "async_memcpy: src and dst must have the same number of elements, got src=",
      src.numel(),
      ", dst=",
      dst.numel());
  TORCH_CHECK(
      src.element_size() == dst.element_size(),
      "async_memcpy: src and dst must have the same element size, got src=",
      src.element_size(),
      ", dst=",
      dst.element_size());
}

}  // namespace

// Device -> Host asynchronous copy
void async_memcpy_d2h(const at::Tensor& src, at::Tensor& dst) {
  check_tensors_for_copy(src, dst);

  TORCH_CHECK(src.is_cuda(), "async_memcpy_d2h: src must be a CUDA tensor");
  TORCH_CHECK(!dst.is_cuda(), "async_memcpy_d2h: dst must be a CPU tensor");

  // For true async behavior, dst should be pinned,但这里不强制要求，只是给出提示。
  // 用户侧可以显式使用 pin_memory=True 来获得最佳性能。

  const void* src_ptr = src.data_ptr();
  void* dst_ptr = dst.data_ptr();
  const size_t num_bytes = static_cast<size_t>(src.numel()) * src.element_size();

  cudaStream_t stream = at::cuda::getCurrentCUDAStream();
  AT_CUDA_CHECK(cudaMemcpyAsync(dst_ptr, src_ptr, num_bytes, cudaMemcpyDeviceToHost, stream));
}

// Host -> Device asynchronous copy
void async_memcpy_h2d(const at::Tensor& src, at::Tensor& dst) {
  check_tensors_for_copy(src, dst);

  TORCH_CHECK(!src.is_cuda(), "async_memcpy_h2d: src must be a CPU tensor");
  TORCH_CHECK(dst.is_cuda(), "async_memcpy_h2d: dst must be a CUDA tensor");

  const void* src_ptr = src.data_ptr();
  void* dst_ptr = dst.data_ptr();
  const size_t num_bytes = static_cast<size_t>(src.numel()) * src.element_size();

  cudaStream_t stream = at::cuda::getCurrentCUDAStream();
  AT_CUDA_CHECK(cudaMemcpyAsync(dst_ptr, src_ptr, num_bytes, cudaMemcpyHostToDevice, stream));
}


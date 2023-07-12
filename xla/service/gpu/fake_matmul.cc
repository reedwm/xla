#include "xla/service/gpu/fake_matmul.h"

#include "third_party/gpus/cuda/include/cuda_runtime_api.h"
#include "xla/stream_executor/gpu/gpu_stream.h"

namespace xla::gpu {

void* GetFakeMatmulKernel();

absl::Status RunFakeMatmul(::tensorflow::se::gpu::GpuStreamHandle stream,
                           void* a, size_t a_size_in_bytes,
                           void* b, size_t b_size_in_bytes,
                           void* c, size_t c_size_in_bytes,
                           void* output, size_t output_size_in_bytes,
                           void* a_scale, void* b_scale, void* c_scale, void* output_scale) {
  int threads_per_block = 1024;
  int num_blocks = (output_size_in_bytes + threads_per_block - 1) / threads_per_block;
  void* kernel = GetFakeMatmulKernel();
  void* kernel_args[] = {&a, &a_size_in_bytes, &b, &b_size_in_bytes, &c, &c_size_in_bytes, &output, &output_size_in_bytes,
                         &a_scale, &b_scale, &c_scale, &output_scale};
  cudaError_t launch_status =
      cudaLaunchKernel(kernel, num_blocks, threads_per_block, kernel_args,
                       0, stream);
  if (launch_status != cudaSuccess) {
    return absl::InternalError(absl::StrCat("Failed to launch kernel: ",
                                            cudaGetErrorString(launch_status)));
  }
  return absl::OkStatus();
}

}
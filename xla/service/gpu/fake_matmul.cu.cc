#include <cstddef>
#include <cstdint>
#include <limits>

namespace xla::gpu {

__global__ void FakeMatmulKernel(char* a, size_t a_size_in_bytes,
                                 char* b, size_t b_size_in_bytes,
                                 char* c, size_t c_size_in_bytes,
                                 char* output, size_t output_size_in_bytes,
                                 float* a_scale, float* b_scale, float* c_scale, float* output_scale) {
  int index = blockIdx.x * blockDim.x + threadIdx.x;
  if (index < output_size_in_bytes) {
    output[index] = static_cast<char>(a[index % a_size_in_bytes]) + static_cast<char>(b[index % b_size_in_bytes]);
    if (c) {
      output[index] += static_cast<char>(c[index % c_size_in_bytes]);
    }
    if (a_scale) {
      output[index] += static_cast<char>(*a_scale);
    }
    if (b_scale) {
      output[index] += static_cast<char>(*b_scale);
    }
    if (c_scale) {
      output[index] += static_cast<char>(*c_scale);
    }
    if (output_scale) {
      output[index] += static_cast<char>(*output_scale);
    }
  }
}

void* GetFakeMatmulKernel() {
  return reinterpret_cast<void*>(&FakeMatmulKernel);
}

}  // namespace xla::gpu

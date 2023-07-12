#ifndef XLA_SERVICE_GPU_FAKE_MATMUL_H_
#define XLA_SERVICE_GPU_FAKE_MATMUL_H_

#include <stddef.h>
#include <stdint.h>

#include "absl/status/status.h"
#include "xla/stream_executor/gpu/gpu_stream.h"

namespace xla::gpu {

// Write arbitrary data into 'output' and read from 'a', 'b', 'c', and the four
// scales. Tests the pointers are valid without actually running a matmul.
absl::Status RunFakeMatmul(::tensorflow::se::gpu::GpuStreamHandle stream,
                           void* a, size_t a_size_in_bytes,
                           void* b, size_t b_size_in_bytes,
                           void* c, size_t c_size_in_bytes,
                           void* output, size_t output_size_in_bytes,
                           void* a_scale, void* b_scale, void* c_scale, void* output_scale);

}

#endif  // XLA_SERVICE_GPU_FAKE_MATMUL_H_

#include <cassert>
#include <iostream>

#include <cuda_runtime.h>

#include <nv_metrics.h>

#define CHECK_CUDA_ERRORS(call)                                                \
  {                                                                            \
    cudaError_t error = call;                                                  \
    if (error != cudaSuccess) {                                                \
      std::cerr << "CUDA error in " << #call << " function (" << __FILE__      \
                << ":" << __LINE__ << "): " << cudaGetErrorString(error)       \
                << std::endl;                                                  \
      exit(EXIT_FAILURE);                                                      \
    }                                                                          \
  }

__global__ void kernel(float *a, float *b, float *c, float n, float scale) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) {
    c[i] = c[i] + scale * (a[i] + b[i]);
  }
}

int main() {
  const int N = 1024;
  float *a, *b, *c;
  float *d_a, *d_b, *d_c;
  const float scale = 2.0f;

  // Allocate memory on host
  a = new float[N];
  b = new float[N];
  c = new float[N];

  // Allocate memory on device
  CHECK_CUDA_ERRORS(cudaMalloc((void **)&d_a, N * sizeof(float)));
  CHECK_CUDA_ERRORS(cudaMalloc((void **)&d_b, N * sizeof(float)));
  CHECK_CUDA_ERRORS(cudaMalloc((void **)&d_c, N * sizeof(float)));

  // Initialize host vectors
  for (int i = 0; i < N; ++i) {
    a[i] = i;
    b[i] = i * 2;
  }

  // Copy host vectors to device
  CHECK_CUDA_ERRORS(
      cudaMemcpy(d_a, a, N * sizeof(float), cudaMemcpyHostToDevice));
  CHECK_CUDA_ERRORS(
      cudaMemcpy(d_b, b, N * sizeof(float), cudaMemcpyHostToDevice));

  // Setup metrics
  std::vector<std::string> metrics = {
      "sm__sass_thread_inst_executed_op_fadd_pred_on.sum",
      "sm__sass_thread_inst_executed_op_fmul_pred_on.sum",
      "sm__sass_thread_inst_executed_op_ffma_pred_on.sum"};

  // Launch configuration
  const int block_size = 256;
  const int grid_size = (N + block_size - 1) / block_size;

  // Start measurement
  nvmetrics::measureMetricsStart(metrics);

  // Launch kernel
  kernel<<<grid_size, block_size>>>(d_a, d_b, d_c, N, scale);

  // Stop measurement
  std::vector<double> result = nvmetrics::measureMetricsStop();
  assert(metrics.size() == result.size());

  // Check for kernel launch errors
  CHECK_CUDA_ERRORS(cudaGetLastError());

  // Print result of the measurement
  for (int i = 0; i < result.size(); i++) {
    std::cout << metrics[i] << ": " << result[i] << std::endl;
  }

  // Copy result back to host
  CHECK_CUDA_ERRORS(
      cudaMemcpy(c, d_c, N * sizeof(float), cudaMemcpyDeviceToHost));

  // Free device memory
  CHECK_CUDA_ERRORS(cudaFree(d_a));
  CHECK_CUDA_ERRORS(cudaFree(d_b));
  CHECK_CUDA_ERRORS(cudaFree(d_c));

  // Free host memory
  delete[] a;
  delete[] b;
  delete[] c;

  return EXIT_SUCCESS;
}

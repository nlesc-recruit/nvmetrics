#include "Metric.hpp"

#include <cuda_runtime.h>
#include <iostream>

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

__global__ void vectorAdd(int *a, int *b, int *c, int n) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) {
    c[i] = a[i] + b[i];
  }
}

int main() {
  const int N = 1000;
  int *a, *b, *c;
  int *d_a, *d_b, *d_c;

  // Allocate memory on host
  a = new int[N];
  b = new int[N];
  c = new int[N];

  // Allocate memory on device
  CHECK_CUDA_ERRORS(cudaMalloc((void **)&d_a, N * sizeof(int)));
  CHECK_CUDA_ERRORS(cudaMalloc((void **)&d_b, N * sizeof(int)));
  CHECK_CUDA_ERRORS(cudaMalloc((void **)&d_c, N * sizeof(int)));

  // Initialize host vectors
  for (int i = 0; i < N; ++i) {
    a[i] = i;
    b[i] = i * 2;
  }

  // Copy host vectors to device
  CHECK_CUDA_ERRORS(
      cudaMemcpy(d_a, a, N * sizeof(int), cudaMemcpyHostToDevice));
  CHECK_CUDA_ERRORS(
      cudaMemcpy(d_b, b, N * sizeof(int), cudaMemcpyHostToDevice));

  // Launch kernel
  int threadsPerBlock = 256;
  int blocksPerGrid = (N + threadsPerBlock - 1) / threadsPerBlock;
  vectorAdd<<<blocksPerGrid, threadsPerBlock>>>(d_a, d_b, d_c, N);

  // Check for kernel launch errors
  CHECK_CUDA_ERRORS(cudaGetLastError());

  // Copy result back to host
  CHECK_CUDA_ERRORS(
      cudaMemcpy(c, d_c, N * sizeof(int), cudaMemcpyDeviceToHost));

  // Verify results
  for (int i = 0; i < N; ++i) {
    if (c[i] != a[i] + b[i]) {
      std::cerr << "Error: " << c[i] << " != " << a[i] + b[i]
                << std::endl;
      exit(EXIT_FAILURE);
    }
  }

  // Free device memory
  CHECK_CUDA_ERRORS(cudaFree(d_a));
  CHECK_CUDA_ERRORS(cudaFree(d_b));
  CHECK_CUDA_ERRORS(cudaFree(d_c));

  // Free host memory
  delete[] a;
  delete[] b;
  delete[] c;

  return 0;
}

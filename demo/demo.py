import nvmetrics


import numpy as np
import pycuda.autoinit
import pycuda.driver as cuda
from pycuda.compiler import SourceModule

module = SourceModule(
    """
__global__ void vector_add(float *a, float *b, float *c, int n, int scale)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n)
        c[idx] += scale * (a[idx] + b[idx]);
}
"""
)

if __name__ == "__main__":
    n = 1024
    scale = 2.0

    # Allocate memory on host
    a = np.random.randn(n).astype(np.float32)
    b = np.random.randn(n).astype(np.float32)
    c = np.zeros_like(a)

    # Allocate memory on device
    a_gpu = cuda.mem_alloc(a.nbytes)
    b_gpu = cuda.mem_alloc(b.nbytes)
    c_gpu = cuda.mem_alloc(b.nbytes)

    cuda.memcpy_htod(a_gpu, a)
    cuda.memcpy_htod(b_gpu, b)

    # Launch configuration
    block_size = 256
    grid_size = (n + block_size - 1) // block_size

    # Setup metrics
    metrics = [
        "sm__sass_thread_inst_executed_op_fadd_pred_on.sum",
        "sm__sass_thread_inst_executed_op_fmul_pred_on.sum",
        "sm__sass_thread_inst_executed_op_ffma_pred_on.sum",
    ]

    # Load kernel
    vector_add = module.get_function("vector_add")

    # Start measurement
    nvmetrics.measureMetricsStart(metrics)

    # Launch kernel
    vector_add(
        a_gpu,
        b_gpu,
        c_gpu,
        np.int32(n),
        np.int32(2),
        block=(block_size, 1, 1),
        grid=(grid_size, 1),
    )

    # Stop measurement
    results = nvmetrics.measureMetricsStop()

    # Print result of the measurement
    for metric, value in zip(metrics, results):
        print(f"{metric}: {value}")

    # Copy result back to the host
    cuda.memcpy_dtoh(c, c_gpu)

    # Free device memory
    a_gpu.free()
    b_gpu.free()
    c_gpu.free()

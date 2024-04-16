import nvmetrics

metrics = [
    "sm__sass_thread_inst_executed_op_fadd_pred_on.sum",
    "sm__sass_thread_inst_executed_op_fmul_pred_on.sum",
    "sm__sass_thread_inst_executed_op_ffma_pred_on.sum"
]

import numpy as np
import pycuda.autoinit
import pycuda.driver as cuda
from pycuda.compiler import SourceModule

mod = SourceModule("""
__global__ void vector_add(float *a, float *b, float *c, int n, int scale)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n)
        c[idx] += scale * (a[idx] + b[idx]);
}
""")

vector_add = mod.get_function("vector_add")

n = 1024

a = np.random.randn(n).astype(np.float32)
b = np.random.randn(n).astype(np.float32)

a_gpu = cuda.mem_alloc(a.nbytes)
b_gpu = cuda.mem_alloc(b.nbytes)
c_gpu = cuda.mem_alloc(b.nbytes)

cuda.memcpy_htod(a_gpu, a)
cuda.memcpy_htod(b_gpu, b)

block_size = 256
grid_size = (n + block_size - 1) // block_size

nvmetrics.measureMetricsStart(metrics)

scale = 2.0
vector_add(a_gpu, b_gpu, c_gpu, np.int32(n), np.int32(2), block=(block_size, 1, 1), grid=(grid_size, 1))

metrics_results = nvmetrics.measureMetricsStop()

for metric, value in zip(metrics, metrics_results):
    print(f"{metric}: {value}")

c = np.zeros_like(a)
cuda.memcpy_dtoh(c, c_gpu)
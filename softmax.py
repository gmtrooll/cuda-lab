import torch
from torch.utils.cpp_extension import load_inline

softmax_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>

#define TILE_SIZE 256

__device__ float warpReduceMax(float val) {
    for (int offset = 16; offset > 0; offset /= 2)
        val = fmaxf(val, __shfl_down_sync(0xffffffff, val, offset));
    return val;
}

__device__ float warpReduceSum(float val) {
    for (int offset = 16; offset > 0; offset /= 2)
        val += __shfl_down_sync(0xffffffff, val, offset);
    return val;
}

__global__ void softmax_kernel(const float *input, float* output, int rows, int cols) {
    int row = blockIdx.x;
    if (row >= rows) return;

    __shared__ float row_max;
    __shared__ float row_sum;

    // Initialize the shared memory for the atomic operations
    if (threadIdx.x == 0) {
        row_max = -FLT_MAX;
        row_sum = 0.0f;
    }

    float thread_max = -FLT_MAX;

    // Find max
    // Reduce cols-values to blockDim.x-values (cols -> 256 values)
    for (int i = threadIdx.x; i < cols; i += blockDim.x) 
        thread_max = fmaxf(thread_max, input[row * cols + i]);

    // Reduce BlockDim.x-values to nwarps-values (256 -> 8 values)
    thread_max = warpReduceMax(thread_max);
    // Only want the values of the first threads of each warp
    if (threadIdx.x % 32 == 0) 
        // Write only the max of all warps
        atomicMax((int*)&row_max, __float_as_int(thread_max));
    __syncthreads();

    // Sum 
    float thread_sum = 0.0f;
    float actual_max = row_max;

    // cols -> 256 values
    for (int i = threadIdx.x; i < cols; i += blockDim.x) 
        thread_sum += expf(input[row * cols + i] - actual_max);

    // 256 -> 8 values
    thread_sum = warpReduceSum(thread_sum);

    // Only values of the first threads of each warp
    if (threadIdx.x % 32 == 0)
        // Add and write the sums of all warps
        atomicAdd(&row_sum, thread_sum);
    __syncthreads();

    float actual_sum = row_sum;

    // Divide and write all values of the entire row
    for (int i = threadIdx.x; i < cols; i += blockDim.x)
        output[row * cols + i] = expf(input[row * cols + i] - actual_max) / actual_sum;
}

torch::Tensor softmax_cuda(torch::Tensor A) {
    int rows = A.size(0);
    int cols = A.size(1);

    auto output = torch::zeros({rows, cols}, A.options());

    dim3 block(TILE_SIZE);
    dim3 grid(rows);

    softmax_kernel<<<grid, block>>>(
        A.data_ptr<float>(),
        output.data_ptr<float>(),
        rows, cols
    );

    return output;
}
"""

cpp_source = "torch::Tensor softmax_cuda(torch::Tensor A);"

# Compile the inline extension
softmax_fused = load_inline(
    name = "softmax_fused",
    cpp_sources = cpp_source,
    cuda_sources = softmax_source,
    functions = ["softmax_cuda"],
    verbose = True
)

# Test dimensions (matching your large scale style)
M, N = 4096, 20000

A = torch.randn(M, N, device="cuda", dtype=torch.float32)

torch.cuda.synchronize()
print("Running custom softmax kernel...")
C_custom = softmax_fused.softmax_cuda(A)
torch.cuda.synchronize()

print("Running PyTorch softmax...")
C_torch = torch.nn.functional.softmax(A, dim=-1)
torch.cuda.synchronize()

print("Running verification...")
if torch.allclose(C_custom, C_torch, atol=1e-5):
    print("good")
else:
    print(f"bad (Max diff: {(C_custom - C_torch).abs().max()})")
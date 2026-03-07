import torch
from torch.utils.cpp_extension import load_inline

row_sum_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>

#define TILE_SIZE 256

__device__ float warpReduceSum(float val) {
    for (int offset = 16; offset > 0; offset /= 2)
        val += __shfl_down_sync(0xffffffff, val, offset);
    return val;
}

__global__ void row_sum_kernel(const float* input, float* output, int rows, int cols) {
    int row = blockIdx.x;
    if (row >= rows) return;

    // Shared memory for the 8 warp winners
    __shared__ float staging[8]; 

    float thread_sum = 0.0f;

    // 1. Grid-stride loop: cols -> 256 partial sums
    for (int i = threadIdx.x; i < cols; i += blockDim.x) 
        thread_sum += input[row * cols + i];

    // 2. Warp reduction: 256 -> 8 warp winners
    thread_sum = warpReduceSum(thread_sum);

    // 3. Staging: Store the 8 winners in shared memory
    if (threadIdx.x % 32 == 0) 
        staging[threadIdx.x / 32] = thread_sum;

    __syncthreads();

    // 4. Final Relay: Only Warp 0 finishes the job
    if (threadIdx.x < 32) {
        float val; 
        if (threadIdx.x < 8)
            val = staging[threadIdx.x];
        else
            val = 0.0f;

        // Final 32-thread reduction (using our 8 values + 24 zeros)
        val = warpReduceSum(val);
        
        if (threadIdx.x == 0)
            output[row] = val;
    }
}

torch::Tensor row_sum_cuda(torch::Tensor A) {
    int rows = A.size(0);
    int cols = A.size(1);

    // Output is a 1D vector (one sum per row)
    auto output = torch::zeros({rows}, A.options());

    dim3 block(TILE_SIZE);
    dim3 grid(rows);

    row_sum_kernel<<<grid, block>>>(
        A.data_ptr<float>(),
        output.data_ptr<float>(),
        rows, cols
    );

    return output;
}
"""

cpp_source = "torch::Tensor row_sum_cuda(torch::Tensor A);"

# Compile the inline extension
row_sum_module = load_inline(
    name = "row_sum_no_atomics",
    cpp_sources = cpp_source,
    cuda_sources = row_sum_source,
    functions = ["row_sum_cuda"],
    verbose = True
)

# Test dimensions
M, N = 4096, 20000 

A = torch.randn(M, N, device="cuda", dtype=torch.float32)

torch.cuda.synchronize()
print("Running custom row_sum (No Atomics)...")
C_custom = row_sum_module.row_sum_cuda(A)
torch.cuda.synchronize()

print("Running PyTorch sum...")
C_torch = torch.sum(A, dim=-1)
torch.cuda.synchronize()

print("Running verification...")
if torch.allclose(C_custom, C_torch, atol=1e-2):
    print("good")
else:
    print(f"bad (Max diff: {(C_custom - C_torch).abs().max()})")
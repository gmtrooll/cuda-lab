#include <torch/extension.h>
#include <cuda_runtime.h>

#define TILE_SIZE 256

__device__ float warpReduceSum(float val) {
    for (int offset = 16; offset > 0; offset /= 2)
        val += __shfl_down_sync(0xffffffff, val, offset);
    return val;
}

__global__ void row_sum(const float* input, float* output, int rows, int cols) {
    int row = blockIdx.x;
    if (row >= rows) return;

    // Shared memory for the 8 warps
    __shared__ float staging[8];

    float thread_sum = 0.0f;

    // cols -> 256 values
    for (int i = threadIdx.x; i < cols; i += blockDim.x) 
        thread_sum += input[row * cols + i];

    // 256 -> 8 values
    thread_sum = warpReduceSum(thread_sum);

    if (threadIdx.x % 32 == 0) 
        staging[threadIdx.x / 32] = thread_sum;

    __syncthreads();

    if (threadIdx.x < 32) {
        float val;
        if (threadIdx.x < 8)
            val = staging[threadIdx.x];
        else
            val = 0.0f;
        val = warpReduceSum(val);
        if (threadIdx.x == 0)
            output[row] = val;
    }
}

torch::Tensor row_sum_cuda(torch::Tensor A) {
    int rows = A.size(0);
    int cols = A.size(1);

    auto output = torch::zeros({rows}, A.options());

    dim3 block(TILE_SIZE);
    dim3 grid(rows);

    row_sum<<<grid, block>>>(
        A.data_ptr<float>(),
        output.data_ptr<float>(),
        rows, cols
    );

    return output;
}
#include <torch/extension.h>
#include <cuda_runtime.h>

#define TILE_SIZE 32

__global__ void tiled_matmul_kernel(const float *A, const float *B, float *C, int M, int N, int K) {
    // Shared memory for tiles of A and B
    __shared__ float As[TILE_SIZE][TILE_SIZE];
    __shared__ float Bs[TILE_SIZE][TILE_SIZE];

    int row = blockIdx.y * TILE_SIZE + threadIdx.y;
    int col = blockIdx.x * TILE_SIZE + threadIdx.x;
    
    float temp = 0;

    // Loop over tiles along the K dimension
    for (int t = 0; t < (K + TILE_SIZE - 1) / TILE_SIZE; ++t) {
        
        // Each thread loads one element of the tile from global memory
        if (row < M && (t * TILE_SIZE + threadIdx.x) < K)
            As[threadIdx.y][threadIdx.x] = A[row * K + t * TILE_SIZE + threadIdx.x];
        else
            As[threadIdx.y][threadIdx.x] = 0.0;

        if (col < N && (t * TILE_SIZE + threadIdx.y) < K)
            Bs[threadIdx.y][threadIdx.x] = B[(t * TILE_SIZE + threadIdx.y) * N + col];
        else
            Bs[threadIdx.y][threadIdx.x] = 0.0;

        // Wait until the entire block has finished loading the tiles
        __syncthreads();

        // Dot product of the tiles in fast shared memory
        for (int i = 0; i < TILE_SIZE; ++i) {
            temp += As[threadIdx.y][i] * Bs[i][threadIdx.x];
        }

        // Wait until everyone is done with the math before we overwrite As/Bs in the next loop
        __syncthreads();
    }

    if (row < M && col < N) {
        C[row * N + col] = temp;
    }
}

torch::Tensor tiled_matmult_cuda(torch::Tensor A, torch::Tensor B) {
    int M = A.size(0);
    int K = A.size(1);
    int N = B.size(1);

    auto C = torch::zeros({M, N}, A.options());

    dim3 block(TILE_SIZE, TILE_SIZE);
    dim3 grid((N + block.x - 1) / block.x, (M + block.y - 1) / block.y);

    tiled_matmul_kernel<<<grid, block>>>(
        A.data_ptr<float>(),
        B.data_ptr<float>(),
        C.data_ptr<float>(),
        M, N, K
    );

    return C;
}
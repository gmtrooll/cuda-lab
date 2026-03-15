#include <torch/extension.h>
#include <cuda_runtime.h>

// Block size
const int BM = 128;
const int BN = 128;
const int BK = 8;

// Tile size for each thread
const int TM = 8;
const int TN = 8;

__global__ void k5_matmult_kernel(const float *A, const float *B, float *C, int M, int N, int K) {
    __shared__ float As[BM][BK];
    __shared__ float Bs[BK][BN];

    float threadResults[TM * TN] = {0.0};
    float regM[TM];
    float regN[TN];

    // Block indices
    const int cRow = blockIdx.y;
    const int cCol = blockIdx.x;

    // Thread indices in the block
    const int threadCol = threadIdx.x % (BN / TN);
    const int threadRow = threadIdx.x / (BN / TN);
    
    // Loop over the K dimension in steps of BK
    for (int t = 0; t < (K + BK - 1) / BK; t++) {
        // Load from gloabl to shared memory
        for (int i = 0; i < BM; i += (256 / BK)) {
            int lRow = i + threadRow;
            int lCol = threadCol;
            int gRow = cRow * BM + lRow;
            int gCol = t * BK + lCol;
            if (gRow < M && gCol < K) 
                As[lRow][lCol] = A[gRow * K + gCol];
            else 
                As[lRow][lCol] = 0.0;
        }
        for (int i = 0; i < BK; i += (256 / BN)) {
            int lRow = i + (threadIdx.x / BN);
            int lCol = threadIdx.x % BN;
            int gRow = t * BK + lRow;
            int gCol = cCol * BN + lCol;
            if (gRow < K && gCol < N) 
                Bs[lRow][lCol] = B[gRow * N + gCol];
            else 
                Bs[lRow][lCol] = 0.0;
        }

        __syncthreads();

        // Compute dot product
        for(int dotIdx = 0; dotIdx < BK; dotIdx++) {
            for (int i = 0; i < TM; i++)
                regM[i] = As[threadRow * TM + i][dotIdx];
            for (int i = 0; i < TN; i++)
                regN[i] = Bs[dotIdx][threadCol * TN + i];

            for (int resIdxM = 0; resIdxM < TM; resIdxM++) 
                for (int resIdxN = 0; resIdxN < TN; resIdxN++)
                    threadResults[resIdxM * TN + resIdxN] += regM[resIdxM] * regN[resIdxN];
        }

        __syncthreads();
    }

    for (int i = 0; i < TM; i++) 
        for (int j = 0; j < TN; j++) {
            int row = cRow * BM + threadRow * TM + i;
            int col = cCol * BN + threadCol * TN + j;
            if (row < M && col < N) {
                C[row * N + col] = threadResults[i * TN + j];
            }
        }
}

torch::Tensor k5_matmult_cuda(torch::Tensor A, torch::Tensor B) {
    int M = A.size(0);
    int K = A.size(1);
    int N = B.size(1);

    auto C = torch::zeros({M, N}, A.options());

    dim3 block(256); 
    dim3 grid((N + BN - 1) / BN, (M + BM - 1) / BM);

    k5_matmult_kernel<<<grid, block>>>(
        A.data_ptr<float>(),
        B.data_ptr<float>(),
        C.data_ptr<float>(),
        M, N, K
    );

    return C;
}
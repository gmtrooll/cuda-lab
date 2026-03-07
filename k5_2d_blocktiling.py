import torch
from torch.utils.cpp_extension import load_inline

k5_matmult_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>

const int BM = 128;
const int BN = 128;
const int BK = 8;
const int TM = 8;
const int TN = 8;

__global__ void k5_matmult_kernel(const float *A, const float *B, float *C, int M, int N, int K) {
    __shared__ float As[BM][BK];
    __shared__ float Bs[BK][BN];

    float threadResults[TM * TN] = {0.0};
    float regM[TM];
    float regN[TN];

    const int cRow = blockIdx.y;
    const int cCol = blockIdx.x;

    const int threadCol = threadIdx.x % (BN / TN);
    const int threadRow = threadIdx.x / (BN / TN);
    
    for (int t = 0; t < (K + BK - 1) / BK; t++) {
        // LOAD A: 128x8 = 1024 elements. 256 threads. 
        // Each thread loads exactly 4 elements.
        for (int i = 0; i < 4; i++) {
            // Map the 256 threads to a 128x8 grid
            // tid 0-7 handle row 0, tid 8-15 handle row 1, etc.
            int tid = threadIdx.x + i * 256;
            int lRow = tid / BK; 
            int lCol = tid % BK;
            int gRow = cRow * BM + lRow;
            int gCol = t * BK + lCol;
            
            if (gRow < M && gCol < K) As[lRow][lCol] = A[gRow * K + gCol];
            else As[lRow][lCol] = 0.0;
        }

        // LOAD B: 8x128 = 1024 elements. 256 threads.
        // Each thread loads exactly 4 elements.
        for (int i = 0; i < 4; i++) {
            // Map the 256 threads to an 8x128 grid
            int tid = threadIdx.x + i * 256;
            int lRow = tid / BN;
            int lCol = tid % BN;
            int gRow = t * BK + lRow;
            int gCol = cCol * BN + lCol;

            if (gRow < K && gCol < N) Bs[lRow][lCol] = B[gRow * N + gCol];
            else Bs[lRow][lCol] = 0.0;
        }

        __syncthreads();

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

    for (int i = 0; i < TM; i++) {
        for (int j = 0; j < TN; j++) {
            int row = cRow * BM + threadRow * TM + i;
            int col = cCol * BN + threadCol * TN + j;
            if (row < M && col < N) {
                C[row * N + col] = threadResults[i * TN + j];
            }
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
"""

cpp_source = "torch::Tensor k5_matmult_cuda(torch::Tensor A, torch::Tensor B);"

k5_matmult = load_inline(
    name = "k5_matmult",
    cpp_sources = cpp_source,
    cuda_sources = k5_matmult_source,
    functions = ["k5_matmult_cuda"],
    verbose = True
)

M, N, K = 20000, 20000, 20000

A = torch.randn(M, K, device="cuda", dtype = torch.float32)
B = torch.randn(K, N, device="cuda", dtype = torch.float32)

torch.cuda.synchronize()
print("Running custom kernel")
C_custom = k5_matmult.k5_matmult_cuda(A, B)
torch.cuda.synchronize()

print("Running pyTorch matmult")
C_torch = torch.matmul(A, B)
torch.cuda.synchronize()

print("Running verification")
if torch.allclose(C_custom, C_torch, atol=1e-2):
    print("good")
else:
    print("bad")
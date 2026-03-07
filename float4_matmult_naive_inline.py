import torch
from torch.utils.cpp_extension import load_inline

float4_matmult_naive_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>

__global__ void float4_matmult_naive_kernel(float *A, float *B_t, float *C, int M, int N, int K){
    const int row = blockIdx.y * blockDim.y + threadIdx.y;
    const int col = blockIdx.x * blockDim.x + threadIdx.x;

    if(row <M && col < N){
        float temp = 0;
        int i = 0;

        for(; i <= K - 4; i += 4){
            float4 a_vals = reinterpret_cast<const float4*>(&A[row * K + i])[0];
            float4 bt_vals = reinterpret_cast<const float4*>(&B_t[col * K + i])[0];

            temp += a_vals.x * bt_vals.x;
            temp += a_vals.y * bt_vals.y;
            temp += a_vals.z * bt_vals.z;
            temp += a_vals.w * bt_vals.w;
        }

        for(; i < K; i++){
            temp += A[row * K + i] * B_t[col * K + i];
        }

        C[row * N + col] = temp;
    }
}

torch::Tensor float4_matmult_naive_cuda(torch::Tensor A, torch::Tensor B) {
    int M = A.size(0);
    int K = A.size(1);
    int N = B.size(1);

    auto B_t = B.t().contiguous();

    auto C = torch::zeros({M, N}, A.options());

    dim3 block(32, 32);
    dim3 grid((N + block.x - 1) / block.x, (M + block.y - 1) / block.y);

    float4_matmult_naive_kernel<<<grid, block>>>(
        A.contiguous().data_ptr<float>(),
        B_t.data_ptr<float>(),
        C.data_ptr<float>(),
        M, N, K
    );

    return C;
}
"""

cpp_source = "torch::Tensor float4_matmult_naive_cuda(torch::Tensor A, torch::Tensor B);"

float4_naive = load_inline(
    name = "float4_naive_matmult",
    cpp_sources = cpp_source,
    cuda_sources = float4_matmult_naive_source,
    functions = ["float4_matmult_naive_cuda"],
    verbose = True
)

M, N, K = 20000, 20000, 20000

A = torch.randn(M, K, device="cuda", dtype=torch.float32)
B = torch.randn(K, N, device="cuda", dtype=torch.float32)

torch.cuda.synchronize()
print("Running custom naive kernel...")
C_custom = float4_naive.float4_matmult_naive_cuda(A, B)
torch.cuda.synchronize()

print("Running pyTorch matmult")
C_torch = torch.matmul(A, B)

print("Running verification")
if torch.allclose(C_custom, C_torch, atol=1e-3):
    print("good")
else:
    print("bad")
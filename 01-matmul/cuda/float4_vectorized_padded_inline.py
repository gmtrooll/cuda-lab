import torch
from torch.utils.cpp_extension import load_inline
import torch.nn.functional as F

float4_matmult_naive_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>

__global__ void float4_matmult_naive_kernel(float *A, float *B_t, float *C, int M, int N, int K){
    const int row = blockIdx.y * blockDim.y + threadIdx.y;
    const int col = blockIdx.x * blockDim.x + threadIdx.x;

    if(row <M && col < N){
        float temp = 0;

        for(int i = 0; i <= K - 4; i += 4){
            float4 a_vals = reinterpret_cast<const float4*>(&A[row * K + i])[0];
            float4 bt_vals = reinterpret_cast<const float4*>(&B_t[col * K + i])[0];

            temp += a_vals.x * bt_vals.x;
            temp += a_vals.y * bt_vals.y;
            temp += a_vals.z * bt_vals.z;
            temp += a_vals.w * bt_vals.w;
        }

        C[row * N + col] = temp;
    }
}

torch::Tensor float4_matmult_naive_cuda(torch::Tensor A_padded, torch::Tensor B_t_padded, int M_orig, int N_orig) {
    int M = A_padded.size(0);
    int K = A_padded.size(1);
    int N = B_t_padded.size(0);

    auto C_padded = torch::zeros({M, N}, A_padded.options());

    dim3 block(32, 32);
    dim3 grid((N + block.x - 1) / block.x, (M + block.y - 1) / block.y);

    float4_matmult_naive_kernel<<<grid, block>>>(
        A_padded.data_ptr<float>(),
        B_t_padded.data_ptr<float>(),
        C_padded.data_ptr<float>(),
        M, N, K
    );

    // Slice back to original size
    return C_padded.index({torch::indexing::Slice(0, M_orig), torch::indexing::Slice(0, N_orig)});
}
"""

cpp_source = "torch::Tensor float4_matmult_naive_cuda(torch::Tensor A_padded, torch::Tensor B_t_padded, int M_orig, int N_orig);"

float4_naive = load_inline(
    name = "float4_naive_matmult",
    cpp_sources = cpp_source,
    cuda_sources = float4_matmult_naive_source,
    functions = ["float4_matmult_naive_cuda"],
    verbose = True
)

def run_padded_matmult(A, B):
    M_orig, K_orig = A.shape
    _, N_orig = B.shape

    M = (M_orig + 3) // 4 * 4
    K = (K_orig + 3) // 4 * 4
    N = (N_orig + 3) // 4 * 4

    A_padded = F.pad(A, (0, K - K_orig, 0, M - M_orig)).contiguous()
    B_padded = F.pad(B, (0, N - N_orig, 0, K - K_orig))

    B_t_padded = B_padded.t().contiguous()

    return float4_naive.float4_matmult_naive_cuda(A_padded, B_t_padded, M_orig, N_orig)

M, N, K = 5001, 5002, 5003

A = torch.randn(M, K, device="cuda", dtype=torch.float32)
B = torch.randn(K, N, device="cuda", dtype=torch.float32)

torch.cuda.synchronize()
print("Running custom naive kernel...")
C_custom = run_padded_matmult(A, B)
torch.cuda.synchronize()

print("Running pyTorch matmult")
C_torch = torch.matmul(A, B)

print("Running verification")
if torch.allclose(C_custom, C_torch, atol=1e-3):
    print("good")
else:
    print("bad")
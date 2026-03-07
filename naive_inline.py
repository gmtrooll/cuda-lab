import torch
from torch.utils.cpp_extension import load_inline

# 1. Naive Kernel logic + the C++ Bridge
naive_matmul_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>

// NAIVE KERNEL
__global__ void naive_matmult_kernel(float *A, float *B, float *C, int M, int N, int K) {
    const int row = blockIdx.y * blockDim.y + threadIdx.y;
    const int col = blockIdx.x * blockDim.x + threadIdx.x;

    if(row < M && col < N){
        float temp = 0;
        for(int i = 0; i < K; i++)
            temp += A[row * K + i] * B[i * N + col];
        C[row * N + col] = temp;
    }
}

// THE BRIDGE FUNCTION (The link between Python and C++)
torch::Tensor naive_matmult_cuda(torch::Tensor A, torch::Tensor B) {
    // A is (M x K), B is (K x N)
    int M = A.size(0);
    int K = A.size(1);
    int N = B.size(1);

    // Allocate output matrix C on the same device as A
    auto C = torch::zeros({M, N}, A.options());

    // Execution configuration
    dim3 block(32, 32);
    dim3 grid((N + block.x - 1) / block.x, (M + block.y - 1) / block.y);

    // Launch the kernel using data_ptr to get the raw memory addresses
    naive_matmult_kernel<<<grid, block>>>(
        A.contiguous().data_ptr<float>(), 
        B.contiguous().data_ptr<float>(), 
        C.data_ptr<float>(), 
        M, N, K
    );

    return C;
}
"""

# 2. Compile the code inline
cpp_source = "torch::Tensor naive_matmult_cuda(torch::Tensor A, torch::Tensor B);"

fast_naive = load_inline(
    name="naive_matmul_ext",
    cpp_sources=cpp_source,
    cuda_sources=naive_matmul_source,
    functions=["naive_matmult_cuda"],
    verbose=True #
)

# 3. Test with PyTorch Tensors
def run_test():
    M, N, K = 20000, 20000, 20000
    
    # Create tensors directly on GPU
    A = torch.randn(M, K, device="cuda", dtype=torch.float32)
    B = torch.randn(K, N, device="cuda", dtype=torch.float32)

    print("Running custom naive kernel...")
    # Call the C++ function
    C_custom = fast_naive.naive_matmult_cuda(A, B)

    print("Running PyTorch native matmul (for verification)...")
    C_torch = torch.matmul(A, B)

    # Verification
    print("Running verification")
    if torch.allclose(C_custom, C_torch, atol=1e-3):
        print("Success! Custom Naive Kernel matches PyTorch result.")
    else:
        print("Verification failed.")

if __name__ == "__main__":
    run_test()
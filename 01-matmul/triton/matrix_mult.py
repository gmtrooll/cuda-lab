import triton
import triton.language as tl
import torch

DEVICE = torch.device('cpu')

@triton.jit
def matrix_mult_kernel(
    x_ptr, y_ptr, out_ptr,
    M, K, N,
    stride_x_m, stride_x_k,
    stride_y_k, stride_y_n,
    stride_out_m, stride_out_n,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    GROUP_SIZE_M: tl.constexpr,
                       ):
    
    pid = tl.program_id(0)
    block_num_M = triton.cdiv(M, BLOCK_SIZE_M)
    block_num_N = triton.cdiv(N, BLOCK_SIZE_N)
    num_pid_in_group = GROUP_SIZE_M * block_num_N
    group_id = pid // num_pid_in_group
    first_pid_m = group_id * GROUP_SIZE_M
    group_size_m = min(block_num_M - first_pid_m, GROUP_SIZE_M)
    pid_m = first_pid_m + (pid % num_pid_in_group) % group_size_m
    pid_n = (pid % num_pid_in_group) // group_size_m

    offs_am = (pid_m * BLOCK_SIZE_M) + tl.arange(0, BLOCK_SIZE_M)
    offs_bn = (pid_n * BLOCK_SIZE_N) + tl.arange(0, BLOCK_SIZE_N)
    offs_k = tl.arange(0, BLOCK_SIZE_K)

    a_ptrs = x_ptr + (offs_am[:, None] * stride_x_m + offs_k[None, :] * stride_x_k)
    b_ptrs = y_ptr + (offs_k[:, None] * stride_y_k + offs_bn[None, :] * stride_y_n)

    mask_m = offs_am[:, None] < M
    mask_n = offs_bn[None, :] < N

    accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
    for k in range(0, tl.cdiv(K, BLOCK_SIZE_K)):
        mask_k = (k * BLOCK_SIZE_K + offs_k) < K
        a = tl.load(a_ptrs, mask=mask_m & mask_k[None, :], other=0.0)
        b = tl.load(b_ptrs, mask=mask_k[:, None] & mask_n, other=0.0)
        accumulator = tl.dot(a, b, accumulator)
        a_ptrs += BLOCK_SIZE_K * stride_x_k
        b_ptrs += BLOCK_SIZE_K * stride_y_k

    offs_cm = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_cn = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    c_ptrs = out_ptr + (stride_out_m * offs_cm[:, None] + stride_out_n * offs_cn[None, :])
    c_mask = (offs_cm[:, None] < M) & (offs_cn[None, :] < N)
    C = accumulator.to(tl.float16)
    tl.store(c_ptrs, C, mask=c_mask)

def matrix_mult(x:torch.tensor, y:torch.tensor):
    M = x.size(0)
    K = x.size(1)
    N = y.size(1)
    output = torch.empty(M,N, dtype = torch.float16, device = x.device)
    assert x.device == DEVICE and y.device == DEVICE and output.device == DEVICE

    grid = lambda META: (
        triton.cdiv(M, META['BLOCK_SIZE_M']) * triton.cdiv(N, META['BLOCK_SIZE_N']),
    )

    matrix_mult_kernel[grid](
        x, y, output,
        M, K, N,
        x.stride(0), x.stride(1),
        y.stride(0), y.stride(1),
        output.stride(0), output.stride(1),
        BLOCK_SIZE_M=128, BLOCK_SIZE_K=32, BLOCK_SIZE_N=128,
        GROUP_SIZE_M=2
    )

    return output


torch.manual_seed(0)
M = 512
N = 512
K = 512
x = torch.rand((M, K), device=DEVICE)
y = torch.rand((K, N), device=DEVICE)
output_torch = torch.matmul(x, y)
output_triton = matrix_mult(x, y)
print(output_torch)
print(output_triton)
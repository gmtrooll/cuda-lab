import torch

import triton
import triton.language as tl

DEVICE = triton.runtime.driver.active.get_active_torch_device()


@triton.jit
def softmax_kernel(x_ptr,
                   output_ptr,
                   columns,
                   stride_r,
                   BLOCK_SIZE: tl.constexpr,
                    ):
    pid = tl.program_id(0)
    offsets = pid * stride_r + tl.arange(0, BLOCK_SIZE)
    col_offs = tl.arange(0, BLOCK_SIZE)
    mask = col_offs < columns
    x = tl.load(x_ptr + offsets, mask = mask, other = -float('inf'))
    row_max = tl.max(x, axis = 0)
    numerator = tl.exp(x - row_max)
    denominator = tl.sum(numerator, axis = 0)
    output = numerator / denominator
    tl.store(output_ptr + offsets, output, mask = mask)
    
def softmax(x:torch.Tensor):
    output = torch.empty_like(x)
    assert x.device == DEVICE and output.device == DEVICE

    rows = output.size(0)
    columns = output.size(1)
    stride_r = output.stride(0)

    # grid = lambda meta: (rows, )

    softmax_kernel[rows,](x, output, columns, stride_r, BLOCK_SIZE = triton.next_power_of_2(columns))

    return output

torch.manual_seed(0)
size = 100
x = torch.rand(size, size, device = DEVICE)
output = softmax(x)
print(x)
print(output)
output_torch = torch.softmax(x, dim=1)
print(f'The maximum difference: {torch.max(torch.abs(output - output_torch))}')
assert torch.allclose(output, output_torch) # Standard way to verify
import torch 

import triton
import triton.language as tl

DEVICE = triton.runtime.driver.active.get_active_torch_device()


@triton.jit
def matrix_1D_kernel(x_ptr,
                      y_ptr,
                      output_ptr,
                      n_elements,
                      ADD: tl.constexpr,
                      BLOCK_SIZE: tl.constexpr,
                      ):
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    x = tl.load(x_ptr + offsets, mask = mask)
    y = tl.load(y_ptr + offsets, mask = mask)
    if ADD:
        output = x + y
    else:
        output = x * y
    tl.store(output_ptr + offsets, output, mask = mask)
    
@triton.jit
def matrix_2D_kernel(x_ptr,
                     y_ptr,
                     output_ptr,
                     rows,
                     columns,
                     stride_r,
                     ADD: tl.constexpr,
                     BLOCK_SIZE_R: tl.constexpr,
                     BLOCK_SIZE_C: tl.constexpr,
                     ):
    pid_r = tl.program_id(0)
    pid_c = tl.program_id(1)
    offs_r = pid_r * BLOCK_SIZE_R + tl.arange(0, BLOCK_SIZE_R)
    offs_c = pid_c * BLOCK_SIZE_C + tl.arange(0, BLOCK_SIZE_C)

    offsets = (offs_r[:, None] * stride_r + offs_c[None, :])
    mask = (offs_r[:, None] < rows) & (offs_c[None, :] < columns)

    x = tl.load(x_ptr + offsets, mask = mask)
    y = tl.load(y_ptr + offsets, mask = mask)

    if ADD:
        output = x + y
    else:
        output = x * y
    tl.store(output_ptr + offsets, output, mask = mask)


def run_1D(x:torch.Tensor, y: torch.Tensor, op="add"):
    output = torch.empty_like(x)
    assert x.device == DEVICE and y.device == DEVICE and output.device == DEVICE
    n_elements = output.numel()

    grid = lambda meta: (triton.cdiv(n_elements, meta['BLOCK_SIZE']),)

    matrix_1D_kernel[grid](x, 
                           y, 
                           output, 
                           n_elements, 
                           1 if op=="add" else 0, 
                           BLOCK_SIZE = 1024)

    return output

def run_2D(x:torch.Tensor, y:torch.Tensor, op="add"):
    output = torch.empty_like(x)
    assert x.device == DEVICE and y.device == DEVICE and output.device == DEVICE
    rows = output.size(0)
    columns = output.size(1)
    stride_r = output.stride(0)

    grid = lambda meta: (triton.cdiv(rows, meta['BLOCK_SIZE_R']),
                         triton.cdiv(columns, meta['BLOCK_SIZE_C']),)
    matrix_2D_kernel[grid](x,
                            y,
                            output, 
                            rows, 
                            columns, 
                            stride_r, 
                            1 if op=="add" else 0, 
                            BLOCK_SIZE_R = 32, 
                            BLOCK_SIZE_C = 32)
    
    return output

torch.manual_seed(0)
size_r = 1024
size_c = 1024
x = torch.rand(size_r, size_c, device = DEVICE)
y = torch.rand(size_r, size_c, device = DEVICE)
output_add_torch = x + y
output_mult_torch = x * y
output_add_1D_triton = run_1D(x, y)
output_add_2D_triton = run_2D(x, y)
output_mult_1D_triton = run_1D(x, y, "mult")
output_mult_2D_triton = run_2D(x, y, "mult") 
print(output_add_torch)
print(output_add_1D_triton)
print(output_add_2D_triton)
print(output_mult_torch)
print(output_mult_1D_triton)
print(output_mult_2D_triton)
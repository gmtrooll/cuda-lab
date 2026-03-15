import triton
import triton.language as tl
import torch

DEVICE = torch.device('cpu')

@triton.jit
def matrix_seeded_dropout_kernel(x_ptr,
                                 out_ptr,
                                 p_ptr,
                                 seed_ptr,
                                 rows,
                                 cols,
                                 BLOCK_SIZE: tl.constexpr,):
    pid = tl.program_id(0)
    row_seed = tl.load(seed_ptr + pid)
    row_p = tl.load(p_ptr + pid)
    row_ptr = x_ptr + pid * cols
    col_offs = tl.arange(0, BLOCK_SIZE)
    mask = col_offs < cols

    x = tl.load(row_ptr + col_offs, mask = mask)
    random = tl.rand(row_seed, pid * cols + col_offs)
    x_keep = random > row_p
    output = tl.where(x_keep, x / (1 - row_p), 0.0)
    tl.store(out_ptr + pid * cols + col_offs, output, mask = mask)
    

def matrix_seeded_dropout(x: torch.Tensor, p, seed):
    output = torch.empty_like(x)
    assert x.is_contiguous()
    rows, cols = x.shape
    matrix_seeded_dropout_kernel[rows,](x, output, p, seed, rows, cols, BLOCK_SIZE = triton.next_power_of_2(cols))
    return output
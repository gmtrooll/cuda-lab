import triton
import triton.language as tl
import torch

DEVICE = torch.device('cpu')

@triton.jit
def matrix_seeded_dropout_kernel(
    x_ptr, out_ptr, p_ptr, seed_ptr,
    stride_row,
    rows, cols,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
):
    pid_row = tl.program_id(0)
    pid_col = tl.program_id(1)

    row_offsets = pid_row * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    col_offsets = pid_col * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)

    row_seeds = tl.load(seed_ptr + row_offsets, mask = row_offsets < rows)
    row_ps = tl.load(p_ptr + row_offsets, mask = row_offsets < rows)

    offsets_2d = row_offsets[:, None] * stride_row + col_offsets[None, :]
    mask_2d = (row_offsets[:, None] < rows) & (col_offsets[None, :] < cols)

    x = tl.load(x_ptr + offsets_2d, mask = mask_2d)

    rand_offsets = row_offsets[:, None] * cols + col_offsets[None, :]
    random = tl.rand(row_seeds[:, None], rand_offsets)

    keep = random > row_ps[:, None]

    output = tl.where(keep, x / (1.0 - row_ps[:, None]), 0.0)

    tl.store(out_ptr + offsets_2d, output, mask=mask_2d)


def matrix_seeded_dropout(x, p, seed):
    rows, cols = x.shape
    output = torch.empty_like(x)

    grid = lambda META: (
        triton.cdiv(rows, META['BLOCK_SIZE_M']), 
        triton.cdiv(cols, META['BLOCK_SIZE_N'])
    )

    matrix_seeded_dropout_kernel[grid](
        x, output, p, seed,
        x.stride(0),
        rows, cols,
        BLOCK_SIZE_M = 4,
        BLOCK_SIZE_N = 256,
    )
    return output


torch.manual_seed(0)
M = 512
N = 512
x = torch.rand((M, N), device = DEVICE)
p_vec = torch.full((M,), 0.5, device=DEVICE)
seeds = torch.arange(M, device=DEVICE, dtype=torch.int32)
print(x)
print(matrix_seeded_dropout(x, p_vec, seeds))
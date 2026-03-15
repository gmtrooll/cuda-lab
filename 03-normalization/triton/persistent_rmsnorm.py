import triton
import triton.language as tl
import torch

DEVICE = torch.device('cpu')

@triton.jit
def RMSNorm_kernel(x_ptr,
                   out_ptr,
                   rows,
                   cols,
                   BLOCK_SIZE: tl.constexpr,
                   ):
    start_row = tl.program_id(0)
    stride = tl.num_programs(0)
    '''
    for row_idx in range(start_row, rows, stride):
    # for row_idx in tl.range(start_row, rows, stride):
        row_ptr = x_ptr + row_idx * cols
        col_offs = tl.arange(0, BLOCK_SIZE)
        mask = col_offs < cols

        x = tl.load(row_ptr + col_offs, mask = mask)
        rms = tl.sqrt(tl.sum(x * x, axis = 0) / cols + 1e-6)
        output = x / rms

        tl.store(out_ptr + row_idx * cols + col_offs, output, mask = mask)
        '''
    while start_row < rows:
        row_start_ptr = x_ptr + start_row * cols
        col_offs = tl.arange(0, BLOCK_SIZE)
        mask = col_offs < cols

        # Load
        x = tl.load(row_start_ptr + col_offs, mask=mask, other=0.0)
        
        # Math
        x_sq = x * x
        row_sum = tl.sum(x_sq, axis=0)
        rms = tl.sqrt(row_sum / cols + 1e-6)
        output = x / rms

        # Store
        out_row_ptr = out_ptr + start_row * cols
        tl.store(out_row_ptr + col_offs, output, mask=mask)
        
        # 3. Advance to the next row in the grid-stride
        start_row += stride


def RMSNorm(x:torch.Tensor):
    rows, cols = x.shape

    output = torch.empty_like(x)

    # device = torch.cuda.current_device()
    # properties = torch.cuda.get_device_properties(device)
    # num_sms = properties.multi_processor_count
    num_sms = 28 # nvidia 3060

    occupancy = 2

    RMSNorm_kernel[num_sms * occupancy,](
        x, output, rows, cols, BLOCK_SIZE = triton.next_power_of_2(cols)
    )

    return output



size = 1500

torch.manual_seed(0)
size = 1500
x = torch.rand(size, size, device=DEVICE)
y = RMSNorm(x)
print(y)
import torch
from torch.utils.cpp_extension import load_inline

blur_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>

// 16x16 = 256 threads
#define BDIM 16
// 3x3 filter => radius of 1
#define RADIUS 1
// Shared memory dimension (16 + 2*1 = 18)
#define SDIM (BDIM + 2 * RADIUS)

__global__ void box_blur_2d_kernel(const float* input, float* output, int width, int height) {
    __shared__ float tile[SDIM][SDIM];

    // Global pixel coordinates
    int x = blockIdx.x * BDIM + threadIdx.x;
    int y = blockIdx.y * BDIM + threadIdx.y;
    // Local pixel coordinates
    int lx = threadIdx.x + RADIUS;
    int ly = threadIdx.y + RADIUS;

    // Loading data
    // Primary pixel
    int g_x = min(max(x, 0), width - 1);
    int g_y = min(max(y, 0), height - 1);
    tile[ly][lx] = input[g_y * width + g_x];

    if (threadIdx.x < RADIUS) {
        // Left
        tile[ly][lx - RADIUS] = input[g_y * width + min(max(x - RADIUS, 0), width - 1)];
        // Right
        tile[ly][lx + BDIM] = input[g_y * width + min(max(x + BDIM, 0), width - 1)];
    }
    if (threadIdx.y < RADIUS) {
        // Top
        tile[ly - RADIUS][lx] = input[min(max(y - RADIUS, 0), height - 1) * width + g_x];
        // Bottom
        tile[ly + BDIM][lx] = input[min(max(y + BDIM, 0), height - 1) * width + g_x];
    }

    // Corners
    if(threadIdx.x < RADIUS && threadIdx.y < RADIUS) {
        tile[ly - RADIUS][lx - RADIUS] = input[min(max(y - RADIUS, 0), height - 1) * width + min(max(x - RADIUS, 0), width - 1)];
        tile[ly - RADIUS][lx + BDIM]   = input[min(max(y - RADIUS, 0), height - 1) * width + min(max(x + BDIM, 0), width - 1)];
        tile[ly + BDIM][lx - RADIUS]   = input[min(max(y + BDIM, 0), height - 1) * width + min(max(x - RADIUS, 0), width - 1)];
        tile[ly + BDIM][lx + BDIM]     = input[min(max(y + BDIM, 0), height - 1) * width + min(max(x + BDIM, 0), width - 1)];
    }
    __syncthreads();

    // Compute average
    if(x < width && y < height) {
        float sum = 0.0f;
        for(int dy = -RADIUS; dy <= RADIUS; dy++) 
            for (int dx = -RADIUS; dx <= RADIUS; dx++)
                sum += tile[ly + dy][lx + dx];
        output[y * width + x] = sum / 9.0f;
    }
}

torch::Tensor blur_cuda(torch::Tensor input) {
    int height = input.size(0);
    int width = input.size(1);
    auto output = torch::empty_like(input);

    dim3 block(BDIM, BDIM);
    dim3 grid((width + BDIM - 1) / BDIM, (height + BDIM - 1) / BDIM);

    box_blur_2d_kernel<<<grid, block>>>(
        input.data_ptr<float>(),
        output.data_ptr<float>(),
        width, height
    );

    return output;
}
"""

cpp_source = "torch::Tensor blur_cuda(torch::Tensor input);"

blur_module = load_inline(
    name="box_blur_2d",
    cpp_sources=cpp_source,
    cuda_sources=blur_source,
    functions=["blur_cuda"],
    verbose=True
)

H, W = 1024, 1024
img = torch.randn(H, W, device="cuda", dtype=torch.float32)

torch.cuda.synchronize()
custom_blurred = blur_module.blur_cuda(img)
torch.cuda.synchronize()

pad = torch.nn.ReplicationPad2d(1)
img_padded = pad(img.unsqueeze(0).unsqueeze(0))
kernel = torch.ones((1, 1, 3, 3), device="cuda") / 9.0
torch_blurred = torch.nn.functional.conv2d(img_padded, kernel, padding=0).squeeze()

if torch.allclose(custom_blurred, torch_blurred, atol=1e-5):
    print("good")
else:
    print(f"bad. Max diff: {(custom_blurred - torch_blurred).abs().max()}")
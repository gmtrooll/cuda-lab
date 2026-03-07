import torch
from torch.utils.cpp_extension import load_inline
from PIL import Image
import numpy as np

blur_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>

#define BDIM 16
#define RADIUS 1
#define SDIM (BDIM + 2 * RADIUS)

__global__ void box_blur_2d_kernel(const float* input, float* output, int width, int height) {
    __shared__ float tile[SDIM][SDIM];
    int x = blockIdx.x * BDIM + threadIdx.x;
    int y = blockIdx.y * BDIM + threadIdx.y;
    int lx = threadIdx.x + RADIUS;
    int ly = threadIdx.y + RADIUS;

    int g_x = min(max(x, 0), width - 1);
    int g_y = min(max(y, 0), height - 1);
    tile[ly][lx] = input[g_y * width + g_x];

    if (threadIdx.x < RADIUS) {
        tile[ly][lx - RADIUS] = input[g_y * width + min(max(x - RADIUS, 0), width - 1)];
        tile[ly][lx + BDIM] = input[g_y * width + min(max(x + BDIM, 0), width - 1)];
    }
    if (threadIdx.y < RADIUS) {
        tile[ly - RADIUS][lx] = input[min(max(y - RADIUS, 0), height - 1) * width + g_x];
        tile[ly + BDIM][lx] = input[min(max(y + BDIM, 0), height - 1) * width + g_x];
    }
    if(threadIdx.x < RADIUS && threadIdx.y < RADIUS) {
        tile[ly - RADIUS][lx - RADIUS] = input[min(max(y - RADIUS, 0), height - 1) * width + min(max(x - RADIUS, 0), width - 1)];
        tile[ly - RADIUS][lx + BDIM]   = input[min(max(y - RADIUS, 0), height - 1) * width + min(max(x + BDIM, 0), width - 1)];
        tile[ly + BDIM][lx - RADIUS]   = input[min(max(y + BDIM, 0), height - 1) * width + min(max(x - RADIUS, 0), width - 1)];
        tile[ly + BDIM][lx + BDIM]     = input[min(max(y + BDIM, 0), height - 1) * width + min(max(x + BDIM, 0), width - 1)];
    }
    __syncthreads();

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
    box_blur_2d_kernel<<<grid, block>>>(input.data_ptr<float>(), output.data_ptr<float>(), width, height);
    return output;
}
"""

cpp_source = "torch::Tensor blur_cuda(torch::Tensor input);"
blur_module = load_inline(name="box_blur_real", cpp_sources=cpp_source, cuda_sources=blur_source, functions=["blur_cuda"])

# load and preprocess image
path = "image.jpg" 
img_pill = Image.open(path).convert('L') # 'L' converts to grayscale
img_np = np.array(img_pill).astype(np.float32) / 255.0 # normalize to 0.0 - 1.0

# move to GPU
input_tensor = torch.from_numpy(img_np).cuda()

# run kernel
torch.cuda.synchronize()
output_tensor = blur_module.blur_cuda(input_tensor)
torch.cuda.synchronize()

# convert back to image and save
output_np = output_tensor.cpu().numpy()
output_np = (output_np * 255.0).clip(0, 255).astype(np.uint8)
result_img = Image.fromarray(output_np)
result_img.save("blurred_result.png")
print("Saved blurred_result.png")
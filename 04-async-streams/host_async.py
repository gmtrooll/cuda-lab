import torch
import time
from torch.utils.cpp_extension import load_inline

# Simple scale kernel
cuda_source = """
#include <cuda_runtime.h>
#include <stdint.h>

__global__ void scale_kernel(float* data, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) data[i] = data[i] * 2.0f;
}

// cant pass cudaStream_t directly, so we pass as intptr_t and cast back
void launch_scale_raw(intptr_t data_ptr, int n, intptr_t stream_ptr) {
    int threads = 256;
    int blocks = (n + threads - 1) / threads;
    cudaStream_t stream = (cudaStream_t)stream_ptr;

    // cant use data.data_ptr<float>() because when calling the second kernel with "data_async[mid:]" 
    // it still returns the base pointer as well. we use raw memory adress instead
    scale_kernel<<<blocks, threads, 0, stream>>>((float*)data_ptr, n);
}
"""

cpp_source = "void launch_scale_raw(intptr_t data_ptr, int n, intptr_t stream_ptr);"

scale_mod = load_inline(
    name="scale_async_v5",
    cpp_sources=cpp_source,
    cuda_sources=cuda_source,
    functions=["launch_scale_raw"],
    verbose=False
)

N = 50 * 1024 * 1024 
data_sync = torch.randn(N, device="cuda")
data_async = data_sync.clone() # identical data

s1 = torch.cuda.Stream()
s2 = torch.cuda.Stream()
start_event = torch.cuda.Event(enable_timing=True)
end_event = torch.cuda.Event(enable_timing=True)

# TEST 1: SYNC
torch.cuda.synchronize()
start_event.record()
scale_mod.launch_scale_raw(data_sync.data_ptr(), N, torch.cuda.current_stream().cuda_stream)
end_event.record()
torch.cuda.synchronize()
print(f"Sync Time:  {start_event.elapsed_time(end_event):.3f} ms")

# TEST 2: ASYNC 
mid = N // 2
torch.cuda.synchronize()
start_event.record()

with torch.cuda.stream(s1):
    # pass pointer to index 0
    scale_mod.launch_scale_raw(data_async.data_ptr(), mid, s1.cuda_stream)

with torch.cuda.stream(s2):
    # pass pointer to index 'mid' (ptr + mid * 4 bytes)
    offset_ptr = data_async.data_ptr() + (mid * 4) 
    scale_mod.launch_scale_raw(offset_ptr, N - mid, s2.cuda_stream)

# record event in both streams and wait
end_event.record(s2)
torch.cuda.synchronize()
print(f"Async Time: {start_event.elapsed_time(end_event):.3f} ms")

# VERIFICATION 
if torch.allclose(data_sync, data_async):
    print("✅ Results Match!")
else:
    matches = torch.isclose(data_sync, data_async).sum().item()
    print(f"❌ Mismatch! ({matches}/{N} matched)")
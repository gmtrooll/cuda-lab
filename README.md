# cuda-lab

CUDA learning repo. I'm writing kernels from scratch to actually understand GPU programming, not just calling library functions. Every kernel is tested against PyTorch or CPU results to make sure it's correct.

---

## Skills / Concepts Covered

| Concept | Where | Notes |
|---|---|---|
| Thread / block indexing | [00-hello-cuda/](00-hello-cuda/) | Grid & block launch config, `cudaDeviceSynchronize`, device `printf` |
| Global memory matmul | [01-matmul/naive.cu](01-matmul/naive.cu) | 2D thread mapping, row-major indexing, boundary checks |
| Shared memory tiling | [01-matmul/tiled.cu](01-matmul/tiled.cu) | `__shared__` tiles, `__syncthreads`, tile-loop over K |
| Vectorized loads (`float4`) | [01-matmul/float4_vectorized_inline.py](01-matmul/float4_vectorized_inline.py) | `reinterpret_cast<float4*>`, B-transpose for coalescing, padding for non-aligned dims |
| 2D register tiling | [01-matmul/blocktiling_2d.cu](01-matmul/blocktiling_2d.cu) | BM/BN/BK block params, register accumulators, 256 threads computing 8x8 output tiles |
| Warp shuffle reductions | [02-reductions/](02-reductions/) | `__shfl_down_sync`, staged warp-to-block reduction, `atomicAdd` / `atomicMax` |
| Numerically stable softmax | [02-reductions/softmax.cu](02-reductions/softmax.cu) | Fused max-subtract-exp-sum-divide, row-parallel, single kernel launch |
| 2D stencil with halo | [03-image-processing/](03-image-processing/) | Shared memory halo loading (edges + corners), 2D block config, real image demo |
| CUDA streams | [04-async-streams/](04-async-streams/) | Multi-stream concurrent kernel launch, raw pointer math, `intptr_t` bridge |
| PyTorch inline extensions | All `*_inline.py` files | `load_inline`, C++ bridge functions, `data_ptr<float>()` |

---

## Folder Breakdown

### `00-hello-cuda/`
First CUDA program. Launches a kernel, prints from the GPU, syncs back to CPU. Nothing fancy, just making sure the toolchain works.

### `01-matmul/` - Matrix Multiplication
This is the main progression, optimizing matmul step by step:

1. **`naive`** - One thread per output element. Works but slow, completely bottlenecked by global memory.
2. **`tiled`** - 32x32 tiles in shared memory. Cuts global memory accesses by ~32x.
3. **`float4_vectorized`** - 128-bit vector loads with B-transpose trick. Also has a padded version for weird matrix sizes.
4. **`blocktiling_2d`** - Each thread computes an 8x8 output chunk using registers. 128x128 blocks, BK=8 inner loop. Fastest one I've written so far.

Each kernel has a standalone `.cu` and a `_inline.py` with PyTorch integration.

### `02-reductions/` - Parallel Reductions
- **`row_sum`** - Row-wise reduction: grid-stride loop -> warp shuffle -> shared memory staging -> final warp reduction. No atomics needed for this one.
- **`softmax`** - Full softmax in one kernel: find row max, subtract for numerical stability, exp, sum, normalize.

### `03-image-processing/` - Image Kernels
2D box blur with shared memory halos. Each block loads a tile plus its 1px border (including corners), then computes the 3x3 average. There's a demo that actually blurs a real `.jpg`.

### `04-async-streams/` - Concurrency
Splits a big array in half and runs two kernels on separate CUDA streams at the same time. Shows how to pass raw pointers with offset arithmetic.

---

## How to Run

Standalone `.cu` files (need `nvcc`):
```bash
nvcc -o output 01-matmul/naive.cu
./output
```

PyTorch inline extensions (need PyTorch with CUDA):
```bash
python 01-matmul/naive_inline.py
```

All kernels verify themselves against PyTorch or CPU baselines.

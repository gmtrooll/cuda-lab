# cuda-lab

My GPU kernel development repo. I write kernels from scratch in both CUDA and Triton to understand what's actually happening on the GPU, not just calling library functions. Every kernel is tested against PyTorch or CPU baselines.

---

## Skills / Concepts Covered

| Concept | Kernel | CUDA | Triton | Key Techniques |
|---|---|---|---|---|
| **Basics** | Hello World | [.cu](00-basics/cuda/hello_cuda.cu) | | Grid/block launch, `cudaDeviceSynchronize`, device `printf` |
| **Basics** | Vector Addition | | [.py](00-basics/triton/vector_addition.py) | `tl.program_id`, pointer arithmetic, masking, benchmarking |
| **Basics** | 1D/2D Element-wise | | [.py](00-basics/triton/1d_2d_matrix.py) | 1D vs 2D grid launch, `tl.constexpr` branching, strides |
| **MatMul** | Naive | [.cu](01-matmul/cuda/naive.cu) | | One thread per output, 2D indexing, boundary checks |
| **MatMul** | Tiled | [.cu](01-matmul/cuda/tiled.cu) | | `__shared__` 32x32 tiles, `__syncthreads`, tile-loop over K |
| **MatMul** | float4 Vectorized | [.py](01-matmul/cuda/float4_vectorized_inline.py) | | `reinterpret_cast<float4*>`, B-transpose, 128-bit coalesced loads |
| **MatMul** | float4 + Padded | [.py](01-matmul/cuda/float4_vectorized_padded_inline.py) | | Padding non-aligned dims to multiples of 4, post-kernel slice-back |
| **MatMul** | 2D Block Tiling | [.cu](01-matmul/cuda/blocktiling_2d.cu) | | BM=128, BN=128, BK=8, 8x8 register accumulators, 256 threads per block |
| **MatMul** | Grouped Tiling | | [.py](01-matmul/triton/matrix_mult.py) | `GROUP_SIZE_M` for L2 cache locality, `tl.dot` accumulator, stride-based pointers |
| **Reductions** | Row Sum | [.cu](02-reductions/cuda/row_sum.cu) | | `__shfl_down_sync`, warp->shared->warp staged reduction, no atomics |
| **Reductions** | Softmax | [.cu](02-reductions/cuda/softmax.cu) | [.py](02-reductions/triton/softmax.py) | Fused max-exp-sum-div, `atomicMax` + warp reduce (CUDA) vs `tl.max`/`tl.sum` (Triton) |
| **Normalization** | Layer Norm | | [.py](03-normalization/triton/layer_normalization.py) | Fwd + bwd fused kernels, mean/var/rstd, `atomic_cas` locking for dW/dB |
| **Normalization** | Persistent RMSNorm | | [.py](03-normalization/triton/persistent_rmsnorm.py) | Persistent kernel pattern, grid-stride while-loop, `tl.num_programs` |
| **Attention** | Flash Attention | | [.py](04-attention/triton/flash_attention.py) | Causal masking, online softmax, `tl.make_block_ptr`, multi-stage inner loop |
| **Regularization** | Seeded Dropout (1D) | | [.py](05-regularization/triton/seeded_dropout_simple.py) | `tl.rand`, per-row seeds, inverted dropout scaling |
| **Regularization** | Seeded Dropout (2D) | | [.py](05-regularization/triton/seeded_dropout_modern.py) | 2D grid launch, per-row dropout rates, 2D random offset indexing |
| **Image** | 2D Box Blur | [.cu](06-image-processing/cuda/box_blur_2d.cu) | | Shared memory halo loading (edges + corners), 3x3 stencil, real image demo |
| **Streams** | Multi-Stream Async | [.py](07-async-streams/cuda/host_async.py) | | Split-array concurrency, `intptr_t` bridge, raw pointer offset arithmetic |
| **Integration** | PyTorch Inline Extensions | All `*_inline.py` | | `load_inline`, C++ bridge functions, `data_ptr<float>()`, JIT compilation |

---

## Folder Breakdown

### `00-basics/` - Getting Started
Hello world kernel in CUDA, vector addition and element-wise operations in Triton.

### `01-matmul/` - Matrix Multiplication
The optimization progression for matmul:
1. **`naive`** - One thread per output element. Bottlenecked by global memory.
2. **`tiled`** - 32x32 tiles in shared memory, ~32x fewer global memory accesses.
3. **`float4_vectorized`** - 128-bit vector loads with B-transpose. Padded variant for non-aligned sizes.
4. **`blocktiling_2d`** - Each thread computes 8x8 output using registers. 128x128 blocks, BK=8.
5. **`matrix_mult`** (Triton) - Grouped tiling with `tl.dot` and L2-friendly scheduling.

### `02-reductions/` - Parallel Reductions
- **`row_sum`** - Grid-stride -> warp shuffle -> shared staging -> final warp reduction.
- **`softmax`** - Both CUDA (warp shuffles + atomics) and Triton (`tl.max`/`tl.sum`) implementations.

### `03-normalization/` - Normalization Layers
- **`layer_normalization`** - Forward and backward pass, fused kernels, `atomic_cas` locking.
- **`persistent_rmsnorm`** - Persistent kernel using grid-stride loop with `tl.num_programs`.

### `04-attention/` - Attention Mechanisms
Flash attention with causal masking, online softmax, and `tl.make_block_ptr`.

### `05-regularization/` - Dropout
Seeded dropout with per-row seeds and dropout rates. Simple (1D) and modern (2D grid) versions.

### `06-image-processing/` - Image Kernels
2D box blur with shared memory halo loading. Includes a real image demo.

### `07-async-streams/` - Concurrency
Multi-stream kernel launch splitting work across two CUDA streams.

---

## How to Run

Standalone `.cu` files (need `nvcc`):
```bash
nvcc -o output 01-matmul/cuda/naive.cu
./output
```

PyTorch CUDA inline extensions (need PyTorch with CUDA):
```bash
python 01-matmul/cuda/naive_inline.py
```

Triton kernels (need `triton`):
```bash
python 02-reductions/triton/softmax.py
```

All kernels verify themselves against PyTorch or CPU baselines.

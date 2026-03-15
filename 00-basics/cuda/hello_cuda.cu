#include <iostream>
#include <cstdio>

// This is the "Kernel" - it runs on the GPU
__global__ void gpu_kernel() {
    // printf works in modern CUDA kernels!
    printf("Hello from the GPU! I am thread %d in block %d\n", threadIdx.x, blockIdx.x);
}

int main() {
    std::cout << "Hello from the CPU!" << std::endl;

    // Launch the kernel with 2 blocks, each having 4 threads
    // Total of 8 GPU threads running in parallel
    gpu_kernel<<<2, 4>>>();

    // CRITICAL: The CPU doesn't wait for the GPU by default. 
    // We must force a sync so we can see the output before the program ends.
    cudaDeviceSynchronize();

    std::cout << "GPU is done cooking. CPU closing down." << std::endl;
    return 0;
}
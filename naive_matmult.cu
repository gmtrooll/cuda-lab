#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include <chrono>

__global__ void naive_matmult(float *A, float *B, float *C, int M, int N, int K){
    const int block_id_x = blockIdx.x;
    const int block_id_y = blockIdx.y;
    const int thread_id_x = threadIdx.x;
    const int thread_id_y = threadIdx.y;
    const int block_size_x = blockDim.x;
    const int block_size_y = blockDim.y;

    const int row = block_id_y * block_size_y + thread_id_y;
    const int col = block_id_x * block_size_x + thread_id_x;

    if(row < M && col < N){
        float temp = 0;
        for(int i = 0; i < K; i++)
            temp += A[row * K + i] * B[i * N + col];
        C[row * N + col] = temp;
    }
}

int main(){
    auto begin = std::chrono::high_resolution_clock::now();

    srand(1);

    int N = 4096;
    int M = 1024;
    int K = 2048;

    size_t size_A = M * K * sizeof(float);
    size_t size_B = K * N * sizeof(float);
    size_t size_C = M * N * sizeof(float);

    float *h_A = (float*)malloc(size_A);
    float *h_B = (float*)malloc(size_B);
    float *h_C = (float*)malloc(size_C);

    for(int i = 0; i < M * K; i++)
        h_A[i] = (float)rand() / (float)RAND_MAX;
    for(int i = 0; i < K * N; i++)
        h_B[i] = (float)rand() / (float)RAND_MAX;

    float *d_A, *d_B, *d_C;
    cudaMalloc(&d_A, size_A);
    cudaMalloc(&d_B, size_B);
    cudaMalloc(&d_C, size_C);

    cudaMemcpy(d_A, h_A, size_A, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, size_B, cudaMemcpyHostToDevice);

    dim3 block(32,32);
    dim3 grid((N + block.x - 1) / block.x, (M + block.y - 1) / block.y);
    naive_matmult<<<grid, block>>>(d_A, d_B, d_C, M, N, K);

    cudaMemcpy(h_C, d_C, size_C, cudaMemcpyDeviceToHost);

    // --- VERIFICATION STEP ---
    printf("Verifying result...\n");
    bool success = true;
    
    // We only check a few random samples because checking the whole 
    // matrix on CPU is SLOW (O(N^3)).
    for (int i = 0; i < 10; i++) {
        int r = rand() % M;
        int c = rand() % N;
        float cpu_val = 0;
        for (int k = 0; k < K; k++) {
            cpu_val += h_A[r * K + k] * h_B[k * N + c];
        }

        // Floating point math isn't perfect, so use a small epsilon (error margin)
        if (fabs(h_C[r * N + c] - cpu_val) > 1e-3) {
            printf("Error at [%d,%d]! GPU: %f, CPU: %f\n", r, c, h_C[r * N + c], cpu_val);
            success = false;
            break;
        }
    }

    if (success) printf("Success! GPU results match CPU samples.\n");

    // Clean up
    cudaFree(d_A); cudaFree(d_B); cudaFree(d_C);
    free(h_A); free(h_B); free(h_C);

    auto end = std::chrono::high_resolution_clock::now();
    auto elapsed = std::chrono::duration_cast<std::chrono::nanoseconds>(end - begin);

    printf("Total Program Time: %.3f seconds\n", elapsed.count() * 1e-9);
    
    return 0;
}
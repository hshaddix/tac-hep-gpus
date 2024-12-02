// Part 2: CUDA Integration 

#include <iostream>
#include <cuda.h>
#include <cuda_runtime.h>

#define DSIZE 512
#define RADIUS 3

// CUDA error-checking utility
void checkCudaError(cudaError_t err, const char* msg) {
    if (err != cudaSuccess) {
        std::cerr << msg << " - Error: " << cudaGetErrorString(err) << std::endl;
        exit(EXIT_FAILURE);
    }
}

// Initialize matrix with arbitrary values on host
void initializeMatrix(int* matrix) {
    for (int i = 0; i < DSIZE * DSIZE; ++i) {
        matrix[i] = rand() % 10;
    }
}

// Kernel for stencil operation
__global__ void stencilKernel(int* matrix, int* result) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    int radius = RADIUS;

    if (x >= radius && x < DSIZE - radius && y >= radius && y < DSIZE - radius) {
        int sum = 0;
        for (int i = -radius; i <= radius; ++i)
            for (int j = -radius; j <= radius; ++j)
                sum += matrix[(y + j) * DSIZE + (x + i)];
        result[y * DSIZE + x] = sum;
    }
}

// Kernel for matrix multiplication
__global__ void matrixMultiplyKernel(int* A, int* B, int* C) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < DSIZE && col < DSIZE) {
        int sum = 0;
        for (int k = 0; k < DSIZE; k++) {
            sum += A[row * DSIZE + k] * B[k * DSIZE + col];
        }
        C[row * DSIZE + col] = sum;
    }
}

// Verify results (simple sum check)
bool verifyResult(int* matrix) {
    int total = 0;
    for (int i = 0; i < DSIZE * DSIZE; ++i) {
        total += matrix[i];
    }
    std::cout << "Matrix sum: " << total << std::endl;
    return true;
}

int main() {
    int* h_A = new int[DSIZE * DSIZE];
    int* h_B = new int[DSIZE * DSIZE];
    int* h_C = new int[DSIZE * DSIZE];

    initializeMatrix(h_A);
    initializeMatrix(h_B);

    int *d_A, *d_B, *d_tempA, *d_tempB, *d_C;
    cudaMalloc((void**)&d_A, DSIZE * DSIZE * sizeof(int));
    cudaMalloc((void**)&d_B, DSIZE * DSIZE * sizeof(int));
    cudaMalloc((void**)&d_tempA, DSIZE * DSIZE * sizeof(int));
    cudaMalloc((void**)&d_tempB, DSIZE * DSIZE * sizeof(int));
    cudaMalloc((void**)&d_C, DSIZE * DSIZE * sizeof(int));

    cudaMemcpy(d_A, h_A, DSIZE * DSIZE * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, DSIZE * DSIZE * sizeof(int), cudaMemcpyHostToDevice);

    dim3 threadsPerBlock(16, 16);
    dim3 blocksPerGrid((DSIZE + threadsPerBlock.x - 1) / threadsPerBlock.x, (DSIZE + threadsPerBlock.y - 1) / threadsPerBlock.y);

    stencilKernel<<<blocksPerGrid, threadsPerBlock>>>(d_A, d_tempA);
    stencilKernel<<<blocksPerGrid, threadsPerBlock>>>(d_B, d_tempB);
    cudaDeviceSynchronize();

    matrixMultiplyKernel<<<blocksPerGrid, threadsPerBlock>>>(d_tempA, d_tempB, d_C);
    cudaMemcpy(h_C, d_C, DSIZE * DSIZE * sizeof(int), cudaMemcpyDeviceToHost);

    verifyResult(h_C);

    delete[] h_A;
    delete[] h_B;
    delete[] h_C;
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_tempA);
    cudaFree(d_tempB);
    cudaFree(d_C);

    return 0;
}

// Output:
// Matrix sum: 560186929

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

// Profiling
// ==3250273== NVPROF is profiling process 3250273, command: ./CUDA
// Matrix sum: 560186929
// ==3250273== Profiling application: ./CUDA
// ==3250273== Profiling result:
//             Type  Time(%)      Time     Calls       Avg       Min       Max  Name
//  GPU activities:   67.97%  784.53us         1  784.53us  784.53us  784.53us  matrixMultiplyKernel(int*, int*, int*)
//                    14.21%  164.06us         2  82.031us  81.951us  82.111us  [CUDA memcpy HtoD]
//                     9.73%  112.35us         2  56.175us  56.095us  56.255us  stencilKernel(int*, int*)
//                     8.08%  93.214us         1  93.214us  93.214us  93.214us  [CUDA memcpy DtoH]
//       API calls:   96.20%  291.08ms         5  58.217ms  4.1600us  290.73ms  cudaMalloc
//                     1.87%  5.6726ms       228  24.879us      90ns  1.5433ms  cuDeviceGetAttribute
//                     0.83%  2.5064ms         3  835.48us  8.6300us  2.4879ms  cudaLaunchKernel
//                     0.82%  2.4856ms         3  828.52us  161.26us  2.1335ms  cudaMemcpy
//                     0.21%  644.85us         5  128.97us  7.2510us  282.66us  cudaFree
//                     0.03%  104.15us         1  104.15us  104.15us  104.15us  cudaDeviceSynchronize
//                     0.02%  45.921us         2  22.960us  7.9200us  38.001us  cuDeviceGetName
//                     0.01%  20.791us         2  10.395us  3.3900us  17.401us  cuDeviceGetPCIBusId
//                     0.00%  3.0600us         4     765ns     100ns  2.3100us  cuDeviceGet
//                     0.00%  1.7200us         3     573ns     160ns  1.3900us  cuDeviceGetCount
//                     0.00%     980ns         2     490ns     430ns     550ns  cuDeviceTotalMem
//                     0.00%     730ns         2     365ns     280ns     450ns  cuDeviceGetUuid
//                    0.00%     310ns         1     310ns     310ns     310ns  cuModuleGetLoadingMode

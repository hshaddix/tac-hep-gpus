// Part 3: Optimized CUDA 

#include <iostream>
#include <cuda.h>
#include <cuda_runtime.h>

#define DSIZE 512
#define RADIUS 3
#define TILE_SIZE 16  // Tile size for shared memory

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

// Kernel for stencil operation using shared memory
__global__ void stencilKernel(int* matrix, int* result) {
    __shared__ int tile[TILE_SIZE + 2 * RADIUS][TILE_SIZE + 2 * RADIUS];
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    int tx = threadIdx.x + RADIUS;
    int ty = threadIdx.y + RADIUS;

    if (x < DSIZE && y < DSIZE) {
        // Load data into shared memory, including halo regions
        tile[ty][tx] = matrix[y * DSIZE + x];
        if (threadIdx.x < RADIUS) {
            tile[ty][tx - RADIUS] = (x >= RADIUS) ? matrix[y * DSIZE + x - RADIUS] : 0;
            tile[ty][tx + TILE_SIZE] = (x + TILE_SIZE < DSIZE) ? matrix[y * DSIZE + x + TILE_SIZE] : 0;
        }
        if (threadIdx.y < RADIUS) {
            tile[ty - RADIUS][tx] = (y >= RADIUS) ? matrix[(y - RADIUS) * DSIZE + x] : 0;
            tile[ty + TILE_SIZE][tx] = (y + TILE_SIZE < DSIZE) ? matrix[(y + TILE_SIZE) * DSIZE + x] : 0;
        }
        __syncthreads();

        // Compute the stencil if within bounds
        if (x >= RADIUS && x < DSIZE - RADIUS && y >= RADIUS && y < DSIZE - RADIUS) {
            int sum = 0;
            for (int i = -RADIUS; i <= RADIUS; i++) {
                for (int j = -RADIUS; j <= RADIUS; j++) {
                    sum += tile[ty + j][tx + i];
                }
            }
            result[y * DSIZE + x] = sum;
        }
    }
}

// Kernel for matrix multiplication using shared memory
__global__ void matrixMultiplyKernel(int* A, int* B, int* C) {
    __shared__ int tileA[TILE_SIZE][TILE_SIZE];
    __shared__ int tileB[TILE_SIZE][TILE_SIZE];

    int row = blockIdx.y * TILE_SIZE + threadIdx.y;
    int col = blockIdx.x * TILE_SIZE + threadIdx.x;

    int sum = 0;
    for (int i = 0; i < DSIZE / TILE_SIZE; ++i) {
        // Load tiles from A and B into shared memory
        tileA[threadIdx.y][threadIdx.x] = A[row * DSIZE + i * TILE_SIZE + threadIdx.x];
        tileB[threadIdx.y][threadIdx.x] = B[(i * TILE_SIZE + threadIdx.y) * DSIZE + col];
        __syncthreads();

        // Perform partial matrix multiplication for this tile
        for (int k = 0; k < TILE_SIZE; ++k) {
            sum += tileA[threadIdx.y][k] * tileB[k][threadIdx.x];
        }
        __syncthreads();
    }

    if (row < DSIZE && col < DSIZE) {
        C[row * DSIZE + col] = sum;
    }
}

// Utility function to verify the result (simple sum check)
bool verifyResult(int* matrix) {
    int total = 0;
    for (int i = 0; i < DSIZE * DSIZE; ++i) {
        total += matrix[i];
    }
    std::cout << "Matrix sum: " << total << std::endl;
    return true;  // Placeholder verification
}

int main() {
    // Allocate host memory
    int* h_A = new int[DSIZE * DSIZE];
    int* h_B = new int[DSIZE * DSIZE];
    int* h_C = new int[DSIZE * DSIZE];

    initializeMatrix(h_A);
    initializeMatrix(h_B);

    // Allocate device memory
    int *d_A, *d_B, *d_tempA, *d_tempB, *d_C;
    cudaMalloc((void**)&d_A, DSIZE * DSIZE * sizeof(int));
    cudaMalloc((void**)&d_B, DSIZE * DSIZE * sizeof(int));
    cudaMalloc((void**)&d_tempA, DSIZE * DSIZE * sizeof(int));
    cudaMalloc((void**)&d_tempB, DSIZE * DSIZE * sizeof(int));
    cudaMalloc((void**)&d_C, DSIZE * DSIZE * sizeof(int));

    // Create CUDA streams
    cudaStream_t stream1, stream2;
    cudaStreamCreate(&stream1);
    cudaStreamCreate(&stream2);

    // Copy data to device asynchronously
    cudaMemcpyAsync(d_A, h_A, DSIZE * DSIZE * sizeof(int), cudaMemcpyHostToDevice, stream1);
    cudaMemcpyAsync(d_B, h_B, DSIZE * DSIZE * sizeof(int), cudaMemcpyHostToDevice, stream2);

    dim3 threadsPerBlock(TILE_SIZE, TILE_SIZE);
    dim3 blocksPerGrid((DSIZE + TILE_SIZE - 1) / TILE_SIZE, (DSIZE + TILE_SIZE - 1) / TILE_SIZE);

    // Launch stencil kernels on matrices A and B asynchronously
    stencilKernel<<<blocksPerGrid, threadsPerBlock, 0, stream1>>>(d_A, d_tempA);
    stencilKernel<<<blocksPerGrid, threadsPerBlock, 0, stream2>>>(d_B, d_tempB);

    // Synchronize streams before proceeding to matrix multiplication
    cudaStreamSynchronize(stream1);
    cudaStreamSynchronize(stream2);

    // Launch matrix multiplication kernel on a single stream
    matrixMultiplyKernel<<<blocksPerGrid, threadsPerBlock>>>(d_tempA, d_tempB, d_C);

    // Copy result back to host
    cudaMemcpy(h_C, d_C, DSIZE * DSIZE * sizeof(int), cudaMemcpyDeviceToHost);

    // Verify result
    verifyResult(h_C);

    // Clean up
    delete[] h_A;
    delete[] h_B;
    delete[] h_C;
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_tempA);
    cudaFree(d_tempB);
    cudaFree(d_C);
    cudaStreamDestroy(stream1);
    cudaStreamDestroy(stream2);

    return 0;
}

// Output: 
// Matrix sum: 1295711823

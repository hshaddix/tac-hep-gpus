#include <stdio.h>
#include <time.h>
#include <cuda_runtime.h>

const int DSIZE = 256;        // Size of the NxN matrix
const float A_val = 3.0f;     // Constant for matrix A
const float B_val = 2.0f;     // Constant for matrix B

// Error checking macro
#define cudaCheckErrors(msg)                                   \
   do {                                                        \
       cudaError_t __err = cudaGetLastError();                 \
       if (__err != cudaSuccess) {                             \
           fprintf(stderr, "Fatal error: %s (%s at %s:%d)\n",  \
                   msg, cudaGetErrorString(__err),             \
                   __FILE__, __LINE__);                        \
           fprintf(stderr, "*** FAILED - ABORTING\n");         \
           exit(1);                                            \
       }                                                       \
   } while (0)

// Print a matrix (for debugging/verification)
void print_matrix(const float *matrix, int size, int blockSize) {
    printf("\nMatrix:\n");
    for (int i = 0; i < blockSize; i++) {
        for (int j = 0; j < blockSize; j++) {
            printf("%f ", matrix[i * size + j]);
        }
        printf("\n");
    }
    printf("\n");
}

// Square matrix multiplication on CPU : C = A * B
void matrix_mul_cpu(const float *A, const float *B, float *C, int size) {
    for (int i = 0; i < size; i++) {
        for (int j = 0; j < size; j++) {
            float temp = 0;
            for (int k = 0; k < size; k++) {
                temp += A[i * size + k] * B[k * size + j]; // Row of A * Column of B
            }
            C[i * size + j] = temp;
        }
    }
}

// Square matrix multiplication on GPU : C = A * B
__global__ void matrix_mul_gpu(const float *A, const float *B, float *C, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x; // Column index
    int idy = blockIdx.y * blockDim.y + threadIdx.y; // Row index

    if (idx < size && idy < size) {
        float temp = 0;
        for (int i = 0; i < size; i++) {
            temp += A[idy * size + i] * B[i * size + idx]; // Row of A * Column of B
        }
        C[idy * size + idx] = temp;
    }
}

int main() {
    float *h_A, *h_B, *h_C, *d_A, *d_B, *d_C;

    // Timing variables
    clock_t t0, t1, t2, t3;
    double t1sum = 0.0, t2sum = 0.0, t3sum = 0.0;

    // Start timing for initialization
    t0 = clock();

    // N*N matrices defined in 1 dimension
    h_A = new float[DSIZE * DSIZE];
    h_B = new float[DSIZE * DSIZE];
    h_C = new float[DSIZE * DSIZE];

    // Initialize the matrices
    for (int i = 0; i < DSIZE * DSIZE; i++) {
        h_A[i] = A_val;
        h_B[i] = B_val;
        h_C[i] = 0;
    }

    // Print A and B before multiplication
    printf("Matrix A (First 5x5 elements):");
    print_matrix(h_A, DSIZE, 5);
    printf("Matrix B (First 5x5 elements):");
    print_matrix(h_B, DSIZE, 5);

    // End initialization timing
    t1 = clock();
    t1sum = ((double)(t1 - t0)) / CLOCKS_PER_SEC;
    printf("Init took %f seconds. Begin compute\n", t1sum);

    // Allocate device memory
    cudaMalloc(&d_A, DSIZE * DSIZE * sizeof(float));
    cudaCheckErrors("Failed to allocate device memory for A");
    
    cudaMalloc(&d_B, DSIZE * DSIZE * sizeof(float));
    cudaCheckErrors("Failed to allocate device memory for B");
    
    cudaMalloc(&d_C, DSIZE * DSIZE * sizeof(float));
    cudaCheckErrors("Failed to allocate device memory for C");

    // Copy data from host to device
    cudaMemcpy(d_A, h_A, DSIZE * DSIZE * sizeof(float), cudaMemcpyHostToDevice);
    cudaCheckErrors("Failed to copy A to device");
    
    cudaMemcpy(d_B, h_B, DSIZE * DSIZE * sizeof(float), cudaMemcpyHostToDevice);
    cudaCheckErrors("Failed to copy B to device");

    // Specify block and grid dimensions
    dim3 block(16, 16);  // 16x16 threads per block
    dim3 grid((DSIZE + block.x - 1) / block.x, (DSIZE + block.y - 1) / block.y);

    // GPU matrix multiplication
    cudaEvent_t start, stop;
    float gpu_time = 0;
    
    // Start timing for GPU execution
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start);

    // Launch kernel
    matrix_mul_gpu<<<grid, block>>>(d_A, d_B, d_C, DSIZE);
    cudaCheckErrors("Kernel launch failed");

    // Stop GPU timing
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&gpu_time, start, stop);
    printf("GPU Compute took %f milliseconds\n", gpu_time);

    // Copy results back to host
    cudaMemcpy(h_C, d_C, DSIZE * DSIZE * sizeof(float), cudaMemcpyDeviceToHost);
    cudaCheckErrors("Failed to copy C to host");

    // End GPU timing
    t2 = clock();
    t2sum = ((double)(t2 - t1)) / CLOCKS_PER_SEC;
    printf("GPU Compute took %f seconds\n", t2sum);

    // Print C after GPU multiplication
    printf("Matrix C (First 5x5 elements) after GPU multiplication:");
    print_matrix(h_C, DSIZE, 5);

    // CPU matrix multiplication
    t3 = clock();
    matrix_mul_cpu(h_A, h_B, h_C, DSIZE);
    t3sum = ((double)(clock() - t3)) / CLOCKS_PER_SEC;
    printf("CPU Compute took %f seconds\n", t3sum);

    // Print C after CPU multiplication
    printf("Matrix C (First 5x5 elements) after CPU multiplication:");
    print_matrix(h_C, DSIZE, 5);

    // Free memory
    delete[] h_A;
    delete[] h_B;
    delete[] h_C;
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);

    return 0;
}

#include <stdio.h>
#include <cuda_runtime.h>

const int DSIZE = 40960;        // Size of vectors
const int block_size = 256;     // Block size
const int grid_size = DSIZE / block_size; // Grid size

__global__ void vector_swap(float *A, float *B, int size) {
    // Calculate the global index of the thread
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    // Swap elements if index is within bounds
    if (idx < size) {
        float temp = A[idx];
        A[idx] = B[idx];
        B[idx] = temp;
    }
}

int main() {
    float *h_A, *h_B, *d_A, *d_B;

    // Allocate host memory
    h_A = new float[DSIZE];
    h_B = new float[DSIZE];

    // Initialize vectors A and B
    for (int i = 0; i < DSIZE; i++) {
        h_A[i] = rand() / (float)RAND_MAX;  // Random values between 0 and 1
        h_B[i] = rand() / (float)RAND_MAX;  // Random values between 0 and 1
    }

    // Print a few elements before the swap
    printf("Before Swap:\n");
    printf("A[0] = %f, B[0] = %f\n", h_A[0], h_B[0]);
    printf("A[DSIZE-1] = %f, B[DSIZE-1] = %f\n", h_A[DSIZE-1], h_B[DSIZE-1]);

    // Allocate device memory
    cudaMalloc((void**)&d_A, DSIZE * sizeof(float));
    cudaMalloc((void**)&d_B, DSIZE * sizeof(float));

    // Copy data from host to device
    cudaMemcpy(d_A, h_A, DSIZE * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, DSIZE * sizeof(float), cudaMemcpyHostToDevice);

    // Launch the kernel for swapping
    vector_swap<<<grid_size, block_size>>>(d_A, d_B, DSIZE);

    // Copy back the swapped data from device to host
    cudaMemcpy(h_A, d_A, DSIZE * sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(h_B, d_B, DSIZE * sizeof(float), cudaMemcpyDeviceToHost);

    // Print a few elements after the swap
    printf("\nAfter Swap:\n");
    printf("A[0] = %f, B[0] = %f\n", h_A[0], h_B[0]);
    printf("A[DSIZE-1] = %f, B[DSIZE-1] = %f\n", h_A[DSIZE-1], h_B[DSIZE-1]);

    // Free host and device memory
    delete[] h_A;
    delete[] h_B;
    cudaFree(d_A);
    cudaFree(d_B);

    return 0;
}

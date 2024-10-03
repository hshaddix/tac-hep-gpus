#include <stdio.h>
#include <cuda_runtime.h>

const int DSIZE_X = 256; // Matrix width (M)
const int DSIZE_Y = 256; // Matrix height (N)

// Kernel to add two matrices
__global__ void add_matrix(float *A, float *B, float *C, int width, int height)
{
    // Calculate 2D index using thread and block indices
    int idx = blockIdx.x * blockDim.x + threadIdx.x; // Column index
    int idy = blockIdx.y * blockDim.y + threadIdx.y; // Row index

    // Convert the 2D index [i,j] into a 1D index: [i * width + j]
    int index = idy * width + idx;

    // Add the two matrices if the index is within bounds
    if (idx < width && idy < height) {
        C[index] = A[index] + B[index];
    }
}

int main()
{
    // Matrix sizes
    int matrix_size = DSIZE_X * DSIZE_Y;
    int mem_size = matrix_size * sizeof(float);

    // Host memory allocation
    float *h_A = new float[matrix_size];
    float *h_B = new float[matrix_size];
    float *h_C = new float[matrix_size];

    // Initialize matrices with some values
    for (int i = 0; i < DSIZE_Y; i++) {
        for (int j = 0; j < DSIZE_X; j++) {
            int index = i * DSIZE_X + j;
            h_A[index] = rand() / (float)RAND_MAX;  // Random values between 0 and 1
            h_B[index] = rand() / (float)RAND_MAX;  // Random values between 0 and 1
            h_C[index] = 0;  // Initialize result matrix to 0
        }
    }

    // Device memory allocation
    float *d_A, *d_B, *d_C;
    cudaMalloc((void**)&d_A, mem_size);
    cudaMalloc((void**)&d_B, mem_size);
    cudaMalloc((void**)&d_C, mem_size);

    // Copy data from host to device
    cudaMemcpy(d_A, h_A, mem_size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, mem_size, cudaMemcpyHostToDevice);

    // Define block size and grid size
    dim3 blockSize(16, 16); // 16x16 threads per block
    dim3 gridSize((DSIZE_X + blockSize.x - 1) / blockSize.x, (DSIZE_Y + blockSize.y - 1) / blockSize.y); // Ensure complete coverage of the matrix

    // Launch the kernel
    add_matrix<<<gridSize, blockSize>>>(d_A, d_B, d_C, DSIZE_X, DSIZE_Y);

    // Copy back the result from device to host
    cudaMemcpy(h_C, d_C, mem_size, cudaMemcpyDeviceToHost);

    // Verify by printing a few elements
    printf("A[0] + B[0] = %f + %f = %f\n", h_A[0], h_B[0], h_C[0]);
    printf("A[DSIZE_X-1] + B[DSIZE_X-1] = %f + %f = %f\n", h_A[DSIZE_X-1], h_B[DSIZE_X-1], h_C[DSIZE_X-1]);

    // Free memory
    delete[] h_A;
    delete[] h_B;
    delete[] h_C;
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);

    return 0;
}

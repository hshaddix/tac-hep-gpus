#include <stdio.h>
#include <algorithm>

using namespace std;

#define N 64
#define RADIUS 2
#define BLOCK_SIZE 32

__global__ void stencil_2d(int *in, int *out) {
    // Shared memory array (larger than block to account for halos)
    __shared__ int temp[BLOCK_SIZE + 2 * RADIUS][BLOCK_SIZE + 2 * RADIUS];

    // Global and local indices
    int gindex_x = blockIdx.x * blockDim.x + threadIdx.x; // global x-index
    int gindex_y = blockIdx.y * blockDim.y + threadIdx.y; // global y-index
    int lindex_x = threadIdx.x + RADIUS;  // local x-index in shared memory
    int lindex_y = threadIdx.y + RADIUS;  // local y-index in shared memory

    // Read input elements into shared memory
    int size = N + 2 * RADIUS;
    temp[lindex_x][lindex_y] = in[gindex_x + gindex_y * size];

    // Handle the halo elements
    if (threadIdx.x < RADIUS) {
        temp[lindex_x - RADIUS][lindex_y] = in[(gindex_x - RADIUS) + gindex_y * size];  // Left halo
        temp[lindex_x + BLOCK_SIZE][lindex_y] = in[(gindex_x + BLOCK_SIZE) + gindex_y * size];  // Right halo
    }
    if (threadIdx.y < RADIUS) {
        temp[lindex_x][lindex_y - RADIUS] = in[gindex_x + (gindex_y - RADIUS) * size];  // Top halo
        temp[lindex_x][lindex_y + BLOCK_SIZE] = in[gindex_x + (gindex_y + BLOCK_SIZE) * size];  // Bottom halo
    }

    // Wait for all threads to finish loading data into shared memory
    __syncthreads();

    // Apply the stencil: 1 center + 4 neighbors (left, right, top, bottom)
    int result = 0;
    for (int offset = -RADIUS; offset <= RADIUS; offset++) {
        if (offset != 0) {
            result += temp[lindex_x + offset][lindex_y];  // Horizontal neighbors
            result += temp[lindex_x][lindex_y + offset];  // Vertical neighbors
        }
    }
    result += temp[lindex_x][lindex_y]; // Add center value

    // Store the result
    out[gindex_x + gindex_y * size] = result;
}

void fill_ints(int *x, int n) {
    fill_n(x, n, 1);  // Fill the array with 1s
}

int main(void) {

    int *in, *out;           // host copies of input and output arrays
    int *d_in, *d_out;       // device copies of input and output arrays

    // Alloc space for host copies and setup values
    int size = (N + 2 * RADIUS) * (N + 2 * RADIUS) * sizeof(int);
    in = (int *)malloc(size);
    out = (int *)malloc(size);
    fill_ints(in, (N + 2 * RADIUS) * (N + 2 * RADIUS));
    fill_ints(out, (N + 2 * RADIUS) * (N + 2 * RADIUS));

    // Alloc space for device copies
    cudaMalloc((void **)&d_in, size);
    cudaMalloc((void **)&d_out, size);

    // Copy input to device
    cudaMemcpy(d_in, in, size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_out, out, size, cudaMemcpyHostToDevice);

    // Launch stencil_2d() kernel on GPU
    int gridSize = (N + BLOCK_SIZE - 1) / BLOCK_SIZE;
    dim3 grid(gridSize, gridSize);
    dim3 block(BLOCK_SIZE, BLOCK_SIZE);

    stencil_2d<<<grid, block>>>(d_in + RADIUS * (N + 2 * RADIUS) + RADIUS, d_out + RADIUS * (N + 2 * RADIUS) + RADIUS);

    // Copy result back to host
    cudaMemcpy(out, d_out, size, cudaMemcpyDeviceToHost);

    // Error Checking
    for (int i = 0; i < N + 2 * RADIUS; ++i) {
        for (int j = 0; j < N + 2 * RADIUS; ++j) {
            if (i < RADIUS || i >= N + RADIUS) {
                if (out[j + i * (N + 2 * RADIUS)] != 1) {
                    printf("Mismatch at index [%d,%d], was: %d, should be: %d\n", i, j, out[j + i * (N + 2 * RADIUS)], 1);
                    return -1;
                }
            } else if (j < RADIUS || j >= N + RADIUS) {
                if (out[j + i * (N + 2 * RADIUS)] != 1) {
                    printf("Mismatch at index [%d,%d], was: %d, should be: %d\n", i, j, out[j + i * (N + 2 * RADIUS)], 1);
                    return -1;
                }
            } else {
                if (out[j + i * (N + 2 * RADIUS)] != 1 + 4 * RADIUS) {
                    printf("Mismatch at index [%d,%d], was: %d, should be: %d\n", i, j, out[j + i * (N + 2 * RADIUS)], 1 + 4 * RADIUS);
                    return -1;
                }
            }
        }
    }

    // Cleanup
    free(in);
    free(out);
    cudaFree(d_in);
    cudaFree(d_out);
    printf("Success!\n");

    return 0;
}


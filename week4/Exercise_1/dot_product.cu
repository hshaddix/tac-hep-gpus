#include <stdio.h>
#include <time.h>

#define BLOCK_SIZE 32

const int DSIZE = 256;
const int a = 1;
const int b = 1;

// error checking macro
#define cudaCheckErrors()                                       \
	do {                                                        \
		cudaError_t __err = cudaGetLastError();                 \
		if (__err != cudaSuccess) {                             \
			fprintf(stderr, "Error:  %s at %s:%d \n",           \
			cudaGetErrorString(__err),__FILE__, __LINE__);      \
			fprintf(stderr, "*** FAILED - ABORTING***\n");      \
			exit(1);                                            \
		}                                                       \
	} while (0)


// CUDA kernel that runs on the GPU
__global__ void dot_product(const int *A, const int *B, int *C, int N) {
	// Get thread index
	int tid = blockIdx.x * blockDim.x + threadIdx.x;

	// Perform dot product only if within bounds
	if (tid < N) {
		// Each thread computes A[tid] * B[tid] and accumulates into C using atomicAdd
		atomicAdd(C, A[tid] * B[tid]);
	}
}


int main() {
	
	// Create the device and host pointers
	int *h_A, *h_B, *h_C, *d_A, *d_B, *d_C;

	// Fill in the host pointers 
	h_A = new int[DSIZE];
	h_B = new int[DSIZE];
	h_C = new int;
	for (int i = 0; i < DSIZE; i++){
		h_A[i] = a;   // Set all elements to 'a'
		h_B[i] = b;   // Set all elements to 'b'
	}

	*h_C = 0; // Initialize host result to 0

	// Allocate device memory 
	cudaMalloc((void**)&d_A, DSIZE * sizeof(int));
	cudaMalloc((void**)&d_B, DSIZE * sizeof(int));
	cudaMalloc((void**)&d_C, sizeof(int));
	cudaCheckErrors(); // Check for any memory allocation errors

	// Copy the host arrays to GPU memory
	cudaMemcpy(d_A, h_A, DSIZE * sizeof(int), cudaMemcpyHostToDevice);
	cudaMemcpy(d_B, h_B, DSIZE * sizeof(int), cudaMemcpyHostToDevice);
	cudaMemcpy(d_C, h_C, sizeof(int), cudaMemcpyHostToDevice);  // Copy the initial value of h_C to d_C
	cudaCheckErrors(); // Check for any memory copy errors

	// Define block/grid dimensions and launch kernel
	int numBlocks = (DSIZE + BLOCK_SIZE - 1) / BLOCK_SIZE;
	dot_product<<<numBlocks, BLOCK_SIZE>>>(d_A, d_B, d_C, DSIZE);
	cudaCheckErrors(); // Check for kernel launch errors

	// Copy results back to host
	cudaMemcpy(h_C, d_C, sizeof(int), cudaMemcpyDeviceToHost);
	cudaCheckErrors(); // Check for any memory copy errors

	// Verify result
	int expected = DSIZE * a * b;
	if (*h_C == expected) {
		printf("SUCCESS: Dot product is correct!\n");
		printf("Computed dot product: %d\n", *h_C);
	} else {
		printf("ERROR: Dot product is incorrect.\n");
		printf("Expected: %d, but got: %d\n", expected, *h_C);
	}

	// Free allocated memory
	cudaFree(d_A);
	cudaFree(d_B);
	cudaFree(d_C);
	delete[] h_A;
	delete[] h_B;
	delete h_C;

	return 0;
}


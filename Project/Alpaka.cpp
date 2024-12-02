// Part 4: Alpaka 

#include <alpaka/alpaka.hpp>
#include <iostream>
#include <vector>

#define DSIZE 512
#define RADIUS 3

using namespace alpaka;

template<typename TAcc>
ALPAKA_FN_ACC void stencilKernel(const TAcc& acc, int* matrix, int* result) {
    const int x = blockIdx.x * blockDim.x + threadIdx.x;
    const int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x >= RADIUS && x < DSIZE - RADIUS && y >= RADIUS && y < DSIZE - RADIUS) {
        int sum = 0;
        for (int i = -RADIUS; i <= RADIUS; ++i)
            for (int j = -RADIUS; j <= RADIUS; ++j)
                sum += matrix[(y + j) * DSIZE + (x + i)];
        result[y * DSIZE + x] = sum;
    }
}

template<typename TAcc>
ALPAKA_FN_ACC void matrixMultiplyKernel(const TAcc& acc, int* A, int* B, int* C) {
    const int row = blockIdx.y * blockDim.y + threadIdx.y;
    const int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < DSIZE && col < DSIZE) {
        int sum = 0;
        for (int k = 0; k < DSIZE; ++k) {
            sum += A[row * DSIZE + k] * B[k * DSIZE + col];
        }
        C[row * DSIZE + col] = sum;
    }
}

void initializeMatrix(std::vector<int>& matrix) {
    for (int i = 0; i < DSIZE * DSIZE; ++i) {
        matrix[i] = rand() % 10;
    }
}

void verifyResult(const std::vector<int>& matrix) {
    int total = 0;
    for (int i = 0; i < DSIZE * DSIZE; ++i) {
        total += matrix[i];
    }
    std::cout << "Matrix sum: " << total << std::endl;
}

int main() {
    using Dim = dim::DimInt<2>;
    using Idx = uint32_t;
    using Acc = acc::AccCpuSerial<Dim, Idx>;

    constexpr Idx blockDim = 16;
    constexpr Idx blocksPerGrid = (DSIZE + blockDim - 1) / blockDim;
    Vec<Dim, Idx> threadsPerBlock(blockDim, blockDim);
    Vec<Dim, Idx> blocksPerGrid(blocksPerGrid, blocksPerGrid);

    std::vector<int> h_A(DSIZE * DSIZE);
    std::vector<int> h_B(DSIZE * DSIZE);
    std::vector<int> h_C(DSIZE * DSIZE);

    initializeMatrix(h_A);
    initializeMatrix(h_B);

    dev::DevCpu dev;
    Queue<Acc> queue(dev);

    auto d_A = allocBuf<int, Idx>(dev, Vec<Dim, Idx>(DSIZE, DSIZE));
    auto d_B = allocBuf<int, Idx>(dev, Vec<Dim, Idx>(DSIZE, DSIZE));
    auto d_tempA = allocBuf<int, Idx>(dev, Vec<Dim, Idx>(DSIZE, DSIZE));
    auto d_tempB = allocBuf<int, Idx>(dev, Vec<Dim, Idx>(DSIZE, DSIZE));
    auto d_C = allocBuf<int, Idx>(dev, Vec<Dim, Idx>(DSIZE, DSIZE));

    queue.enqueue(memcpy(d_A, h_A.data()));
    queue.enqueue(memcpy(d_B, h_B.data()));

    // Stencil and matrix multiplication operations using Alpaka kernels
    queue.enqueue(kernel::createTaskKernel<Acc>(blocksPerGrid, threadsPerBlock, stencilKernel<Acc>, d_A, d_tempA));
    queue.enqueue(kernel::createTaskKernel<Acc>(blocksPerGrid, threadsPerBlock, stencilKernel<Acc>, d_B, d_tempB));
    queue.enqueue(kernel::createTaskKernel<Acc>(blocksPerGrid, threadsPerBlock, matrixMultiplyKernel<Acc>, d_tempA, d_tempB, d_C));

    queue.enqueue(memcpy(h_C.data(), d_C));
    verifyResult(h_C);

    return 0;
}

// Could not get to run :(

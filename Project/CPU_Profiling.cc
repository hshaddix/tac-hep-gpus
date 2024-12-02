/// End of class Project 
// Work done by: Hayden Shaddix 
//

// Part 1: C++ and Profiling 

#include <iostream>
#include <vector>
#include <cstdlib>
#include <chrono>

#define DSIZE 512

using namespace std;

// Initialize matrix with random values
void initializeMatrix(vector<vector<int>>& matrix) {
    for (int i = 0; i < DSIZE; ++i)
        for (int j = 0; j < DSIZE; ++j)
            matrix[i][j] = rand() % 10;
}

// 2D stencil function
void applyStencil(vector<vector<int>>& matrix) {
    vector<vector<int>> temp = matrix; 
    int radius = 3;

    for (int i = radius; i < DSIZE - radius; ++i) {
        for (int j = radius; j < DSIZE - radius; ++j) {
            int sum = 0;
            for (int di = -radius; di <= radius; ++di)
                for (int dj = -radius; dj <= radius; ++dj)
                    sum += matrix[i + di][j + dj];
            temp[i][j] = sum;
        }
    }

    matrix = temp;
}

// Matrix multiplication
void matrixMultiply(const vector<vector<int>>& A, const vector<vector<int>>& B, vector<vector<int>>& C) {
    for (int i = 0; i < DSIZE; ++i) {
        for (int j = 0; j < DSIZE; ++j) {
            C[i][j] = 0;
            for (int k = 0; k < DSIZE; ++k)
                C[i][j] += A[i][k] * B[k][j];
        }
    }
}

// Verify results
bool verifyResult(const vector<vector<int>>& matrix) {
    int total = 0;
    for (const auto& row : matrix)
        for (const auto& elem : row)
            total += elem;

    cout << "Matrix sum: " << total << endl;
    return true;
}

int main() {
    vector<vector<int>> A(DSIZE, vector<int>(DSIZE));
    vector<vector<int>> B(DSIZE, vector<int>(DSIZE));
    vector<vector<int>> C(DSIZE, vector<int>(DSIZE));

    initializeMatrix(A);
    initializeMatrix(B);

    auto start = chrono::high_resolution_clock::now();
    applyStencil(A);
    applyStencil(B);
    auto end = chrono::high_resolution_clock::now();
    cout << "Time for stencil: " << chrono::duration_cast<chrono::milliseconds>(end - start).count() << " ms\n";

    start = chrono::high_resolution_clock::now();
    matrixMultiply(A, B, C);
    end = chrono::high_resolution_clock::now();
    cout << "Time for matrix multiplication: " << chrono::duration_cast<chrono::milliseconds>(end - start).count() << " ms\n";

    verifyResult(C);

    return 0;
}

// Output: 
// Time for stencil: 260 ms
// Time for matrix multiplication: 3106 ms
// Matrix sum: -648442996

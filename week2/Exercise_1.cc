// Code Written by: Hayden Shaddix 
// Exercise 1 


#include <iostream>

// Function to swap two integers
void swap(int &a, int &b) {
    int temp = a;
    a = b;
    b = temp;
}

int main() {
    // Arrays A and B with 10 integers each
    int A[10] = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10};
    int B[10] = {10, 9, 8, 7, 6, 5, 4, 3, 2, 1};

    // Print arrays before swapping
    std::cout << "Before swapping:\n";
    std::cout << "A: ";
    for (int i = 0; i < 10; i++) {
        std::cout << A[i] << " ";
    }
    std::cout << "\nB: ";
    for (int i = 0; i < 10; i++) {
        std::cout << B[i] << " ";
    }
    std::cout << std::endl;

    // Swapping the values between arrays A and B
    for (int i = 0; i < 10; i++) {
        swap(A[i], B[i]);
    }

    // Print arrays after swapping
    std::cout << "\nAfter swapping:\n";
    std::cout << "A: ";
    for (int i = 0; i < 10; i++) {
        std::cout << A[i] << " ";
    }
    std::cout << "\nB: ";
    for (int i = 0; i < 10; i++) {
        std::cout << B[i] << " ";
    }
    std::cout << std::endl;

    return 0;
}


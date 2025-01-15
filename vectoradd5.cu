#include <cuda_runtime.h>
#include <iostream>
#include <chrono>
#include <thread>

// CUDA kernel
__global__ void vectorAdd(const float* a, const float* b, float* c, int N) {
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    if (i < N) {
        c[i] = a[i] + b[i];
    }
}

int main() {
    const int N = 1024;
    const int size = N * sizeof(float);

    // Host memory allocation
    float *h_a, *h_b, *h_c;
    h_a = (float*)malloc(size);
    h_b = (float*)malloc(size);
    h_c = (float*)malloc(size);

    // Initialize host arrays
    for (int i = 0; i < N; ++i) {
        h_a[i] = i * 1.0f;
        h_b[i] = i * 2.0f;
    }

    // Device memory allocation
    float *d_a, *d_b, *d_c;
    cudaMalloc((void**)&d_a, size);
    cudaMalloc((void**)&d_b, size);
    cudaMalloc((void**)&d_c, size);

    // Copy data from host to device
    cudaMemcpy(d_a, h_a, size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, h_b, size, cudaMemcpyHostToDevice);

    // Define thread hierarchy
    int threadsPerBlock = 256;
    int blocksPerGrid = (N + threadsPerBlock - 1) / threadsPerBlock;

    while (true) {
        // Launch kernel
        vectorAdd<<<blocksPerGrid, threadsPerBlock>>>(d_a, d_b, d_c, N);

        // Check for kernel errors
        cudaError_t kernel_err = cudaGetLastError();
        if (kernel_err != cudaSuccess) {
            std::cerr << "CUDA kernel launch error: " << cudaGetErrorString(kernel_err) << std::endl;
            break;
        }

        // Copy results back to host
        cudaMemcpy(h_c, d_c, size, cudaMemcpyDeviceToHost);

        // Verify and output results
        bool success = true;
        std::cout << "Results (First 10 Elements):" << std::endl;
        for (int i = 0; i < 10; ++i) { // Print first 10 results
            std::cout << "c[" << i << "] = " << h_c[i] << " (Expected: " << h_a[i] + h_b[i] << ")" << std::endl;
            if (h_c[i] != h_a[i] + h_b[i]) {
                success = false;
            }
        }

        if (!success) {
            std::cerr << "Results verification failed!" << std::endl;
            break;
        } else {
            std::cout << "Results verification succeeded!" << std::endl;
        }

        // Memory usage report
        size_t freeMem, totalMem;
        cudaMemGetInfo(&freeMem, &totalMem);
        std::cout << "Memory Report:" << std::endl;
        std::cout << "  Free memory: " << freeMem / (1024.0 * 1024.0) << " MB" << std::endl;
        std::cout << "  Total memory: " << totalMem / (1024.0 * 1024.0) << " MB" << std::endl;
        std::cout << "  Memory used: " << (totalMem - freeMem) / (1024.0 * 1024.0) << " MB" << std::endl;

        // Wait for 90 seconds before the next iteration
        std::cout << "Waiting for 90 seconds..." << std::endl;
        std::this_thread::sleep_for(std::chrono::seconds(90));
    }

    // Free memory
    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_c);
    free(h_a);
    free(h_b);
    free(h_c);

    std::cout << "Exiting program..." << std::endl;
    return 0;
}

#include "cudaIF.h"

#include <cuda_runtime.h>

#include <iostream>
#include <opencv2/core/cuda.hpp>
#include <opencv2/cudawarping.hpp>

#include "params.h"

unsigned char *devInput, *devOutput;

// カーネル関数の宣言
__global__ void concentricGaussianBlur(unsigned char* input, unsigned char* output, int width, int height, int centerX,
                                       int centerY, float maxRadius, int minKernelSize, int maxKernelSize);

int cudaIF::initialize() {
    if (cv::cuda::getCudaEnabledDeviceCount() <= 0) {
        std::cerr << "CUDA Device not found" << std::endl;
        return -1;
    }
    cudaError_t errCuda = cudaMalloc((void**)&devInput, maxImageSize);
    if (errCuda != cudaSuccess) {
        std::cerr << "CUDA Memory Allocation Error: " << cudaGetErrorString(errCuda) << std::endl;
        return -1;
    }

    errCuda = cudaMalloc((void**)&devOutput, maxImageSize);
    if (errCuda != cudaSuccess) {
        std::cerr << "CUDA Memory Allocation Error: " << cudaGetErrorString(errCuda) << std::endl;
        return -1;
    }
    initialized = true;
    return 0;
}

cudaIF::~cudaIF() {
    if (initialized) {
        // ピン留めメモリの解放（終了時）
        cudaFreeHost(devInput);
        cudaFreeHost(devOutput);
    }
}

// CUDAカーネルを呼び出す関数
void cudaIF::runBlur(cv::Mat& input, cv::Mat& output, int centerX, int centerY, float maxRadius, int minKernelSize,
                     int maxKernelSize) {
    int width = input.cols;
    int height = input.rows;
    int imageSize = width * height * 3;
    output = cv::Mat::zeros(input.rows, input.cols, input.type());

    // 通常通り、デバイスメモリへのコピーやカーネル呼び出しを行う
    cudaError_t err = cudaMemcpy(devInput, input.data, imageSize, cudaMemcpyHostToDevice);
    if (err != cudaSuccess) {
        std::cerr << "CUDA input Memory Copy Error: " << cudaGetErrorString(err) << std::endl;
        return;
    }
    // CUDAでスレッドを設定
    dim3 threadsPerBlock(16, 16);
    dim3 blocksPerGrid((width + threadsPerBlock.x - 1) / threadsPerBlock.x,
                       (height + threadsPerBlock.y - 1) / threadsPerBlock.y);
    // デバッグ: スレッドとブロックの設定を確認
    // std::cout << "Threads per block: (" << threadsPerBlock.x << ", " << threadsPerBlock.y << ")" << std::endl;
    // std::cout << "Blocks per grid: (" << blocksPerGrid.x << ", " << blocksPerGrid.y << ")" << std::endl;

    // カーネルの呼び出し
    concentricGaussianBlur<<<blocksPerGrid, threadsPerBlock>>>(devInput, devOutput, width, height, centerX, centerY,
                                                               maxRadius, minKernelSize, maxKernelSize);

    // エラー処理
    cudaError_t error = cudaGetLastError();
    if (error != cudaSuccess) {
        std::cerr << "CUDA Kernel Launch Error: " << cudaGetErrorString(error) << std::endl;
        return;
    }

    cudaDeviceSynchronize();
    err = cudaMemcpy(output.data, devOutput, imageSize, cudaMemcpyDeviceToHost);
    if (err != cudaSuccess) {
        std::cerr << "CUDA output Memory Copy Error: " << cudaGetErrorString(err) << std::endl;
        return;
    }
}

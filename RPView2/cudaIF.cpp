#include "cudaIF.h"

#include <cuda_runtime.h>

#include <iostream>
#include <opencv2/core/cuda.hpp>
#include <opencv2/cudawarping.hpp>

#include "params.h"

unsigned char *devInput, *devOutput;

// �J�[�l���֐��̐錾
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
        // �s�����߃������̉���i�I�����j
        cudaFreeHost(devInput);
        cudaFreeHost(devOutput);
    }
}

// CUDA�J�[�l�����Ăяo���֐�
void cudaIF::runBlur(cv::Mat& input, cv::Mat& output, int centerX, int centerY, float maxRadius, int minKernelSize,
                     int maxKernelSize) {
    int width = input.cols;
    int height = input.rows;
    int imageSize = width * height * 3;
    output = cv::Mat::zeros(input.rows, input.cols, input.type());

    // �ʏ�ʂ�A�f�o�C�X�������ւ̃R�s�[��J�[�l���Ăяo�����s��
    cudaError_t err = cudaMemcpy(devInput, input.data, imageSize, cudaMemcpyHostToDevice);
    if (err != cudaSuccess) {
        std::cerr << "CUDA input Memory Copy Error: " << cudaGetErrorString(err) << std::endl;
        return;
    }
    // CUDA�ŃX���b�h��ݒ�
    dim3 threadsPerBlock(16, 16);
    dim3 blocksPerGrid((width + threadsPerBlock.x - 1) / threadsPerBlock.x,
                       (height + threadsPerBlock.y - 1) / threadsPerBlock.y);
    // �f�o�b�O: �X���b�h�ƃu���b�N�̐ݒ���m�F
    // std::cout << "Threads per block: (" << threadsPerBlock.x << ", " << threadsPerBlock.y << ")" << std::endl;
    // std::cout << "Blocks per grid: (" << blocksPerGrid.x << ", " << blocksPerGrid.y << ")" << std::endl;

    // �J�[�l���̌Ăяo��
    concentricGaussianBlur<<<blocksPerGrid, threadsPerBlock>>>(devInput, devOutput, width, height, centerX, centerY,
                                                               maxRadius, minKernelSize, maxKernelSize);

    // �G���[����
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

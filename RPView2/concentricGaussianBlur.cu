#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#define PI 3.14159265358979323846f

__device__ float gaussian(float x, float sigma) {
    return expf(-(x * x) / (2 * sigma * sigma)) / (sqrtf(2 * PI) * sigma);
}

__global__ void concentricGaussianBlur(unsigned char* input, unsigned char* output, int width, int height, int centerX,
                                       int centerY, float maxRadius, int minKernelSize, int maxKernelSize) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x >= width || y >= height) {
        return;
    }

    int idx = (y * width + x) * 3;  // カラー画像なので、ピクセルごとに3チャンネル

    // メモリ範囲チェック
    if (idx + 2 >= width * height * 3) {
        return;  // メモリ範囲外へのアクセスを防止
    }

    // R, G, B の3つのチャンネルを処理
    for (int c = 0; c < 3; c++) {
#if 0
        output[idx + c] = input[idx + c];
#else
        float dx = x - centerX;
        float dy = y - centerY;
        float distance = sqrtf(dx * dx + dy * dy);

        // 距離に基づいてカーネルサイズを決定
        int kernelSize;
        if (distance > maxRadius) {
            kernelSize = 1;
        } else {
            // 距離に基づいてカーネルサイズを線形にスケーリング
            float normalizedDistance = distance / maxRadius;
            kernelSize = minKernelSize + (int)((maxKernelSize - minKernelSize) * normalizedDistance);
        }

        if (kernelSize % 2 == 0) {
            kernelSize++;
        }

        float sigma = kernelSize / 3.0f;
        float sum = 0.0f;
        float weightSum = 0.0f;
        int halfKernel = kernelSize / 2;

        for (int ky = -halfKernel; ky <= halfKernel; ky++) {
            for (int kx = -halfKernel; kx <= halfKernel; kx++) {
                int nx = min(max(x + kx, 0), width - 1);
                int ny = min(max(y + ky, 0), height - 1);
                int neighborIdx = (ny * width + nx) * 3 + c;  // 隣接ピクセルのチャンネルを参照

                float weight = gaussian(sqrtf(kx * kx + ky * ky), sigma);

                sum += input[neighborIdx] * weight;
                weightSum += weight;
            }
        }

        output[idx + c] = static_cast<unsigned char>(sum / weightSum);  // 各チャンネルを処理
#endif
    }
}

// CUDAカーネルの呼び出し関数
extern "C" void runConcentricGaussianBlur(unsigned char* input, unsigned char* output, int width, int height,
                                          int centerX, int centerY, float maxRadius, int minKernelSize,
                                          int maxKernelSize) {
    dim3 threadsPerBlock(16, 16);
    dim3 blocksPerGrid((width + threadsPerBlock.x - 1) / threadsPerBlock.x,
                       (height + threadsPerBlock.y - 1) / threadsPerBlock.y);

    concentricGaussianBlur<<<blocksPerGrid, threadsPerBlock>>>(input, output, width, height, centerX, centerY,
                                                               maxRadius, minKernelSize, maxKernelSize);

    cudaDeviceSynchronize();
}

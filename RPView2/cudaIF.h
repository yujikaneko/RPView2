#pragma once

#include <opencv2/opencv.hpp>

class cudaIF {
   public:
    int initialize();
    ~cudaIF();
    void runBlur(cv::Mat& input, cv::Mat& output, int centerX, int centerY, float maxRadius, int minKernelSize,
                 int maxKernelSize);

   private:
    bool initialized{};
};

#include "gazeIF.h"

#include "SRanipal.h"
#include "SRanipal_Enums.h"
#include "SRanipal_Eye.h"
#include "SRanipal_NotRelease.h"
#include "params.h"
#pragma comment(lib, "SRanipal.lib")
using namespace ViveSR;

// gazeIFクラスのstaticメンバー変数の定義と初期化
std::mutex gazeIF::mtx;
float gazeIF::gazeLeft[countMax][2] = {};   // 配列の初期化
float gazeIF::gazeRight[countMax][2] = {};  // 配列の初期化
uint32_t gazeIF::count = 0;

int gazeIF::initialize() {
    auto error = ViveSR::anipal::Initial(ViveSR::anipal::Eye::ANIPAL_TYPE_EYE, NULL);
    if (error == ViveSR::Error::WORK) {
        ViveSR::anipal::Eye::RegisterEyeDataCallback(callback);
    } else {
        std::cerr << "Failed to ViveSR::anipal::Initial " << error << std::endl;
        return -1;
    }
    return 0;
}

int gazeIF::getCurrent(float* left, float* right) {
    if (!left || !right) {
        return -1;
    }

    float gazeLeftLocal[2] = {0.0, 0.0}, gazeRightLocal[2] = {0.0, 0.0};
    mtx.lock();
    for (uint32_t i = 0; i < count; i++) {
        gazeLeftLocal[0] += gazeLeft[i][0];
        gazeLeftLocal[1] += gazeLeft[i][1];
        gazeRightLocal[0] += gazeRight[i][0];
        gazeRightLocal[1] += gazeRight[i][1];
    }
    if (count != 0) {
        gazeLeftLocal[0] /= static_cast<float>(count);
        gazeLeftLocal[1] /= static_cast<float>(count);
        gazeRightLocal[0] /= static_cast<float>(count);
        gazeRightLocal[1] /= static_cast<float>(count);
    }
    count = 0;
    mtx.unlock();
    return 0;
}

void gazeIF::callback(ViveSR::anipal::Eye::EyeData const& eye_data) {
    mtx.lock();
    if (count < countMax) {
        gazeLeft[count][0] = eye_data.verbose_data.left.gaze_direction_normalized.elem_[0];
        gazeLeft[count][1] = eye_data.verbose_data.left.gaze_direction_normalized.elem_[1];
        gazeRight[count][0] = eye_data.verbose_data.right.gaze_direction_normalized.elem_[0];
        gazeRight[count][1] = eye_data.verbose_data.right.gaze_direction_normalized.elem_[1];
        count++;
    }
    mtx.unlock();
}

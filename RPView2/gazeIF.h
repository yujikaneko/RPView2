#pragma once
#include <iostream>
#include <mutex>

#include "SRanipal.h"
#include "SRanipal_Enums.h"
#include "SRanipal_Eye.h"

const int countMax = 16;
class gazeIF {
   public:
    int initialize();
    int getCurrent(float* left, float* right);
    static void callback(ViveSR::anipal::Eye::EyeData const& eye_data);

   private:
    static std::mutex mtx;
    static float gazeLeft[countMax][2], gazeRight[countMax][2];
    static uint32_t count;
};
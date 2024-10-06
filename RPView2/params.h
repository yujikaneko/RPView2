#pragma once

constexpr size_t maxWidth = 1920;
constexpr size_t maxHeight = 1920;
constexpr size_t maxImageSize = maxWidth * maxHeight * 3 * sizeof(unsigned char);

constexpr double zoom = 0.75;
constexpr float fov = 98.0;  // VIVE Pro Eye‚ÌŽ‹–ìŠp
constexpr float fov_half = fov / 2.0;
constexpr int maxRadiusi = 300;
constexpr float maxRadius = static_cast<float>(maxRadiusi);
constexpr int minKernelSize = 3;
constexpr int maxKernelSize = 31;


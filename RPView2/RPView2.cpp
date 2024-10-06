#include <Windows.h>
#include <openvr.h>

#include <iostream>
#include <mutex>
#include <opencv2/opencv.hpp>
#define GLFW_INCLUDE_NONE
#include <GL/glew.h>
#include <GLFW/glfw3.h>
#define GLFW_EXPOSE_NATIVE_WIN32
#include <GLFW/glfw3native.h>

#include "cudaIF.h"
#include "gazeIF.h"
#include "params.h"

// OpenGLテクスチャを作成
void createTexture(GLuint& textureID, const cv::Mat& image) {
    if (image.empty()) {
        std::cerr << "Image is empty, cannot create texture." << std::endl;
        return;
    }

    cv::Mat imageRGBA, imageFlipped;
    cv::cvtColor(image, imageRGBA, cv::COLOR_BGR2RGBA);  // OpenCVのBGRをRGBAに変換
    cv::flip(imageRGBA, imageFlipped, 0);                // 画像を上下反転

    glGenTextures(1, &textureID);
    glBindTexture(GL_TEXTURE_2D, textureID);

    // テクスチャデータの設定
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, imageFlipped.cols, imageFlipped.rows, 0, GL_RGBA, GL_UNSIGNED_BYTE,
                 imageFlipped.data);

    // テクスチャパラメータの設定
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
}

// OpenGLテクスチャを更新
void updateTexture(GLuint textureID, const cv::Mat& image) {
    if (image.empty()) {
        std::cerr << "Image is empty, cannot update texture." << std::endl;
        return;
    }

    cv::Mat imageRGBA, imageFlipped;
    cv::cvtColor(image, imageRGBA, cv::COLOR_BGR2RGBA);  // OpenCVのBGRをRGBAに変換
    cv::flip(imageRGBA, imageFlipped, 0);                // 画像を上下反転

    glBindTexture(GL_TEXTURE_2D, textureID);
    glTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0, imageFlipped.cols, imageFlipped.rows, GL_RGBA, GL_UNSIGNED_BYTE,
                    imageFlipped.data);
}

// 画像を特定のパーセントで縮小する関数
cv::Mat resizeImage(const cv::Mat& inputImage, double percent) {
    cv::Mat outputImage;
    // 新しいサイズを計算
    int newWidth = static_cast<int>(inputImage.cols * percent);
    int newHeight = static_cast<int>(inputImage.rows * percent);
    cv::Size newSize(newWidth, newHeight);

    // 画像のリサイズ
    cv::resize(inputImage, outputImage, newSize);

    return outputImage;
}

// 画像処理で視野狭窄をシミュレートする
void runImageProcessing(const cv::Mat& src, cv::Mat& dst, cudaIF& cuda, float* gaze) {
    cv::Mat tempImage = cv::Mat::zeros(src.cols, src.rows, src.type());
    cv::Mat resizedImage = resizeImage(src, zoom);
    int centerX = src.cols / 2 + 100;
    int centerY = src.rows / 2 - 0;
    resizedImage(cv::Rect(0, 0, resizedImage.cols, resizedImage.rows))
        .copyTo(tempImage(cv::Rect(centerX - resizedImage.cols / 2, centerY - resizedImage.rows / 2, resizedImage.cols,
                                   resizedImage.rows)));
    int gazeX = centerX - static_cast<int>(gaze[0] * (centerX / tan(fov_half * CV_PI / 180.0)));
    int gazeY = centerY - static_cast<int>(gaze[1] * (centerY / tan(fov_half * CV_PI / 180.0)));
    gazeX = std::min(1900, std::max(20, gazeX));
    gazeY = std::min(1900, std::max(20, gazeY));
    // 視線方向の中心
    cv::Point gazeCenter(gazeX, gazeY);

    // gazeX, gazeY を中心に 2maxRadiusi x 2maxRadiusi の矩形を切り出す
    int rectX = std::max(0, gazeX - maxRadiusi);
    int rectY = std::max(0, gazeY - maxRadiusi);
    cv::Rect roi(rectX, rectY, 2 * maxRadiusi, 2 * maxRadiusi);
    cv::Mat cropped = tempImage(roi).clone();

    // 円形マスクを作成し、領域を円形に切り出す
    cv::Mat mask = cv::Mat::zeros(cropped.size(), CV_8UC1);
    cv::circle(mask, cv::Point(maxRadiusi, maxRadiusi), maxRadiusi, cv::Scalar(255), -1);

    // 円形領域にガウシアンブラーを適用
    cv::Mat blurredCropped = cropped.clone();
    cuda.runBlur(cropped, blurredCropped, maxRadiusi, maxRadiusi, maxRadius, minKernelSize, maxKernelSize);

    cv::Mat resizedInput1 = resizeImage(tempImage, 0.05);
    dst = resizeImage(resizedInput1, 20.0);

    // マスクを適用して円形部分だけを抽出
    blurredCropped.copyTo(dst(roi), mask);

    cv::line(dst, cv::Point(gazeX, gazeY - 20), cv::Point(gazeX, gazeY + 20), cv::Scalar(0, 0, 255),
             5);  // 垂直線
    cv::line(dst, cv::Point(gazeX - 20, gazeY), cv::Point(gazeX + 20, gazeY), cv::Scalar(0, 0, 255),
             5);  // 水平線
}

// ステレオ画像を左目と右目に分割
void splitStereoImage(const cv::Mat& stereoImage, cv::Mat& leftImage, cv::Mat& rightImage, cudaIF& cuda, gazeIF& gaze) {
    int width = stereoImage.cols / 2;
    int height = stereoImage.rows;
    float gazeLeftLocal[2] = {0.0, 0.0}, gazeRightLocal[2] = {0.0, 0.0};
    gaze.getCurrent(gazeLeftLocal, gazeRightLocal);

    // 左目領域
    cv::Mat rawLeftImage = stereoImage(cv::Rect(0, 0, width, height));
    runImageProcessing(rawLeftImage, leftImage, cuda, gazeLeftLocal);

    // 右目領域
    cv::Mat rawRightImage = stereoImage(cv::Rect(width, 0, width, height));
    runImageProcessing(rawRightImage, rightImage, cuda, gazeRightLocal);
}

int main() {
    cudaIF cudaif;
    if (cudaif.initialize() < 0) {
        return -1;
    }

    gazeIF gazeif;
    if (gazeif.initialize() < 0) {
        return -1;
    }

    // OpenVRの初期化
    vr::EVRInitError eError;
    vr::IVRSystem* vr_system = vr::VR_Init(&eError, vr::EVRApplicationType::VRApplication_Scene);
    if (!vr_system) {
        std::cerr << "Failed to initialize OpenVR" << std::endl;
        return -1;
    }

    // GLFWの初期化
    if (!glfwInit()) {
        std::cerr << "Failed to initialize GLFW" << std::endl;
        vr::VR_Shutdown();
        return -1;
    }

    // OpenGLコンテキストの作成
    glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 3);
    glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 3);
    glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);
    GLFWwindow* window = glfwCreateWindow(300, 300, "OpenVR OpenGL", nullptr, nullptr);
    if (!window) {
        std::cerr << "Failed to create GLFW window" << std::endl;
        glfwTerminate();
        vr::VR_Shutdown();
        return -1;
    }
    glfwMakeContextCurrent(window);

    // GLEWの初期化とエラーチェック
    GLenum err = glewInit();
    if (err != GLEW_OK) {
        std::cerr << "Failed to initialize GLEW: " << glewGetErrorString(err) << std::endl;
        glfwDestroyWindow(window);
        glfwTerminate();
        vr::VR_Shutdown();
        return -1;
    }

    // VRヘッドセットのプロパティを取得
    vr::TrackedDevicePose_t poses[vr::k_unMaxTrackedDeviceCount];
    vr::VRCompositor()->WaitGetPoses(poses, vr::k_unMaxTrackedDeviceCount, nullptr, 0);

    // ステレオカメラのキャプチャ初期化
    cv::VideoCapture cap(0, cv::CAP_DSHOW);  // カメラIDは環境に応じて調整
    if (!cap.isOpened()) {
        std::cerr << "Failed to open stereo camera" << std::endl;
        glfwDestroyWindow(window);
        glfwTerminate();
        vr::VR_Shutdown();
        return -1;
    }

    // カメラの解像度を設定
    cap.set(cv::CAP_PROP_FRAME_WIDTH, 3840);
    cap.set(cv::CAP_PROP_FRAME_HEIGHT, 1080);
    cap.set(cv::CAP_PROP_FPS, 30);
    cap.set(cv::CAP_PROP_FOURCC, cv::VideoWriter::fourcc('M', 'J', 'P', 'G'));

    // キャプチャしたフレームを保持するための変数
    cv::Mat frame;
    cv::Mat leftEyeImage, rightEyeImage;

    // OpenGLテクスチャを保持する変数
    GLuint leftEyeTexture = 0, rightEyeTexture = 0;
    bool texturesCreated = false;

    // cv::VideoWriter video("vr_output.avi", cv::VideoWriter::fourcc('M', 'J', 'P', 'G'), 14, cv::Size(1920, 1920));

    // デバッグ用にウィンドウを作成
    cv::namedWindow("left Camera", cv::WINDOW_NORMAL);

    // メインループ
    while (!glfwWindowShouldClose(window)) {
        // カメラからフレームをキャプチャ
        cap >> frame;
        auto t1 = std::chrono::high_resolution_clock::now();

        if (frame.empty()) {
            std::cerr << "Failed to capture frame from stereo camera" << std::endl;
            continue;
        }

        // ステレオ画像を左右に分割
        splitStereoImage(frame, leftEyeImage, rightEyeImage, cudaif, gazeif);

        // デバッグ用にウィンドウに表示
        cv::imshow("left Camera", leftEyeImage);

        // video.write(leftEyeImage);
        auto key = cv::waitKey(1);
        if (key == 27) {  // ESCキーが押された場合
            break;
        }

        if (!texturesCreated) {
            // テクスチャがまだ作成されていない場合は作成する
            createTexture(leftEyeTexture, leftEyeImage);
            createTexture(rightEyeTexture, rightEyeImage);
            texturesCreated = true;
        } else {
            // 既存のテクスチャを更新する
            updateTexture(leftEyeTexture, leftEyeImage);
            updateTexture(rightEyeTexture, rightEyeImage);
        }

        // OpenVR用のテクスチャ作成
        vr::Texture_t leftEyeVrTexture = {(void*)(uintptr_t)leftEyeTexture, vr::TextureType_OpenGL,
                                          vr::ColorSpace_Auto};
        vr::Texture_t rightEyeVrTexture = {(void*)(uintptr_t)rightEyeTexture, vr::TextureType_OpenGL,
                                           vr::ColorSpace_Auto};

        // トラッキングデバイスのポーズを取得
        vr::VRCompositor()->WaitGetPoses(poses, vr::k_unMaxTrackedDeviceCount, nullptr, 0);

        // 左目の画像を送信
        auto ret = vr::VRCompositor()->Submit(vr::Eye_Left, &leftEyeVrTexture);
        if (ret != vr::VRCompositorError_None) {
            std::cerr << "Submit Left Error: " << ret << std::endl;
        }

        // 右目の画像を送信
        ret = vr::VRCompositor()->Submit(vr::Eye_Right, &rightEyeVrTexture);
        if (ret != vr::VRCompositorError_None) {
            std::cerr << "Submit Right Error: " << ret << std::endl;
        }

        auto t5 = std::chrono::high_resolution_clock::now();

        std::chrono::duration<double> d1 = t5 - t1;
        std::cout << "d1:" + std::to_string(d1.count()) << "sec" << std::endl;

        // ウィンドウイベント処理
        glfwPollEvents();
    }

    // クリーンアップ
    // video.release();
    cap.release();
    cv::destroyAllWindows();
    glfwDestroyWindow(window);
    glfwTerminate();
    vr::VR_Shutdown();
    return 0;
}

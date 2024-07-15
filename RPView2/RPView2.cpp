#include <mutex>
#include <iostream>
#include <openvr.h>
#include <opencv2/opencv.hpp>
#include <Windows.h>
#define GLFW_INCLUDE_NONE
#include <GL/glew.h>
#include <GLFW/glfw3.h>
#define GLFW_EXPOSE_NATIVE_WIN32
#include <GLFW/glfw3native.h>
#include "SRanipal.h"
#include "SRanipal_Eye.h"
#include "SRanipal_Enums.h"
#include "SRanipal_NotRelease.h"
#pragma comment (lib, "SRanipal.lib")
using namespace ViveSR;

// OpenGLテクスチャを作成
void createTexture(GLuint& textureID, const cv::Mat& image) {
	if (image.empty()) {
		std::cerr << "Image is empty, cannot create texture." << std::endl;
		return;
	}

	cv::Mat imageRGBA, imageFlipped;
	cv::cvtColor(image, imageRGBA, cv::COLOR_BGR2RGBA); // OpenCVのBGRをRGBAに変換
	cv::flip(imageRGBA, imageFlipped, 0); // 画像を上下反転

	glGenTextures(1, &textureID);
	glBindTexture(GL_TEXTURE_2D, textureID);

	// テクスチャデータの設定
	glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, imageFlipped.cols, imageFlipped.rows, 0, GL_RGBA, GL_UNSIGNED_BYTE, imageFlipped.data);

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
	cv::cvtColor(image, imageRGBA, cv::COLOR_BGR2RGBA); // OpenCVのBGRをRGBAに変換
	cv::flip(imageRGBA, imageFlipped, 0); // 画像を上下反転

	glBindTexture(GL_TEXTURE_2D, textureID);
	glTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0, imageFlipped.cols, imageFlipped.rows, GL_RGBA, GL_UNSIGNED_BYTE, imageFlipped.data);
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

void applyConcentricBlur(cv::Mat& image, cv::Point center, const std::vector<int>& radii, const std::vector<int>& blurSizes) {
	// 画像のコピーを作成
	cv::Mat originalImage = image.clone();

	// 初期設定で最も外側のブラー画像を作成
	cv::Mat previousBlurredImage = image.clone();
	int blurSize = blurSizes.back();

	// 縮小・拡大によるブラー適用
	int scaleFactor = blurSize;
	cv::Mat smallImage, blurredImage;
	cv::resize(originalImage, smallImage, cv::Size(), 1.0 / scaleFactor, 1.0 / scaleFactor, cv::INTER_LINEAR);
	cv::resize(smallImage, previousBlurredImage, originalImage.size(), 0, 0, cv::INTER_LINEAR);

	for (int i = radii.size() - 1; i >= 0; --i) {
		// 現在のブラーサイズ
		blurSize = blurSizes[i];

		if (blurSize != 0) {
			// 縮小・拡大による効率的なブラー適用
			scaleFactor = blurSize;
			cv::resize(originalImage, smallImage, cv::Size(), 1.0 / scaleFactor, 1.0 / scaleFactor, cv::INTER_LINEAR);
			cv::resize(smallImage, blurredImage, originalImage.size(), 0, 0, cv::INTER_LINEAR);
		}

		// マスクを作成
		cv::Mat mask = cv::Mat::zeros(image.size(), CV_8UC1);
		cv::circle(mask, center, radii[i], cv::Scalar(255), -1);

		if (i > 0) {
			cv::circle(mask, center, radii[i - 1], cv::Scalar(0), -1);
		}

		// ブラー画像を合成
		cv::Mat temp;
		if (blurSize == 0) {
			originalImage.copyTo(temp, mask);
		}
		else {
			blurredImage.copyTo(temp, mask);
		}
		temp.copyTo(previousBlurredImage, mask);
	}

	// 最終的な合成画像を出力
	image = previousBlurredImage.clone();
}

constexpr int gazeCountMax = 16;
std::mutex m;
float gazeLeft[gazeCountMax][2], gazeRight[gazeCountMax][2];
uint32_t gazeCount = 0;

// ステレオ画像を左目と右目に分割
void splitStereoImage(const cv::Mat& stereoImage, cv::Mat& leftImage, cv::Mat& rightImage) {
	float gazeLeftLocal[2] = { 0.0, 0.0 }, gazeRightLocal[2] = { 0.0, 0.0 };
	uint32_t gazeCountLocal = 0;
	m.lock();
	for (int i = 0; i < gazeCount; i++) {
		gazeLeftLocal[0] += gazeLeft[i][0];
		gazeLeftLocal[1] += gazeLeft[i][1];
		gazeRightLocal[0] += gazeRight[i][0];
		gazeRightLocal[1] += gazeRight[i][1];
	}
	gazeCountLocal = gazeCount;
	if (gazeCount != 0) {
		gazeLeftLocal[0] /= static_cast<float>(gazeCount);
		gazeLeftLocal[1] /= static_cast<float>(gazeCount);
		gazeRightLocal[0] /= static_cast<float>(gazeCount);
		gazeRightLocal[1] /= static_cast<float>(gazeCount);
	}
	gazeCount = 0;
	m.unlock();
	//std::cout << "gazeCountLocal " << gazeCountLocal << std::endl;
	int width = stereoImage.cols / 2;
	int height = stereoImage.rows;
	double zoom = 0.75;
	float fov = 98.0; // VIVE Pro Eyeの視野角
	float fov_half = fov / 2.0;

	// 同心円の半径と対応するブラーサイズ
	std::vector<int> radii = { 200, 220, 240, 270, 300, 330, 360, 400 };
	std::vector<int> blurSizes = { 0, 3, 5, 7, 9, 13, 17, 31 }; // ブラーサイズを大きく設定
	// 左目領域
	{
		leftImage = cv::Mat::zeros(1920, 1920, stereoImage.type());
		cv::Mat rawImage = stereoImage(cv::Rect(0, 0, width, height));
		cv::Mat resizedImage = resizeImage(rawImage, zoom);
		int centerX = 1920 / 2 + 100;
		int centerY = 1920 / 2 - 0;
		resizedImage(cv::Rect(0, 0, resizedImage.cols, resizedImage.rows))
			.copyTo(leftImage(cv::Rect(centerX - resizedImage.cols / 2,
				centerY - resizedImage.rows / 2,
				resizedImage.cols, resizedImage.rows)));
		int gazeX = centerX - static_cast<int>(gazeLeftLocal[0] * (centerX / tan(fov_half * CV_PI / 180.0)));
		int gazeY = centerY - static_cast<int>(gazeLeftLocal[1] * (centerY / tan(fov_half * CV_PI / 180.0)));
		//int gazeX = centerX - static_cast<int>(gazeLeftLocal[0] * 1920.0 / 2.0);
		gazeX = min(1900, max(20, gazeX));
		//int gazeY = centerY - static_cast<int>(gazeLeftLocal[1] * 1920.0 / 2.0);
		gazeY = min(1900, max(20, gazeY));
		// 視線方向の中心
		cv::Point gazeCenter(gazeX, gazeY);
		applyConcentricBlur(leftImage, gazeCenter, radii, blurSizes);
		cv::line(leftImage, cv::Point(gazeX, gazeY - 20), cv::Point(gazeX, gazeY + 20), cv::Scalar(0, 0, 255), 5); // 垂直線
		cv::line(leftImage, cv::Point(gazeX - 20, gazeY), cv::Point(gazeX + 20, gazeY), cv::Scalar(0, 0, 255), 5); // 水平線
	}
	// 右目領域
	{
		rightImage = cv::Mat::zeros(1920, 1920, stereoImage.type());
		cv::Mat rawImage = stereoImage(cv::Rect(width, 0, width, height));
		cv::Mat resizedImage = resizeImage(rawImage, zoom);
		int centerX = 1920 / 2 - 100;
		int centerY = 1920 / 2 + 0;
		resizedImage(cv::Rect(0, 0, resizedImage.cols, resizedImage.rows))
			.copyTo(rightImage(cv::Rect(centerX - resizedImage.cols / 2,
				centerY - resizedImage.rows / 2,
				resizedImage.cols, resizedImage.rows)));
		int gazeX = centerX - static_cast<int>(gazeRightLocal[0] * (centerX / tan(fov_half * CV_PI / 180.0)));
		int gazeY = centerY - static_cast<int>(gazeRightLocal[1] * (centerY / tan(fov_half * CV_PI / 180.0)));
		//int gazeX = centerX - static_cast<int>(gazeRightLocal[0] * 1920.0 / 2.0);
		gazeX = min(1900, max(20, gazeX));
		//int gazeY = centerY - static_cast<int>(gazeRightLocal[1] * 1920.0 / 2.0);
		gazeY = min(1900, max(20, gazeY));
		// 視線方向の中心
		cv::Point gazeCenter(gazeX, gazeY);
		applyConcentricBlur(rightImage, gazeCenter, radii, blurSizes);
		cv::line(rightImage, cv::Point(gazeX, gazeY - 20), cv::Point(gazeX, gazeY + 20), cv::Scalar(0, 0, 255), 5); // 垂直線
		cv::line(rightImage, cv::Point(gazeX - 20, gazeY), cv::Point(gazeX + 20, gazeY), cv::Scalar(0, 0, 255), 5); // 水平線
	}
}

void checkFocus(HWND hwnd) {
	HWND foregroundWindow = GetForegroundWindow();
	if (foregroundWindow == hwnd) {
		//std::cerr << "The GLFW window has focus." << std::endl;
	}
	else {
		std::cerr << "The GLFW window does NOT have focus." << std::endl;
		SetForegroundWindow(hwnd);
		SetFocus(hwnd);
	}
}

void TestEyeCallback(ViveSR::anipal::Eye::EyeData const& eye_data) {
	m.lock();
	if (gazeCount < gazeCountMax) {
		gazeLeft[gazeCount][0] = eye_data.verbose_data.left.gaze_direction_normalized.elem_[0];
		gazeLeft[gazeCount][1] = eye_data.verbose_data.left.gaze_direction_normalized.elem_[1];
		gazeRight[gazeCount][0] = eye_data.verbose_data.right.gaze_direction_normalized.elem_[0];
		gazeRight[gazeCount][1] = eye_data.verbose_data.right.gaze_direction_normalized.elem_[1];
		gazeCount++;
	}
	m.unlock();
	//float openness = eye_data.verbose_data.right.eye_openness;
	//if (gaze[0] != 0.0) {
	//	printf("[Eye callback] Gaze: %.2f %.2f %.2f %.2f\n", gaze[0], gaze[1], gaze[2], openness);
	//}
}

int main() {
	auto error = ViveSR::anipal::Initial(ViveSR::anipal::Eye::ANIPAL_TYPE_EYE, NULL);
	if (error == ViveSR::Error::WORK) {
		ViveSR::anipal::Eye::RegisterEyeDataCallback(TestEyeCallback);
	}
	else {
		std::cerr << "Failed to ViveSR::anipal::Initial " << error << std::endl;
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

	// ウィンドウハンドルの取得
	//HWND hwnd = glfwGetWin32Window(window);

	// VRヘッドセットのプロパティを取得
	vr::TrackedDevicePose_t poses[vr::k_unMaxTrackedDeviceCount];
	vr::VRCompositor()->WaitGetPoses(poses, vr::k_unMaxTrackedDeviceCount, nullptr, 0);

	// ステレオカメラのキャプチャ初期化
	cv::VideoCapture cap(0, cv::CAP_DSHOW); // カメラIDは環境に応じて調整
	if (!cap.isOpened()) {
		std::cerr << "Failed to open stereo camera" << std::endl;
		glfwDestroyWindow(window);
		glfwTerminate();
		vr::VR_Shutdown();
		return -1;
	}

	// カメラの解像度を設定
	cap.set(cv::CAP_PROP_FRAME_WIDTH, 3840); // 1920x2 = 3840
	cap.set(cv::CAP_PROP_FRAME_HEIGHT, 1080);
	cap.set(cv::CAP_PROP_FPS, 30); // 例えば30fpsに設定
	cap.set(cv::CAP_PROP_FOURCC, cv::VideoWriter::fourcc('M', 'J', 'P', 'G'));
	// 設定が反映されているか確認
	//double width = cap.get(cv::CAP_PROP_FRAME_WIDTH);
	//double height = cap.get(cv::CAP_PROP_FRAME_HEIGHT);
	//double fps = cap.get(cv::CAP_PROP_FPS);
	//double siz = cap.get(cv::CAP_PROP_BUFFERSIZE);
	//std::cout << "Resolution: " << width << "x" << height << ", FPS: " << fps << ", SIZE: " << siz << std::endl;


	// キャプチャしたフレームを保持するための変数
	cv::Mat frame;
	cv::Mat leftEyeImage, rightEyeImage;

	// OpenGLテクスチャを保持する変数
	GLuint leftEyeTexture = 0, rightEyeTexture = 0;
	bool texturesCreated = false;

	// デバッグ用にウィンドウを作成
	cv::namedWindow("left Camera", cv::WINDOW_NORMAL);
	cv::namedWindow("right Camera", cv::WINDOW_NORMAL);

	// メインループ
	while (!glfwWindowShouldClose(window)) {
		auto t0 = std::chrono::high_resolution_clock::now();
		// カメラからフレームをキャプチャ
		cap >> frame;
		auto t1 = std::chrono::high_resolution_clock::now();

		if (frame.empty()) {
			std::cerr << "Failed to capture frame from stereo camera" << std::endl;
			continue;
		}

		//ViveSR::anipal::Eye::EyeData eye_data;
		//ViveSR::anipal::Eye::GetEyeData(&eye_data);
		//float* gazeLeft = eye_data.verbose_data.left.gaze_direction_normalized.elem_;
		//float* gazeRight = eye_data.verbose_data.right.gaze_direction_normalized.elem_;

		// ステレオ画像を左右に分割
		splitStereoImage(frame, leftEyeImage, rightEyeImage);
		auto t2 = std::chrono::high_resolution_clock::now();

		// デバッグ用にウィンドウに表示
		cv::imshow("left Camera", leftEyeImage);
		cv::imshow("right Camera", rightEyeImage);

		// OpenCVウィンドウイベントの処理
		// 特定のキーが押されたら左目用の画像を保存
		int key = cv::waitKey(1);
		if (key == '1') {
			cv::imwrite("left_eye_image_1.jpg", leftEyeImage);
		}
		if (key == '2') {
			cv::imwrite("left_eye_image_2.jpg", leftEyeImage);
		}
		if (key == '3') {
			cv::imwrite("left_eye_image_3.jpg", leftEyeImage);
		}
		if (key == 27) { // ESCキーが押された場合
			break;
		}

		if (!texturesCreated) {
			// テクスチャがまだ作成されていない場合は作成する
			createTexture(leftEyeTexture, leftEyeImage);
			createTexture(rightEyeTexture, rightEyeImage);
			texturesCreated = true;
		}
		else {
			// 既存のテクスチャを更新する
			updateTexture(leftEyeTexture, leftEyeImage);
			updateTexture(rightEyeTexture, rightEyeImage);
		}

		// OpenVR用のテクスチャ作成
		vr::Texture_t leftEyeVrTexture = { (void*)(uintptr_t)leftEyeTexture, vr::TextureType_OpenGL, vr::ColorSpace_Auto };
		vr::Texture_t rightEyeVrTexture = { (void*)(uintptr_t)rightEyeTexture, vr::TextureType_OpenGL, vr::ColorSpace_Auto };
		auto t3 = std::chrono::high_resolution_clock::now();

		// フォーカスの確認と取得
		//checkFocus(hwnd);

		// トラッキングデバイスのポーズを取得
		vr::VRCompositor()->WaitGetPoses(poses, vr::k_unMaxTrackedDeviceCount, nullptr, 0);
		auto t4 = std::chrono::high_resolution_clock::now();

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
		std::chrono::duration<double> d1 = t2 - t1;
		std::cout << "d1:" + std::to_string(d1.count()) << "sec"<< std::endl;

		// ウィンドウイベント処理
		glfwPollEvents();
	}

	// クリーンアップ
	cap.release();
	cv::destroyAllWindows();
	glfwDestroyWindow(window);
	glfwTerminate();
	vr::VR_Shutdown();
	ViveSR::anipal::Release(ViveSR::anipal::Eye::ANIPAL_TYPE_EYE);

	return 0;
}

# RPView2 - Visual Field Constriction VR Simulator

RPView2 is a VR simulator designed to provide an experience of visual field constriction, such as that caused by retinitis pigmentosa. The project aims to help users understand the impact of visual field loss by simulating the experience in a VR environment.

## Features
- **Visual Field Constriction Simulation**: Simulates the condition where only the central part of the visual field is visible, with the peripheral vision gradually blurring.
- **Real-Time Gaze Tracking**: Uses real-time gaze tracking data to update the visual experience dynamically.

## Requirements

### Hardware
- **VR Headset**: [HTC Vive Pro Eye](https://www.vive.com/jp/product/vive-pro-eye/overview/)
- **PC with NVidia GPU**: CUDA-supported NVIDIA graphics card for GPU acceleration.

### Software
- **OS**: Windows 11 Home (23H2, 22631.4112)
- **Development Environment**: Visual Studio 2022 Version 17.11.2
- **Programming Language**: C++, CUDA

### Libraries and SDKs
The following libraries and SDKs are required:
- **OpenVR**: 2.5.1, Supports VR headsets in a SteamVR environment
- **OpenCV**: 4.10.0, For image processing and computer vision
- **GLFW**: 2.2.0#3, For window management with OpenGL
- **GLEW**: 3.4, For managing OpenGL extensions
- **Vive SRanipal SDK**: 1.3.3.0, For utilizing the eye-tracking features of the HTC Vive Pro Eye
- **CUDA Toolkit**: Version 12.6, For GPU-accelerated image processing

## Installation

1. Clone this repository:
   ```bash
   git clone https://github.com/yujikaneko/RPView2.git
   ```

2. Install the required libraries. Refer to the following links for guidance:
   - [OpenVR](https://github.com/ValveSoftware/openvr)
     - environment variable `OPENVR` to the OpenVR folder
   - [OpenCV](https://opencv.org/)
     - environment variable `OPENCV` to the OpenCV folder
   - [GLFW](https://www.glfw.org/)
   - [GLEW](http://glew.sourceforge.net/)
   - [Vive SRanipal SDK](https://developer.vive.com/resources/knowledgebase/vive-sranipal-sdk/)
     - environment variable `SRanipal` to the SRanipal folder
   - [CUDA Toolkit]()
     - Install the CUDA Toolkit, and set the environment variable `CUDA_PATH`.

3. Open the `RPView2.sln` project file in Visual Studio.

4. Configure the project settings by setting the include directories and library directories, and link the necessary libraries.

5. Build and run the project.

## Usage

1. Start the simulator and put on the HTC Vive Pro Eye.
2. The stereo camera captures the left and right fields of view, and visual field constriction is applied.
3. As you move your gaze, the visual field dynamically adjusts, with objects in the peripheral vision becoming blurred based on your gaze direction.

## Notes

- This simulator is intended for educational and research purposes only and should not be used for medical diagnosis.
- Ensure the HTC Vive Pro Eye is properly fitted for accurate eye-tracking.

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

## Contributing

Bug reports and improvement suggestions are welcome via GitHub Issues or Pull Requests.

## Author

- **Yuji Kaneko** - Developer

## Contact

For questions or support, please contact [yuji.kaneko.t1@dc.tohoku.ac.jp](yuji.kaneko.t1@dc.tohoku.ac.jp).

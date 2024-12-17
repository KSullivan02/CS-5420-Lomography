#include <opencv2/opencv.hpp>
#include <iostream>
#include <cmath>

// Globals
cv::Mat inputImage, colorFilteredImage, finalImage;
double colorParam = 0.1; // Initial color parameter (x0.01)
int vignetteParam = 100; // Initial vignette radius percentage

// Function to center window using dynamic screen size detection
void centerWindow(const std::string& windowName, int windowWidth, int windowHeight) {
    // Create a temporary full-screen window to detect screen size
    cv::namedWindow("Temp", cv::WINDOW_NORMAL);
    cv::setWindowProperty("Temp", cv::WND_PROP_FULLSCREEN, cv::WINDOW_FULLSCREEN);
    
    // Retrieve the screen resolution
    cv::Rect screenSize = cv::getWindowImageRect("Temp");

    // Close the temporary window
    cv::destroyWindow("Temp");

    // Calculate position to center the window
    int posX = (screenSize.width - windowWidth) / 2;
    int posY = (screenSize.height - windowHeight) / 2;

    // Move the specified window
    cv::moveWindow(windowName, posX, posY);
}

// Apply the color filter
void applyColorFilter(int, void*) {
    // Create a LUT (Look-Up Table) for the red channel
    cv::Mat lut(1, 256, CV_8UC1);
    for (int i = 0; i < 256; ++i) {
        lut.at<uchar>(0, i) = cv::saturate_cast<uchar>(256 / (1 + std::exp(-((i / 256.0) - 0.5) / colorParam)));
    }

    // Decompose channels
    std::vector<cv::Mat> channels;
    cv::split(inputImage, channels);

    // Apply LUT to the red channel
    cv::LUT(channels[2], lut, channels[2]);

    // Merge channels back
    cv::merge(channels, colorFilteredImage);
    cv::imshow("Lomography", colorFilteredImage);
}

// Apply the vignette effect
void applyVignetteFilter(int, void*) {
    if (colorFilteredImage.empty()) {
        colorFilteredImage = inputImage.clone();
    }

    // Create halo matrix
    cv::Mat halo(inputImage.size(), CV_32FC3, cv::Scalar(0.75, 0.75, 0.75));
    int maxRadius = std::min(inputImage.rows, inputImage.cols) / 2;
    int radius = std::max(1, (vignetteParam * maxRadius) / 100); // Ensure radius >= 1

    // Draw a filled circle
    cv::circle(halo, cv::Point(inputImage.cols / 2, inputImage.rows / 2), radius, cv::Scalar(1, 1, 1), -1);

    // Ensure valid kernel size for blur
    int blurKernelSize = std::max(1, radius) | 1; // Make it odd
    cv::blur(halo, halo, cv::Size(blurKernelSize, blurKernelSize));

    // Blend the halo with the color-filtered image
    cv::Mat colorFilteredFloat, resultFloat;
    colorFilteredImage.convertTo(colorFilteredFloat, CV_32FC3, 1.0 / 255.0);
    cv::multiply(colorFilteredFloat, halo, resultFloat);

    // Convert back to 8-bit
    resultFloat.convertTo(finalImage, CV_8UC3, 255.0);
    cv::imshow("Lomography", finalImage);
}

// Trackbar update functions
void onColorTrackbar(int value, void*) {
    colorParam = std::max(0.08, static_cast<double>(value) / 100.0); // Enforce minimum value
    applyColorFilter(0, nullptr);
    applyVignetteFilter(0, nullptr);
}

void onVignetteTrackbar(int value, void*) {
    vignetteParam = value;
    applyVignetteFilter(0, nullptr);
}

int main(int argc, char** argv) {
    try {
        // Parse command-line arguments
        if (argc != 2) {
            std::cerr << "Usage: lomo <image_path>\n";
            return -1;
        }
        std::string imagePath = argv[1];

        // Load the input image
        inputImage = cv::imread(imagePath, cv::IMREAD_COLOR);
        if (inputImage.empty()) {
            std::cerr << "Error: Could not load image " << imagePath << "\n";
            return -1;
        }

        // Create OpenCV window
        cv::namedWindow("Lomography", cv::WINDOW_AUTOSIZE);
        centerWindow("Lomography", inputImage.cols, inputImage.rows);

        // Create trackbars
        cv::createTrackbar("Color Param (x0.01)", "Lomography", nullptr, 20, onColorTrackbar);
        cv::setTrackbarPos("Color Param (x0.01)", "Lomography", static_cast<int>(colorParam * 100));

        cv::createTrackbar("Vignette Radius (%)", "Lomography", nullptr, 100, onVignetteTrackbar);
        cv::setTrackbarPos("Vignette Radius (%)", "Lomography", vignetteParam);

        // Initial display
        applyColorFilter(0, nullptr);
        applyVignetteFilter(0, nullptr);

        // Event loop
        while (true) {
            char key = static_cast<char>(cv::waitKey(1));
            if (key == 'q') {
                break; // Quit program
            } else if (key == 's') {
                // Save the result image
                cv::imwrite("lomography_result.jpg", finalImage);
                std::cout << "Result saved as lomography_result.jpg\n";
                break;
            }
        }
    } catch (const cv::Exception& e) {
        std::cerr << "OpenCV Error: " << e.what() << "\n";
    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << "\n";
    }

    return 0;
}

#include <opencv2/opencv.hpp>
#include <iostream>

int main() {
    cv::VideoCapture cap(0, cv::CAP_V4L2);

    if (!cap.isOpened()) {
        std::cerr << "Error: Could not open camera\n";
        return -1;
    }

    cv::Mat frame;
    while (true) {
        cap >> frame;
        if (frame.empty()) {
            std::cerr << "Error: Empty frame\n";
            break;
        }
        cv::imshow("Camera Feed", frame);
        if (cv::waitKey(30) >= 0) break;
    }
    return 0;
}

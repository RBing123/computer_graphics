#include <opencv2/opencv.hpp>
#include <iostream>

cv::Point start_point;
bool drawing = false;

void draw_line(int event, int x, int y, int flags, void* param) {
    static cv::Mat image = *((cv::Mat*)param);

    if (event == cv::EVENT_LBUTTONDOWN) {
        drawing = true;
        start_point = cv::Point(x, y);
    }
    else if (event == cv::EVENT_MOUSEMOVE && drawing) {
        cv::Mat temp_img = image.clone();
        cv::line(temp_img, start_point, cv::Point(x, y), cv::Scalar(255, 0, 0), 2);
        cv::imshow("Draw Line", temp_img);
    }
    else if (event == cv::EVENT_LBUTTONUP) {
        drawing = false;
        cv::line(image, start_point, cv::Point(x, y), cv::Scalar(255, 0, 0), 2);
        cv::imshow("Draw Line", image);
    }
}

int main() {
    cv::Mat image = cv::Mat::zeros(512, 512, CV_8UC3);
    cv::namedWindow("Draw Line");
    cv::setMouseCallback("Draw Line", draw_line, &image);

    while (true) {
        cv::imshow("Draw Line", image);
        if (cv::waitKey(1) == 27) break;  // Press ESC to exit
    }

    cv::destroyAllWindows();
    return 0;
}

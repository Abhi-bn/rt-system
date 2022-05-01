
#include <opencv2/opencv.hpp>

void expandRectBy(cv::Rect &inputRect, cv::Rect &outputRect, float by) {
    cv::Rect rec(inputRect);
    rec.x -= inputRect.width * by;
    rec.y -= inputRect.height * by;
    rec.width += inputRect.width * by * 2;
    rec.height += inputRect.height * by * 2;
    outputRect = rec;
}

bool isInside(const cv::Rect &mainRect, const cv::Rect &checkingRect) {
    return (checkingRect & mainRect) == checkingRect;
}
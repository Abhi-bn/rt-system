/*
 * Created on Wed Mar 30 2022
 *
 * Author: Abhinava B N
 */

#include "VideoReader.hpp"

cv::VideoCapture VideoReader::READER = cv::VideoCapture();

std::string VideoReader::VIDEO_PATH = "";

bool VideoReader::open_capture() {
    READER = cv::VideoCapture(std::string(VIDEO_PATH));
    return READER.isOpened();
}

bool VideoReader::fetch_frame(cv::Mat& frame) {
    if (!READER.isOpened() && !open_capture()) {
        throw std::string("Error reading file");
    }
    return READER.read(frame);
}

void VideoReader::set_video_path(std::string path) { VIDEO_PATH = path; }
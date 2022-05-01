/*
 * Created on Wed Mar 30 2022
 *
 * Author: Abhinava B N
 */
#pragma once

#ifndef VIDEOREADER_H  // include guard
#define VIDEOREADER_H

#include <opencv2/opencv.hpp>

const std::string VIDEO_PATH = "data/traffic_origin.mp4";

class VideoReader {
    static cv::VideoCapture READER;
    static std::string VIDEO_PATH;
    static bool open_capture();

   public:
    static void set_video_path(std::string);
    static bool fetch_frame(cv::Mat&);
};
#endif /* VIDEOREADER_H */
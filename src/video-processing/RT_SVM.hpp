/*
 * Created on Wed Mar 30 2022
 *
 * Author: Abhinava B N
 */

#pragma once

#ifndef RT_SVM_H // include guard
#define RT_SVM_H

#include <opencv2/opencv.hpp>
#include <string>
#include <vector>

#include "BaseModel.hpp"
#include "SVMModelOpencv.hpp"

class RT_SVM : BaseModel
{
    cv::HOGDescriptor hog;
    // cv::Ptr<cv::ml::SVM> svm;
    SVMModelOpencv svm_model;
    cv::Ptr<cv::BackgroundSubtractorMOG2> bg;
    void detection();

    template <typename T>
    void pre_processing(cv::Mat input, T &f)
    {
        resize(input, input, cv::Size(128, 64));
        hog.compute(input, f);
    }

public:
    RT_SVM();
    virtual void load_model(std::string s);
    void training(const std::vector<std::string> &, const std::vector<float> &);
    void get_foreground(const cv::Mat &, cv::Mat &);
    cv::Mat inference(const cv::Mat &in);
    cv::Mat inference(const cv::Mat &in, cv::Mat &);
    void convulation(const cv::Mat &image, std::vector<cv::Rect> &rects, float dh, float dw);
    ~RT_SVM();
};
#endif /* RT_SVM_H */
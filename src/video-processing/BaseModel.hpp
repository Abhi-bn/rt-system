/*
 * Created on Wed Mar 30 2022
 *
 * Author: Abhinava B N
 */
#pragma once

#ifndef BASE_MODEL_H  // include guard
#define BASE_MODEL_H

#include <map>
#include <opencv2/opencv.hpp>
#include <string>
#include <vector>

class BaseModel {
    bool loaded = false;
    template <typename T>
    void pre_processing(cv::Mat input, T& f) {
        return;
    }

   public:
    virtual void load_model(std::string);

    cv::Mat toMat(std::vector<std::vector<float>>);

    virtual void training(const std::vector<std::string>&){};
};
#endif /* BASE_MODEL_H */
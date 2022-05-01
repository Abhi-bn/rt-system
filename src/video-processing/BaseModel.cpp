/*
 * Created on Wed Mar 30 2022
 *
 * Author: Abhinava B N
 */
#include "BaseModel.hpp"
using namespace cv;

void BaseModel::load_model(std::string model_path) { loaded = true; }

cv::Mat BaseModel::toMat(std::vector<std::vector<float>> vec) {
    assert(vec.size() != 0 && vec[0].size() != 0);

    cv::Mat in = cv::Mat(Size(vec[0].size(), vec.size()), CV_32FC1);

    for (size_t i = 0; i < vec.size(); i++)
        for (size_t j = 0; j < vec[i].size(); j++) in.at<float>(i, j) = vec[i][j];

    return in;
}
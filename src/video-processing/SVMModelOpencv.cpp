#include "SVMModelOpencv.hpp"

#include <iostream>

using namespace cv;
using namespace cv::ml;
SVMModelOpencv::SVMModelOpencv() { this->svm = cv::ml::SVM::create(); }

SVMModelOpencv::~SVMModelOpencv() {
    // delete this->param;
    // delete this->svm;
}

bool SVMModelOpencv::isEmpty() { return this->svm == nullptr; }

void SVMModelOpencv::training(std::vector<float> labels,
                              std::vector<std::vector<float>> embds) {
    svm->setType(cv::ml::SVM::C_SVC);
    svm->setC(10);
    svm->setKernel(cv::ml::SVM::LINEAR);
    svm->setTermCriteria(TermCriteria(TermCriteria::MAX_ITER, 1000, 1e-6));
    std::vector<float> info;
    info.push_back((float)std::count(labels.begin(), labels.end(), 1) / (float)labels.size());
    info.push_back((float)std::count(labels.begin(), labels.end(), -1) / (float)labels.size());

    cv::Mat weights(1, 2, CV_32FC1);
    for (size_t i = 0; i < info.size(); i++) {
        weights.at<float>(0, i) = info[i];
    }
    svm->setClassWeights(weights);
    svm->setNu(0.001);

    cv::Mat matAngles(embds.size(), embds.at(0).size(), CV_32FC1);
    for (int i = 0; i < matAngles.rows; ++i)
        for (int j = 0; j < matAngles.cols; ++j)
            matAngles.at<float>(i, j) = embds.at(i).at(j);

    Mat labelsMat(labels.size(), 1, CV_32S);

    for (size_t i = 0; i < labels.size(); i++)
        labelsMat.at<int>(0, i) = int(labels[i]);

    svm->train(matAngles, ROW_SAMPLE, labelsMat);
    save_model("models/HOGModel.svmopencv");
}

int SVMModelOpencv::recognise(std::vector<float> embds) {
    cv::Mat image(embds.size(), 1, CV_32FC1, embds.data()), im, out;
    transpose(image, im);
    float response = svm->predict(im);
    return response;
}

int SVMModelOpencv::save_model(std::string path) {
    svm->save(path);
    return true;
}

void SVMModelOpencv::load_model(std::string path) {
    this->svm = svm->load(path.c_str());
}

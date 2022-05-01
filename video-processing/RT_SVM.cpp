/*
 * Created on Wed Mar 30 2022
 *
 * Author: Abhinava B N
 */

#include "RT_SVM.hpp"

#include "Helper.hpp"
#include "SVMModel.hpp"

using namespace cv;
using namespace std;
using namespace cv::ml;
RNG rng(12345);
RT_SVM::RT_SVM(std::string model_path) {
    hog = cv::HOGDescriptor(Size(128, 64), Size(16, 16), Size(8, 8), Size(8, 8),
                            9);
    svm_model = SVMModelOpencv();
    svm_model.load_model(model_path);
    bg = cv::createBackgroundSubtractorMOG2(100, 50, true);
}

void RT_SVM::training(const std::vector<std::string>& datasets_path, const std::vector<float>& labels) {
    std::vector<std::vector<float>> feature_set;
    for (size_t i = 0; i < datasets_path.size(); i++) {
        cv::Mat im = imread(datasets_path[i], IMREAD_GRAYSCALE);
        resize(im, im, Size(128, 64));
        std::vector<float> feature;
        pre_processing(im, feature);
        feature_set.push_back(feature);
    }
    svm_model.training(labels, feature_set);
}

RT_SVM::~RT_SVM() {}

void RT_SVM::load_model(std::string s) {
    svm_model.load_model(s);
}

void RT_SVM::convulation(const cv::Mat& image, std::vector<cv::Rect>& rects, float dh, float dw) {
    cv::Size w_s = cv::Size(128, 64);
    cv::Mat dimage;
    resize(image, dimage, Size(0, 0), dh, dw);
    for (int i = w_s.height; i < dimage.rows - w_s.height; i += w_s.height / 8) {
        for (int j = w_s.width; j < dimage.cols - w_s.width; j += w_s.width / 8) {
            Rect rect = Rect(j, i, w_s.width, w_s.height);
            std::vector<float> des;
            // imshow("asdsad", dimage(rect));
            // waitKey(30);
            pre_processing(dimage(rect), des);
            int label = svm_model.recognise(des);
            if (label == 1) {
                imshow("out", dimage(rect));
                rects.push_back(Rect(j / dw, i / dh, w_s.width / dw, w_s.height / dh));
            }
        }
    }
}

void RT_SVM::get_foreground(const cv::Mat& input, cv::Mat& output) {
    cv::Mat image;
    // GaussianBlur(input, image, Size(5, 5), 1, 1);
    bg->apply(input, image);
    // cv::Mat kernel = cv::Mat::ones(Size(3, 3), CV_8UC1);
    // morphologyEx(image, image, MORPH_OPEN, kernel);

    vector<vector<Point>> contours;
    vector<cv::Vec4i> hierarchy;
    findContours(image, contours, hierarchy, RETR_TREE, CHAIN_APPROX_SIMPLE);
    Mat drawing = Mat::zeros(image.size(), CV_8UC3);
    Mat gray = Mat::zeros(image.size(), CV_8UC1);

    for (size_t i = 0; i < contours.size(); i++) {
        if (contours[i].size() < 50)
            continue;

        cv::Rect rect = boundingRect(contours[i]);
        // if ((float)contours[i].size() / (float)rect.area() > 0.1)
        //     continue;

        Scalar color = Scalar(rng.uniform(0, 256), rng.uniform(0, 256), rng.uniform(0, 256));
        drawContours(drawing, contours, (int)i, color, -1, LINE_8, hierarchy, 0);
        drawContours(gray, contours, (int)i, Scalar(255), -1, LINE_8, hierarchy, 0);
    }
    output = gray;
}

void RT_SVM::inference(const cv::Mat& input, const cv::Mat& forground, cv::Mat& output) {
    Mat drawing = input.clone();
    Mat gray = Mat::zeros(forground.size(), CV_8UC1);
    vector<Rect> obj_locations;
    vector<vector<Point>> contours;
    vector<cv::Vec4i> hierarchy;
    findContours(forground, contours, hierarchy, RETR_TREE, CHAIN_APPROX_SIMPLE);
    for (size_t i = 0; i < contours.size(); i++) {
        if (contours[i].size() < 50)
            continue;

        cv::Rect rect = boundingRect(contours[i]);
        expandRectBy(rect, rect, 0.1);
        if (!isInside(cv::Rect(0, 0, input.cols - 1, input.rows - 1), rect)) continue;
        drawContours(gray, contours, (int)i, Scalar(255), -1, LINE_8, hierarchy, 0);

        std::vector<float> des;
        pre_processing(input(rect), des);
        int label = svm_model.recognise(des);
        if (label != 1) continue;
        obj_locations.push_back(rect);
    }

    for (size_t i = 0; i < obj_locations.size(); i++) {
        rectangle(drawing, obj_locations[i], cv::Scalar(255, 0, 0));
    }
    output = drawing;
    // imshow("gray", forground);
    // imshow("drawing", drawing);
    // imshow("gray1", gray);
    // waitKey(1);
}
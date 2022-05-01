//
//  svm.h
//
//  Created by Abhinava BN on 4/December/2021.
//  Copyright © 2021. All rights reserved.
//

#ifndef svm_h
#define svm_h

#include <algorithm>
#include <map>
#include <opencv2/opencv.hpp>
#include <string>
#include <vector>

#include "svm.h"

class SVMModelOpencv {
    int face_number;
    cv::Ptr<cv::ml::SVM> svm;
    void prepareSVMProblem(std::vector<double> labels,
                           std::vector<std::vector<float>> embds,
                           svm_problem *);

   public:
    SVMModelOpencv();
    ~SVMModelOpencv();
    void training(std::vector<float> labels,
                  std::vector<std::vector<float>> embds);
    int recognise(std::vector<float> embds);
    int save_model(std::string path);
    void load_model(std::string path);
    bool isEmpty();
};

#endif /* svm_h */
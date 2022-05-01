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
#include <string>
#include <vector>

#include "svm.h"

class SVMModel {
    int face_number;
    svm_model *model = nullptr;
    svm_parameter *param;
    std::vector<svm_node> converTo1DSVMNode(const std::vector<float> &embd);
    svm_node **convertTo2DSVMNode(std::vector<std::vector<svm_node>> nodes);
    void prepareSVMProblem(std::vector<double> labels,
                           std::vector<std::vector<float>> embds,
                           svm_problem *);

   public:
    SVMModel();
    ~SVMModel();
    void training(std::vector<double> labels,
                  std::vector<std::vector<float>> embds);
    int recognise(std::vector<float> embds);
    int save_model(std::string path);
    void load_model(std::string path);
    bool isEmpty();
};

#endif /* svm_h */
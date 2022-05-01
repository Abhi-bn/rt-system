#include "SVMModel.hpp"

#include <iostream>

SVMModel::SVMModel() { this->face_number = 0; }

SVMModel::~SVMModel() {
    // delete this->param;
    delete this->model;
}

bool SVMModel::isEmpty() { return this->model == nullptr; }

std::vector<svm_node> SVMModel::converTo1DSVMNode(
    const std::vector<float> &embd) {
    auto v = std::vector<svm_node>();
    for (size_t i = 0; i < embd.size(); i++) {
        auto node = svm_node();
        node.index = i + 1;
        node.value = embd[i];
        v.push_back(node);
    }
    auto end = svm_node();
    end.index = -1;
    end.value = -1;
    v.push_back(end);
    return v;
}

svm_node **SVMModel::convertTo2DSVMNode(
    std::vector<std::vector<svm_node>> nodes) {
    svm_node **dsvm_node = new svm_node *[nodes.size()];
    for (size_t i = 0; i < nodes.size(); i++) {
        dsvm_node[i] = (svm_node *)malloc(nodes[i].size() * sizeof(svm_node));
        for (size_t j = 0; j < nodes[i].size(); j++) {
            dsvm_node[i][j] = nodes[i][j];
        }
    }
    return dsvm_node;
}

void SVMModel::prepareSVMProblem(std::vector<double> labels,
                                 std::vector<std::vector<float>> embds,
                                 svm_problem *prob) {
    std::vector<std::vector<svm_node>> nodes;
    for (size_t i = 0; i < labels.size(); i++) {
        nodes.push_back(converTo1DSVMNode(embds[i]));
    }
    prob->l = (int)labels.size();
    prob->y = labels.data();
    auto s = convertTo2DSVMNode(nodes);

    prob->x = s;
    for (int i = 0; i < prob->l; i++) {
        std::cout << prob->x[i]->index << std::endl;
    }
}

void SVMModel::training(std::vector<double> labels,
                        std::vector<std::vector<float>> embds) {
    auto prob = new svm_problem();
    this->param = new svm_parameter();
    this->param->C = 10;
    // svm_types: enum { C_SVC, NU_SVC, ONE_CLASS, EPSILON_SVR, NU_SVR }
    this->param->svm_type = NU_SVC;
    this->param->nu = 0.1;
    // kernel_types: enum { LINEAR, POLY, RBF, SIGMOID, PRECOMPUTED }
    this->param->kernel_type = LINEAR;
    double *info = new double[2];
    info[1] = std::count(labels.begin(), labels.end(), -1) / labels.size();
    info[0] = std::count(labels.begin(), labels.end(), 1) / labels.size();
    int *label = new int[2];
    label[1] = -1;
    label[0] = 1;
    // this->param->nr_weight = 1;
    // this->param->weight_label = label;
    // this->param->weight = info;
    this->param->eps = 1e-6;
    this->param->degree = 3;
    this->param->probability = int(true);
    this->param->shrinking = int(false);
    prepareSVMProblem(labels, embds, prob);
    this->model = svm_train(prob, this->param);
    save_model("models/HOGModel.libsvm");
}

int SVMModel::recognise(std::vector<float> embds) {
    auto prob_estimates = new double[26];
    int label = svm_predict_probability(
        this->model, this->converTo1DSVMNode(embds).data(), prob_estimates);
    if (label == 1) {
        std::cout << "catch";
    }
    return label;
}

int SVMModel::save_model(std::string path) {
    return svm_save_model(path.c_str(), this->model);
}

void SVMModel::load_model(std::string path) {
    this->model = svm_load_model(path.c_str());
}

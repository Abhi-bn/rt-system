/*
 * Created on Wed Mar 30 2022
 *
 * Author: Abhinava B N
 */

#include <condition_variable>
#include <mutex>
#include <opencv2/opencv.hpp>
#include <thread>

#include "process_system.hpp"
#include "video-processing/RT_SVM.hpp"

using namespace std;
using namespace cv;

void prep_data(vector<std::string>& img_path, vector<float>& labels) {
    vector<string> pos;
    glob("photos/processed/*", pos);
    vector<string> neg;
    glob("photos/neg/*", neg);

    for (size_t i = 0; i < pos.size(); i++) {
        img_path.push_back(pos.at(i));
        labels.push_back(1);
    }

    for (size_t i = 0; i < neg.size(); i++) {
        img_path.push_back(neg.at(i));
        labels.push_back(-1);
    }
}

int main(int argc, char* argv[]) {
    ProcessSystem* ps = new ProcessSystem("result/", "models/HOGModel.svmopencv");
    ps->load_data("photos/test/*");
    for (int i = 0; i < 10; i++) {
        thread t1(&ProcessSystem::get_frame, ps);
        thread t2(&ProcessSystem::get_foreground, ps);
        thread t3(&ProcessSystem::inference, ps);
        t1.join();
        t2.join();
        t3.join();
    }
}

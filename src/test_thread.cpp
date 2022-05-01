/*
 * Created on Wed Mar 30 2022
 *
 * Author: Abhinava B N
 */

#include <condition_variable>
#include <mutex>
#include <opencv2/opencv.hpp>
#include <thread>

#include "video-processing/RT_SVM.hpp"
#include "video-processing/VideoReader.hpp"
using namespace std;
using namespace cv;

#define DEBUG 1

class ProcessSystem {
    std::string frame_store_path;
    RT_SVM* rt;
    std::vector<cv::String> all_frames;
    int frame_index = 0;
    std::mutex frame_read, foreground_lock;
    cv::Mat fresh_frame, fresh_frame_copy, foreground_frame;
    condition_variable frame_read_var, foreground_frame_var;

    void locked_execution(std::function<void()> func, std::mutex& locker, condition_variable& var, std::string debug = string()) {
        std::unique_lock<mutex> cl(locker);
        func();
        cl.unlock();
        var.notify_one();
#if DEBUG
        cout << debug << endl;
#endif
    }

    void waited_execution(const cv::Mat& frame, std::mutex& locker, condition_variable& var, std::string debug = string()) {
        std::unique_lock<mutex> cl(locker);
        var.wait(cl, [&] {
#if DEBUG
            cout << debug << endl;
#endif
            return !frame.empty();
        });
    }

   public:
    ProcessSystem(std::string frame_store_path, std::string model_path) {
        rt = new RT_SVM(model_path);
        this->frame_store_path = frame_store_path;
    }

    void load_data(std::string path_pattern) {
        glob(path_pattern, all_frames);
    }

    void get_frame() {
        auto update_fresh_frame = [&]() {
            this->fresh_frame = cv::imread(this->all_frames[this->frame_index++]);
#if DEBUG
            this_thread::sleep_for(chrono::seconds(1));
#endif
        };

        locked_execution(update_fresh_frame, frame_read, frame_read_var, "notified: get_frame");
        cout << "done: get_frame" << endl;
    }

    void get_foreground() {
        waited_execution(fresh_frame, frame_read, frame_read_var, "waiting: get_foreground");
        fresh_frame_copy = fresh_frame.clone();
        fresh_frame.release();

        auto update_foreground_frame = [&]() {
            this->rt->get_foreground(this->fresh_frame_copy, this->foreground_frame);
#if DEBUG
            this_thread::sleep_for(chrono::seconds(1));
#endif
        };

        locked_execution(update_foreground_frame, foreground_lock, foreground_frame_var);
        cout << "done: get_foreground" << endl;
    }

    void inference() {
        waited_execution(foreground_frame, foreground_lock, foreground_frame_var, "waiting: inference");
        // auto update_foreground_frame = [&]() {

        //     this_thread::sleep_for(chrono::seconds(5));
        // };
        Mat foreground_frame_copy = foreground_frame.clone(), output;
        foreground_frame.release();
        this->rt->inference(this->fresh_frame_copy, foreground_frame_copy, output);
        imwrite(frame_store_path + std::to_string(frame_index) + ".png", output);
        // locked_execution(update_foreground_frame, foreground_lock, foreground_frame_var);
        cout << "done: inference" << endl;
    }
};

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

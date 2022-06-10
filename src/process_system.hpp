/*
 * Created on Wed Mar 30 2022
 *
 * Author: Abhinava B N
 */
#pragma once

#ifndef PROCESS_SYSTEM_H  // include guard
#define PROCESS_SYSTEM_H

#include <condition_variable>
#include <mutex>
#include <opencv2/opencv.hpp>
#include <thread>

#include "video-processing/RT_SVM.hpp"

using namespace std;
using namespace cv;

#define DEBUG 0

class ProcessSystem {
    bool exit = false;
    std::string frame_store_path;
    RT_SVM* rt;
    std::vector<cv::String> all_frames;
    int frame_index = 0;
    std::mutex frame_read, foreground_lock, storeImg_lock;
    std::map<string, cv::Mat> storeImage;
    cv::Mat fresh_frame, fresh_frame_copy, fresh_frame_copy_copy, foreground_frame;
    condition_variable frame_read_var, foreground_frame_var;

    void locked_execution(std::function<void()> func, std::mutex& locker, condition_variable& var, bool notify = true, std::string debug = string()) {
        std::unique_lock<mutex> cl(locker);
        func();
        cl.unlock();
        if (notify)
            var.notify_all();
#if DEBUG
        cout << debug << endl;
#endif
    }

    void waited_execution(const cv::Mat& frame, std::mutex& locker, condition_variable& var, bool extra = false, const cv::Mat& extra_frame = cv::Mat(), std::string debug = string()) {
        // return;
        std::unique_lock<mutex> cl(locker);
        var.wait(cl, [&] {
#if DEBUG
            cout << debug << endl;
#endif
            if (extra) {
                return !frame.empty() && !extra_frame.empty();
            }
            return !frame.empty();
        });
    }

   public:
    ProcessSystem(std::string frame_store_path, std::string model_path) {
        rt = new RT_SVM(model_path);
        this->frame_store_path = frame_store_path;
    }

    bool should_exit() {
        return exit;
    }

    void load_data(std::string path_pattern) {
        glob(path_pattern, all_frames);
        cout << all_frames.size() << endl;
    }

    void get_frame() {
        auto update_fresh_frame = [&]() {
            if (this->frame_index >= this->all_frames.size() - 1) {
                this->exit = true;
            }
            this->fresh_frame = cv::imread(this->all_frames[this->frame_index++]);
            resize(this->fresh_frame, this->fresh_frame, cv::Size(), 0.7, 0.7);
#if DEBUG
            this_thread::sleep_for(chrono::seconds(1));
#endif
        };

        locked_execution(update_fresh_frame, frame_read, frame_read_var, "notified: get_frame");
        cout << "done: get_frame" << endl;
    }

    void get_foreground() {
        waited_execution(this->fresh_frame, this->frame_read, this->frame_read_var, false, cv::Mat(), "waiting: get_foreground");

        locked_execution([&]() {
            this->fresh_frame_copy = this->fresh_frame.clone();
            this->fresh_frame.release();
        },
                         this->frame_read, this->frame_read_var, false);

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
        waited_execution(this->foreground_frame, this->foreground_lock, foreground_frame_var, true, this->fresh_frame_copy, "waiting: inference");
        Mat foreground_frame_copy = foreground_frame.clone(), output;
        locked_execution([&]() {
            this->fresh_frame_copy_copy = fresh_frame_copy.clone();
            this->fresh_frame_copy.release();
            this->foreground_frame.release();
        },
                         foreground_lock, foreground_frame_var, false);

        this->rt->inference(this->fresh_frame_copy_copy, foreground_frame_copy, output);

        locked_execution([&]() {
            if (output.empty()) {
                cout << "inference: wasted!!!" << endl;
            }
            cout << "\ninference: frame number: " + to_string(this->frame_index) << endl;
            auto t = std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::system_clock::now().time_since_epoch()).count();
            storeImage[to_string(t)] = output.clone();
        },
                         storeImg_lock, foreground_frame_var);
        cout << "done: inference" << endl;
    }

    void store_processed_image() {
        std::vector<string> removed;
        for (std::map<string, cv::Mat>::iterator it = storeImage.begin(); it != storeImage.end(); ++it) {
            string index_number = it->first;
            Mat get_img = it->second.clone();
            removed.push_back(index_number);
            if (get_img.empty()) {
                cout << "store_processed_image: wasted!!!" << endl;
                continue;
            }
            imwrite(frame_store_path + index_number + ".png", get_img);
        }

        locked_execution([&]() {
            string deleted = "";
            for (int i = 0; i < removed.size(); i++) {
                storeImage.erase(removed[i]);
                deleted += " " + removed[i] + " ";
            }
            if (removed.size() > 0)
                cout << "\nstore_processed_image; deleted:" + deleted << endl;
        },
                         storeImg_lock, foreground_frame_var);
    }
};
#endif /* PROCESS_SYSTEM_H */
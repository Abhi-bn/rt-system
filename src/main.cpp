/*
 * Created on Wed Mar 30 2022
 *
 * Author: Abhinava B N
 */

#ifdef LITMUS_H
#include <litmus.h>
#endif

#include <stdio.h>
#include <sys/types.h>
#include <unistd.h>

#include <opencv2/opencv.hpp>
#include <thread>

#include "video-processing/RT_SVM.hpp"
#include "video-processing/VideoReader.hpp"

#define PERIOD ms2ns(1000)
#define DEADLINE ms2ns(1500)
#define EXEC_COST ms2ns(500)

#define CALL(exp)                                     \
    do {                                              \
        int ret;                                      \
        ret = exp;                                    \
        if (ret != 0)                                 \
            fprintf(stderr, "%s failed: %m\n", #exp); \
        else                                          \
            fprintf(stderr, "%s ok.\n", #exp);        \
    } while (0)

void prep_data(std::vector<std::string> &img_path, std::vector<float> &labels) {
    std::vector<cv::String> pos;
    cv::glob("photos/processed/*", pos);
    std::vector<cv::String> neg;
    cv::glob("photos/neg/*", neg);

    for (size_t i = 0; i < pos.size(); i++) {
        img_path.push_back(pos.at(i));
        labels.push_back(1);
    }

    for (size_t i = 0; i < neg.size(); i++) {
        img_path.push_back(neg.at(i));
        labels.push_back(-1);
    }
}

void test_data(RT_SVM *rt) {
    std::vector<cv::String> test;
    cv::glob("data/test/*", test);
    for (size_t i = 0; i < test.size(); i++) {
        cv::Mat res = cv::imread(test[i]);
        cv::resize(res, res, cv::Size(), 0.5, 0.5);
        cv::Mat out = rt->inference(res);
        // cv::imwrite("result/" + std::to_string(i) + ".jpg", out);
        break;
    }
}
class ProcessSystem {
    RT_SVM *rt;
    void get_foreground() {
        rt->get_foreground()
    }
};
void job1(rt_task params, RT_SVM *rt) {
    CALL(set_rt_task_param(gettid(), &params));
    CALL(task_mode(LITMUS_RT_TASK));
    CALL(wait_for_ts_release());
    int do_exit = 0;
    do {
        sleep_next_period();
        test_data(rt);
        std::cout << "Done job1" << std::endl;
        printf("%d\n", do_exit);
    } while (!do_exit);
}
void job2(rt_task params, RT_SVM *rt) {
    CALL(set_rt_task_param(gettid(), &params));
    CALL(task_mode(LITMUS_RT_TASK));
    CALL(wait_for_ts_release());
    int do_exit = 0;
    do {
        sleep_next_period();
        test_data(rt);
        std::cout << "Done job2" << std::endl;
        printf("%d\n", do_exit);
    } while (!do_exit);
}
int main(int argc, char *argv[]) {
    int do_exit = 0;
    struct rt_task params1;
    struct rt_task params2;

    init_rt_task_param(&params1);
    init_rt_task_param(&params2);
    params1.exec_cost = ms2ns(100);
    params1.cpu = 0;
    params1.period = ms2ns(200);
    params1.relative_deadline = ms2ns(200);
    params1.release_policy = TASK_PERIODIC;

    params2.exec_cost = ms2ns(100);
    params2.cpu = 1;
    params2.period = ms2ns(200);
    params2.relative_deadline = ms2ns(200);
    params2.release_policy = TASK_PERIODIC;

    CALL(init_litmus());
    RT_SVM *rt = new RT_SVM();
    rt->load_model("models/HOGModel.svmopencv");
    std::thread first(job1, params1, rt);
    std::thread second(job2, params2, rt);
    first.join();
    second.join();
    // char buffer[80];
    // CALL(read(STDIN_FILENO, buffer, sizeof(buffer)));

    // CALL(task_mode(BACKGROUND_TASK));

    // rt.training(img_path, labels);

    // VideoReader vr = VideoReader();
    // vr.set_video_path(argv[1]);
    // Mat frame;
    // do
    // {
    //     sleep_next_period();
    //     test_data();
    //     printf("%d\n", do_exit);
    // } while (!do_exit);

    // CALL(task_mode(BACKGROUND_TASK));
    // CALL(wait_for_ts_release());
    // while (true) {
    //     if (!vr.fetch_frame(frame)) {
    //         cout << "Video Ended!!!\n";
    //         break;
    //     }
    //     rotate(frame, frame, ROTATE_180);
    //     resize(frame, frame, Size(0, 0), 0.5, 0.5);

    //     rt.inference(frame);
    //     // imshow("asdsd", frame);

    //     if (waitKey(30) == 27) break;
    // }
}
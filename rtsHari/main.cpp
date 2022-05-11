#include <glob.h>
#include <litmus.h>
#include <stdio.h>
#include <sys/types.h>
#include <unistd.h>

#include <condition_variable>
#include <mutex>
#include <opencv2/opencv.hpp>
#include <thread>
// using namespace cv;
// using namespace std;

int i = 0;
int do_exit = 0;

std::vector<cv::String> image_path = {"data/aero1.jpg", "data/aero3.jpg", "data/left.jpg"};

#define EXEC_COST ms2ns(8)
#define PERIOD ms2ns(38)
#define DEADLINE ms2ns(20)

#define EXEC_COST2 ms2ns(8)
#define PERIOD2 ms2ns(38)
#define DEADLINE2 ms2ns(28)

#define EXEC_COST3 ms2ns(10)
#define PERIOD3 ms2ns(38)
#define DEADLINE3 ms2ns(38)

#define CALL(exp)                                     \
    do {                                              \
        int ret;                                      \
        ret = exp;                                    \
        if (ret != 0)                                 \
            fprintf(stderr, "%s failed: %m\n", #exp); \
        else                                          \
            fprintf(stderr, "%s ok.\n", #exp);        \
    } while (0)

std::mutex frame_read, foreground_lock;
std::condition_variable frame_read_var, foreground_frame_var;

void locked_execution(std::function<void()> func, std::mutex& locker, std::condition_variable& var, bool notify = true, std::string debug = std::string()) {
    std::unique_lock<std::mutex> cl(locker);
    func();
    cl.unlock();
    if (notify)
        var.notify_all();
#if DEBUG
    cout << debug << endl;
#endif
}

void waited_execution(const cv::Mat& frame, std::mutex& locker, std::condition_variable& var, bool extra = false, const cv::Mat& extra_frame = cv::Mat(), std::string debug = std::string()) {
    std::unique_lock<std::mutex> cl(locker);
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
cv::Mat fresh_frame, equlize_frame;
void jobRgbToHsv(rt_task* params) {
    CALL(init_rt_thread());
    init_rt_task_param(params);
    params->cpu = 0;
    params->exec_cost = EXEC_COST;

    params->period = PERIOD;

    params->relative_deadline = DEADLINE;

    CALL(set_rt_task_param(gettid(), params));
    CALL(task_mode(LITMUS_RT_TASK));
    CALL(wait_for_ts_release());

    do {
        sleep_next_period();

        std::cout << "Step1\n";

        auto update_fresh_frame = []() {
            cv::Mat img = imread(image_path[i++], cv::IMREAD_COLOR);
            std::cout << img.size() << std::endl;

            cv::Mat ycrcb;

            cv::cvtColor(img, ycrcb, cv::COLOR_BGR2HSV);

            std::cout << "Step1 write\n";

            cv::String imagePath1 = "Step1-" + std::to_string(i) + ".jpg";
            fresh_frame = ycrcb;
            // cout << i << "\n";
            // imwrite(imagePath1, ycrcb);
#if DEBUG
            this_thread::sleep_for(chrono::seconds(1));
#endif
        };
        locked_execution(update_fresh_frame, frame_read, frame_read_var, "notified: get_frame");
        if (i == image_path.size())
            do_exit = true;

    } while (!do_exit);
}

void jobEqualize(rt_task* params) {
    CALL(init_rt_thread());
    init_rt_task_param(params);
    params->cpu = 0;
    params->exec_cost = EXEC_COST2;
    params->period = PERIOD2;

    params->relative_deadline = DEADLINE2;

    CALL(set_rt_task_param(gettid(), params));
    CALL(task_mode(LITMUS_RT_TASK));
    CALL(wait_for_ts_release());

    do {
        sleep_next_period();
        waited_execution(fresh_frame, frame_read, frame_read_var, false, cv::Mat(), "waiting: get_foreground");

        std::cout << "Step2\n";

        // String imagePath2 = "Step2-" + std::to_string(i) + ".jpg";

        // cout << "Step2 write\n";

        // imwrite(imagePath2, ycrcb);

        auto update_foreground_frame = [&]() {
            cv::String imagePath1 = "Step1-" + std::to_string(i) + ".jpg";

            cv::Mat ycrcb = fresh_frame.clone();
            fresh_frame.release();

            std::vector<cv::Mat> channels;

            cv::split(ycrcb, channels);

            cv::equalizeHist(channels[0], channels[0]);

            cv::merge(channels, ycrcb);
            equlize_frame = ycrcb;
        };
        locked_execution(update_foreground_frame, foreground_lock, foreground_frame_var);
    } while (!do_exit);
}

void jobHsvToRgb(rt_task* params) {
    CALL(init_rt_thread());
    init_rt_task_param(params);
    params->cpu = 0;
    params->exec_cost = EXEC_COST3;

    params->period = PERIOD3;

    params->relative_deadline = DEADLINE3;

    CALL(set_rt_task_param(gettid(), params));
    CALL(task_mode(LITMUS_RT_TASK));
    CALL(wait_for_ts_release());

    do {
        sleep_next_period();
        waited_execution(equlize_frame, foreground_lock, foreground_frame_var, false, cv::Mat(), "waiting: jobHsvToRgb");

        cv::String imagePath2 = "Step2-" + std::to_string(i) + ".jpg";
        std::cout << "Step3\n";

        cv::Mat ycrcb = equlize_frame.clone();
        equlize_frame.release();

        cv::Mat result;
        cvtColor(ycrcb, result, CV_HSV2BGR);
        cv::String imagePath3 = "Step3-" + std::to_string(i) + ".jpg";

        imwrite(imagePath3, result);

    } while (!do_exit);
}

int main(int argc, char* argv[]) {
    cv::glob("data/*", image_path);
    CALL(init_litmus());

    struct rt_task params;
    struct rt_task params2;
    struct rt_task params3;

    std::thread T1(jobRgbToHsv, &params);

    std::thread T2(jobEqualize, &params2);

    std::thread T3(jobHsvToRgb, &params3);

    T1.join();

    T2.join();

    T3.join();

    return 0;
}

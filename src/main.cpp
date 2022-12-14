/*
 * Created on Wed Mar 30 2022
 *
 * Author: Abhinava B N
 */

#include <litmus.h>
#include <stdio.h>
#include <sys/types.h>
#include <unistd.h>

#include <opencv2/opencv.hpp>
#include <thread>

#include "process_system.hpp"

#define CALL(exp)                                     \
    do {                                              \
        int ret;                                      \
        ret = exp;                                    \
        if (ret != 0)                                 \
            fprintf(stderr, "%s failed: %m\n", #exp); \
        else                                          \
            fprintf(stderr, "%s ok.\n", #exp);        \
    } while (0)

void fetch_frame(ProcessSystem *ps) {
    init_rt_thread();
    struct rt_task params;
    init_rt_task_param(&params);
    params.exec_cost = ms2ns(10);
    params.cpu = 0;
    params.period = ms2ns(33);
    params.relative_deadline = ms2ns(15);
    params.budget_policy = QUANTUM_ENFORCEMENT;
    params.release_policy = TASK_SPORADIC;
    params.cls = RT_CLASS_HARD;

    CALL(set_rt_task_param(gettid(), &params));
    CALL(task_mode(LITMUS_RT_TASK));
    CALL(wait_for_ts_release());

    do {
        sleep_next_period();
        ps->get_frame();
    } while (!ps->should_exit());
}
void fetch_foreground(ProcessSystem *ps) {
    init_rt_thread();
    struct rt_task params;
    init_rt_task_param(&params);
    params.exec_cost = ms2ns(13);
    params.cpu = 0;
    params.period = ms2ns(33);
    params.relative_deadline = ms2ns(28);
    params.budget_policy = QUANTUM_ENFORCEMENT;
    params.release_policy = TASK_SPORADIC;
    params.cls = RT_CLASS_HARD;

    CALL(set_rt_task_param(gettid(), &params));
    CALL(task_mode(LITMUS_RT_TASK));
    CALL(wait_for_ts_release());
    do {
        sleep_next_period();
        ps->get_foreground();
    } while (!ps->should_exit());
}

void fetch_inference(ProcessSystem *ps) {
    CALL(init_rt_thread());
    struct rt_task params;
    init_rt_task_param(&params);
    params.exec_cost = ms2ns(10);
    params.cpu = 0;
    params.period = ms2ns(33);
    params.relative_deadline = ms2ns(33);
    params.budget_policy = QUANTUM_ENFORCEMENT;
    params.release_policy = TASK_PERIODIC;
    params.cls = RT_CLASS_HARD;

    CALL(set_rt_task_param(gettid(), &params));
    CALL(task_mode(LITMUS_RT_TASK));
    CALL(wait_for_ts_release());
    do {
        sleep_next_period();
        ps->inference();
    } while (!ps->should_exit());
}

void store_image(ProcessSystem *ps) {
    CALL(init_rt_thread());
    struct rt_task params;
    init_rt_task_param(&params);
    params.exec_cost = ms2ns(33);
    params.period = ms2ns(33);
    params.cpu = 2;
    // params.relative_deadline = ms2ns(300);
    params.budget_policy = NO_ENFORCEMENT;
    // params.release_policy = TASK_PERIODIC;
    params.cls = RT_CLASS_BEST_EFFORT;

    CALL(set_rt_task_param(gettid(), &params));
    CALL(task_mode(LITMUS_RT_TASK));
    CALL(wait_for_ts_release());
    do {
        sleep_next_period();
        ps->store_processed_image();
    } while (!ps->should_exit());
    ps->store_processed_image();
}

int main(int argc, char *argv[]) {
    CALL(init_litmus());

    ProcessSystem *ps = new ProcessSystem("result/", "models/HOGModel.svmopencv");
    ps->load_data("data/test/*");

    std::thread T1(fetch_frame, ps);
    std::thread T2(fetch_foreground, ps);
    std::thread T3(fetch_inference, ps);
    std::thread T4(store_image, ps);
    T1.join();
    T2.join();
    T3.join();
    T4.join();
}
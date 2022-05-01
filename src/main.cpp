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

#include "process_system.hpp"

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

void fetch_frame(ProcessSystem *ps) {
    struct rt_task params;
    init_rt_task_param(&params);
    params1.exec_cost = ms2ns(100);
    params1.cpu = 0;
    params1.period = ms2ns(200);
    params1.relative_deadline = ms2ns(200);
    params1.release_policy = TASK_PERIODIC;

    CALL(set_rt_task_param(gettid(), &params));
    CALL(task_mode(LITMUS_RT_TASK));
    CALL(wait_for_ts_release());

    do {
        sleep_next_period();
        ps->get_frame();
        std::cout << "Done job1" << std::endl;
    } while (!ps->should_exit());
}
void fetch_foreground(ProcessSystem *ps) {
    struct rt_task params;
    init_rt_task_param(&params);
    params1.exec_cost = ms2ns(100);
    params1.cpu = 0;
    params1.period = ms2ns(200);
    params1.relative_deadline = ms2ns(200);
    params1.release_policy = TASK_PERIODIC;

    CALL(set_rt_task_param(gettid(), &params));
    CALL(task_mode(LITMUS_RT_TASK));
    CALL(wait_for_ts_release());
    int do_exit = 0;
    do {
        sleep_next_period();
        ps->get_foreground();
        std::cout << "Done job2" << std::endl;
        printf("%d\n", do_exit);
    } while (!ps->should_exit());
}

void fetch_inference(ProcessSystem *ps) {
    struct rt_task params;
    init_rt_task_param(&params);
    params1.exec_cost = ms2ns(100);
    params1.cpu = 0;
    params1.period = ms2ns(200);
    params1.relative_deadline = ms2ns(200);
    params1.release_policy = TASK_PERIODIC;

    CALL(set_rt_task_param(gettid(), &params));
    CALL(task_mode(LITMUS_RT_TASK));
    CALL(wait_for_ts_release());
    int do_exit = 0;
    do {
        sleep_next_period();
        ps->get_foreground();
        std::cout << "Done job2" << std::endl;
        printf("%d\n", do_exit);
    } while (!ps->should_exit());
}

int main(int argc, char *argv[]) {
    CALL(init_litmus());

    ProcessSystem *ps = new ProcessSystem("result/", "models/HOGModel.svmopencv");
    ps->load_data("photos/test/*");

    std::thread first(fetch_frame, ps);
    std::thread second(fetch_foreground, ps);
    std::thread third(fetch_inference, ps);
    first.join();
    second.join();
    third.join();
}
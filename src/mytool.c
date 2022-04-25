#include <stdio.h>
#include <stdlib.h>

/* include the main liblitmus header */
#include <litmus.h>

int main(int argc, char** argv)
{
        struct rt_task param;
        int ready_rt_tasks, all_rt_tasks;

        init_litmus();
        init_rt_task_param(&param);

        /* Normally you would configure the task here, enter real-time mode, and start
         * the real-time computation, but we'll simply call a liblitmus function and
         * quit here as the goal is simply to test whether we can successfully link
         * against liblitmus.
         */
/* query the LITMUS^RT proc interface */
        if (!read_litmus_stats(&ready_rt_tasks, &all_rt_tasks)) {
                perror("read_litmus_stats");
                exit(1);
        }

        printf("There are %d real-time tasks waiting for a synchronous "
                "task system release.\n", ready_rt_tasks);
        printf("There are %d real-time tasks in total.\n", all_rt_tasks);

        return 0;
}

#include <condition_variable>
#include <iostream>
#include <mutex>
#include <opencv2/opencv.hpp>
#include <thread>
using namespace std;
using namespace cv;

Mat image;
mutex rdy_mtx;
condition_variable rdy_cond_var;

void job1(int a) {
    this_thread::sleep_for(chrono::seconds(5));
    {
        lock_guard<mutex> lg(rdy_mtx);
        image = Mat::ones(Size(3, 3), CV_8UC1);
    }
    rdy_cond_var.notify_one();
    this_thread::sleep_for(chrono::seconds(5));
    rdy_cond_var.notify_one();
    cout << "done with job1" << endl;
}
void job2(int b) {
    cout << "Started waiting....";
    unique_lock<mutex> ul(rdy_mtx);
    rdy_cond_var.wait(ul, [] {
        cout << "notified!!!" << endl;
        return !image.empty();
    });
    ul.unlock();

    cout << image << endl;

    rdy_cond_var.wait(ul, [] {
        cout << "notified!!!" << endl;
        return !image.empty();
    });
    cout << image << endl;
}
int main(int argc, char *argv[]) {
    thread t1(job1, 1);
    thread t2(job2, 2);
    t1.join();
    t2.join();
}
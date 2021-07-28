// test_cmd2.cpp : This file contains the 'main' function. Program execution begins and ends there.
//

#include <iostream>

#include "HandDetection.h"
#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <Image.h>
#include <iostream>
#include <opencv2/dnn.hpp>
#include <CF2Detection.h>
#include <CFDetectionParams.h>
#include <EcvFileSystem.h>

using namespace std;
using namespace cv;
using namespace hkc;

struct HandTF2MBNet320Params :public CFDetectionParams {
    HandTF2MBNet320Params() :

        //CFDetectionParams(R"(detection_models\checkpoint_mb2_320\saved_model)",
        CFDetectionParams(R"(assets\model\hand\detection_model\saved_model)",
            cv::Size(320, 320), 0.5f, CFDetectionOutputIndexs(5, 1, 4, 2), 1, 3, "hava khobe inja") {}

};

using ModelParams = HandTF2MBNet320Params;
using ModelParamsP = std::shared_ptr< ModelParams>;

void testImageOV() {
    const std::string image_filename = R"(E:\Database\HandDetection\1.png)";
    cv::Mat image = cv::imread(image_filename, 1);

    HandDetection hand_detector;
    std::vector<BaseDetectionOV::Result> result;
    auto t1 = std::chrono::steady_clock::now();
    for (int i = 0; i < 10; i++)
        result = hand_detector.detect(image);
    auto t2 = std::chrono::steady_clock::now();
    cout << std::chrono::duration<double, std::milli>(t2 - t1).count() / 10 << endl;

    hand_detector.draw(image, result);
    ecv::imshowScale("view", image, cv::Size(640, 640));
    cv::waitKey(0);

}


void testImageTF() {
    const std::string image_filename = R"(E:\Database\HandDetection\1.png)";
    cv::Mat image = cv::imread(image_filename, 1);

    CF2DetectionP hand_detector;
    ModelParamsP detector_params_;

    detector_params_ = std::make_shared< ModelParams>();
    detector_params_->model_path_ = ecv::getFolderAppSide<std::string>(detector_params_->model_path_);
    hand_detector = std::make_shared< CF2Detection>(detector_params_);

    hand_detector->setLabels({ "","Fist", "Like", "One", "Palm", "Two" });
    cv::Mat f;
    fitOnSizeMat3(image, f, cv::Size(640, 640));

    auto t1 = std::chrono::steady_clock::now();
    CFDetectionResultPS regions;
    for (int i = 0; i < 10; i++)
        regions = hand_detector->detection(f);
    auto t2 = std::chrono::steady_clock::now();

    cout << std::chrono::duration<double, std::milli>(t2 - t1).count() / 10 << endl;
    cv::Scalar region_color = CV_RGB(255, 0, 0);
    

    hand_detector->draw(f, regions, region_color, 5,2,2);

    ecv::imshowScale("view", f, cv::Size(640, 640));
    cv::waitKey(0);

}

void testVideoOV() {
    const std::string video_filename = R"(E:\Database\HandDetection\20210521_190043.mp4)";
    cv::VideoCapture capture(video_filename);
    HandDetection hand_detector;

    hand_detector.setLabels({"","Fist", "Like", "One", "Palm", "Two"});
    int i = 0;
    std::vector<BaseDetectionOV::Result> pre_result;
    while (capture.isOpened()) {
        i++;
        auto t11 = std::chrono::steady_clock::now();
        cv::Mat frame;
        capture >> frame;
        if (frame.empty())
            break;
        std::vector<BaseDetectionOV::Result> result;
        if (i % 1 == 0) {

            auto t1 = std::chrono::steady_clock::now();
            result = hand_detector.detect(frame);
            auto t2 = std::chrono::steady_clock::now();
            cout << std::chrono::duration<double, std::milli>(t2 - t1).count()  << endl;
            pre_result = result;

        }
       // else result = pre_result;
        hand_detector.draw(frame, result);
        ecv::imshowScale("view", frame, cv::Size(640, 640));
        cv::waitKey(0);
        std::this_thread::sleep_until(t11 + std::chrono::milliseconds(1000 / 30));

    }

    

}



void testVideoTF() {
    const std::string video_filename = R"(E:\Database\HandDetection\20210521_190043.mp4)";
    cv::VideoCapture capture(video_filename);

    CF2DetectionP hand_detector;
    ModelParamsP detector_params_;

    detector_params_ = std::make_shared< ModelParams>();
    detector_params_->model_path_ = ecv::getFolderAppSide<std::string>(detector_params_->model_path_);
    hand_detector = std::make_shared< CF2Detection>(detector_params_);

    hand_detector->setLabels({ "","Fist", "Like", "One", "Palm", "Two" });
  /*  cv::Mat f;
    fitOnSizeMat3(image, f, cv::Size(640, 640));*/


    int i = 0;
    std::vector<BaseDetectionOV::Result> pre_result;
    while (capture.isOpened()) {
        i++;
        auto t11 = std::chrono::steady_clock::now();
        cv::Mat frame;
        capture >> frame;
        if (frame.empty())
            break;
        
        CFDetectionResultPS regions;
        auto t1 = std::chrono::steady_clock::now();
        regions = hand_detector->detection(frame);
        auto t2 = std::chrono::steady_clock::now();
        cout << std::chrono::duration<double, std::milli>(t2 - t1).count() << endl;

         cv::Scalar region_color = CV_RGB(255, 0, 0);
         hand_detector->draw(frame, regions, region_color, 5, 2, 2);

        ecv::imshowScale("view", frame, cv::Size(640, 640));
        cv::waitKey(0);
        std::this_thread::sleep_until(t11 + std::chrono::milliseconds(1000 / 30));

    }



}
int main()
{
    //testImageOV();
    testVideoOV();
   // testVideoAsync();

    //testImageTF();
    //testVideoTF();
}


#pragma once
#include "ov_utility.h"
#include "encrypt.h"
#include <inference_engine.hpp>
#include <opencv2/core.hpp>
#include "ov_global.h"
#include <opencv2/imgproc.hpp>

using CoreP = std::shared_ptr<ie::Core>;
class OV_EXPORT DetectionParams
{
public:
    DetectionParams(const std::string& topo_name,
        const std::string& path_to_model,
        const std::string& path_to_bin,
        const std::string& device_for_inference,
        int max_batch,
        bool is_batch_dynamic,
        bool is_async,
        bool do_raw_output_messages,
        float detection_threshold = 0.6f,
        bool encrypt_mode = false,
        ecv::EncryptMethod encrypt_method = ecv::EncryptMethod::method2,
        const std::string& key_ = "hava khobe inja");
    virtual ~DetectionParams() {}

    std::string topo_name_;
    std::string path_to_model_;
    std::string path_to_bin_;
    std::string device_for_inference_;
    int max_batch_;
    bool is_batch_dynamic_;
    bool is_async_;
    bool do_raw_output_messages_;
    float detection_threshold_;
    bool encrypt_mode_;
    ecv::EncryptMethod encrypt_method_;
    std::string key_;
   
};
using DetectionParamsP = std::shared_ptr< DetectionParams>;

class OV_EXPORT BaseDetectionOV {
public:
    struct Result {
        int label_;
        float confidence_;
        cv::Rect location_;
    };
    using ResultS = std::vector< Result>;
    
    
    BaseDetectionOV(const DetectionParamsP& params,const CoreP& core = nullptr);
    virtual ~BaseDetectionOV() {}

    
    ie::ExecutableNetwork net_;
    ie::ExecutableNetwork* operator ->();

    std::vector<Result> results_;

    void submitRequest();
    virtual void wait();
    bool enabled() const;
    void printPerformanceCounts(std::string fullDeviceName);
    ie::CNNNetwork read(const ie::Core& core);

    

    void enqueue(const cv::Mat& frame);
    void fetchResults();

    std::vector<Result> detect(const cv::Mat& image);
    void draw(cv::Mat& frame, const std::vector<Result>& results, const cv::Scalar& color = CV_RGB(0,255,0), int thickness = 12, int font_thickness =2,
        float font_scale = 3.f);

    void setLabels(const std::vector<std::string>& labels);
protected:
    std::vector<std::string> labels_;
    CoreP core_;
    DetectionParamsP params_;
    
    ie::InferRequest request_;
    mutable bool enabling_checked_;
    mutable bool enabled_;

    size_t network_input_width_;
    size_t network_input_height_;

    std::string input_;
    std::string output_;

    int max_proposal_count_;
    int object_size_;
    std::string labels_output_;


    int enqued_frames_;
    float width_;
    float height_;

    bool results_fetched_;
    
   
    void checkInput(ie::CNNNetwork& network);
    void checkOutput(ie::CNNNetwork& network);

    void createCore();

};

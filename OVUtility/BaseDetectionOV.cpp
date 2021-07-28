#include "BaseDetectionOV.h"
#include <samples/slog.hpp>
#include <samples/ocv_common.hpp>
#include <encrypt.h>
#include <EcvFileSystem.h>
#include <EcvUtility.h>
#include <EcvDetectionResult.h>
#include <iostream>

using namespace InferenceEngine;
using namespace std;

DetectionParams::DetectionParams(const std::string& topo_name, const std::string& path_to_model, const std::string& path_to_bin, 
    const std::string& device_for_inference,
    int max_batch, bool is_batch_dynamic, bool is_async, bool do_raw_output_messages, float detection_threshold, 
    bool encrypt_mode, ecv::EncryptMethod encrypt_method, const std::string& key_):
    topo_name_{ topo_name},
    path_to_model_{ path_to_model},
    path_to_bin_{ path_to_bin },
    device_for_inference_{ device_for_inference},
    max_batch_{ max_batch},
    is_batch_dynamic_{ is_batch_dynamic }, 
    is_async_{ is_async }, 
    do_raw_output_messages_{ do_raw_output_messages }, 
    detection_threshold_{ detection_threshold },
    encrypt_mode_{ encrypt_mode }, 
    encrypt_method_{ encrypt_method }, 
    key_{ key_ }
{
}

BaseDetectionOV::BaseDetectionOV(const DetectionParamsP& params, const CoreP& core ):
    params_{params},
    core_{core},
    enabling_checked_(false), 
    enabled_(false),
    max_proposal_count_(0),
    object_size_(0),
    network_input_width_(0),
    network_input_height_(0),
    enqued_frames_(0),
    width_(0),
    height_(0),
    results_fetched_(false)

     {
    createCore();
   
}



ie::ExecutableNetwork* BaseDetectionOV::operator ->() {
    return &net_;
}

void BaseDetectionOV::submitRequest() {
    if (!enqued_frames_)
        return;
    enqued_frames_ = 0;
    results_fetched_ = false;
    results_.clear();


    if (!enabled() ) 
        return;
    if (params_->is_async_) 
        request_.StartAsync();
    
    else 
        request_.Infer();
    
}

void BaseDetectionOV::wait() {
    if (!enabled() || !request_ || !params_->is_async_)
        return;
    request_.Wait(ie::InferRequest::WaitMode::RESULT_READY);
}

bool BaseDetectionOV::enabled() const {
    if (!enabling_checked_) {
        enabled_ = !params_->path_to_model_.empty();
        if (!enabled_) {
            //slog::info << topo_name_ << " DISABLED" << slog::endl;
        }
        enabling_checked_ = true;
    }
    return enabled_;
}

void BaseDetectionOV::printPerformanceCounts(std::string fullDeviceName) {
    if (!enabled()) 
        return;
    
    //slog::info << "Performance counts for " << topo_name_ << slog::endl << slog::endl;
    ::printPerformanceCounts(request_, std::cout, fullDeviceName, false);
}



ie::CNNNetwork BaseDetectionOV::read(const ie::Core& core) {


    ie::CNNNetwork network;
    if (params_->encrypt_mode_) {
        //std::string bin_filename = ecv::changeFileExtension(params_->path_to_model_, ".dat2");

        std::vector<unsigned char> model;
        // ecv::readBinaryFile(path_to_model_, model);
        ecv::decryptFile2Data(params_->path_to_model_, model, ecv::str2CharVec(params_->key_), params_->encrypt_method_);

        std::vector<unsigned char> weights;
        //ecv::readBinaryFile(bin_filename, weights);
        ecv::decryptFile2Data(params_->path_to_bin_, weights, ecv::str2CharVec(params_->key_), params_->encrypt_method_);//"hava khobe inja"


        std::string strModel(model.begin(), model.end());

        network = core.ReadNetwork(strModel,
            InferenceEngine::make_shared_blob<uint8_t>({ InferenceEngine::Precision::U8,
                {weights.size()}, InferenceEngine::C }, weights.data()));

    }
    else {
        network = core.ReadNetwork(params_->path_to_model_);
    }



    /** Set batch size to 1 **/
    //slog::info << "Batch size is set to " << max_batch_ << slog::endl;
    network.setBatchSize(params_->max_batch_);

    checkInput(network);
    checkOutput(network);

    return network;
}


void BaseDetectionOV::checkInput(ie::CNNNetwork& network)
{
    // slog::info << "Checking Face Detection network inputs" << slog::endl;
    InputsDataMap input_info(network.getInputsInfo());
    if (input_info.size() != 1) {
        throw std::logic_error("Face Detection network should have only one input");
    }
    InputInfo::Ptr input_info_first = input_info.begin()->second;
    input_info_first->setPrecision(Precision::U8);

    const SizeVector input_dims = input_info_first->getTensorDesc().getDims();
    network_input_height_ = input_dims[2];
    network_input_width_ = input_dims[3];

    //slog::info << "Loading Face Detection model to the " << device_for_inference_ << " device" << slog::endl;
    input_ = input_info.begin()->first;

}

void BaseDetectionOV::checkOutput(ie::CNNNetwork& network)
{
    // ---------------------------Check outputs ------------------------------------------------------------
    //slog::info << "Checking Face Detection network outputs" << slog::endl;
    OutputsDataMap output_info(network.getOutputsInfo());
    if (output_info.size() == 1) {
        DataPtr& output = output_info.begin()->second;
        output_ = output_info.begin()->first;

        const SizeVector output_dims = output->getTensorDesc().getDims();
        max_proposal_count_ = output_dims[2];
        object_size_ = output_dims[3];

        if (object_size_ != 7) {
            throw std::logic_error("Face Detection network output layer should have 7 as a last dimension");
        }

        if (output_dims.size() != 4) {
            throw std::logic_error("Face Detection network output dimensions not compatible shoulld be 4, but was " +
                std::to_string(output_dims.size()));
        }
        output->setPrecision(Precision::FP32);
    }
    else {
        for (const auto& output_layer : output_info) {
            const SizeVector output_dims = output_layer.second->getTensorDesc().getDims();
            if (output_dims.size() == 2 && output_dims.back() == 5) {
                output_ = output_layer.first;
                max_proposal_count_ = output_dims[0];
                object_size_ = output_dims.back();
                output_layer.second->setPrecision(Precision::FP32);
            }
            else if (output_dims.size() == 1 && output_layer.second->getPrecision() == Precision::I32) {
                labels_output_ = output_layer.first;
            }
        }
        if (output_.empty() || labels_output_.empty()) {
            throw std::logic_error("Face Detection network must contain ether single DetectionOutput or "
                "'boxes' [nx5] and 'labels' [n] at least, where 'n' is a number of detected objects.");
        }
    }

}

void BaseDetectionOV::createCore()
{
    if (core_ == nullptr) {
        core_ = std::make_shared<ie::Core>();

        Load(*this).into(*core_, params_->device_for_inference_, false);
    }
}


void BaseDetectionOV::enqueue(const cv::Mat& frame) {
    if (!enabled()) return;

    if (!request_) {
        request_ = net_.CreateInferRequest();
    }

    width_ = static_cast<float>(frame.cols);
    height_ = static_cast<float>(frame.rows);

    Blob::Ptr  inputBlob = request_.GetBlob(input_);

    matU8ToBlob<uint8_t>(frame, inputBlob);

    enqued_frames_ = 1;
}



void BaseDetectionOV::fetchResults() {
    
    if (!enabled())
        return ;

    results_.clear();
    if (results_fetched_)
        return ;

    results_fetched_ = true;
    LockedMemory<const void> output_mapped = as<MemoryBlob>(request_.GetBlob(output_))->rmap();
    const float* detections = output_mapped.as<float*>();

    if (!labels_output_.empty()) {
        LockedMemory<const void> labels_mapped = as<MemoryBlob>(request_.GetBlob(labels_output_))->rmap();
        const int32_t* labels = labels_mapped.as<int32_t*>();

        for (int i = 0; i < max_proposal_count_ && object_size_ == 5; i++) {
            Result r;
            r.label_ = labels[i];
            r.confidence_ = detections[i * object_size_ + 4];

            if (r.confidence_ <= params_->detection_threshold_ && !params_->do_raw_output_messages_) {
                continue;
            }

            r.location_.x = static_cast<int>(detections[i * object_size_ + 0] / network_input_width_ * width_);
            r.location_.y = static_cast<int>(detections[i * object_size_ + 1] / network_input_height_ * height_);
            r.location_.width = static_cast<int>(detections[i * object_size_ + 2] / network_input_width_ * width_ - r.location_.x);
            r.location_.height = static_cast<int>(detections[i * object_size_ + 3] / network_input_height_ * height_ - r.location_.y);

            //// Make square and enlarge face bounding box for more robust operation of face analytics networks
            //int bb_width = r.location_.width;
            //int bb_height = r.location_.height;

            //int bb_center_x = r.location_.x + bb_width / 2;
            //int bb_center_y = r.location_.y + bb_height / 2;

            //int max_of_sizes = std::max(bb_width, bb_height);

            //int bb_new_width = static_cast<int>(params()->bb_enlarge_coefficient_ * max_of_sizes);
            //int bb_new_height = static_cast<int>(params()->bb_enlarge_coefficient_ * max_of_sizes);

            //r.location_.x = bb_center_x - static_cast<int>(std::floor(params()->bb_dx_coefficient_ * bb_new_width / 2));
            //r.location_.y = bb_center_y - static_cast<int>(std::floor(params()->bb_dy_coefficient_ * bb_new_height / 2));

            //r.location_.width = bb_new_width;
            //r.location_.height = bb_new_height;

            if (params_->do_raw_output_messages_) {
                std::cout << "[" << i << "," << r.label_ << "] element, prob = " << r.confidence_ <<
                    "    (" << r.location_.x << "," << r.location_.y << ")-(" << r.location_.width << ","
                    << r.location_.height << ")"
                    << ((r.confidence_ > params_->detection_threshold_) ? " WILL BE RENDERED!" : "") << std::endl;
            }
            if (r.confidence_ > params_->detection_threshold_) {
                results_.push_back(r);
            }
        }
    }

    for (int i = 0; i < max_proposal_count_ && object_size_ == 7; i++) {
        float image_id = detections[i * object_size_ + 0];
        if (image_id < 0) {
            break;
        }
        Result r;
        r.label_ = static_cast<int>(detections[i * object_size_ + 1]);
        r.confidence_ = detections[i * object_size_ + 2];
        if (r.confidence_ <= params_->detection_threshold_ && !params_->do_raw_output_messages_) {
            continue;
        }
 /*       cout << "label:" << r.label_ << endl;
        cout << "confidence:" << r.confidence_ << endl;*/


        r.location_.x = static_cast<int>(detections[i * object_size_ + 3] * width_);
        r.location_.y = static_cast<int>(detections[i * object_size_ + 4] * height_);
        r.location_.width = static_cast<int>(detections[i * object_size_ + 5] * width_ - r.location_.x);
        r.location_.height = static_cast<int>(detections[i * object_size_ + 6] * height_ - r.location_.y);

       // // Make square and enlarge face bounding box for more robust operation of face analytics networks
       // int bb_width = r.location_.width;
       // int bb_height = r.location_.height;

       // int bb_center_x = r.location_.x + bb_width / 2;
       // int bb_center_y = r.location_.y + bb_height / 2;

       // int max_of_sizes = std::max(bb_width, bb_height);

       // int bb_new_width = static_cast<int>(params_->bb_enlarge_coefficient_ * max_of_sizes);
       // int bb_new_height = static_cast<int>(params_->bb_enlarge_coefficient_ * max_of_sizes);

       // /*       r.location.x = bb_center_x - static_cast<int>(std::floor(bb_dx_coefficient * bb_new_width / 2));
       //        r.location.y = bb_center_y - static_cast<int>(std::floor(bb_dy_coefficient * bb_new_height / 2));

       //        r.location.width = bb_new_width;
       //        r.location.height = bb_new_height;
       //*/

       // r.location_.x = bb_center_x - static_cast<int>(std::floor(bb_width / 2));
       // r.location_.y = bb_center_y - static_cast<int>(std::floor(bb_height / 2));

       // r.location_.width = bb_width;
       // r.location_.height = bb_height;



        if (params_->do_raw_output_messages_) {
            /*std::cout << "[" << i << "," << r.label_ << "] element, prob = " << r.confidence_ <<
                "    (" << r.location_.x << "," << r.location_.y << ")-(" << r.location_.width << ","
                << r.location_.height << ")"
                << ((r.confidence_ > params_->detection_threshold_) ? " WILL BE RENDERED!" : "") << std::endl;*/
        }
        if (r.confidence_ > params_->detection_threshold_) {
            results_.push_back(r);
        }
    }
}

std::vector<BaseDetectionOV::Result> BaseDetectionOV::detect(const cv::Mat& image)
{
    enqueue(image);
    submitRequest();

    
    wait();
    fetchResults();
    return results_;


}

void BaseDetectionOV::draw(cv::Mat& frame, const std::vector<Result>& results, const cv::Scalar& color, int thickness, int font_thickness , float font_scale )
{
   if (labels_.size())
    for (auto& result : results)
           ecv::drawTitleRect(frame, result.location_,labels_[result.label_], color, thickness,font_thickness,font_scale);

   else 
   for (auto& result : results)
       cv::rectangle(frame, result.location_, color, thickness);

    
}

void BaseDetectionOV::setLabels(const std::vector<std::string>& labels)
{
    labels_ = labels;
}





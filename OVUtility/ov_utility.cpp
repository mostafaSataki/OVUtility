#include "ov_utility.h"
#include "BaseDetectionOV.h"

using namespace InferenceEngine;

Load::Load(BaseDetectionOV& detector) : 
    detector_(detector) {
}

void Load::into(ie::Core& ie, const std::string& device_name, bool enable_dynamic_batch) const {
    if (detector_.enabled()) {
        std::map<std::string, std::string> config = { };

        bool is_possible_dyn_batch = device_name.find("CPU") != std::string::npos ||
            device_name.find("GPU") != std::string::npos;

        if (enable_dynamic_batch && is_possible_dyn_batch) {
            config[PluginConfigParams::KEY_DYN_BATCH_ENABLED] = PluginConfigParams::YES;
        }

        detector_.net_ = ie.LoadNetwork(detector_.read(ie), device_name, config);
    }
}


CallStat::CallStat() :
    number_of_calls_(0),
    total_duration_(0.0), 
    last_call_duration_(0.0), 
    smoothed_duration_(-1.0) {
}

double CallStat::getSmoothedDuration() {
    // Additional check is needed for the first frame while duration of the first
    // visualisation is not calculated yet.
    if (smoothed_duration_ < 0) {
        auto t = std::chrono::high_resolution_clock::now();
        return std::chrono::duration_cast<ms>(t - last_call_start_).count();
    }
    return smoothed_duration_;
}

double CallStat::getTotalDuration() {
    return total_duration_;
}

double CallStat::getLastCallDuration() {
    return last_call_duration_;
}

void CallStat::calculateDuration() {

    auto t = std::chrono::high_resolution_clock::now();

    last_call_duration_ = std::chrono::duration_cast<ms>(t - last_call_start_).count();
    number_of_calls_++;
    total_duration_ += last_call_duration_;
    
    if (smoothed_duration_ < 0) 
        smoothed_duration_ = last_call_duration_;
    
    double alpha = 0.1;
    smoothed_duration_ = smoothed_duration_ * (1.0 - alpha) + last_call_duration_ * alpha;
}

void CallStat::setStartTime() {
    last_call_start_ = std::chrono::high_resolution_clock::now();
}


void Timer::start(const std::string& name) {
    if (_timers.find(name) == _timers.end()) {
        _timers[name] = CallStat();
    }
    _timers[name].setStartTime();
}

void Timer::finish(const std::string& name) {
    auto& timer = (*this)[name];
    timer.calculateDuration();
}

CallStat& Timer::operator[](const std::string& name) {
    if (_timers.find(name) == _timers.end()) {
        throw std::logic_error("No timer with name " + name + ".");
    }
    return _timers[name];
}

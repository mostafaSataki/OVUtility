#pragma once
#include <inference_engine.hpp>

#include <chrono>

namespace ie = InferenceEngine;

class BaseDetectionOV;

struct Load {
	BaseDetectionOV& detector_;

	explicit Load(BaseDetectionOV& detector);

	void into(ie::Core& ie, const std::string& device_name, bool enable_dynamic_batch = false) const;
};

class CallStat {
public:
	typedef std::chrono::duration<double, std::ratio<1, 1000>> ms;

	CallStat();

	double getSmoothedDuration();
	double getTotalDuration();
	double getLastCallDuration();
	void calculateDuration();
	void setStartTime();

private:
	size_t number_of_calls_;
	double total_duration_;
	double last_call_duration_;
	double smoothed_duration_;
	std::chrono::time_point<std::chrono::high_resolution_clock> last_call_start_;
};

class Timer {
public:
	void start(const std::string& name);
	void finish(const std::string& name);
	CallStat& operator[](const std::string& name);

private:
	std::map<std::string, CallStat> _timers;
};
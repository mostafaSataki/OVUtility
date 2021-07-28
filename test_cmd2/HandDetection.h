#pragma once

#include <opencv2/core.hpp>
#include <BaseDetectionOV.h>

class HandDetectionParams :public DetectionParams {
public:
    HandDetectionParams();

};

using HandDetectionParamsP = std::shared_ptr< HandDetectionParams>;

class HandDetection : public  BaseDetectionOV
{
public:
    HandDetection();
};


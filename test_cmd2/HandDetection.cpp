#include "HandDetection.h"
#include <EcvFileSystem.h>

HandDetectionParams::HandDetectionParams():
	DetectionParams("HandDetection",ecv::folderAppSide<std::string>(R"(assets\model\saved_model.xml)"), ecv::folderAppSide<std::string>(R"(assets\model\saved_model.bin)"),"CPU",1,false,false,false,0.5)
{
}

HandDetection::HandDetection():
	BaseDetectionOV(std::make_shared<HandDetectionParams>())
{
}

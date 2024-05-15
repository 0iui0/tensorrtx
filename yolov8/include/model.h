#pragma once
#include <assert.h>
#include <string>
#include "NvInfer.h"

nvinfer1::IHostMemory* buildEngineYolov8Pose(nvinfer1::IBuilder* builder, nvinfer1::IBuilderConfig* config,
                                             nvinfer1::DataType dt, const std::string& wts_path, float& gd, float& gw,
                                             int& max_channels);

nvinfer1::IHostMemory* buildEngineYolov8PoseP6(nvinfer1::IBuilder* builder, nvinfer1::IBuilderConfig* config,
                                               nvinfer1::DataType dt, const std::string& wts_path, float& gd, float& gw,
                                               int& max_channels);

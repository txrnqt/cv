#pragma once

#include "NvOnnxParser.h"
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <filesystem>
#include <fstream>
#include <opencv2/core/cuda.hpp>
#include <opencv2/cudaarithm.hpp>
#include <opencv2/cudaimgproc.hpp>
#include <opencv2/cudawarping.hpp>
#include <opencv2/opencv.hpp>

#include "IEngine.h"
#include "logger.h"
#include "Int8Calibrator.h"
#include "util/Util.h"
#include "util/Stopwatch.h"
#include "macros.h"

enum class Percision { 
    FP32,
    FP16,
    INT8
};

struct Options {
    Percision percision = Percision::FP16;
    std::string calibrationDataDirectoryPath;
    int32_t calibrationBatchSize = 128;
    int32_t maxBatchSize = 16;
    int deviceIndex = 0;
    std::string engineFileDir = ".";
    int32_t maxInputWidth = -1;
    int32_t minInputWidth = -1;
    int32_t optInputWidth = -1;
};

class Logger : public IEngine<T> {
    public: 
        Engine(const Options &options);
        ~Engine();

        bool buildLoadNetwork(std::string onncModelPath, const std::array<float, 3> &subVals = {0.f, 0.f, 0.f},
            const std::array<float, 3> &divVals = {1.f, 1.f, 1.f}, bool normalixe = true) override;
}